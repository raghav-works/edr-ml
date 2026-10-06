"""
Cortex-Static feature extractor — EMBER v3 / EMBER2024-compatible, 2568-dim.
=============================================================================

Formula-compatible with the real EMBER2024 / "thrember" reference feature
extractor (Joyce et al., 2025; FutureComputing4AI/EMBER2024, Apache-2.0):
same 12 feature groups, same per-group dimension budget, same fixed
vocabularies (regex categories, machine/subsystem/characteristics lists,
data-directory order, pefile-warning normalization), same hashing schemes.
This is a from-scratch reimplementation, not a copy of that codebase, but it
deliberately matches its formulas exactly rather than improvising -- the
2568-dim vector produced here from raw PE bytes at inference time must be
numerically compatible, column for column, with the vector produced by
features/ember2024_adapter.py from an EMBER2024 training record. Any
divergence between the two would silently corrupt a feature group after
deployment: the model learns one semantics for a column at training time and
gets fed a different one at inference time, and that kind of skew doesn't
show up in offline eval.

Feature groups (sum = 2568):
    general            7    size, entropy, is_pe, first 4 raw bytes
    histogram        256    normalized byte histogram
    byteentropy      256    2D byte/local-entropy joint histogram
    strings          177    string stats + printable-char dist + 77 fixed IOC/keyword regex counts
    header            74    DOS/COFF/Optional header fields (fixed-vocabulary categoricals)
    section          224    per-section stats, feature-hashed
    imports         1282    hashed import libraries + lib:function pairs
    exports          129    hashed export function names
    datadirectories   34    size/RVA of 16 data directories + 2 summary stats
    richheader        33    hashed (compid, count) pairs from the Rich header
    authenticode       8    Authenticode signature summary
    pefilewarnings    88    fixed-vocabulary pefile parse-warning bag + count

Requires: pefile, numpy, scikit-learn (FeatureHasher), and signify>=0.9,<0.10
(Authenticode parsing). signify is NOT optional: without it the 8-dim
authenticode group would zero out, skewing static scores on signed binaries,
so PEFeatureExtractor() raises at construction if it is missing.

Feature-group degradation (review item 6): a group that raises during
extraction is no longer silently swallowed into a zero vector. It is logged
at WARNING and reported -- feature_vector_with_report() returns the list of
degraded groups, and self_test() checks the extractor against a known-good
signed PE at startup. CRITICAL_FEATURE_GROUPS names the groups whose zero-fill
fabricates or erases a primary maliciousness signal; the pipeline routes a
scan with any degraded critical group to NEEDS_REVIEW rather than trusting the
score.
"""

from __future__ import annotations

import hashlib
import io
import logging
import math
import re
from abc import ABC, abstractmethod
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.feature_extraction import FeatureHasher

logger = logging.getLogger("cortex.features.pe")

try:
    import pefile
    _PEFILE_AVAILABLE = True
except ImportError:
    _PEFILE_AVAILABLE = False

# signify 0.9.x renamed the PE entry point: the old `SignedPEFile` (used up
# to ~0.8.x, which cortex-endpoint pins and monkey-patches) is now
# `AuthenticodeFile`, opened via `AuthenticodeFile.from_stream(...)` and
# iterated with `iter_signatures()` instead of `iter_signed_datas()`. We port
# forward to the installed 0.9.x API rather than pinning back to 0.8.1 (that
# version needs a CertificateStore.__getitem__ monkey-patch).
#
# If this import fails, _SIGNIFY_AVAILABLE stays False and PEFeatureExtractor()
# raises at construction (see __init__) -- deliberately a HARD failure, not a
# silent all-zeros authenticode group. Silent degradation here is a
# train/serve skew that inflates static scores on signed binaries; it
# happened once (against signify 0.9.2's rename) and went undetected until a
# manual 5-file scan comparison against cortex-endpoint caught it. A future
# signify rename must break loudly. (requirements.txt pins signify>=0.9,<0.10.)
try:
    from signify.authenticode import AuthenticodeFile
    _SIGNIFY_AVAILABLE = True
except ImportError:
    _SIGNIFY_AVAILABLE = False

EMBER2024_FEATURE_COUNT = 2568
_WARNINGS_FILE = Path(__file__).parent / "pefile_warnings.txt"

# Feature groups whose silent zero-fill fabricates or erases a PRIMARY
# maliciousness signal (review item 6). If any of these degrades on a live
# scan, the static score is untrustworthy in an unknown direction and the
# pipeline routes the scan to NEEDS_REVIEW instead of acting on it
# (inference/pipeline.py). The three EXCLUDED groups -- exports, richheader,
# pefilewarnings -- are the ones where an all-zero vector is also a common
# LEGITIMATE value (an EXE with no exports, a non-MSVC binary with no Rich
# header, a clean parse with no warnings), so their degradation is
# in-distribution and recorded for telemetry only.
CRITICAL_FEATURE_GROUPS = frozenset({
    "general", "histogram", "byteentropy", "strings",
    "header", "section", "imports", "datadirectories", "authenticode",
})

# Known-good signed Windows PE for PEFeatureExtractor.self_test(). Single
# source of truth -- the committed test fixture, not a duplicated copy.
_SELFTEST_PE = (
    Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "pe_samples" / "sample_signed64.exe"
)


class FeatureGroup(ABC):
    name: str = ""
    dim: int = 0

    @abstractmethod
    def raw_features(self, bytez: bytes, pe: Any) -> Any: ...

    @abstractmethod
    def process_raw_features(self, raw: Any) -> np.ndarray: ...

    def __repr__(self) -> str:
        return f"{self.name}({self.dim})"


# ─────────────────────────────────────────────────────────────────────────
class GeneralFileInfo(FeatureGroup):
    name, dim = "general", 3 + 4

    def raw_features(self, bytez, pe):
        size = len(bytez)
        counts = Counter(bytez)
        entropy = 0.0
        for c in counts.values():
            p = c / size if size else 0.0
            if p > 0:
                entropy -= p * math.log2(p)
        start_bytes = [bytez[i] if i < size else 0 for i in range(4)]
        return {"size": size, "entropy": entropy, "is_pe": int(pe is not None), "start_bytes": start_bytes}

    def process_raw_features(self, raw):
        return np.hstack([raw["size"], raw["entropy"], raw["is_pe"], raw["start_bytes"]]).astype(np.float32)


class ByteHistogram(FeatureGroup):
    name, dim = "histogram", 256

    def raw_features(self, bytez, pe):
        return np.bincount(np.frombuffer(bytez, dtype=np.uint8), minlength=256).tolist()

    def process_raw_features(self, raw):
        c = np.array(raw, dtype=np.float32)
        s = c.sum()
        return c / s if s > 0 else c


class ByteEntropyHistogram(FeatureGroup):
    name, dim = "byteentropy", 256

    def __init__(self, window: int = 2048, step: int = 1024):
        self.window, self.step = window, step

    def _entropy_bin_counts(self, block: np.ndarray):
        c = np.bincount(block >> 4, minlength=16)
        p = c.astype(np.float64) / self.window
        nz = p > 0
        H = float(-np.sum(p[nz] * np.log2(p[nz]))) * 2.0
        hbin = int(H * 2)
        if hbin == 16:
            hbin = 15
        return hbin, c

    def raw_features(self, bytez, pe):
        out = np.zeros((16, 16), dtype=np.int64)
        a = np.frombuffer(bytez, dtype=np.uint8)
        if a.shape[0] < self.window:
            if a.shape[0] > 0:
                hbin, c = self._entropy_bin_counts(a)
                out[hbin, :] += c
        else:
            shape = (a.shape[0] - self.window + 1, self.window)
            strides = (a.strides[0], a.strides[0])
            blocks = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)[:: self.step, :]
            for block in blocks:
                hbin, c = self._entropy_bin_counts(block)
                out[hbin, :] += c
        return out.flatten().tolist()

    def process_raw_features(self, raw):
        c = np.array(raw, dtype=np.float32)
        s = c.sum()
        return c / s if s > 0 else c


class StringExtractor(FeatureGroup):
    """Fixed 77-regex IOC/keyword dictionary + fixed sorted-index encoding.

    This exact vocabulary (not a hashed bag of raw strings) is what
    EMBER2024's own `string_counts` field is -- keeping it here identical to
    the reference implementation is what lets features/ember2024_adapter.py
    consume a training record's precomputed string_counts directly and get
    the same 76 [sic -- see dim comment] category columns this class would
    produce from live PE bytes.
    """

    name, dim = "strings", 3 + 96 + 1 + 77  # numstrings/avlength/printables + dist(96) + entropy + 77 categories
    _ALLSTRINGS = re.compile(rb"[\x20-\x7f]{5,}")

    # Verbatim from the EMBER2024 reference extractor (thrember/features.py)
    # so live-parsed PE bytes and EMBER2024 training records land on the
    # exact same category vocabulary.
    _REGEXES = {
        "url": re.compile(r"\b(?:http|https|ftp):\/\/[a-zA-Z0-9-._~:?#[\]@!$&'()*+,;=]+"),
        "ipv4_addr": re.compile(r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"),
        "ipv6_addr": re.compile(r"\b(?:[A-Fa-f0-9]{1,4}:){7}[A-Fa-f0-9]{1,4}\b|\b(?:[A-Fa-f0-9]{1,4}:){1,7}:\b|\b:[A-Fa-f0-9]{1,4}(?::[A-Fa-f0-9]{1,4}){1,6}\b"),
        "mac_addr": re.compile(r"\b(?:[0-9A-Fa-f]{2}[:-]){5}(?:[0-9A-Fa-f]{2})\b"),
        "email_addr": re.compile(r"\b(?:[0-9A-Fa-f]{2}[:-]){5}(?:[0-9A-Fa-f]{2})\b"),
        "btc_wallet": re.compile(r"[13][a-km-zA-HJ-NP-Z1-9]{25,34}"),
        "file_path": re.compile(r"\bC:/"),
        "dos_msg": re.compile(r"!This program "),
        "registry_key": re.compile(r"\b(?:KHEY_|KHLM|HKCU)"),
        "/dev/": re.compile(r"/dev/"),
        "/proc/": re.compile(r"/proc/"),
        "/bin/": re.compile(r"/bin/"),
        "/usr/": re.compile(r"/usr/"),
        "/tmp/": re.compile(r"/tmp/"),
        "/URI": re.compile(r"/URI"),
        "/FlateDecode": re.compile(r"/FlateDecode"),
        "/EmbeddedFile": re.compile(r"/EmbeddedFile"),
        "html": re.compile(r"html", re.IGNORECASE),
        "javascript": re.compile(r"javascript", re.IGNORECASE),
        "<script": re.compile(r"<script", re.IGNORECASE),
        ".click(": re.compile(r".click", re.IGNORECASE),
        "onlick": re.compile(r"onclick", re.IGNORECASE),
        "powershell": re.compile(r"powershell", re.IGNORECASE),
        "Invoke-Expression": re.compile(r"Invoke-Expression"),
        "Invoke-Command": re.compile(r"Invoke-Command"),
        "Start-process": re.compile(r"Start-process"),
        "get": re.compile(r"GET /", re.IGNORECASE),
        "post": re.compile(r"POST /", re.IGNORECASE),
        "http": re.compile(r"HTTP/", re.IGNORECASE),
        "http://": re.compile(r"http://", re.IGNORECASE),
        "https://": re.compile(r"https://", re.IGNORECASE),
        "ftp": re.compile(r"ftp:", re.IGNORECASE),
        "useragent": re.compile(r"User-Agent", re.IGNORECASE),
        "cookie": re.compile(r"cookie", re.IGNORECASE),
        "internet": re.compile(r"internet", re.IGNORECASE),
        "download": re.compile(r"download", re.IGNORECASE),
        "connect": re.compile(r"connect", re.IGNORECASE),
        "base64": re.compile(r"base64", re.IGNORECASE),
        "base64string": re.compile(r"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"),
        "crypt": re.compile(r"crypt"),
        "encode": re.compile(r"encode", re.IGNORECASE),
        "decode": re.compile(r"decode", re.IGNORECASE),
        "cache": re.compile(r"cache", re.IGNORECASE),
        "certificate": re.compile(r"certificate", re.IGNORECASE),
        "clipboard": re.compile(r"clipboard", re.IGNORECASE),
        "command": re.compile(r"command", re.IGNORECASE),
        "create": re.compile(r"create", re.IGNORECASE),
        "debug": re.compile(r"debug", re.IGNORECASE),
        "delete": re.compile(r"delete", re.IGNORECASE),
        "desktop": re.compile(r"desktop", re.IGNORECASE),
        "directory": re.compile(r"directory", re.IGNORECASE),
        "disk": re.compile(r"disk", re.IGNORECASE),
        "environment": re.compile(r"environment", re.IGNORECASE),
        "enum": re.compile(r"enum", re.IGNORECASE),
        "exit": re.compile(r"exit", re.IGNORECASE),
        "file": re.compile(r"file", re.IGNORECASE),
        "hostname": re.compile(r"hostname", re.IGNORECASE),
        "install": re.compile(r"install", re.IGNORECASE),
        "hidden": re.compile(r"hidden", re.IGNORECASE),
        "keyboard": re.compile(r"keyboard", re.IGNORECASE),
        "memory": re.compile(r"memory", re.IGNORECASE),
        "module": re.compile(r"module", re.IGNORECASE),
        "mutex": re.compile(r"mutex", re.IGNORECASE),
        "password": re.compile(r"password", re.IGNORECASE),
        "privilege": re.compile(r"privilege", re.IGNORECASE),
        "process": re.compile(r"process", re.IGNORECASE),
        "remote": re.compile(r"remote", re.IGNORECASE),
        "resource": re.compile(r"resource", re.IGNORECASE),
        "security": re.compile(r"security", re.IGNORECASE),
        "service": re.compile(r"service", re.IGNORECASE),
        "shell": re.compile(r"shell", re.IGNORECASE),
        "snapshot": re.compile(r"snapshot", re.IGNORECASE),
        "system": re.compile(r"system", re.IGNORECASE),
        "thread": re.compile(r"thread", re.IGNORECASE),
        "token": re.compile(r"token", re.IGNORECASE),
        "wallet": re.compile(r"wallet", re.IGNORECASE),
        "window": re.compile(r"window", re.IGNORECASE),
    }
    _REGEX_IDXS = {k: i for i, k in enumerate(sorted(_REGEXES))}

    def raw_features(self, bytez, pe):
        allstrings = self._ALLSTRINGS.findall(bytez)
        allstrings_ascii = [s.decode() for s in allstrings]
        if allstrings:
            lengths = [len(s) for s in allstrings]
            avlength = sum(lengths) / len(lengths)
            shifted = [b - 0x20 for b in b"".join(allstrings)]
            c = np.bincount(shifted, minlength=96)
            csum = int(c.sum())
            p = c.astype(np.float64) / csum
            nz = p > 0
            entropy = float(-np.sum(p[nz] * np.log2(p[nz])))
        else:
            avlength, c, csum, entropy = 0.0, np.zeros(96, dtype=np.int64), 0, 0.0

        string_counts: Dict[str, int] = {}
        for s in allstrings_ascii:
            for k, r in self._REGEXES.items():
                if r.search(s):
                    string_counts[k] = string_counts.get(k, 0) + 1
        string_counts = dict(sorted(string_counts.items()))

        return {
            "numstrings": len(allstrings),
            "avlength": avlength,
            "printabledist": c.tolist(),
            "printables": csum,
            "entropy": entropy,
            "string_counts": string_counts,
        }

    def process_raw_features(self, raw):
        divisor = float(raw["printables"]) if raw["printables"] > 0 else 1.0
        cat_counts = np.zeros(len(self._REGEX_IDXS), dtype=np.float32)
        for category, count in raw["string_counts"].items():
            idx = self._REGEX_IDXS.get(category)
            if idx is not None:
                cat_counts[idx] = count
        return np.hstack([
            raw["numstrings"], raw["avlength"], raw["printables"],
            np.asarray(raw["printabledist"], dtype=np.float32) / divisor,
            raw["entropy"], cat_counts,
        ]).astype(np.float32)


class HeaderFileInfo(FeatureGroup):
    name, dim = "header", 74

    # Fixed vocabularies (verbatim from the reference extractor) so this
    # extractor's categorical encodings don't drift if pefile's own
    # constant tables ever change.
    _MACHINE_TYPES = [
        "IMAGE_FILE_MACHINE_UNKNOWN", "IMAGE_FILE_MACHINE_I386", "IMAGE_FILE_MACHINE_R3000",
        "IMAGE_FILE_MACHINE_R4000", "IMAGE_FILE_MACHINE_R10000", "IMAGE_FILE_MACHINE_WCEMIPSV2",
        "IMAGE_FILE_MACHINE_ALPHA", "IMAGE_FILE_MACHINE_SH3", "IMAGE_FILE_MACHINE_SH3DSP",
        "IMAGE_FILE_MACHINE_SH3E", "IMAGE_FILE_MACHINE_SH4", "IMAGE_FILE_MACHINE_SH5",
        "IMAGE_FILE_MACHINE_ARM", "IMAGE_FILE_MACHINE_THUMB", "IMAGE_FILE_MACHINE_ARMNT",
        "IMAGE_FILE_MACHINE_AM33", "IMAGE_FILE_MACHINE_POWERPC", "IMAGE_FILE_MACHINE_POWERPCFP",
        "IMAGE_FILE_MACHINE_IA64", "IMAGE_FILE_MACHINE_MIPS16", "IMAGE_FILE_MACHINE_ALPHA64",
        "IMAGE_FILE_MACHINE_AXP64", "IMAGE_FILE_MACHINE_MIPSFPU", "IMAGE_FILE_MACHINE_MIPSFPU16",
        "IMAGE_FILE_MACHINE_TRICORE", "IMAGE_FILE_MACHINE_CEF", "IMAGE_FILE_MACHINE_EBC",
        "IMAGE_FILE_MACHINE_RISCV32", "IMAGE_FILE_MACHINE_RISCV64", "IMAGE_FILE_MACHINE_RISCV128",
        "IMAGE_FILE_MACHINE_LOONGARCH32", "IMAGE_FILE_MACHINE_LOONGARCH64", "IMAGE_FILE_MACHINE_AMD64",
        "IMAGE_FILE_MACHINE_M32R", "IMAGE_FILE_MACHINE_ARM64", "IMAGE_FILE_MACHINE_CEE",
    ]
    _SUBSYSTEM_TYPES = [
        "IMAGE_SUBSYSTEM_UNKNOWN", "IMAGE_SUBSYSTEM_NATIVE", "IMAGE_SUBSYSTEM_WINDOWS_GUI",
        "IMAGE_SUBSYSTEM_WINDOWS_CUI", "IMAGE_SUBSYSTEM_OS2_CUI", "IMAGE_SUBSYSTEM_POSIX_CUI",
        "IMAGE_SUBSYSTEM_NATIVE_WINDOWS", "IMAGE_SUBSYSTEM_WINDOWS_CE_GUI", "IMAGE_SUBSYSTEM_EFI_APPLICATION",
        "IMAGE_SUBSYSTEM_EFI_BOOT_SERVICE_DRIVER", "IMAGE_SUBSYSTEM_EFI_RUNTIME_DRIVER",
        "IMAGE_SUBSYSTEM_EFI_ROM", "IMAGE_SUBSYSTEM_XBOX", "IMAGE_SUBSYSTEM_WINDOWS_BOOT_APPLICATION",
    ]
    _IMAGE_CHARACTERISTICS = [
        "RELOCS_STRIPPED", "EXECUTABLE_IMAGE", "LINE_NUMS_STRIPPED", "LOCAL_SYMS_STRIPPED",
        "AGGRESIVE_WS_TRIM", "LARGE_ADDRESS_AWARE", "16BIT_MACHINE", "BYTES_REVERSED_LO",
        "32BIT_MACHINE", "DEBUG_STRIPPED", "REMOVABLE_RUN_FROM_SWAP", "NET_RUN_FROM_SWAP",
        "SYSTEM", "DLL", "UP_SYSTEM_ONLY", "BYTES_REVERSED_HI",
    ]
    _DLL_CHARACTERISTICS = [
        "HIGH_ENTROPY_VA", "DYNAMIC_BASE", "FORCE_INTEGRITY", "NX_COMPAT", "NO_ISOLATION",
        "NO_SEH", "NO_BIND", "APPCONTAINER", "WDM_DRIVER", "GUARD_CF", "TERMINAL_SERVER_AWARE",
    ]
    _DOS_MEMBERS = [
        "e_magic", "e_cblp", "e_cp", "e_crlc", "e_cparhdr", "e_minalloc", "e_maxalloc",
        "e_ss", "e_sp", "e_csum", "e_ip", "e_cs", "e_lfarlc", "e_ovno", "e_oemid",
        "e_oeminfo", "e_lfanew",
    ]
    _MACHINE_IDX = {mt: i for i, mt in enumerate(_MACHINE_TYPES)}
    _SUBSYSTEM_IDX = {st: i for i, st in enumerate(_SUBSYSTEM_TYPES)}

    def raw_features(self, bytez, pe):
        raw = {
            "coff": {"timestamp": 0, "machine": "", "number_of_sections": 0, "number_of_symbols": 0,
                     "sizeof_optional_header": 0, "pointer_to_symbol_table": 0, "characteristics": []},
            "optional": {"magic": 0, "subsystem": "", "major_image_version": 0, "minor_image_version": 0,
                         "major_linker_version": 0, "minor_linker_version": 0, "major_operating_system_version": 0,
                         "minor_operating_system_version": 0, "major_subsystem_version": 0,
                         "minor_subsystem_version": 0, "sizeof_code": 0, "sizeof_headers": 0, "sizeof_image": 0,
                         "sizeof_initialized_data": 0, "sizeof_uninitialized_data": 0, "sizeof_stack_reserve": 0,
                         "sizeof_stack_commit": 0, "sizeof_heap_reserve": 0, "sizeof_heap_commit": 0,
                         "address_of_entrypoint": 0, "base_of_code": 0, "image_base": 0, "section_alignment": 0,
                         "checksum": 0, "number_of_rvas_and_sizes": 0, "dll_characteristics": []},
            "dos": {m: 0 for m in self._DOS_MEMBERS},
        }
        if pe is None:
            return raw
        # Intentionally NOT wrapped in try/except: a pefile failure parsing the
        # headers must propagate to PEFeatureExtractor.raw_features(), which
        # records "header" in degraded_groups (review item 6). Swallowing it
        # here returned a partially/all-zero header dict that read as a
        # "stripped, no-mitigations" binary -- an invented maliciousness signal.
        fh, oh = pe.FILE_HEADER, pe.OPTIONAL_HEADER
        raw["coff"]["timestamp"] = fh.TimeDateStamp
        raw["coff"]["machine"] = pefile.MACHINE_TYPE.get(fh.Machine, "IMAGE_FILE_MACHINE_UNKNOWN")
        raw["coff"]["number_of_sections"] = fh.NumberOfSections
        raw["coff"]["number_of_symbols"] = fh.NumberOfSymbols
        raw["coff"]["sizeof_optional_header"] = fh.SizeOfOptionalHeader
        raw["coff"]["pointer_to_symbol_table"] = fh.PointerToSymbolTable
        raw["coff"]["characteristics"] = [k[11:] for k, v in fh.__dict__.items()
                                           if k.startswith("IMAGE_FILE_") and v]
        raw["optional"]["magic"] = oh.Magic
        raw["optional"]["subsystem"] = pefile.SUBSYSTEM_TYPE.get(oh.Subsystem, "IMAGE_SUBSYSTEM_UNKNOWN")
        raw["optional"]["major_image_version"] = oh.MajorImageVersion
        raw["optional"]["minor_image_version"] = oh.MinorImageVersion
        raw["optional"]["major_linker_version"] = oh.MajorLinkerVersion
        raw["optional"]["minor_linker_version"] = oh.MinorLinkerVersion
        raw["optional"]["major_operating_system_version"] = oh.MajorOperatingSystemVersion
        raw["optional"]["minor_operating_system_version"] = oh.MinorOperatingSystemVersion
        raw["optional"]["major_subsystem_version"] = oh.MajorSubsystemVersion
        raw["optional"]["minor_subsystem_version"] = oh.MinorSubsystemVersion
        raw["optional"]["sizeof_code"] = oh.SizeOfCode
        raw["optional"]["sizeof_headers"] = oh.SizeOfHeaders
        raw["optional"]["sizeof_image"] = oh.SizeOfImage
        raw["optional"]["sizeof_initialized_data"] = oh.SizeOfInitializedData
        raw["optional"]["sizeof_uninitialized_data"] = oh.SizeOfUninitializedData
        raw["optional"]["sizeof_stack_reserve"] = oh.SizeOfStackReserve
        raw["optional"]["sizeof_stack_commit"] = oh.SizeOfStackCommit
        raw["optional"]["sizeof_heap_reserve"] = oh.SizeOfHeapReserve
        raw["optional"]["sizeof_heap_commit"] = oh.SizeOfHeapCommit
        raw["optional"]["address_of_entrypoint"] = oh.AddressOfEntryPoint
        raw["optional"]["base_of_code"] = oh.BaseOfCode
        raw["optional"]["image_base"] = oh.ImageBase
        raw["optional"]["section_alignment"] = oh.SectionAlignment
        raw["optional"]["checksum"] = oh.CheckSum
        raw["optional"]["number_of_rvas_and_sizes"] = oh.NumberOfRvaAndSizes
        raw["optional"]["dll_characteristics"] = [k[25:] for k, v in oh.__dict__.items()
                                                   if k.startswith("IMAGE_DLLCHARACTERISTICS_") and v]
        dos_dict = pe.DOS_HEADER.dump_dict()
        for member in self._DOS_MEMBERS:
            if dos_dict.get(member, {}).get("Value") is not None:
                raw["dos"][member] = dos_dict[member]["Value"]
        return raw

    def process_raw_features(self, raw):
        coff, optional = raw["coff"], raw["optional"]
        return np.hstack([
            coff["timestamp"], coff["number_of_sections"], coff["number_of_symbols"],
            coff["sizeof_optional_header"], coff["pointer_to_symbol_table"],
            self._MACHINE_IDX.get(coff["machine"], 0),
            self._SUBSYSTEM_IDX.get(optional["subsystem"], 0),
            optional["major_image_version"], optional["minor_image_version"],
            optional["major_linker_version"], optional["minor_linker_version"],
            optional["major_operating_system_version"], optional["minor_operating_system_version"],
            optional["major_subsystem_version"], optional["minor_subsystem_version"],
            optional["sizeof_code"], optional["sizeof_headers"], optional["sizeof_image"],
            optional["sizeof_initialized_data"], optional["sizeof_uninitialized_data"],
            optional["sizeof_stack_reserve"], optional["sizeof_stack_commit"],
            optional["sizeof_heap_reserve"], optional["sizeof_heap_commit"],
            optional["address_of_entrypoint"], optional["base_of_code"], optional["image_base"],
            optional["section_alignment"], optional["checksum"], optional["number_of_rvas_and_sizes"],
            [1.0 if ch in coff["characteristics"] else 0.0 for ch in self._IMAGE_CHARACTERISTICS],
            [1.0 if ch in optional["dll_characteristics"] else 0.0 for ch in self._DLL_CHARACTERISTICS],
            [raw["dos"][m] for m in self._DOS_MEMBERS],
        ]).astype(np.float32)


class SectionInfo(FeatureGroup):
    name, dim = "section", 11 + 50 + 50 + 50 + 50 + 10 + 3

    def raw_features(self, bytez, pe):
        if pe is None:
            return {}
        entry_section = ""
        aoep = pe.OPTIONAL_HEADER.AddressOfEntryPoint
        for s in pe.sections:
            if s.contains_rva(aoep):
                entry_section = s.Name.strip(b"\x00").decode(errors="ignore").lower()
        i = 0
        while entry_section == "" and i < len(pe.sections):
            if pe.sections[i].Characteristics & 0x20000000 > 0:
                entry_section = pe.sections[i].Name.strip(b"\x00").decode(errors="ignore").lower()
            i += 1

        raw = {"entry": entry_section}
        raw["sections"] = [
            {
                "name": s.Name.strip(b"\x00").decode(errors="ignore").lower(),
                "size": s.SizeOfRawData,
                "entropy": s.get_entropy(),
                "vsize": s.Misc_VirtualSize,
                "size_ratio": s.SizeOfRawData / len(bytez),
                "vsize_ratio": s.SizeOfRawData / max(s.Misc_VirtualSize, 1),
                "props": [sc[10:] for sc, _ in pefile.section_characteristics if s.__dict__[sc]],
            }
            for s in pe.sections
        ]
        raw["overlay"] = {"size": 0, "size_ratio": 0.0, "entropy": 0.0}
        overlay = pe.get_overlay()
        if overlay is not None:
            overlay_size = len(overlay)
            entropy = 0.0
            for x in Counter(bytearray(overlay)).values():
                p_x = x / overlay_size
                entropy -= p_x * math.log2(p_x)
            raw["overlay"] = {"size": overlay_size, "size_ratio": overlay_size / len(bytez), "entropy": entropy}
        return raw

    def process_raw_features(self, raw):
        if not raw:
            return np.zeros(self.dim, dtype=np.float32)
        secs = raw["sections"]
        n_rx = sum(1 for s in secs if "MEM_READ" in s["props"] and "MEM_EXECUTE" in s["props"])
        n_w = sum(1 for s in secs if "MEM_WRITE" in s["props"])
        entropies = [s["entropy"] for s in secs] + [raw["overlay"]["entropy"], 0]
        size_ratios = [s["size_ratio"] for s in secs] + [raw["overlay"]["size_ratio"], 0]
        vsize_ratios = [s["vsize_ratio"] for s in secs] + [0]
        general = [
            len(secs), sum(1 for s in secs if s["size"] == 0), sum(1 for s in secs if s["name"] == ""),
            n_rx, n_w, max(entropies), min(entropies), max(size_ratios), min(size_ratios),
            max(vsize_ratios), min(vsize_ratios),
        ]
        h = lambda pairs: FeatureHasher(50, input_type="pair").transform([pairs]).toarray()[0]
        sizes_h = h([(s["name"], s["size"]) for s in secs])
        vsize_h = h([(s["name"], s["vsize"]) for s in secs])
        entropy_h = h([(s["name"], s["entropy"]) for s in secs])
        chars = [f"{s['name']}:{p}" for s in secs for p in s["props"]]
        chars_h = FeatureHasher(50, input_type="string").transform([chars]).toarray()[0]
        entry_h = FeatureHasher(10, input_type="string").transform([[raw["entry"]]]).toarray()[0]
        return np.hstack([
            general, sizes_h, vsize_h, entropy_h, chars_h, entry_h,
            raw["overlay"]["size"], raw["overlay"]["size_ratio"], raw["overlay"]["entropy"],
        ]).astype(np.float32)


class ImportsInfo(FeatureGroup):
    name, dim = "imports", 2 + 256 + 1024

    def raw_features(self, bytez, pe):
        imports: Dict[str, List[str]] = {}
        if pe is None or not hasattr(pe, "DIRECTORY_ENTRY_IMPORT"):
            return imports
        for entry in pe.DIRECTORY_ENTRY_IMPORT:
            dll_name = entry.dll.decode(errors="ignore") if entry.dll else ""
            imports[dll_name] = []
            for imp in entry.imports:
                if imp.name:
                    imports[dll_name].append(imp.name.decode(errors="ignore")[:10000])
                elif imp.ordinal is not None:
                    imports[dll_name].append(f"{dll_name}:ordinal{imp.ordinal}")
        return imports

    def process_raw_features(self, raw):
        if not raw:
            return np.zeros(self.dim, dtype=np.float32)
        libraries = list({lib.lower() for lib in raw.keys()})
        libs_h = FeatureHasher(256, input_type="string", alternate_sign=False).transform([libraries]).toarray()[0]
        # Note: for ordinal-style entries, `e` already embeds "<dll>:ordinalN",
        # so this deliberately produces "<dll>:<dll>:ordinalN" -- that's what
        # the reference extractor does, and EMBER2024's training data was
        # generated with this exact (not "fixed") behavior.
        pairs = [lib.lower() + ":" + e for lib, elist in raw.items() for e in elist]
        pairs_h = FeatureHasher(1024, input_type="string", alternate_sign=False).transform([pairs]).toarray()[0]
        return np.hstack([len(pairs), len(libraries), libs_h, pairs_h]).astype(np.float32)


class ExportsInfo(FeatureGroup):
    name, dim = "exports", 1 + 128

    def raw_features(self, bytez, pe):
        names: List[str] = []
        if pe is not None and hasattr(pe, "DIRECTORY_ENTRY_EXPORT"):
            for sym in pe.DIRECTORY_ENTRY_EXPORT.symbols:
                if sym.name:
                    names.append(sym.name.decode(errors="ignore")[:10000])
                elif sym.ordinal is not None:
                    names.append(f"ordinal{sym.ordinal}")
        return names

    def process_raw_features(self, raw):
        if not raw:
            return np.zeros(self.dim, dtype=np.float32)
        h = FeatureHasher(128, input_type="string").transform([raw]).toarray()[0]
        # raw is the list of exported symbol names; the leading scalar is the
        # export COUNT (matches ImportsInfo / RichHeader / PEFormatWarnings,
        # which all prepend a real count). len(h) was always 128 -- the
        # FeatureHasher width -- i.e. a dead constant feature slot.
        # TODO(feature-parity test, TECHNICAL_NOTES.md open item #1): the parity test
        # MUST assert this slot (absolute index feature_2276) == len(export
        # names) on BOTH the live-PE path here AND the EMBER2024 adapter path
        # (features/ember2024_adapter.py). Correctness on the adapter side
        # depends on EMBER2024's record "exports" field being a list of
        # symbol-name strings -- confirm when the parquet is regenerated.
        return np.hstack([len(raw), h]).astype(np.float32)


class DataDirectories(FeatureGroup):
    name, dim = "datadirectories", 16 * 2 + 2

    _NAME_ORDER = [
        "EXPORT", "IMPORT", "RESOURCE", "EXCEPTION", "SECURITY", "BASERELOC", "DEBUG", "COPYRIGHT",
        "GLOBALPTR", "TLS", "LOAD_CONFIG", "BOUND_IMPORT", "IAT", "DELAY_IMPORT", "COM_DESCRIPTOR", "RESERVED",
    ]

    def raw_features(self, bytez, pe):
        if pe is None:
            return []
        out = [{"has_relocs": int(pe.has_relocs()), "has_dynamic_relocs": int(pe.has_dynamic_relocs())}]
        for dd in pe.OPTIONAL_HEADER.DATA_DIRECTORY:
            out.append({
                "name": str(dd.name).replace("IMAGE_DIRECTORY_ENTRY_", ""),
                "size": dd.Size,
                "virtual_address": dd.VirtualAddress,
            })
        return out

    def process_raw_features(self, raw):
        if not raw:
            return np.zeros(self.dim, dtype=np.float32)
        features = np.zeros(2 * len(self._NAME_ORDER) + 2, dtype=np.float32)
        # Matches the reference extractor's loop bound exactly (excludes the
        # last directory entry, "RESERVED") for train/serve parity.
        for i in range(1, len(raw) - 1):
            idx = self._NAME_ORDER.index(raw[i]["name"])
            features[2 * idx] = raw[i]["size"]
            features[2 * idx + 1] = raw[i]["virtual_address"]
        features[-2] = raw[0]["has_relocs"]
        features[-1] = raw[0]["has_dynamic_relocs"]
        return features


class RichHeader(FeatureGroup):
    name, dim = "richheader", 1 + 32

    def raw_features(self, bytez, pe):
        if pe is not None and getattr(pe, "RICH_HEADER", None) is not None:
            return list(pe.RICH_HEADER.values)
        return []

    def process_raw_features(self, raw):
        if not raw:
            return np.zeros(self.dim, dtype=np.float32)
        n_pairs = len(raw) // 2
        pairs = [(str(raw[i]), raw[i + 1]) for i in range(0, len(raw) - 1, 2)]
        h = FeatureHasher(32, input_type="pair").transform([pairs]).toarray()[0]
        return np.hstack([n_pairs, h]).astype(np.float32)


class AuthenticodeSignature(FeatureGroup):
    name, dim = "authenticode", 8

    def raw_features(self, bytez, pe):
        raw = {"num_certs": 0, "self_signed": 0, "empty_program_name": 0, "no_countersigner": 0,
               "parse_error": 0, "chain_max_depth": 0, "latest_signing_time": 0.0, "signing_time_diff": 0.0}
        if pe is None or not _SIGNIFY_AVAILABLE:
            return raw
        try:
            af = AuthenticodeFile.from_stream(io.BytesIO(bytez))
            # signify 0.9.x: iter_signatures() yields AuthenticodeSignature
            # objects (0.8.x: iter_signed_datas() -> SignedData). Both expose
            # .signer_info and .certificates; feature semantics below are
            # unchanged from the 0.8.x version.
            for sig in af.iter_signatures():
                raw["num_certs"] += 1
                if sig.signer_info.program_name is None:
                    raw["empty_program_name"] = 1
                cs = sig.signer_info.countersigner
                if cs is not None:
                    t = cs.signing_time.timestamp()
                    if t >= raw["latest_signing_time"]:
                        raw["latest_signing_time"] = t
                    raw["signing_time_diff"] = t - pe.FILE_HEADER.TimeDateStamp
                else:
                    raw["no_countersigner"] = 1
                certs = list(sig.certificates)
                raw["chain_max_depth"] = max(raw["chain_max_depth"], len(certs))
                for cert in certs[:-1]:
                    if cert.issuer == cert.subject:
                        raw["self_signed"] = 1
        except Exception:
            raw["parse_error"] = 1
        return raw

    def process_raw_features(self, raw):
        if not raw:
            return np.zeros(self.dim, dtype=np.float32)
        return np.array([raw[k] for k in
                          ["num_certs", "self_signed", "empty_program_name", "no_countersigner",
                           "parse_error", "chain_max_depth", "latest_signing_time", "signing_time_diff"]],
                         dtype=np.float32)


class PEFormatWarnings(FeatureGroup):
    """Normalizes raw pefile warning strings against a fixed 87-entry
    prefix/suffix vocabulary (features/pefile_warnings.txt, verbatim from
    the reference extractor) so warning text differences don't fragment the
    hash space -- matches how EMBER2024's own `pefilewarnings` field was
    generated."""

    name, dim = "pefilewarnings", 87 + 1

    def __init__(self, warnings_file: Path = _WARNINGS_FILE):
        self.warning_prefixes: set[str] = set()
        self.warning_suffixes: set[str] = set()
        self.warning_ids: Dict[str, int] = {}
        if warnings_file.exists():
            with open(warnings_file) as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if line.startswith("..."):
                        self.warning_suffixes.add(line[3:])
                    else:
                        self.warning_prefixes.add(line[:-3])
                    self.warning_ids[line] = i

    def raw_features(self, bytez, pe):
        if pe is None:
            return []
        warnings_norm = set()
        for warning in set(pe.get_warnings()):
            for suf in self.warning_suffixes:
                if warning.endswith(suf):
                    warnings_norm.add("..." + suf)
                    break
            else:
                for pre in self.warning_prefixes:
                    if warning.startswith(pre):
                        warnings_norm.add(pre + "...")
                        break
                else:
                    logger.debug("Unknown pefile warning: %s", warning)
        return sorted(warnings_norm)

    def process_raw_features(self, raw):
        if not raw:
            return np.zeros(self.dim, dtype=np.float32)
        ids = [0.0] * self.dim
        for warning_norm in raw:
            idx = self.warning_ids.get(warning_norm)
            if idx is not None:
                ids[idx] = 1.0
        ids[-1] = len(raw)
        return np.array(ids, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────
class PEFeatureExtractor:
    """EMBER2024-compatible 2568-dim extractor built on `pefile`."""

    def __init__(self) -> None:
        if not _PEFILE_AVAILABLE:
            raise ImportError("pefile is required: pip install pefile")
        if not _SIGNIFY_AVAILABLE:
            raise ImportError(
                "signify>=0.9,<0.10 is required for the authenticode feature group "
                "(this module targets the 0.9.x AuthenticodeFile API). A missing or "
                "too-old signify would silently zero 8 features and skew every static "
                "score on signed binaries -- failing hard instead of degrading. "
                "pip install 'signify>=0.9,<0.10'"
            )
        self._groups: List[FeatureGroup] = [
            GeneralFileInfo(), ByteHistogram(), ByteEntropyHistogram(), StringExtractor(),
            HeaderFileInfo(), SectionInfo(), ImportsInfo(), ExportsInfo(),
            DataDirectories(), RichHeader(), AuthenticodeSignature(), PEFormatWarnings(),
        ]
        self.dim = sum(g.dim for g in self._groups)
        assert self.dim == EMBER2024_FEATURE_COUNT, f"dim mismatch: {self.dim} != {EMBER2024_FEATURE_COUNT}"

    def parse(self, bytez: bytes) -> Optional["pefile.PE"]:
        """Full pefile parse, or None if `bytez` is not a PE. The pipeline
        parses once with this and hands the object to truncation_findings()
        and feature_vector_with_report(pe=...); the caller owns and closes it."""
        try:
            return pefile.PE(data=bytez, fast_load=False)
        except Exception as exc:
            logger.info("pefile parse failed: %s", exc)
            return None

    _parse = parse  # backward-compatible private name

    def raw_features(self, bytez: bytes, *, pe: Optional["pefile.PE"] = None,
                     _degraded: Optional[list] = None) -> Dict[str, Any]:
        """Per-group raw feature extraction.

        `_degraded` (keyword-only, internal): if a list is passed, the name of
        every group that could not be extracted cleanly is appended to it --
        review item 6. A group lands in `_degraded` when its extraction raised
        (and was retried with pe=None, i.e. degraded to defaults) OR, for the
        authenticode group, when signify set parse_error=1. Public callers
        (features/ember2024_adapter.py, tests) pass nothing and get the same
        dict as before.

        `pe` (keyword-only): an already-parsed pefile.PE for these exact
        bytes, so the file is not parsed twice. When given, the caller owns
        it and this method does not close it.
        """
        owns_pe = pe is None
        if owns_pe:
            pe = self.parse(bytez)
        out = {"sha256": hashlib.sha256(bytez).hexdigest()}
        for g in self._groups:
            try:
                out[g.name] = g.raw_features(bytez, pe)
            except Exception:
                # A raised feature group is NOT normal data. Log loudly (not
                # debug), record it, and fall back to the pe=None defaults so
                # the scan can still produce a vector -- the pipeline decides
                # what a degraded critical group means for the verdict.
                logger.warning("feature group '%s' raw extraction failed; using degraded defaults",
                               g.name, exc_info=True)
                if _degraded is not None and g.name not in _degraded:
                    _degraded.append(g.name)
                try:
                    out[g.name] = g.raw_features(bytez, None)
                except Exception:
                    logger.warning("feature group '%s' degraded retry also failed; zero-filling",
                                   g.name, exc_info=True)
                    out[g.name] = None  # process_raw_features() zero-fills a None group
        # authenticode's own try/except sets parse_error=1 rather than raising,
        # so the loop above never sees it as a failure -- surface it here.
        auth = out.get("authenticode")
        if isinstance(auth, dict) and auth.get("parse_error"):
            if _degraded is not None and "authenticode" not in _degraded:
                _degraded.append("authenticode")
        if owns_pe and pe is not None:
            try:
                pe.close()
            except Exception:
                pass
        return out

    def process_raw_features(self, raw: Dict[str, Any], *, _degraded: Optional[list] = None) -> np.ndarray:
        """Vectorize per-group raw features into the 2568-dim vector.

        `_degraded` (keyword-only, internal): names of groups whose
        vectorization raised and had to be zero-filled are appended (review
        item 6). Public signature is unchanged for callers that pass nothing.
        """
        parts = []
        for g in self._groups:
            try:
                part = g.process_raw_features(raw[g.name])
                # A group can also "fail" WITHOUT raising: e.g. a None raw
                # group (double extraction failure) makes ByteHistogram return
                # a 0-d NaN scalar instead of a 256-vector. Validate shape and
                # finiteness so a malformed part is caught here, not as a
                # cryptic length mismatch when the model is called.
                if np.shape(part) != (g.dim,) or not np.isfinite(part).all():
                    raise ValueError(
                        f"group '{g.name}' produced shape {np.shape(part)} "
                        f"(expected ({g.dim},)) or non-finite values"
                    )
            except Exception:
                logger.warning("feature group '%s' vectorization failed; zero-filling %d dims",
                               g.name, g.dim, exc_info=True)
                if _degraded is not None and g.name not in _degraded:
                    _degraded.append(g.name)
                part = np.zeros(g.dim, dtype=np.float32)
            parts.append(part)
        return np.hstack(parts).astype(np.float32)

    def feature_vector(self, bytez: bytes) -> np.ndarray:
        return self.feature_vector_with_report(bytez)[0]

    def feature_vector_with_report(self, bytez: bytes, *, pe: Optional["pefile.PE"] = None,
                                   ) -> "tuple[np.ndarray, list[str]]":
        """Like feature_vector(), but also returns the sorted list of feature
        groups that had to be degraded (raised and fell back to zeros/defaults)
        -- review item 6. An empty list means every group extracted cleanly.
        `pe`: optional pre-parsed object for `bytez` (see raw_features()).
        """
        degraded: list[str] = []
        raw = self.raw_features(bytez, pe=pe, _degraded=degraded)
        vec = self.process_raw_features(raw, _degraded=degraded)
        return vec, sorted(set(degraded))

    def _group_offset(self, name: str) -> int:
        off = 0
        for g in self._groups:
            if g.name == name:
                return off
            off += g.dim
        raise KeyError(name)

    def self_test(self, pe_path: "str | Path | None" = None) -> list[str]:
        """Run the extractor against a known-good signed Windows PE and return
        a list of failure descriptions (empty == healthy). Does NOT raise --
        the caller decides severity (CortexPipeline.__init__ raises on a
        critical-group failure, warns on a non-critical one).

        This catches, at startup, the class of regression where a pefile or
        signify API change silently disables a feature group -- which would
        otherwise only show up as skewed production scores.
        """
        path = Path(pe_path) if pe_path is not None else _SELFTEST_PE
        if not path.is_file():
            return [f"reference_pe_not_found:{path}"]
        bytez = path.read_bytes()
        try:
            vec, degraded = self.feature_vector_with_report(bytez)
        except Exception as exc:  # pragma: no cover - defensive
            return [f"extraction_raised:{exc!r}"]

        failures: list[str] = [f"degraded_group:{g}" for g in degraded]
        if vec.shape != (EMBER2024_FEATURE_COUNT,):
            failures.append(f"vector_shape:{vec.shape}")
        if not np.isfinite(vec).all():
            failures.append("vector_nonfinite")
        auth_off = self._group_offset("authenticode")
        if not vec[auth_off:auth_off + 8].any():
            failures.append("authenticode_all_zero_on_signed_pe")
        imports_off = self._group_offset("imports")
        if vec[imports_off] <= 0:
            failures.append("imports_pair_count_zero")
        hist_off = self._group_offset("histogram")
        if abs(float(vec[hist_off:hist_off + 256].sum()) - 1.0) > 1e-4:
            failures.append("histogram_not_normalised")
        if float(vec[0]) != float(len(bytez)):
            failures.append("general_size_mismatch")
        return failures

    def is_valid_pe(self, bytez: bytes) -> bool:
        """Rule-based PE validation gate (stage 3 of the architecture).
        Parses the file; the pipeline uses parse() directly instead so the
        parsed object can be reused."""
        return self.parse(bytez) is not None


def truncation_findings(pe: "pefile.PE", file_size: int) -> List[str]:
    """docs/CODE_REVIEW.md F17: structural signs that the file on disk is
    cut short. Returns machine-parseable detail codes (empty == not
    truncated):

      pe_truncated:section_raw_beyond_eof:<bytes>  -- some section with
          SizeOfRawData > 0 has PointerToRawData + SizeOfRawData past EOF;
          <bytes> is the largest overrun. Zero tolerance.
      pe_truncated:headers_beyond_eof              -- SizeOfHeaders > file size
      pe_truncated:certificate_table_beyond_eof    -- the security directory
          (a FILE OFFSET, not an RVA) has Size > 0 and ends past EOF

    Measured before adoption (docs/TECHNICAL_NOTES.md): 0 hits on 126 benign
    PEs. pefile's own warnings are deliberately not used.
    """
    findings: List[str] = []
    overruns = [
        s.PointerToRawData + s.SizeOfRawData - file_size
        for s in pe.sections
        if s.SizeOfRawData > 0 and s.PointerToRawData + s.SizeOfRawData > file_size
    ]
    if overruns:
        findings.append(f"pe_truncated:section_raw_beyond_eof:{max(overruns)}")
    if pe.OPTIONAL_HEADER.SizeOfHeaders > file_size:
        findings.append("pe_truncated:headers_beyond_eof")
    data_dirs = pe.OPTIONAL_HEADER.DATA_DIRECTORY
    if len(data_dirs) > 4:  # IMAGE_DIRECTORY_ENTRY_SECURITY
        sec = data_dirs[4]
        if sec.Size > 0 and sec.VirtualAddress + sec.Size > file_size:
            findings.append("pe_truncated:certificate_table_beyond_eof")
    return findings
