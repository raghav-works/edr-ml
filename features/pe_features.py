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

Requires: pefile, numpy, scikit-learn (FeatureHasher). Optional: signify (for
Authenticode parsing) — degrades gracefully to zeros if unavailable.
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

try:
    from signify.authenticode import SignedPEFile
    from signify import exceptions as signify_exceptions
    _SIGNIFY_AVAILABLE = True
except ImportError:
    _SIGNIFY_AVAILABLE = False

EMBER2024_FEATURE_COUNT = 2568
_WARNINGS_FILE = Path(__file__).parent / "pefile_warnings.txt"


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
        try:
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
        except Exception:
            logger.debug("HeaderFileInfo extraction failed", exc_info=True)
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
        return np.hstack([len(h), h]).astype(np.float32)


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
            spe = SignedPEFile(io.BytesIO(bytez))
            for sd in spe.iter_signed_datas():
                raw["num_certs"] += 1
                if sd.signer_info.program_name is None:
                    raw["empty_program_name"] = 1
                cs = sd.signer_info.countersigner
                if cs is not None:
                    t = cs.signing_time.timestamp()
                    if t >= raw["latest_signing_time"]:
                        raw["latest_signing_time"] = t
                    raw["signing_time_diff"] = t - pe.FILE_HEADER.TimeDateStamp
                else:
                    raw["no_countersigner"] = 1
                certs = sd.certificates
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
        self._groups: List[FeatureGroup] = [
            GeneralFileInfo(), ByteHistogram(), ByteEntropyHistogram(), StringExtractor(),
            HeaderFileInfo(), SectionInfo(), ImportsInfo(), ExportsInfo(),
            DataDirectories(), RichHeader(), AuthenticodeSignature(), PEFormatWarnings(),
        ]
        self.dim = sum(g.dim for g in self._groups)
        assert self.dim == EMBER2024_FEATURE_COUNT, f"dim mismatch: {self.dim} != {EMBER2024_FEATURE_COUNT}"

    def _parse(self, bytez: bytes) -> Optional["pefile.PE"]:
        try:
            return pefile.PE(data=bytez, fast_load=False)
        except Exception as exc:
            logger.info("pefile parse failed: %s", exc)
            return None

    def raw_features(self, bytez: bytes) -> Dict[str, Any]:
        pe = self._parse(bytez)
        out = {"sha256": hashlib.sha256(bytez).hexdigest()}
        for g in self._groups:
            try:
                out[g.name] = g.raw_features(bytez, pe)
            except Exception:
                logger.debug("group '%s' failed", g.name, exc_info=True)
                out[g.name] = g.raw_features(bytez, None)
        if pe is not None:
            try:
                pe.close()
            except Exception:
                pass
        return out

    def process_raw_features(self, raw: Dict[str, Any]) -> np.ndarray:
        parts = []
        for g in self._groups:
            try:
                parts.append(g.process_raw_features(raw[g.name]))
            except Exception:
                logger.debug("process failed for '%s'", g.name, exc_info=True)
                parts.append(np.zeros(g.dim, dtype=np.float32))
        return np.hstack(parts).astype(np.float32)

    def feature_vector(self, bytez: bytes) -> np.ndarray:
        return self.process_raw_features(self.raw_features(bytez))

    def is_valid_pe(self, bytez: bytes) -> bool:
        """Rule-based PE validation gate (stage 3 of the architecture)."""
        return self._parse(bytez) is not None
