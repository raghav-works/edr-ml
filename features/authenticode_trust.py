"""
Standalone Authenticode trust-chain verification -- NOT an ML feature.
========================================================================

features/pe_features.py's AuthenticodeSignature feature group extracts an
8-dim SUMMARY of a file's signature (cert count, self-signed flag, presence
of a countersigner, ...) as one input among 2568 to the static LightGBM
model. It deliberately never calls signify's real verification -- it only
enumerates certificates via iter_signatures() for those summary stats.

This module answers a different, harder question: does this file's
Authenticode signature chain up to a certificate this machine's real trust
store actually trusts? It exists to support the known-file allowlist
(OPEN_ITEMS.md, "Static false-positive severity cluster"): a file that
chains to a genuinely trusted root is treated as pre-verified and
short-circuits static's ML judgment entirely via pipeline.py's step 3b.
This module has no dependency on PEFeatureExtractor / AuthenticodeSignature
and is never called from them, and they are never called from here --
the two co-exist as separate consumers of the same underlying library.

Requires signify>=0.9,<0.10, same as pe_features.py (see that module's
header for why the version is pinned to the 0.9.x AuthenticodeFile API).
Unlike PEFeatureExtractor, this module does not guard the import behind an
availability flag -- there is no partial-functionality mode for a trust
verification check to degrade into; if signify is missing, importing this
module fails immediately and loudly, which is the correct behaviour.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass

from signify.authenticode import TRUSTED_CERTIFICATE_STORE, AuthenticodeFile
from signify.exceptions import AuthenticodeNotSignedError, SignifyError

logger = logging.getLogger("cortex.features.authenticode_trust")


@dataclass(frozen=True)
class AuthenticodeTrustResult:
    trusted: bool
    # "verified" | "not_signed" | "parse_error" | "chain_untrusted"
    reason: str


def verify_trusted_chain(bytez: bytes) -> AuthenticodeTrustResult:
    """
    Real Authenticode chain verification against signify's bundled copy of
    Microsoft's actual Authenticode root Certificate Trust List
    (TRUSTED_CERTIFICATE_STORE, built from the `mscerts` package's
    authroot.stl -- 566 real roots confirmed loaded in this repo's venv,
    not a stand-in). This calls signify's AuthenticodeFile.verify(), which
    validates digest consistency, key usage, and full chain-of-trust -- NOT
    the presence-only check features/pe_features.py::AuthenticodeSignature
    performs for its ML feature.

    Three-way outcome taxonomy, confirmed against this repo's own PE
    fixtures (tests/test_authenticode_trust.py):
      - "not_signed"     -- AuthenticodeNotSignedError: no signature present,
                            or the file is too truncated/malformed for
                            signify to locate one (the two are
                            indistinguishable to signify and are reported
                            identically here; a truncated signed binary
                            reading as "not signed" is an honest description
                            of what was actually found, not a bug).
      - "parse_error"    -- the bytes don't parse as a recognizable signed
                            file structure at all (AuthenticodeFile.from_stream
                            itself raises), or verification raises something
                            outside signify's own exception hierarchy.
      - "chain_untrusted" -- a signature IS present and was parsed, but chain
                            verification against TRUSTED_CERTIFICATE_STORE
                            failed (self-signed, untrusted issuer, expired,
                            digest mismatch, etc.).
      - "verified"       -- verify() succeeded: a full, real chain from this
                            file's signature to a genuinely trusted
                            Microsoft/CA root.

    Never raises. Every non-"verified" outcome is treated identically by
    callers (fall through to static's normal ML judgment) -- the reason
    string is for audit/telemetry only, not branching logic. This function
    only ever produces a POSITIVE allowlist signal (trusted=True); it is
    never used to justify a MORE severe verdict than the ML path would have
    produced on its own.
    """
    try:
        af = AuthenticodeFile.from_stream(io.BytesIO(bytez))
    except Exception:
        logger.debug("authenticode trust check: file did not parse", exc_info=True)
        return AuthenticodeTrustResult(False, "parse_error")

    try:
        af.verify(trusted_certificate_store=TRUSTED_CERTIFICATE_STORE)
        return AuthenticodeTrustResult(True, "verified")
    except AuthenticodeNotSignedError as exc:
        logger.debug("authenticode trust check: not signed (%s)", exc)
        return AuthenticodeTrustResult(False, "not_signed")
    except SignifyError as exc:
        logger.debug("authenticode trust check: chain untrusted (%s)", exc)
        return AuthenticodeTrustResult(False, "chain_untrusted")
    except Exception:
        logger.debug("authenticode trust check: unexpected verification error", exc_info=True)
        return AuthenticodeTrustResult(False, "parse_error")
