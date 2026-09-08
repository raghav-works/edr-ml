#!/usr/bin/env bash
# Regenerates the checked-in PE fixtures used by
# tests/test_static_feature_parity.py. Not run automatically by the test
# suite or CI -- these are one-time-generated, frozen binary fixtures, and
# this script exists only as provenance for how they were produced and to
# let a maintainer regenerate them if ever needed (e.g. to add a new
# fixture). It requires osslsigncode and openssl, and a Python launcher-stub
# source (distlib, PSF License, vendored inside pip and setuptools) to seed
# the unsigned samples.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

DISTLIB_STUBS="${DISTLIB_STUBS:-$(python3 -c '
import pip._vendor.distlib as d, os
candidate = os.path.dirname(d.__file__)
if not os.path.exists(os.path.join(candidate, "t32.exe")):
    raise SystemExit(
        f"{candidate} has no launcher stubs (some pip vendoring strips "
        "non-Python files); pass DISTLIB_STUBS=<dir> explicitly, e.g. a "
        "site-packages/pip/_vendor/distlib directory that still has them"
    )
print(candidate)
')}"

# Three tiny PSF-licensed launcher-stub PEs (distlib, vendored by pip and
# setuptools). These are real, minimal Windows executables, so they exercise
# every EMBER2024 feature group's happy path (headers, imports, sections,
# rich header) without carrying any actual behavior or license baggage.
cp "$DISTLIB_STUBS/t32.exe" sample_cli32.exe
cp "$DISTLIB_STUBS/t64.exe" sample_cli64.exe
cp "$DISTLIB_STUBS/w64.exe" sample_gui64.exe

# A self-signed Authenticode-signed PE, to keep exercising
# AuthenticodeSignature feature extraction. No timestamp server is used, so
# generation is fully offline and does not depend on an external service
# being reachable.
openssl req -x509 -newkey rsa:2048 -keyout signing_key.pem -out signing_cert.pem \
    -days 36500 -nodes -subj "/CN=Cortex Endpoint Test Fixture/O=Test Only"

osslsigncode sign \
    -certs signing_cert.pem -key signing_key.pem \
    -n "Cortex Endpoint test fixture (not a real signed binary)" \
    -in sample_gui64.exe -out sample_signed64.exe

rm -f signing_key.pem signing_cert.pem
echo "Fixtures written to $HERE"
