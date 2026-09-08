# PE fixtures for `tests/test_static_feature_parity.py`

Small, behaviour-free Windows PE files used to exercise every EMBER2024
feature group's happy path. Frozen binary fixtures — `generate_fixtures.sh`
is checked in as **provenance only**, not run by the test suite or CI.

Copied verbatim from the `malware-ml` repo's identical fixture set
(sha256-matched at copy time).

| file | what it is | licence |
|---|---|---|
| `sample_cli32.exe` | distlib `t32.exe` launcher stub | PSF License (distlib, vendored by pip/setuptools) |
| `sample_cli64.exe` | distlib `t64.exe` launcher stub | PSF License |
| `sample_gui64.exe` | distlib `w64.exe` launcher stub | PSF License |
| `sample_signed64.exe` | `sample_gui64.exe` re-signed with a throwaway self-signed cert (`CN=Cortex Endpoint Test Fixture`, `O=Test Only`, no timestamp server) | PSF License + self-signed test cert |

None of these carry real behaviour or a real vendor signature. The signed
one exists solely so `AuthenticodeSignature` extraction keeps being
exercised (it parses one certificate; `parse_error` stays 0).

To regenerate (needs `osslsigncode` + `openssl` + a distlib with launcher
stubs): `./generate_fixtures.sh`.
