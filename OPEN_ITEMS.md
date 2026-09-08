# Open items

Running tracker for known-but-deferred work. Blocking review items 1–6 are
done (see git log); this is what remains.

## Structural

- **Feature-parity harness for `pe_features.py`** — MVP DONE
  (`tests/test_static_feature_parity.py`, item 7): vector contract,
  determinism, signed-binary authenticode, ExportsInfo count slot, and
  adapter-passthrough parity, on committed synthetic PE fixtures. Two
  sub-items remain:
  - **Real EMBER2024 record schema check.** The adapter assumes an
    EMBER2024 record's group dicts (`record["general"]`,
    `record["header"]["coff"]`, ...) have the exact keys/shape that
    `pe_features.raw_features()` produces. The MVP only proves the adapter
    faithfully processes a dict `raw_features` itself made. Verifying
    against a *real* record needs a small (~100-record) pull from
    `joyce8/EMBER2024` on HF — NOT a full parquet regeneration. Needed
    before trusting the adapter beyond "faithful passthrough".
  - **`thrember` skew-quantification cross-check.** Compare the live
    `pe_features` vector against the gold-standard `thrember` extractor on
    real PEs, per group, to quantify the documented "skews toward
    malicious" gap. BLOCKED in this repo's main venv: `thrember` (a git
    install from the EMBER2024 repo) imports the pre-0.9 `signify`
    authenticode API, and item 4 pinned `signify>=0.9,<0.10`. The two
    cannot coexist — this check needs a separate venv (or to run inside
    the `malware-ml` venv, which already has `thrember` + old `signify`).
- **Baseline `pytest` suite** — item 8. `tests/` scaffold (`conftest.py`,
  `pytest.ini`) exists from item 7. Still to add:
  `tests/test_policy_engine.py` (the `decide()` truth table run by hand all
  session) and a pytest wrapper around `scripts/verify_onnx_parity.py`.

## Deferred to a deliberate retrain pass

- **Threshold-selection optimistic bias** — memory/network operating
  thresholds are derived on `val+test` combined and then held-out metrics
  are quoted on that same pool. Fix = a 3-way split (separate
  threshold-selection set). Fold into the next retrain, not a standalone
  re-run.
- **`pe_features.py`-vs-thrember feature-fidelity skew** — cortex-ml's live
  vector skews toward "malicious"; this is the real cause behind the interim
  cap on Cortex-Static's BLOCK authority (`policy_engine.decide()` "INTERIM
  CAP"). Closing it needs the parity harness first, then a static retrain +
  threshold re-derivation + real-world re-validation.
- **`ExportsInfo` count fix** (commit `bb1ade0`) delivers no model change
  until the next static retrain regenerates the EMBER2024 parquets.

## Cleanup

- **Unreachable `None` guards** — `MEMORY/NETWORK/EMULATION_MALICIOUS_MIN`
  keep `Optional[float]` typing and `is None -> NotImplementedError` guards
  in `*_verdict_from_score()`. Since `policy_engine` now loads thresholds
  from YAML (loader yields a float or raises), those paths are dead.
- **`config/` packaging** — `policy_engine` loads `config/thresholds.yaml`
  via `parents[1]/config`; fine running from the tree, but there is no
  `pyproject.toml`/`setup.py`, so packaging would need to bundle `config/`.
- **`thresholds.yaml` `sequence.*` / `file.*`** keys are informational only;
  `api_tokenizer.MAX_SEQ_LEN` / `pipeline.MAX_FILE_SIZE_BYTES` still hold
  their own constants.

## Docs pass (batch, not piecemeal)

- `README.md` — calibration-saturation section (~lines 176/852) still
  describes pre-fix behaviour; head/flow already updated.
- `PROJECT_HISTORY_REPORT.md` — uncommitted edits pending.
- `Architecture.txt` §10 — carries a `STALE` flag; the Cortex-Network
  paragraph predates the item-6 wiring and the calibration refit.
