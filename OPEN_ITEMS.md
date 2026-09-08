# Open items

Running tracker for known-but-deferred work. Blocking review items 1–6 are
done (see git log); this is what remains.

## Structural (in progress / next)

- **Feature-parity harness for `pe_features.py`** — compare live extraction
  against `features/ember2024_adapter.py` on real PE files (incl. a signed
  binary). Must assert the exports-count slot `feature_2276 == len(export
  names)` on both paths (see the `TODO` in `ExportsInfo.process_raw_features`).
- **Baseline `pytest` suite** — none exists. Seed from
  `scripts/verify_onnx_parity.py` and the `decide()` truth-table checks used
  for review items 2 and 6 (→ `tests/test_policy_engine.py`).

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
