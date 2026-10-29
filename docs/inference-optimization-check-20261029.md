# Inference optimization verification — 2026-10-29

No additional training run was launched. Changes preserve model weights and feature semantics:

- Exact imphash indexing avoids comparing unrelated prototype digests.
- Fuzzy evidence caching uses an LRU limit of 4,096 records.
- Directory discovery enforces its entry budget during enumeration, before sorting.

Verification: `PYTHONPATH=src .venv/bin/python -m pytest -q` completed with
329 passed and 94 subtests passed in 11.01 seconds. One Streamlit dependency
warning concerns a deprecated pandas argument.

A CPU inference smoke check used the registered `winmet-relv3-seed44` model and
`tests/fixtures/cape_process_injection.json`, parsed with the routed pipeline's
schema. Prediction took 0.459 seconds, including lazy model loading inside the
predict call. The model abstained on family (`UNKNOWN`) and returned T1055 as
`suspected`. This synthetic fixture checks execution and guarded output only;
it is not family accuracy evidence or a representative latency benchmark.

The full project remains incomplete: hybrid accuracy above 85%, attention
faithfulness and flow ablation requirements still need verified evidence.
