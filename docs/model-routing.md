# Local model routing

The dashboard defaults to **Tự chọn model theo đầu vào**. Routing uses parsed
evidence rather than trusting the uploaded filename to imply execution behavior.

| Parsed input | Registered model |
| --- | --- |
| PE bytes, no execution trace | Signature–Static Attention trained on EMBER2024 |
| CAPE/WinMET JSON with ordered events | Three-branch Hybrid Attention trained on recovered WinMET PE-v2 |
| JSON with only static evidence | Static model; no fabricated execution events |
| Suitable checkpoint not installed | Evidence-only output, family UNKNOWN and explicit unavailable reason |

`configs/inference_models.json` pins each candidate by relative path, byte size
and SHA-256. Paths are relative to the registry file. Checkpoints are loaded once
per input mode and verified before loading. A mixed batch records `model_id`,
`model_mode` and `model_status` on every CSV row. It never silently substitutes
static neural scores for a missing hybrid model, or the reverse. Corrupted or
replaced artifacts fail with an error. Family abstention and the independent
malware-verdict release gate still apply; selecting a model does not qualify it.

The registered artifacts are local research candidates, not release weights.
Ignored model binaries are not included in a fresh Git clone. Train the documented
datasets, then register the resulting checkpoint with its actual checksum/size,
or choose a checkpoint explicitly in the dashboard. For another installation,
`MAL_CLASS_MODEL_REGISTRY` can point to a different registry with the same schema.
Changing that file invalidates the dashboard routing cache. Manual checkpoint
selection overrides automatic routing and retains the existing input guards.

This service reads PE bytes and pre-existing JSON reports only. It does not invoke
an executable, launch an emulator, submit a sample to a remote service, or install
missing weights. Runtime behavior requires an existing execution report from an
isolated sandbox.

Tests use their own temporary registries and tiny checkpoints, so ignored
workstation models cannot change test outcomes or startup cost. A local smoke
check also processed a real setuptools PE launcher and one bounded raw WinMET
report: both selected the expected model and accepted input compatibility while
abstaining on family. That is an integration check, not a benign FPR or family
accuracy benchmark.
