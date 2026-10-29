# Local model routing

The dashboard defaults to **Tự chọn model theo đầu vào**. Routing uses parsed
evidence rather than trusting the uploaded filename to imply execution behavior.

| Parsed input | Registered model |
| --- | --- |
| PE bytes, no execution trace | Signature–Static Attention trained on EMBER2024 |
| CAPE/WinMET JSON with ordered events | Three-branch Hybrid Attention trained on WinMET process-relation v3 |
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

Hybrid registrations can pin `normalization_version`. A missing/different trace
schema returns evidence-only UNKNOWN before loading weights, even when token
coverage would be high. New checkpoints also persist `normalization_versions`
from their dynamic training reports and enforce it inside the pipeline, including
manual selection. An empty list means static-only training; `null` means legacy
unknown provenance, not verified compatibility. Older checkpoints cannot be
retroactively assigned a training schema just because current code uses it.

The current parser uses `cape-process-relations-v3` to resolve documented
current-process pseudo handles. The registry now selects the retrained v3
research candidate, whose checkpoint independently records this same schema.
Older v2 traces are rejected by this registration rather than reinterpreted.
PE-only static routing is unaffected. V3 has its own validation metrics; historical
PE-v2 accuracy is not silently applied to changed graph data.

For direct PE uploads, the bounded local parser and the EMBER feature extractor
are different sources. Matching token vocabulary does not verify that their
feature distributions or predictions are equivalent. The PE parser detects a
bounded certificate table but does not verify an Authenticode signature or trust
chain; the UI and batch output disclose that limitation. Evaluation on paired
PE bytes and EMBER-style features remains necessary.

This service reads PE bytes and pre-existing JSON reports only. It does not invoke
an executable, launch an emulator, submit a sample to a remote service, or install
missing weights. Runtime behavior requires an existing execution report from an
isolated sandbox.

`scripts/probe_macos_parser_sandbox.py` is a local feasibility probe, not an
isolation boundary used by inference. On the development Mac it parsed a
synthetic PE inside a child restricted from connecting to localhost, writing a
file, forking, and reading a harmless marker elsewhere in the user's home. The
profile still permits file metadata access, reads outside the home, and exec;
it has not been integrated into the application or validated against parser
compromise. Do not treat this probe as a sandbox for executing malware.

Tests use their own temporary registries and tiny checkpoints, so ignored
workstation models cannot change test outcomes or startup cost. A local smoke
check also processed a real setuptools PE launcher and one bounded raw WinMET
report: both selected the expected model and accepted input compatibility while
abstaining on family. That is an integration check, not a benign FPR or family
accuracy benchmark.
