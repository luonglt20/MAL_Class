# WinMET: bounded, report-only workflow

Source: [WinMET v3](https://zenodo.org/records/16414116), with source checksums,
sizes and API license metadata pinned in `configs/winmet.lock.json`. Volumes 1
and 2 are individually pinned; volume 2 is a report-only extension and is not
the same as the held-out volume-1 block cohort.
The [record API](https://zenodo.org/api/records/16414116) declares
GPL-3.0-or-later. Keep attribution and applicable dataset license terms with
derived datasets; this metadata does not change the repository's code license.

Install the optional reader and fetch only the volume(s) needed plus labels:

```bash
pip install -e '.[winmet]'
malware-hybrid download-winmet reports_consensus_label.json
malware-hybrid download-winmet WinMET_volume_1.7z
malware-hybrid download-winmet WinMET_volume_2.7z
malware-hybrid prepare-winmet Data/raw/winmet/WinMET_volume_1.7z \
  --labels Data/raw/winmet/reports_consensus_label.json \
  --output artifacts/winmet/prepared_reports.jsonl \
  --max-reports 32 --max-report-mib 64 --raw-budget-mib 512 \
  --output-budget-mib 128 --max-events 2048
```

Downloads are size-bounded, checked against the publisher's MD5 and given a
local SHA-256. Failed downloads are removed; existing artifacts are verified,
not overwritten. Downloads require space for the pinned file plus a 1 GiB
reserve. The label JSON acquired locally has SHA-256
`fe79a49d219988613f67faa580df0f860d9092b64cbb77a4da8494320f7038a8`.

The [official v3 record](https://zenodo.org/records/16414116) lists 31,844
reports across five volumes; volume 2 contains 6,369 report JSON files and is
2,620,638,102 bytes (MD5
`aee86b4591a46c69b0d027de80ff1011`). It expands to about 154 GB, so use the
bounded report preparer and never unpack the volume wholesale. Its AVClass
family labels come from the same separately pinned consensus file.

Preparation verifies both pinned artifacts before decompression. Only regular
JSON report members named with SHA-256 hashes are selected. Archive bytes go
to a bounded in-memory writer, then immediately to compact normalized JSONL;
no archive member or executable is extracted to the filesystem. Per-report,
total decompression and output limits are enforced. The reader runs serially,
releasing each report buffer after parsing. Invalid reports are counted. Output
is finalized only on successful extraction and cannot overwrite existing files.
The sidecar manifest records input checksums, limits and counts.

Current normalization (`cape-pe-locations-v2`) reads both `static.pe` and the
`target.file.pe` layout documented in [Table 2 of the WinMET paper](https://zaguan.unizar.es/record/165301/files/texto_completo.pdf).
It supports mapping-form DLL imports and hexadecimal PE size fields, and restores
nested imphash to metadata for train-only prototype matching. Hash identities and
vendor verdicts are not model tokens. YARA matches use the whole-rule structural
allowlist; cached generic matches without provenance are excluded.

Old prepared caches omitted `target.file.pe`; re-run preparation to recover its
fields. The manifest now reports Signature/Static counts and coverage for imports,
sections and dynamic events. A repeated volume-2 sample with seed 44 retained the
same 37 blocks/2,500 reports/3,671,889 events, but recovered 367,560 atomic tokens
instead of 17,521. New tensorizers preserve observed feature kinds under the
256-token budget, so a long import list cannot consume every section/entropy slot.

Solid archives may need to decode skipped members to reach a selected report;
the total decompression budget also covers these bytes and can abort the run.
Increase limits only after reviewing available memory/disk and the archive
layout. Do not run full-volume extraction: one volume expands to about 154 GB.
This protects against accidental resource exhaustion, not every possible bug
in a decompression/JSON library; parsing untrusted reports is not malware
execution and is not equivalent to a hardened VM sandbox.

The first real diagnostic prefix was prepared from volume 1: 32 reports, 46,417
normalized events, 222 static evidence items, 25 AVClass families, and no invalid
reports. It is archive-order data, not a representative benchmark; family support
is sparse and the prefix carried no explicit ATT&CK/MBC weak labels. Eight reports
had partial Run/RunOnce evidence, but none had a complete same-key open/write
chain. Do not train or quote accuracy from this prefix. AVClass consensus supplies
family labels; missing consensus stays unlabeled, never "benign". CAPE verdicts do
not replace AVClass targets. A production experiment still needs representative
sampling, leakage-controlled splits, independent behavior annotations and benign
dynamic traces.

A larger bounded run on 2026-10-26 prepared the first 1,700 eligible reports in
archive order: 17,967,304,865 decompressed report bytes, 2,494,447 normalized
events, 11,931 static tokens, 1,594 AVClass-labeled reports across 263 families,
and zero invalid reports. Fifty-three families have at least five reports and 29
have at least ten; 106 reports have no AVClass consensus and remain unlabeled, not
benign. The 1.95 GB JSONL is a diagnostic artifact ignored by Git. It is neither
random nor family-stratified, so these support counts do not establish a valid
train/test split. Fifty-eight reports reach the configured 2,048-event cap and
need truncation-coverage review. No report in this cohort carries explicit weak
ATT&CK/MBC labels, so this run does not train or validate the behavior head. Static
evidence is still mostly file metadata, string categories, TLSH/ssdeep presence
and YARA categories—not imports, PE sections, entropy or signer verification.

To reduce archive-prefix bias without extracting the volume, the preparer also
supports deterministic selection from complete 7z solid blocks. On the verified
volume 1 archive used here, 79 blocks are available. The bounded seed-42 run was:

```bash
malware-hybrid prepare-winmet Data/raw/winmet/WinMET_volume_1.7z \
  --labels Data/raw/winmet/reports_consensus_label.json \
  --output artifacts/winmet/volume1_random_blocks_1k.jsonl \
  --max-reports 1000 --max-report-mib 64 --raw-budget-mib 40000 \
  --output-budget-mib 2048 --max-events 2048 \
  --sampling random-solid-blocks --seed 42 --max-solid-blocks 16
```

This selected 1,000 reports from 15 blocks, processed 31.9 GB of block data
within the configured decompression budget, wrote 1.15 GB of normalized JSONL,
and produced 1,480,653 events and 7,001 static tokens with zero invalid reports.
The sample is still clustered by solid block and is not a representative
full-volume sample. The trainer therefore assigns whole blocks to one partition:
663/92/89/156 reports in train/validation/calibration/withheld test, covering
11/1/1/2 blocks. Validation and calibration each depend on a single block, and
the test partition remains unevaluated.

On this one-seed family diagnostic (support threshold 5, ranking weight 0.5),
attention validation mAP/macro-F1/top-1/top-3 was 0.297/0.159/0.431/0.523; the
matched bag model was 0.302/0.092/0.431/0.692. Only 22 family labels had positive
validation support. Attention helped calibrated macro-F1, but not mAP or top-3;
this does not establish a general attention advantage. AVClass is weak family
supervision, and the behavior head stayed disabled. The full JSONL, report and
checkpoint are ignored local artifacts; no executable was extracted or run.
Repeat with additional seeds/blocks and independently annotated family and
behavior labels before making benchmark claims.

The evidence engine was then run over all 1,700 reports. Rule-derived chains
produced positive weak-label support for 12 of the 15 techniques; for example,
T1547.001 had 493 `suspected` and zero `confirmed` chains. These are not manual
annotations, and the “not detected” outcomes are not valid negative labels. The
behavior head therefore remains untrained.

A resource-bounded family pilot used `--max-reports 500 --seed 42` from that
prefix, a 65/10/10/15 deterministic report split, a support-5 family vocabulary,
and a labelwise family-attention head. The 14-family model's held-out set had only
34 labeled reports: mAP 0.249 versus frequency-prior 0.050, top-1 0.412 versus
0.382, top-3 0.735 versus 0.559, but macro-F1 only 0.049. A lower support-2 run
had top-1 0.245 versus its 0.265 prior baseline. This is a promising ranking
signal, not a satisfactory classifier: the cohort is archive-order biased, the
test set is tiny, and family macro-F1 remains poor. The checkpoint is a local,
ignored exploratory artifact at `models/winmet_500_family_support5_probe.pt` and
is not suitable for security decisions. It can be exercised in the dashboard to
verify the report-to-attention inference path, with abstention and behavior-head
warnings preserved.

The CLI and dashboard now expose a deterministic report cap so long JSONL files
are fully scanned without retaining every decoded event graph in memory. The UI
defaults to 600 CAPE/WinMET reports (set 0 only when the machine can hold the full
dataset). Train progress reports batch and validation milestones; each newly best
validation checkpoint is written atomically, then replaced by its calibrated
version after training completes. This preserves the last valid checkpoint if a
later epoch or calibration step fails, but it does not yet resume optimizer state.

## Excluding prior solid blocks for a frozen check

`prepare-winmet --exclude-manifest PATH` can be repeated to keep a new sample out
of every solid block listed by earlier sidecars. The selected and excluded IDs
are preserved in the new sidecar. Example: all previously used random-block
family cohorts (1k, 1.6k and 2.5k reports) are excluded below. The 2.5k sidecar
name contains the historical word `disjoint`, but that cohort overlapped earlier
cohorts; exclusion applies to the full union of all 56 previously used blocks.

```bash
PYTHONPATH=src .venv/bin/python -m malware_hybrid.cli prepare-winmet \
  Data/raw/winmet/WinMET_volume_1.7z \
  --labels Data/raw/winmet/reports_consensus_label.json \
  --output artifacts/winmet/volume1_remaining_blocks_20261027.jsonl \
  --max-reports 5000 --max-report-mib 64 --raw-budget-mib 92160 \
  --output-budget-mib 4096 --max-events 2048 \
  --sampling random-solid-blocks --seed 20261027 --max-solid-blocks 79 \
  --exclude-manifest artifacts/winmet/volume1_random_blocks_1k.jsonl.manifest.json \
  --exclude-manifest artifacts/winmet/volume1_random_blocks_1600_seed44.jsonl.manifest.json \
  --exclude-manifest artifacts/winmet/volume1_random_blocks_2500_seed202600_disjoint.jsonl.manifest.json
```

This left 23 blocks and 1,556 reports (zero invalid), with 2,294,296 events,
10,904 static/signature tokens and a 1.79 GB JSONL. The 23 selected IDs had zero
intersection with the 56 excluded IDs. This is block-disjoint from those prior
cohorts inside the same WinMET volume, not an independent dataset, not a temporal
holdout, and not a representative full-volume benchmark.

The frozen checkpoint can be evaluated on the new cohort without retraining or
threshold fitting:

```bash
PYTHONPATH=src:scripts .venv/bin/python scripts/evaluate_winmet_external_cohort.py \
  artifacts/winmet/volume1_remaining_blocks_20261027.jsonl \
  models/winmet_hybrid3_attention_2500_seed44.pt \
  artifacts/winmet/hybrid3_attention_2500_seed44.json \
  --report artifacts/winmet/hybrid3_attention_2500_seed44_external_blocks_20261027.json \
  --batch-size 8 --device mps
```

The evaluator verifies source/checkpoint fingerprints, reconstructs the original
split and training priors, requires source-block exclusions for the same archive,
and rejects block-ID/sample-hash overlap. Across volumes, it requires verified
archive entries from the same pinned record without requiring foreign block
exclusions. AVClass labels cover 1,456 rows; only 1,106 rows
(76.0%) include a family known to the model, across 48 supported labels. On those
rows, frozen mAP/calibrated macro-F1/top-1/top-3 were 0.1960/0.1657/0.5235/0.7269.
The same weights and thresholds without cross-branch attention scored mAP 0.1711
and calibrated macro-F1 0.1206 (top-1/top-3 lower by 1.63/3.71 points). This is
evidence that attention contributes on held-out blocks, not evidence that the
classifier is accurate enough for security decisions. The family labels are
weak, 350 labeled rows have unseen family names, malware verdicts are absent,
and the behavior head is untrained. Reports and checkpoint are local ignored
artifacts; no malware binary was executed.

Normalization preserves a report-local anonymous identity for each named
file/registry/network/service resource. Semantically similar but different
paths remain separate graph nodes; repeated access to the same path remains
one entity. IDs are used only for equality/edges, not learned graph tokens.
For common process APIs, successful `OpenProcess`/`NtOpenProcess` calls bind a
handle to an explicit target PID within the owning process; successful close
invalidates that binding, and reuse replaces it. For registry APIs, normalized
full names or relative `SubKey` plus a recognized hive identify a key; successful
`RegOpenKeyEx`/`NtOpenKeyEx` and create calls bind output handles to that entity.
`RegSetValueEx`/`NtSetValueKey` resolve through the handle where the report logs
one. Failed acquisition, unknown owners, conflicting PIDs/keys, unknown handles,
and unresolved file/key/socket handles stay unknown. Duplicate/inherited handles,
thread-handle-to-process resolution and complex socket endpoints remain
limitations. The current
`T1547.001` evidence guard covers the standard registry `Run`/`RunOnce` path
subset (including `WOW6432Node`) and same-key open/write chains; it does not
cover startup folders or every autorun location. These behaviors have synthetic
counterexample tests and the argument schema was checked against
the bounded real prefix; remaining conflicts are not forced into a shared target.
Event truncation selects globally earliest calls across all processes before
repeat compression, not a prefix of the first process.

The process-handle argument conventions were cross-checked against the
[CAPEv2 process injection signatures](https://github.com/kevoreilly/CAPEv2/blob/master/modules/signatures/CAPE.py)
and Microsoft's documentation for [OpenProcess](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-openprocess),
[process handles and identifiers](https://learn.microsoft.com/en-us/windows/win32/procthread/process-handles-and-identifiers),
and [NTSTATUS success values](https://learn.microsoft.com/en-us/windows-hardware/drivers/kernel/using-ntstatus-values).

After inspecting a prepared cohort, the ordinary `train` command accepts its
JSONL path. The larger prefix permits exploratory family-head runs, but do not
present archive-order metrics as representative results or claim behavior
classification from this cohort without independently reviewed labels.
