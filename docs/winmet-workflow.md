# WinMET: bounded, report-only workflow

Source: [WinMET v3](https://zenodo.org/records/16414116), with source checksums,
sizes and API license metadata pinned in `configs/winmet.lock.json`.
The [record API](https://zenodo.org/api/records/16414116) declares
GPL-3.0-or-later. Keep attribution and applicable dataset license terms with
derived datasets; this metadata does not change the repository's code license.

Install the optional reader and fetch only one volume plus labels:

```bash
pip install -e '.[winmet]'
malware-hybrid download-winmet reports_consensus_label.json
malware-hybrid download-winmet WinMET_volume_1.7z
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

Preparation verifies both pinned artifacts before decompression. Only regular
JSON report members named with SHA-256 hashes are selected. Archive bytes go
to a bounded in-memory writer, then immediately to compact normalized JSONL;
no archive member or executable is extracted to the filesystem. Per-report,
total decompression and output limits are enforced. The reader runs serially,
releasing each report buffer after parsing. Invalid reports are counted. Output
is finalized only on successful extraction and cannot overwrite existing files.
The sidecar manifest records input checksums, limits and counts.

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
have synthetic counterexample tests and the argument schema was checked against
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
