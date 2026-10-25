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

The initial prefix is for schema/flow diagnostics, not a representative family
benchmark. AVClass consensus supplies family labels; missing consensus stays
unlabeled, never "benign". CAPE verdicts do not replace AVClass targets. Behavior
rules generate weak/derived targets, not independent annotations. A production
experiment still needs representative sampling, leakage-controlled splits,
independent behavior annotations and benign dynamic traces.

Normalization preserves a report-local anonymous identity for each named
file/registry/network/service resource. Semantically similar but different
paths remain separate graph nodes; repeated access to the same path remains
one entity. IDs are used only for equality/edges, not learned graph tokens.
Handles are scoped by process but are not yet resolved to underlying named
objects; handle aliases, handle reuse and complex socket endpoints remain
limitations. Event truncation selects globally earliest calls across all
processes before repeat compression, not a prefix of the first process.

After inspecting a prepared cohort, the ordinary `train` command accepts its
JSONL path. Do not report an accuracy based only on this small diagnostic prefix.
