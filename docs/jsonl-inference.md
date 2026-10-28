# Streaming JSONL inference

The analysis tab now has **Phân loại dataset JSONL (CAPE / prepared)**. This is
inference, not the dataset-training tab. Enter a local `.jsonl` path, choose the
format and a record cap, then download the per-record CSV.

Supported explicit formats:

- **Prepared JSONL của MAL_Class**: records produced by the existing CAPE/WinMET
  preparation pipeline (`sample_hash`, `atomic_evidence`, `events`, `metadata`).
  Reference family/verdict labels and weak TTP labels are removed from the
  inference record. Graphs are rebuilt from events, not taken from serialized
  graph edges. Normalization provenance still passes through the model's schema
  guard. A supplied version string is not proof the underlying data is genuine.
- **Raw CAPE JSON mỗi dòng**: one raw CAPE report object per UTF-8 line, passed
  through the existing normalizer; reference labels are not used for prediction.

This is not an EMBER feature-array reader, an archive extractor, a sandbox or
an executable runner. PE uploads and local PE/report directories keep their
existing separate controls. No API or external sample submission is involved.

## Limits and provenance

Default limits are 8 MiB per line, 256 MiB scanned per operation and at most 500
nonblank records (UI defaults to 100). The library enforces upper bounds even if
called outside the UI. Files larger than the scan budget can still be analyzed
as a clearly reported prefix; they are not loaded wholesale into memory.

Invalid records count against the record cap, blank lines against the byte cap.
An oversized line is drained in 64 KiB chunks up to the scan budget and produces
one error row; it is never split into artificial samples. Later valid records
are retained if the budget permits. A byte limit in the middle of a record is
an error, not a parsed partial JSON object. Symlink final targets and nonregular
files are rejected; nonblocking open prevents waiting on a named pipe.

Prepared events require finite, ordered timestamps and typed scalar context;
the 2,048-event limit is enforced. Input event indices are canonicalized from
order. Atomic evidence numbers must be finite and bounded. Saved graph claims
and reference labels cannot manufacture execution edges or supervised targets.
These checks do not establish semantic correctness of an untrusted prepared
record; prefer datasets produced from verified reports by this project.

The UI shows records/lines/bytes read, original file size, limits, completion,
stop reason and whether file size/mtime changed during the operation. Every CSV
row includes its source line number, per-record failure, routing, abstention
and family result. A `record_limit`/`byte_limit` prefix is explicitly **not** the
whole dataset and is not random or representative sampling.

## Verification on real data (2026-10-28)

The local prepared WinMET file is 2,933,043,067 bytes. A two-record smoke run
read only 1,687,901 bytes and reported `complete=false`, `stop_reason=record_limit`.
Both records passed parsing, used the registered `winmet-relv3-seed44` hybrid
model and abstained at UNKNOWN with explicit reasons. No report failed and no
sample was executed. This verifies streaming/routing, not family accuracy.

The same two-record operation also passed through Streamlit AppTest using the
actual installed model registry (not the test fallback): two hybrid rows,
no exceptions, no per-row errors and a visible `record_limit` warning.

Unit tests cover oversized-line draining, EOF without newline, malformed JSON
and UTF-8, byte/record caps, empty input, nonfinite/unordered events, ignored
ground truth/serialized graph, and symlink/FIFO rejection. Dashboard AppTest
covers mixed valid/invalid rows, CSV configuration and explicit prefix warnings.
Complete regression suite after this addition: 248 tests + 80 subtests pass,
with one third-party Streamlit/Pandas deprecation warning. Model weights,
thresholds and the default model registry were not changed.
