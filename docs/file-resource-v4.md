# Experimental file-resource relations

The optional `--file-handle-resolution` flag for `prepare-winmet` creates
`cape-resource-relations-v4` reports. It links successful, explicitly observed
file opens to later typed file operations by owner-local handle and anonymous
path identity. Successful closes and subsequent typed acquisitions invalidate
stale bindings. Failed opens, invalid handles, unknown owners, relative paths
and conflicting names cannot establish a binding. Windows documents file handles
as process-local with duplication/inheritance exceptions, which this experimental
resolver deliberately does not infer ([Microsoft file handles](https://learn.microsoft.com/en-us/windows/win32/fileio/file-handles),
[NtCreateFile](https://learn.microsoft.com/en-us/windows-hardware/drivers/ddi/ntifs/nf-ntifs-ntcreatefile)).

Existing v3 parsing remains the default. The version guard refuses v4 reports
for a checkpoint trained on v3; no deployed checkpoint or threshold was changed.
This schema requires separate preparation, training and evaluation before use.
Path equality does not prove filesystem object identity across renames, aliases,
deletion/recreation, inherited handles or incomplete sandbox traces.

On four small archive-prefix WinMET reports, the parser recovered 145 previously
unnamed file events. File-event counts changed from 578 to 641 because some
close events were newly identified as file operations; named file events changed
from 249 to 394. These are parser diagnostics on four reports, not representative
coverage or classification accuracy. Event order/context, static evidence and
reference labels were checked unchanged, with repeat collapsing disabled. A
larger eight-report attempt hit the 512 MiB decode guard; the budget was not
raised. No sample was executed and no predictions were evaluated.

Tests cover process isolation, close and handle reuse, conflicting and failed
operations, schema compatibility, and the same-resource bias reaching temporal
attention. The full suite passed 285 tests and 80 subtests. Next, prepare a
matched v4 cohort and evaluate its effect against v3 and relevant baselines;
independent behavior annotations and benign hybrid validation remain necessary.
