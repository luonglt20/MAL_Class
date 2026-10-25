# MalBehavD-V1 auxiliary dynamic sequences

Source: [the dataset authors' public repository](https://github.com/mpasco/MalbehavD-V1),
which declares an MIT license. The CSV is stored locally at
`Data/raw/malbehavd/MalBehavD-V1-dataset.csv`; its byte size and SHA-256 are pinned
in `configs/datasets.lock.json`.

The acquired file has 2,570 rows, 1,285 for each binary label, 2,543 unique
SHA-256 values, 27 repeated-hash rows, and ordered API sequences of length 2–175
(median 37). The header's numeric columns are sequence positions; trailing empty
columns are padding. The dataset contains API names only—no executable bytes,
arguments, return values, process/resource relationships, malware family, or
ATT&CK/MBC labels.

Before any train/validation/test split, deduplicate by SHA-256. Preserve each
sequence as a separate auxiliary dynamic-only sample; never pair it with a PE or
CAPE report from another source to manufacture a hybrid sample. Use it only for
an explicitly labeled sequence/verdict experiment. Missing APIs in these short
sequences are not strong negative evidence for individual ATT&CK techniques, and
the dataset cannot supervise malware-family classification. Its traces also have
unknown API return status, so an adapter must preserve that uncertainty rather
than mark every call successful or failed.

No model has yet been trained or scored from this CSV. The adapter and an
independent, hash-deduplicated evaluation remain required.
