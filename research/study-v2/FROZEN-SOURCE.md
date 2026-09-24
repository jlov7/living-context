# Study-era source snapshot

`frozen-source/` contains exact copies of the nine source files named in
`protocol.json`'s `source_files` map, read from the recorded source commit
`a2a7ed3b6a7bb795bef4082639cf8319354e4526`. The current release
candidate changes package metadata and other maintenance files. Those current
paths must not be mistaken for the code and dependency lock used during the
2026-09-22 model run.

`python -m scripts.verify_study_v2` verifies each snapshot file against the
protocol's SHA-256 value before replaying the published text evidence. This is
source custody, not a new model run. An actual model rerun would require a
separate clean checkout of the recorded commit, the exact third-party model
and tokenizer under their own terms, the frozen runtime dependencies, and the
protocol's full time, memory and cost accounting. The public source package
contains no model or learned-prefix tensor bytes.
