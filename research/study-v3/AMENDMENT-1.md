# Pre-outcome supervisor amendment

This amendment was committed after the model-file preflight, before any
training or inference outcome. The first frozen runner is retained in Git at
`c3b79af`; no model run used it. The positive-control question, answer,
training steps, adapter configuration, four pilot facts, eight questions,
quality gate and 900-second budget are unchanged.

Review found that the first runner could miss memory reports split across log
polls, accept a missing training report, leave subprocesses after a limit,
count a valid raw output despite a worker resource failure, and overcharge
cached-text timing. The amended runner checks complete log snapshots and the
final tail, requires a report for every model worker, kills the owned process
group on a limit, gates quality only after all required workers completed, and
measures cached text around its own call. The training data now uses the same
system/user/assistant chat template as inference; the first runner's
prompt/completion format omitted the system message. Model-free supervisor
tests exercise these boundaries. A denied process-tree RSS read now fails the
resource gate. Cached-text serving time is the cached generation call; the
fresh/cached parity check has its own validation time and is not silently
charged to text serving alone. `protocol.json` records the amended runner
and tests by SHA-256. The host model preflight must be rerun against the
amended bytes before any training begins.
