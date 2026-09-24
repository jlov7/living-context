# Rerunning the static adapter pilot

The [public evidence replay](../../scripts/replay_study_v3.py) checks exact
source/evidence hashes, all eight pilot cells, raw responses, strict scores and
seven worker/resource receipts without MLX, a model, or tensor files:

```sh
python scripts/replay_study_v3.py
```

That is saved-output replay. An independent person has not rerun inference.
To attempt a new model run, first obtain the pinned
`mlx-community/Llama-3.2-1B-Instruct-4bit` snapshot
`08231374eeacb049a0eade7922910865b8fce912` under its own Meta Llama 3.2
Community License. This repository grants no right to redistribute the model
or the separately trained adapter tensors. Use the six model file hashes and
sizes in the frozen v2 `protocol.json`. Use Python 3.13.15, MLX 0.32.2,
MLX-LM 0.31.3 and psutil 7.1.0 on an Apple Metal host to match the local
runtime. The pinned `mlx_lm/lora.py` SHA-256 is in this study's
`protocol.json`; a changed trainer fails preflight. Verify the model license,
package versions, and a new absent output directory before running:

```sh
python experiments/study_v3_qlora.py --preflight --model /path/to/pinned-model --output /path/to/new-output
python experiments/study_v3_qlora.py --run --model /path/to/pinned-model --output /path/to/new-output
python scripts/compare_inference.py --study v3 --candidate /path/to/new-output/pilot-raw.json --report /path/to/new-output/discrepancies.json
```

The runner sets offline environment flags and never requests a model by repo
name. It checks exact frozen source and input bytes before creating output,
then enforces 900 seconds, process-tree RSS of 8 GiB, and a reported MLX peak
of 2 GiB. The control failure stops the pilot. The discrepancy report retains
every reference/candidate response and token sequence, including differences,
missing cells and extra cells. Hardware, MLX numerical behavior and training
order may produce different adapters even if the method and file hashes match.
Publish discrepancies rather than selecting a favorable rerun. The original
local adapter hashes/sizes are in [RESULTS.json](RESULTS.json), but their bytes
are not public, so an outside runner cannot directly infer from the original
adapters or rehash them.
