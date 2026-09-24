# QLoRA source-adapter pilot, preregistered before any model outcome

Identity: `lc-study-v3-qlora`. The v2 embedding soft-prefix development gate
remains closed. This pilot changes the representation: rank-4 LoRA matrices on
the last eight attention layers of a frozen, quantized Llama 3.2 1B model.
The source-derived QA objective trains an adapter per document. It does not
add virtual prefix tokens or use v2 tensors. This is a standard QLoRA method,
not a new adapter algorithm; see [LoRA](https://arxiv.org/abs/2106.09685)
and the [MLX-LM implementation](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md).

The model snapshot is the exact cached v2 revision and six file hashes from
`research/study-v2/protocol.json`. Code, inputs and installed `mlx_lm/lora.py`
are hash checked before a worker starts. The runner uses only a local model
path and sets network-offline environment flags. No model or adapter bytes go
into the public source package.

First train the Cobalt positive control for 200 iterations on its source QA.
Its exact held-in question must yield `{"answer":"CB-73184"}` with strict
single-object parsing under greedy 32-token generation. A failure stops the
study before any pilot question is scored. No easier target, additional seed,
or extra steps may be selected after the result.

After a passing control, train four disjoint source adapters for 100 iterations
each. The four synthetic source facts span routing and permit document
families. The facts are present in source QA training; the eight evaluation
question wordings and labels are frozen separately and never passed as training
inputs. Compare the adapter with the same base reader given no context and
with actual revision-keyed cached source text. All arms receive the same eight
questions, chat template, greedy decoder, 32-token cap and exact JSON answer
contract. Retain every raw response, including malformed output.

The exploratory quality gate requires at least 7/8 adapter exact and no fewer
correct than cached text. If it fails, close this pilot. Even a pass would
only justify a larger, separate preregistered revision study. It would not
establish deletion, unlearning, lifecycle savings, broad generalization, or
product usefulness. Report adapter bytes, cached text bytes, all training and
query wall times, and peak RSS/MLX memory samples. No prices.

The supervisor enforces a cumulative 900-second worker wall limit and 8 GiB
RSS limit. It reads each one-step training report and stops when reported MLX
peak exceeds 2 GiB. Missing memory reports are a failed resource gate. A
model/runtime failure, unavailable GPU, or budget stop is retained as a
non-result, never turned into a quality score. The public evidence can include
only non-sensitive synthetic data, hashes, relative paths and aggregate
hardware descriptors. An outside rerun is not claimed here.
