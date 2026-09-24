# Related work

Living Context's maintained software is a model-free revision catalog, complete-snapshot admission mechanism, text reader and refresh planner. Its learned-context experiments are separately bounded: the corrected soft-prefix study stopped at a negative development gate; the static and revision adapter studies used a few synthetic facts. No lifecycle-cost advantage or general learned-context update algorithm has been established.

The primary-source descriptions below were checked on 2026-09-24. This is a selected reading list, not a systematic search or a novelty assessment. It does not infer the absence of revision-related capabilities from a paper's abstract.

| Work | Relevant scope | Relation to this repository |
| --- | --- | --- |
| [Cartridges at Scale](https://arxiv.org/abs/2606.04557) | Trains modular document KV caches with dynamic distractor mixing and manages large collections across GPU and persistent storage. | Motivates questions about composition and the cost of maintaining learned representations. Living Context's embedding soft prefixes are a different representation; its small experiments are not a reproduction of CAS. |
| [Where Should a Document Live: Context, Representations, or Parameters?](https://arxiv.org/abs/2609.17346) | Compares text, KV-cache representations and parameter adaptation on knowledge-intensive tasks, including retrieval and effects on other capabilities. | Shows why representation choice needs matched accuracy, storage and capability measurements. The local synthetic adapter pilot here does not reproduce that comparison or establish the same conclusions. |
| [Context Distillation as Latent Memory Management](https://arxiv.org/abs/2605.28889) | Stores distilled contexts in modular LoRA adapters, retrieves and routes among memories, and gates their activation. | Adapter-based latent memory and memory selection are prior work. This repository currently provides source-revision and admission boundaries rather than a learned retrieval or routing service. |

## The remaining research question

Can a revision-aware learned-context system preserve current-answer accuracy at lower total lifecycle cost than well-engineered cached text? Answering that requires matched information, tasks and serving conditions, with training, refresh, cache invalidation, storage, loading and inference all counted. The present studies do not establish where such a crossover occurs.

A document-level artifact is not, by itself, proof that learned information is isolated to that document. Any claim about cross-artifact dependencies, deletion or unlearning needs the actual training recipe and gradient routing, plus observations that test the claim. The declared dependency graph in this package is supplied by the caller; it is not an experimentally discovered information-flow graph.

The [research index](INDEX.md) distinguishes the corrected studies from superseded v1 claims. Their negative results and limits remain part of the release.
