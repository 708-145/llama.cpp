# DNA performance: current issues

This note describes performance costs visible in the current CPU implementation.
It distinguishes observed behavior from likely bottlenecks inferred from the
operator structure; the latter need profiling before being treated as measured
results.

## Summary

DNA can save gate/up and down-projection work when many neurons are silent, but
its current operators are row-oriented custom CPU kernels. They still evaluate
the predictor for every neuron and process tokens one at a time. As a result,
the fraction of neurons marked active is not itself a speedup measurement.
Large batches use the dense FFN by default, and DNA also adds calibration,
weight-packing, and persistent-memory costs.

The current results in [DNA.md](DNA.md) report perplexity, predictor calibration correlation and active fractions, plus setup and whole-run times for smaller predictor quants. They do not provide matched dense-versus-DNA throughput or peak-memory measurements. No general speedup claim can be made from those results alone.

## Current costs and limitations

### 1. ✅ Large dense-FFN batches fall back to dense execution

`llama_dna_params::max_batch` defaults to 32. In `build_ffn`, a predicting
dense/shared FFN is switched back to the ordinary dense path when its token
dimension exceeds this value. This is a performance guard, not an apparent
shape/correctness limit in the custom route/down operators.  `LLAMA_DNA_MAX_BATCH` can raise the guard; existing perplexity recipes use 512.

This particularly limits prompt prefill: batches above the configured cap get no DNA skipping in those FFNs, even after calibration. MoE DNA has a separate path and does not use this `max_batch` check.

This is fine for DNA. Future approaches will use sparsity inside matrix-matrix multiplication as well. <future>

References: [`llama_dna_params`](src/llama-dna.h), [`build_ffn`](src/llama-graph.cpp),
and the batch-limit notes and recipes in [DNA.md](DNA.md).

### 2. ✅ Route work remains proportional to every neuron and token

For each token, `op_route` quantizes the input for the predictor and gate/up
dot-product types, then estimates the gate value for every FFN neuron. Only
after that does it skip the true gate/up dot products for neurons classified as silent. Active neurons still require separate row dot products for gate and up.  
The loops are token-major and call row-vector kernels; they do not turn a token batch into a batched matrix multiply.

Thus, increasing sparsity avoids some gate/up reads and dot products, but does not remove the all-neuron predictor scan. At high active fractions, the remaining gate/up work may also leave little room to offset predictor overhead.

This is fine for DNA. Future approaches will use sparsity inside the predictor as well. <future>

Reference: [`op_route`](src/llama-dna.cpp).

### 3. Down projection uses per-neuron AXPY updates

`op_down` initializes each output row from `c0`, scans the full neuron delta for each token, and applies an AXPY for every nonzero delta using the transposed F16 down weights. This avoids computing silent neurons, but does not use a batched matrix-multiply kernel. With larger token counts, the same active weight rows are revisited for each token; whether this is slower than dense execution depends on sparsity, model dimensions, cache behavior, and CPU.

Open TODO: need faster down projection in the sparse regime. 

Reference: [`op_down`](src/llama-dna.cpp).

### 4. Intermediate and persistent memory grow with model/batch size

- The route output (`delta`) contains `n_ff * n_tokens` F32 values, or
  `4 * n_ff * n_tokens` bytes per layer before allocator reuse. For example,
  `n_ff = 11008` and `n_tokens = 512` is about 21.5 MiB for one such tensor.
- After bias calibration, each layer retains `down_t`, an F16 copy with
  `n_ff * n_embd` elements, in addition to the model's original down weights.
- Each enabled layer also retains its packed predictor gate (`pred_q`) and
  per-neuron calibration/prediction data. MoE layers can hold corresponding
  state per expert.

The delta tensor is batch-dependent; the transposed weights and predictor are model/layer-dependent. Batching or removing the batch guard should therefore be measured for peak memory as well as speed.

Open TODO: remove the calibration data after prediction has started. Decision per layer. In low RAM regime calibration can be done with as many layers in parallel as free RAM allows. Start new layers when any calibration ends.  
Open TODO: `down_t` needs to be replaced/implemented by an efficient sparse down transformation. 

References: the `delta` allocation in [`build_ffn`](src/llama-graph.cpp), and `down_t`/`pred_q` storage in [`llama-dna.cpp`](src/llama-dna.cpp).

### 5. Setup and calibration delay the point at which skipping begins

When a layer is enabled, `llama_dna_pack_gate` reads/dequantizes its gate rows and quantizes them into the predictor format. This is a one-time per-layer setup cost and adds predictor storage.

Open TODO: The predictors can be precomputed and stored in the GGUF.

Calibration runs the ordinary dense FFN and adds observers for two phases, each configured by `LLAMA_DNA_CALIB`. Prediction savings therefore begin only after the layer completes both phases; for short prompts or large calibration settings, DNA may spend much or all of the workload calibrating. The Gemma results in [DNA.md](DNA.md) include a four-chunk, `n_calib=4096` run where no layer reached PREDICT.

Open TODO: The calibration result can be saved into a side car file or into the GGUF (needs to reserve metadata fields).

Each run's recovered PPL for chunks 1–8 is recorded in
[DNA.md](DNA.md). Those values are segment estimates derived from cumulative
PPL, not independent evaluator runs for each chunk.

References: [`llama_dna_pack_gate`](src/llama-dna.cpp),
[`op_observe`](src/llama-dna.cpp), and the calibration results in [DNA.md](DNA.md).

## Measurement gaps

The in-tree `LLAMA_DNA_VERBOSE` reporting includes per-layer observe, route, and down timings, but the documented sweeps do not provide matched dense-versus-DNA tokens/s, end-to-end latency, or peak RSS. They also mix different calibration lengths, chunking, and prediction coverage.  Consequently, the net speed benefit and the batch-size crossover are not yet established.

Before changing the default guard or claiming a speedup, compare dense and DNA under the same model, CPU/thread count, context, and input with optimized codepaths for sparse execution.  
Record:

- load/setup and calibration time separately from steady-state prediction;
- prompt throughput and decode throughput/latency;
- peak memory, including graph workspace;
- per-layer active fraction and route/down timings;
- output quality/perplexity under the same calibration and prediction coverage.

