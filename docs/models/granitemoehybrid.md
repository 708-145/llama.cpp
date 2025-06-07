# GraniteMoeHybrid Architecture in llama.cpp

GraniteMoeHybrid is a hybrid architecture that combines standard attention layers with Mamba (SSM) layers, and utilizes a Mixture-of-Experts (MoE) approach for its feed-forward networks, supplemented by a shared MLP.

## GGUF Model Configuration

To use a GraniteMoeHybrid model with `llama.cpp`, the GGUF file must contain specific metadata parameters. These are typically set during the model conversion process (e.g., using an updated `convert_hf_to_gguf.py` script).

### Key Hyperparameters:

The following GGUF keys (under the `{arch}` namespace, which is `granitemoehybrid`) are essential for defining the GraniteMoeHybrid architecture:

*   **`granitemoehybrid.block_count`**: (Integer) The total number of layers in the model. (This is the standard GGUF key `llama.block_count` but namespaced for clarity in documentation; `llama.cpp` will read the generic `block_count` for the specified architecture).
*   **`granitemoehybrid.layer_type.{i}`**: (Array of Integers, where `i` is the layer index from 0 to `block_count - 1`) Specifies the type of each layer.
    *   `0`: Indicates an Attention layer.
    *   `1`: Indicates a Mamba (SSM) layer.
    *   *Example*: `granitemoehybrid.layer_type.0 = 0`, `granitemoehybrid.layer_type.1 = 1`
*   **`granitemoehybrid.shared_intermediate_size`**: (Integer) The intermediate size of the shared MLP block present in each layer. (Corresponds to `hparams.n_ff_shared` in `llama.cpp`).

### Standard MoE Parameters:

These are also required, following general MoE model conventions, and should be namespaced under `granitemoehybrid`:

*   **`granitemoehybrid.expert_count`**: (Integer) Total number of experts available in each MoE layer. (Corresponds to `hparams.n_expert`).
*   **`granitemoehybrid.expert_used_count`**: (Integer) Number of experts to use per token. (Corresponds to `hparams.n_expert_used`).
*   **`granitemoehybrid.expert_feed_forward_length`**: (Integer) Intermediate size of the FFN within each expert. (Corresponds to `hparams.n_ff_exp`).

### Mamba (SSM) Parameters (if Mamba layers are used):

If any layer is specified as a Mamba layer, the following standard SSM GGUF keys are used (namespaced under `granitemoehybrid`):

*   **`granitemoehybrid.ssm.conv_kernel`**: (Integer) Convolution kernel size for Mamba layers.
*   **`granitemoehybrid.ssm.inner_size`**: (Integer) Inner size (expanded dimension) for Mamba layers.
*   **`granitemoehybrid.ssm.state_size`**: (Integer) Dimension of the Mamba state vector.
*   **`granitemoehybrid.ssm.time_step_rank`**: (Integer) Rank for the time step projection in Mamba layers.

### Standard Architecture Parameters:

Standard parameters like `granitemoehybrid.embedding_length`, `granitemoehybrid.attention.head_count`, `granitemoehybrid.attention.head_count_kv`, `granitemoehybrid.attention.layer_norm_rms_epsilon`, `granitemoehybrid.rope.freq_base`, `granitemoehybrid.rope.scaling.type`, etc., are also required as per other Transformer models, but namespaced for this architecture. `llama.cpp` will look for these under the `granitemoehybrid.` prefix.

### Granite-Specific Scaling Parameters:

GraniteMoeHybrid also utilizes scaling factors that can be set in the GGUF file (namespaced):
*   **`granitemoehybrid.logit_scale`**: Scale factor for the final logits.
*   **`granitemoehybrid.residual_scale`**: Scale factor applied to residual connections.
*   **`granitemoehybrid.embedding_scale`**: Scale factor for token embeddings.
*   **`granitemoehybrid.attention.scale`**: Scale factor for attention scores.


## Tensor Naming

The GGUF file should contain tensors corresponding to the architecture. Key GGUF tensor names (as expected by `llama.cpp` after mapping by `convert_hf_to_gguf.py`) include:
*   `token_embd.weight`
*   `output_norm.weight`
*   `output.weight`
*   For each layer `blk.{i}`:
    *   `attn_norm.weight` (This is the input layernorm for the block, applied before either Attention or Mamba)
    *   If Attention Layer:
        *   `attn_q.weight`, `attn_k.weight`, `attn_v.weight`
        *   `attn_output.weight`
        *   Optional biases: `attn_q.bias`, `attn_k.bias`, `attn_v.bias`, `attn_output.bias`
    *   If Mamba Layer:
        *   `ssm_in.weight`
        *   `ssm_conv1d.weight`, `ssm_conv1d.bias`
        *   `ssm_x.weight`
        *   `ssm_dt.weight`, `ssm_dt.bias`
        *   `ssm_a` (Note: no `.weight` suffix)
        *   `ssm_d` (Note: no `.weight` suffix)
        *   `ssm_out.weight`
    *   `ffn_norm.weight` (This is the layernorm applied after the Attention/Mamba block and its residual connection, before the MoE/SharedMLP part)
    *   MoE Router: `ffn_gate_inp.weight`
    *   MoE Experts (per expert): `ffn_gate_exp.weight`, `ffn_up_exp.weight`, `ffn_down_exp.weight`
    *   Shared MLP: `ffn_gate_shexp.weight`, `ffn_up_shexp.weight`, `ffn_down_shexp.weight`

## Implementation Notes

*   The `llama.cpp` implementation dynamically builds the computation graph for each layer based on the `granitemoehybrid.layer_type.{i}` parameter.
*   Mamba layer computations will leverage the existing SSM GGML ops. The current graph-building logic for Mamba layers within GraniteMoeHybrid might be a placeholder; a full, optimized Mamba implementation is complex.
*   The Feed-Forward Network (FFN) part of each layer consists of a sum of the outputs from the Mixture-of-Experts (MoE) block and a Shared MLP block. Both blocks take the same normalized output from the preceding Attention/Mamba block (after its residual connection) as input.
*   Residual connections are applied after the Attention/Mamba block and after the combined MoE+SharedMLP block.
```
