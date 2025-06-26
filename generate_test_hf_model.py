import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import PreTrainedModel, PretrainedConfig
from transformers.utils import add_start_docstrings
from transformers.modeling_outputs import CausalLMOutputWithPast
import os
import json

# --- Configuration ---
class GraniteMoeHybridConfig(PretrainedConfig):
    model_type = "granite_moe_hybrid"

    def __init__(
        self,
        vocab_size=50257, # Standard GPT-2 vocab size
        hidden_size=32, # Small for testing
        num_hidden_layers=3, # e.g., [ATTN, MOE, MAMBA]
        layer_types=["attention", "moe", "mamba"], # Example sequence
        # Attention params
        num_attention_heads=4,
        attention_bias=False,
        # MoE params
        num_local_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=32, # ff_dim for experts
        shared_intermediate_size=0, # ff_dim for shared FFN (if any, 0 means no separate shared FFN)
        # Mamba params
        mamba_d_state=8, # ssm_d_state
        mamba_d_conv=2,  # ssm_d_conv
        mamba_expand=2,  # Expansion factor for mamba_d_inner
        # General params
        rms_norm_eps=1e-6,
        pad_token_id=0, # Often 0 for SentencePiece/GPT-2
        bos_token_id=1, # Often 1
        eos_token_id=2, # Often 2
        hidden_act="silu",
        rope_theta=10000.0,
        attention_multiplier=1.0, # Granite specific
        embedding_multiplier=1.0, # Granite specific
        residual_multiplier=1.0,  # Granite specific
        logits_scaling=1.0,       # Granite specific
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.layer_types = layer_types
        if len(layer_types) != num_hidden_layers:
            raise ValueError("Length of layer_types must match num_hidden_layers")

        self.num_attention_heads = num_attention_heads
        self.attention_bias = attention_bias

        self.num_local_experts = num_local_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size
        self.shared_intermediate_size = shared_intermediate_size

        self.mamba_d_state = mamba_d_state
        self.mamba_d_conv = mamba_d_conv
        self.mamba_expand = mamba_expand
        self.mamba_d_inner = hidden_size * mamba_expand # ssm_d_inner
        self.mamba_dt_rank = (hidden_size + 15) // 16 # ssm_dt_rank, math.ceil(hidden_size / 16)

        self.rms_norm_eps = rms_norm_eps
        self.hidden_act = hidden_act
        self.rope_theta = rope_theta # For RoPE in Attention
        self.attention_multiplier = attention_multiplier
        self.embedding_multiplier = embedding_multiplier
        self.residual_multiplier = residual_multiplier
        self.logits_scaling = logits_scaling

        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)


# --- PyTorch Modules ---

class SimpleRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class SimpleAttention(nn.Module):
    def __init__(self, config: GraniteMoeHybridConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
        # No RoPE implemented for simplicity in generator, GGUF conversion expects it though

    def forward(self, hidden_states, past_key_value=None, attention_mask=None, position_ids=None, output_attentions=False):
        # Simplified forward, no actual attention computation
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        attn_output = self.o_proj(q + k + v) # Dummy combination
        return attn_output, None, past_key_value # attn_weights, new_kv_cache

class SimpleMoE(nn.Module):
    def __init__(self, config: GraniteMoeHybridConfig):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.moe_intermediate_size

        # Simplified gate: just a linear layer
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)

        # For GGUF conversion, it expects specific tensor names like:
        # block_sparse_moe.experts.{idx}.w1.weight
        # block_sparse_moe.experts.{idx}.w3.weight (equivalent to up_proj for us)
        # block_sparse_moe.experts.{idx}.w2.weight (equivalent to down_proj for us)
        # We will rename them during saving or ensure converter handles this.
        # For this generator, we'll use a simpler expert structure.
        # The converter expects separate w1, w3, w2. We simulate this by defining
        # expert as: w1 -> act -> w2, and then we'd need a separate w3.
        # Or, if following Mixtral: gate_proj, up_proj, down_proj.
        # The current `convert_hf_to_gguf.py` for GraniteMoeHybridModel expects:
        # - experts.{xid}.w1.weight  (maps to FFN_GATE_EXP)
        # - experts.{xid}.w3.weight  (maps to FFN_UP_EXP)
        # - experts.{xid}.w2.weight  (maps to FFN_DOWN_EXP)
        # Let's adapt the expert structure to provide these.

        self.experts_w1 = nn.ModuleList([nn.Linear(self.hidden_dim, self.ffn_dim, bias=False) for _ in range(self.num_experts)])
        self.experts_w3 = nn.ModuleList([nn.Linear(self.hidden_dim, self.ffn_dim, bias=False) for _ in range(self.num_experts)]) # up_proj
        self.experts_w2 = nn.ModuleList([nn.Linear(self.ffn_dim, self.hidden_dim, bias=False) for _ in range(self.num_experts)]) # down_proj
        self.act_fn = nn.SiLU() if config.hidden_act == "silu" else nn.GELU()


    def forward(self, hidden_states):
        # Simplified forward: just pass through the first expert
        gating_output = self.gate(hidden_states)
        # In a real MoE, you'd use gating_output to select and combine experts.
        # Here, just use expert 0 for simplicity of generation.
        # expert_0_output = self.experts[0](hidden_states)

        # To match expected GGUF tensor names (w1, w3, w2):
        expert_0_w1_out = self.experts_w1[0](hidden_states)
        expert_0_w3_out = self.experts_w3[0](hidden_states) # This is the "up_proj" part
        activated_out = self.act_fn(expert_0_w1_out * expert_0_w3_out) # Gated MLP style: gate * up
        final_expert_output = self.experts_w2[0](activated_out)

        return final_expert_output

class SimpleMamba(nn.Module):
    def __init__(self, config: GraniteMoeHybridConfig):
        super().__init__()
        self.d_model = config.hidden_size
        self.d_state = config.mamba_d_state
        self.d_conv = config.mamba_d_conv
        self.expand = config.mamba_expand
        self.d_inner = self.d_model * self.expand # config.mamba_d_inner

        # Projections
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False) # Projects to x and z

        # Convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=self.d_conv,
            bias=True, # Mamba typically has bias in conv
            groups=self.d_inner, # Depthwise convolution
            padding=self.d_conv - 1,
        )

        # SSM parameters
        # x_proj projects from d_inner to dt_rank + d_state * 2
        self.dt_rank = config.mamba_dt_rank
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)

        # dt_proj projects from dt_rank to d_inner
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # A_log and D are parameters
        self.A_log = nn.Parameter(torch.randn(self.d_inner, self.d_state)) # Should be initialized carefully
        self.D = nn.Parameter(torch.randn(self.d_inner)) # Should be initialized carefully

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)


    def forward(self, hidden_states):
        # Simplified forward: just pass through projections
        # This is NOT a functional Mamba, just for structure.
        xz = self.in_proj(hidden_states)
        x, z = xz.chunk(2, dim=-1) # Split into two parts

        # Dummy conv application
        x_conv = self.conv1d(x.transpose(1,2)).transpose(1,2) # Needs B,C,L -> B,L,C

        # Dummy x_proj and dt_proj
        x_proj_out = self.x_proj(x_conv)
        dt, B_ssm, C_ssm = torch.split(x_proj_out, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt_out = self.dt_proj(dt)

        # Dummy combination
        # A real Mamba would use a selective scan here with A_log, B_ssm, C_ssm, dt_out, D
        # For structure, just combine them simply.
        y = self.out_proj(x_conv + dt_out) # Very simplified
        return y

class GraniteMoeHybridLayer(nn.Module):
    def __init__(self, config: GraniteMoeHybridConfig, layer_type: str):
        super().__init__()
        self.layer_type = layer_type
        self.hidden_size = config.hidden_size

        self.input_layernorm = SimpleRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # self.post_attention_layernorm = SimpleRMSNorm(config.hidden_size, eps=config.rms_norm_eps) # If using 2 norms

        if self.layer_type == "attention":
            self.block = SimpleAttention(config)
        elif self.layer_type == "moe":
            self.block = SimpleMoE(config)
        elif self.layer_type == "mamba":
            self.block = SimpleMamba(config)
        else:
            raise ValueError(f"Unknown layer type: {self.layer_type}")

    def forward(self, hidden_states, **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.layer_type == "attention":
            hidden_states, _, _ = self.block(hidden_states, **kwargs)
        else:
            hidden_states = self.block(hidden_states)

        hidden_states = residual + hidden_states
        return hidden_states


# --- Model ---
@add_start_docstrings(
    "GraniteMoeHybrid model.",
)
class GraniteMoeHybridPreTrainedModel(PreTrainedModel):
    config_class = GraniteMoeHybridConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = None # Add module names here if they shouldn't be split across devices

    def _init_weights(self, module):
        # Placeholder, actual initialization would be more complex
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, SimpleRMSNorm):
            module.weight.data.fill_(1.0)


class GraniteMoeHybridModel(GraniteMoeHybridPreTrainedModel):
    def __init__(self, config: GraniteMoeHybridConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [GraniteMoeHybridLayer(config, config.layer_types[i]) for i in range(config.num_hidden_layers)]
        )
        self.norm = SimpleRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        self.post_init() # Initialize weights

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids=None,
        attention_mask=None, # Simplified, not fully used by dummy layers
        position_ids=None,   # Simplified, not fully used by dummy layers
        past_key_values=None,# Simplified, not fully used by dummy layers
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # For attention layers, pass through cache and mask related args
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask, # Only used by actual attention
                position_ids=position_ids, # Only used by actual attention with RoPE
                past_key_value=past_key_values[idx] if past_key_values is not None else None,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            # Unpack based on what the layer might return
            if isinstance(layer_outputs, tuple): # Attention layer returns more
                 hidden_states = layer_outputs[0]
                 if use_cache:
                     next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)
                 if output_attentions:
                     all_self_attns += (layer_outputs[1],)
            else: # MoE, Mamba return only hidden_states
                hidden_states = layer_outputs


        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, next_decoder_cache, all_hidden_states, all_self_attns] if v is not None)

        return CausalLMOutputWithPast( # Using CausalLMOutput for consistency, though logits are missing
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class GraniteMoeHybridForCausalLM(GraniteMoeHybridPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = GraniteMoeHybridModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs):
        #This model currently does not support generation with past_key_values.
        #For generation, make sure that use_cache is set to False.
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the first call
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

# --- Renaming and Saving Logic ---
def rename_and_save_model(model, save_directory, config):
    os.makedirs(save_directory, exist_ok=True)

    # Save config
    # Add the 'architectures' field before saving
    config_dict = config.to_dict()
    config_dict["architectures"] = ["GraniteMoeHybridForCausalLM"]

    config_save_path = os.path.join(save_directory, "config.json")
    with open(config_save_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"Saved config with architectures to {config_save_path}")

    # Create state_dict with GGUF-expected names
    # This is crucial for the converter to find the tensors.
    renamed_state_dict = {}
    for name, param in model.named_parameters():
        new_name = name # Default to original name

        # Embeddings and LM head
        if name == "model.embed_tokens.weight": new_name = "model.embed_tokens.weight" # Standard
        elif name == "model.norm.weight": new_name = "model.norm.weight" # Standard
        elif name == "lm_head.weight": new_name = "lm_head.weight" # Standard

        # Layer-specific renaming
        # model.layers.0.input_layernorm.weight -> model.layers.0.input_layernorm.weight
        # model.layers.0.block.q_proj.weight -> model.layers.0.self_attn.q_proj.weight (for attention)
        # model.layers.0.block.gate.weight -> model.layers.0.block_sparse_moe.gate.weight (for moe gate)
        # model.layers.0.block.experts_w1.0.weight -> model.layers.0.block_sparse_moe.experts.0.w1.weight
        # model.layers.0.block.in_proj.weight -> model.layers.0.mamba.in_proj.weight (for mamba)

        parts = name.split('.')
        if len(parts) > 3 and parts[0] == "model" and parts[1] == "layers":
            layer_idx = int(parts[2])
            layer_type = config.layer_types[layer_idx]
            block_part_name = parts[4:] # e.g. ['q_proj', 'weight'] or ['experts_w1', '0', 'weight']

            if parts[3] == "input_layernorm":
                new_name = f"model.layers.{layer_idx}.input_layernorm.{'.'.join(block_part_name)}"
            elif parts[3] == "block":
                if layer_type == "attention":
                    # parts[4] is e.g. 'q_proj', 'k_proj', etc.
                    new_name = f"model.layers.{layer_idx}.self_attn.{'.'.join(block_part_name)}"
                elif layer_type == "moe":
                    if block_part_name[0] == "gate":
                        new_name = f"model.layers.{layer_idx}.block_sparse_moe.gate.weight"
                    elif block_part_name[0].startswith("experts_w"): # experts_w1, experts_w3, experts_w2
                        expert_type_key = block_part_name[0].split('_')[1] # w1, w3, w2
                        expert_idx = block_part_name[1]
                        new_name = f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.{expert_type_key}.weight"
                elif layer_type == "mamba":
                    # parts[4] is e.g. 'in_proj', 'conv1d', 'x_proj', 'dt_proj', 'A_log', 'D', 'out_proj'
                    # For A_log and D, parts[4] will be 'A_log' or 'D', and block_part_name will be ['A_log'] or ['D']
                    # after parts[4:] slicing.
                    if block_part_name[0] in ["A_log", "D"] and len(block_part_name) == 1: # A_log or D parameters
                        mamba_param_name = block_part_name[0]
                        new_name = f"model.layers.{layer_idx}.mamba.{mamba_param_name}"
                    else: # Linear or Conv1D weights/biases
                        mamba_comp_name = block_part_name[0] # e.g. 'in_proj'
                        if len(block_part_name) > 1:
                            attr_name = block_part_name[1] # 'weight' or 'bias'
                            new_name = f"model.layers.{layer_idx}.mamba.{mamba_comp_name}.{attr_name}"
                        else:
                            # This case should ideally not be hit if names are like 'component.weight'
                            # but as a fallback, keep the component name
                            new_name = f"model.layers.{layer_idx}.mamba.{mamba_comp_name}"


        renamed_state_dict[new_name] = param
        if new_name != name:
            print(f"Renamed: {name} -> {new_name}")
        else:
            print(f"Kept: {name}")


    # Save the renamed state_dict
    torch.save(renamed_state_dict, os.path.join(save_directory, "pytorch_model.bin"))

    # Create dummy tokenizer files for GPT-2 style vocab
    # This will be used by the _set_vocab_gpt2 fallback in convert_hf_to_gguf.py

    # vocab.json
    gpt2_vocab = {
        "<|endoftext|>": 0, # Standard EOS/BOS for GPT-2
        "<|pad|>": 1, # Example pad token
        # Fill with dummy tokens up to vocab_size
    }
    for i in range(2, config.vocab_size):
        gpt2_vocab[f"dummytoken{i}"] = i
    with open(os.path.join(save_directory, "vocab.json"), "w") as f:
        json.dump(gpt2_vocab, f)

    # merges.txt (can be empty for basic tokenization)
    with open(os.path.join(save_directory, "merges.txt"), "w") as f:
        # GPT-2 BPE typically has merges. Add a minimal valid merge line.
        f.write("#version: 0.2\n")
        f.write("d u\n") # A single dummy merge rule
        f.write("um m\n")
        f.write("m y\n")
        f.write("y t\n")
        f.write("t o\n")
        f.write("o k\n")
        f.write("k e\n")
        f.write("e n\n")


    # tokenizer_config.json
    tokenizer_config_json_content = {
        "tokenizer_class": "GPT2Tokenizer", # Explicitly use GPT2Tokenizer
        "bos_token": "<|endoftext|>",
        "eos_token": "<|endoftext|>",
        "unk_token": "<|endoftext|>", # GPT-2 often uses EOS as UNK
        "pad_token": "<|pad|>",
        "model_max_length": config.max_position_embeddings if hasattr(config, 'max_position_embeddings') else 2048,
        "add_prefix_space": False, # Typical for GPT-2
    }
    with open(os.path.join(save_directory, "tokenizer_config.json"), "w") as f:
        json.dump(tokenizer_config_json_content, f)

    # Remove tokenizer.model if it exists, to ensure GPT-2 path is taken
    tokenizer_model_path = os.path.join(save_directory, "tokenizer.model")
    if os.path.exists(tokenizer_model_path):
        os.remove(tokenizer_model_path)

    print(f"Saved GPT-2 style dummy tokenizer files and renamed model to {save_directory}")


if __name__ == "__main__":
    # Configuration for the test model
    config = GraniteMoeHybridConfig(
        vocab_size=1000, # Smaller vocab for testing
        hidden_size=32,
        num_hidden_layers=3,
        layer_types=["attention", "moe", "mamba"], # One of each
        # Attention
        num_attention_heads=2,
        # MoE
        num_local_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=32,
        # Mamba
        mamba_d_state=4,
        mamba_d_conv=2,
        mamba_expand=1, # d_inner = hidden_size * 1
        # General
        rms_norm_eps=1e-5,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )

    # Instantiate model
    model = GraniteMoeHybridForCausalLM(config)
    model.eval() # Set to evaluation mode

    # Save model with renamed tensors
    save_path = "./test_hf_granite_moe_hybrid_model"
    rename_and_save_model(model, save_path, config)

    print(f"Test GraniteMoeHybrid HF model generated and saved to {save_path}")
    print("Model config:")
    print(config)

# ```
# Example usage:
#
# from transformers import AutoModelForCausalLM, AutoTokenizer
#
# model_path = "./test_hf_granite_moe_hybrid_model"
#
# # Load model
# model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
# print("Model loaded successfully.")
#
# # Load tokenizer (though it's a dummy one for this test model)
# # For actual use, you'd need a real tokenizer compatible with your vocab.
# try:
#     tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
#     print("Tokenizer loaded successfully.")
# except Exception as e:
#     print(f"Could not load tokenizer (this might be expected for the dummy): {e}")
#
# # Example inference (will likely produce nonsense due to dummy model and tokenizer)
# if 'tokenizer' in locals():
#     try:
#         inputs = tokenizer("Hello, this is a test:", return_tensors="pt")
#         outputs = model.generate(**inputs, max_length=20)
#         print("Generated text:", tokenizer.decode(outputs[0], skip_special_tokens=True))
#     except Exception as e:
#         print(f"Error during generation (expected for dummy tokenizer/model): {e}")
# else:
#     print("Skipping generation example as tokenizer failed to load.")
# ```
