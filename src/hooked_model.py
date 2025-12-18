"""
HookedSparseGPT - TransformerLens-compatible wrapper for SparseGPT.

Provides hook points for interpretability tools like:
- dictionary_learning (SAE training)
- circuit-tracer (attribution graphs)

Based on TransformerLens HookedTransformer interface.
"""

import math
from typing import Optional, Tuple, Union, List, Dict, Callable, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from transformer_lens.hook_points import HookedRootModule, HookPoint

from .config import ModelConfig, SparsityConfig


# =============================================================================
# Configuration Wrapper
# =============================================================================

@dataclass
class HookedSparseGPTConfig:
    """
    TransformerLens-compatible config wrapper.
    
    Maps SparseGPT config attributes to TransformerLens naming conventions.
    """
    # Core dimensions
    n_layers: int
    d_model: int
    n_ctx: int
    d_head: int
    n_heads: int
    d_mlp: int
    d_vocab: int
    
    # Model features
    act_fn: str
    use_rms_norm: bool
    use_attention_sinks: bool
    use_bias: bool
    
    # Naming
    model_name: str = "sparse_gpt"
    tokenizer_name: Optional[str] = None
    
    # Device/dtype
    device: Optional[str] = None
    dtype: torch.dtype = torch.float32
    
    # TransformerLens compatibility
    normalization_type: str = "RMS"  # or "LN"
    default_prepend_bos: bool = False
    tokenizer_prepends_bos: Optional[bool] = None
    
    @classmethod
    def from_sparse_gpt_config(
        cls,
        model_config: ModelConfig,
        tokenizer_name: Optional[str] = None,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
    ) -> "HookedSparseGPTConfig":
        """Create from SparseGPT ModelConfig."""
        return cls(
            n_layers=model_config.n_layer,
            d_model=model_config.d_model,
            n_ctx=model_config.n_ctx,
            d_head=model_config.d_head,
            n_heads=model_config.n_heads,
            d_mlp=model_config.d_mlp,
            d_vocab=model_config.vocab_size,
            act_fn=model_config.activation,
            use_rms_norm=model_config.use_rms_norm,
            use_attention_sinks=model_config.use_attention_sinks,
            use_bias=model_config.use_bias,
            tokenizer_name=tokenizer_name,
            device=device,
            dtype=dtype,
            normalization_type="RMS" if model_config.use_rms_norm else "LN",
        )


# =============================================================================
# Hooked Normalization Layers
# =============================================================================

class HookedRMSNorm(nn.Module):
    """RMSNorm with hook points for scale and normalized output."""
    
    def __init__(self, d_model: int, eps: float = None, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.d_model = d_model
        # Match PyTorch's nn.RMSNorm: when eps=None, use machine epsilon
        self.eps = eps if eps is not None else torch.finfo(dtype).eps
        self.weight = nn.Parameter(torch.ones(d_model, dtype=dtype))
        
        self.hook_scale = HookPoint()  # [batch, pos, 1]
        self.hook_normalized = HookPoint()  # [batch, pos, d_model]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        scale = self.hook_scale(1.0 / rms)
        
        # Normalize
        x_normed = x * scale
        x_normed = self.hook_normalized(x_normed)
        
        return x_normed * self.weight


class HookedLayerNorm(nn.Module):
    """LayerNorm with hook points for scale and normalized output."""
    
    def __init__(self, d_model: int, eps: float = 1e-5, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(d_model, dtype=dtype))
        
        self.hook_scale = HookPoint()  # [batch, pos, 1]
        self.hook_normalized = HookPoint()  # [batch, pos, d_model]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute mean and variance
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        scale = self.hook_scale(1.0 / torch.sqrt(var + self.eps))
        
        # Normalize
        x_normed = (x - mean) * scale
        x_normed = self.hook_normalized(x_normed)
        
        return x_normed * self.weight + self.bias


# =============================================================================
# Hooked MLP
# =============================================================================

class HookedMLP(nn.Module):
    """MLP with hook points for pre and post activation."""
    
    def __init__(self, cfg: HookedSparseGPTConfig):
        super().__init__()
        self.cfg = cfg
        
        self.W_in = nn.Parameter(torch.empty(cfg.d_model, cfg.d_mlp, dtype=cfg.dtype))
        self.W_out = nn.Parameter(torch.empty(cfg.d_mlp, cfg.d_model, dtype=cfg.dtype))
        
        if cfg.use_bias:
            self.b_in = nn.Parameter(torch.zeros(cfg.d_mlp, dtype=cfg.dtype))
            self.b_out = nn.Parameter(torch.zeros(cfg.d_model, dtype=cfg.dtype))
        else:
            self.register_parameter('b_in', None)
            self.register_parameter('b_out', None)
        
        # Activation function
        if cfg.act_fn == "gelu":
            self.act_fn = F.gelu
        else:
            self.act_fn = F.relu
        
        # Hook points
        self.hook_pre = HookPoint()  # [batch, pos, d_mlp] - before activation
        self.hook_post = HookPoint()  # [batch, pos, d_mlp] - after activation
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input projection
        pre_act = x @ self.W_in
        if self.b_in is not None:
            pre_act = pre_act + self.b_in
        pre_act = self.hook_pre(pre_act)
        
        # Activation
        post_act = self.act_fn(pre_act)
        post_act = self.hook_post(post_act)
        
        # Output projection
        out = post_act @ self.W_out
        if self.b_out is not None:
            out = out + self.b_out
        
        return out


# =============================================================================
# Hooked Attention (with Attention Sink support)
# =============================================================================

class HookedAttention(nn.Module):
    """
    Multi-head causal self-attention with full hook points.
    
    Supports attention sinks (learnable per-head bias to a dummy position).
    Implements manual attention computation to expose hook_pattern.
    """
    
    def __init__(self, cfg: HookedSparseGPTConfig, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_head
        self.d_model = cfg.d_model
        
        # QKV projection (combined)
        qkv_dim = 3 * cfg.d_head * cfg.n_heads
        self.W_QKV = nn.Parameter(torch.empty(cfg.d_model, qkv_dim, dtype=cfg.dtype))
        if cfg.use_bias:
            self.b_QKV = nn.Parameter(torch.zeros(qkv_dim, dtype=cfg.dtype))
        else:
            self.register_parameter('b_QKV', None)
        
        # Output projection
        self.W_O = nn.Parameter(torch.empty(cfg.n_heads * cfg.d_head, cfg.d_model, dtype=cfg.dtype))
        if cfg.use_bias:
            self.b_O = nn.Parameter(torch.zeros(cfg.d_model, dtype=cfg.dtype))
        else:
            self.register_parameter('b_O', None)
        
        # Attention sink (learnable logit for dummy position)
        if cfg.use_attention_sinks:
            self.sink_logit = nn.Parameter(torch.zeros(cfg.n_heads, dtype=cfg.dtype))
        else:
            self.register_parameter('sink_logit', None)
        
        # Hook points
        self.hook_q = HookPoint()  # [batch, pos, n_heads, d_head]
        self.hook_k = HookPoint()  # [batch, pos, n_heads, d_head]
        self.hook_v = HookPoint()  # [batch, pos, n_heads, d_head]
        self.hook_attn_scores = HookPoint()  # [batch, n_heads, query_pos, key_pos]
        self.hook_pattern = HookPoint()  # [batch, n_heads, query_pos, key_pos] - after softmax
        self.hook_z = HookPoint()  # [batch, pos, n_heads, d_head] - attention output per head
        self.hook_result = HookPoint()  # [batch, pos, n_heads, d_model] - after W_O per head
        
        # Causal mask (registered as buffer)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(cfg.n_ctx, cfg.n_ctx, dtype=torch.bool))
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        
        # QKV projection
        qkv = x @ self.W_QKV
        if self.b_QKV is not None:
            qkv = qkv + self.b_QKV
        
        # Split into Q, K, V
        qkv_size = self.n_heads * self.d_head
        q, k, v = qkv.split(qkv_size, dim=-1)
        
        # Reshape to [batch, pos, n_heads, d_head]
        q = q.view(B, T, self.n_heads, self.d_head)
        k = k.view(B, T, self.n_heads, self.d_head)
        v = v.view(B, T, self.n_heads, self.d_head)
        
        # Hook Q, K, V
        q = self.hook_q(q)
        k = self.hook_k(k)
        v = self.hook_v(v)
        
        # Transpose to [batch, n_heads, pos, d_head] for attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        scale = 1.0 / math.sqrt(self.d_head)
        
        if self.sink_logit is not None:
            # With attention sinks: prepend dummy KV position
            # k_aug: [B, H, T+1, D], v_aug: [B, H, T+1, D]
            k_sink = torch.zeros((B, self.n_heads, 1, self.d_head), dtype=k.dtype, device=k.device)
            v_sink = torch.zeros((B, self.n_heads, 1, self.d_head), dtype=v.dtype, device=v.device)
            k_aug = torch.cat([k_sink, k], dim=2)
            v_aug = torch.cat([v_sink, v], dim=2)
            
            # Attention scores: [B, H, T, T+1]
            attn_scores = (q @ k_aug.transpose(-2, -1)) * scale
            
            # Build mask: sink (col 0) always visible, causal for real positions
            mask = torch.zeros((T, T + 1), dtype=torch.bool, device=x.device)
            mask[:, 0] = True  # Sink always visible
            mask[:, 1:] = self.causal_mask[:T, :T]  # Causal for real keys
            
            # Apply mask
            attn_scores = attn_scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            
            # Add learnable sink bias to column 0
            sink_bias = self.sink_logit.view(1, self.n_heads, 1, 1)
            attn_scores[:, :, :, 0:1] = attn_scores[:, :, :, 0:1] + sink_bias
            
            # Hook attention scores
            attn_scores = self.hook_attn_scores(attn_scores)
            
            # Softmax
            pattern = F.softmax(attn_scores, dim=-1)
            pattern = self.hook_pattern(pattern)
            
            # Apply attention
            z = pattern @ v_aug  # [B, H, T, D]
        else:
            # Standard causal attention (no sinks)
            attn_scores = (q @ k.transpose(-2, -1)) * scale
            
            # Apply causal mask
            attn_scores = attn_scores.masked_fill(
                ~self.causal_mask[:T, :T].unsqueeze(0).unsqueeze(0),
                float('-inf')
            )
            
            # Hook attention scores
            attn_scores = self.hook_attn_scores(attn_scores)
            
            # Softmax
            pattern = F.softmax(attn_scores, dim=-1)
            pattern = self.hook_pattern(pattern)
            
            # Apply attention
            z = pattern @ v  # [B, H, T, D]
        
        # Reshape z for hook: [B, T, H, D]
        z = z.transpose(1, 2)
        z = self.hook_z(z)
        
        # Reshape for output projection: [B, T, H*D]
        z_flat = z.reshape(B, T, self.n_heads * self.d_head)
        
        # Output projection
        out = z_flat @ self.W_O
        if self.b_O is not None:
            out = out + self.b_O
        
        return out


# =============================================================================
# Hooked Transformer Block
# =============================================================================

class HookedTransformerBlock(nn.Module):
    """Transformer block with full hook points."""
    
    def __init__(self, cfg: HookedSparseGPTConfig, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        
        # Normalization
        norm_cls = HookedRMSNorm if cfg.use_rms_norm else HookedLayerNorm
        self.ln1 = norm_cls(cfg.d_model, dtype=cfg.dtype)
        self.ln2 = norm_cls(cfg.d_model, dtype=cfg.dtype)
        
        # Attention and MLP
        self.attn = HookedAttention(cfg, layer_idx)
        self.mlp = HookedMLP(cfg)
        
        # Block-level hook points
        self.hook_resid_pre = HookPoint()  # [batch, pos, d_model]
        self.hook_resid_mid = HookPoint()  # [batch, pos, d_model]
        self.hook_resid_post = HookPoint()  # [batch, pos, d_model]
        self.hook_attn_in = HookPoint()  # [batch, pos, d_model]
        self.hook_attn_out = HookPoint()  # [batch, pos, d_model]
        self.hook_mlp_in = HookPoint()  # [batch, pos, d_model]
        self.hook_mlp_out = HookPoint()  # [batch, pos, d_model]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-attention residual
        x = self.hook_resid_pre(x)
        
        # Attention sublayer
        attn_in = self.ln1(x)
        attn_in = self.hook_attn_in(attn_in)
        attn_out = self.attn(attn_in)
        attn_out = self.hook_attn_out(attn_out)
        
        # First residual connection
        x = x + attn_out
        x = self.hook_resid_mid(x)
        
        # MLP sublayer
        mlp_in = self.ln2(x)
        mlp_in = self.hook_mlp_in(mlp_in)
        mlp_out = self.mlp(mlp_in)
        mlp_out = self.hook_mlp_out(mlp_out)
        
        # Second residual connection
        x = x + mlp_out
        x = self.hook_resid_post(x)
        
        return x


# =============================================================================
# Main HookedSparseGPT Class
# =============================================================================

class HookedSparseGPT(HookedRootModule):
    """
    TransformerLens-compatible wrapper for SparseGPT models.
    
    Provides all hook points needed for interpretability tools like
    dictionary_learning and circuit-tracer.
    
    Usage:
        model = HookedSparseGPT.from_pretrained("username/model-name")
        logits, cache = model.run_with_cache(tokens)
    """
    
    def __init__(
        self,
        cfg: HookedSparseGPTConfig,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        
        # Embedding
        self.embed = nn.Embedding(cfg.d_vocab, cfg.d_model)
        self.hook_embed = HookPoint()  # [batch, pos, d_model]
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            HookedTransformerBlock(cfg, layer_idx)
            for layer_idx in range(cfg.n_layers)
        ])
        
        # Final layer norm
        norm_cls = HookedRMSNorm if cfg.use_rms_norm else HookedLayerNorm
        self.ln_final = norm_cls(cfg.d_model, dtype=cfg.dtype)
        
        # Unembedding
        self.unembed = nn.Linear(cfg.d_model, cfg.d_vocab, bias=False)
        
        # Setup hook dict (required by HookedRootModule)
        self.setup()
    
    @property
    def W_E(self) -> torch.Tensor:
        """Embedding matrix [d_vocab, d_model]."""
        return self.embed.weight
    
    @property
    def W_U(self) -> torch.Tensor:
        """Unembedding matrix [d_model, d_vocab]."""
        return self.unembed.weight.T
    
    @property
    def b_U(self) -> torch.Tensor:
        """Unembedding bias [d_vocab]."""
        if self.unembed.bias is not None:
            return self.unembed.bias
        return torch.zeros(self.cfg.d_vocab, device=self.W_U.device, dtype=self.W_U.dtype)
    
    def to_tokens(
        self,
        input: Union[str, List[str]],
        prepend_bos: bool = False,
        move_to_device: bool = True,
        truncate: bool = True,
    ) -> Int[torch.Tensor, "batch pos"]:
        """Convert string(s) to tokens."""
        assert self.tokenizer is not None, "Tokenizer not set"
        
        tokens = self.tokenizer(
            input,
            return_tensors="pt",
            padding=True,
            truncation=truncate,
            max_length=self.cfg.n_ctx if truncate else None,
        )["input_ids"]
        
        if prepend_bos and self.tokenizer.bos_token_id is not None:
            bos = torch.full((tokens.shape[0], 1), self.tokenizer.bos_token_id, dtype=tokens.dtype)
            tokens = torch.cat([bos, tokens], dim=1)
        
        if move_to_device and self.cfg.device:
            tokens = tokens.to(self.cfg.device)
        
        return tokens
    
    def forward(
        self,
        input: Union[str, List[str], Int[torch.Tensor, "batch pos"]],
        return_type: Optional[str] = "logits",
        stop_at_layer: Optional[int] = None,
        prepend_bos: bool = False,
    ) -> Union[
        None,
        Float[torch.Tensor, "batch pos d_vocab"],
        Float[torch.Tensor, "batch pos d_model"],
        Tuple[Float[torch.Tensor, "batch pos d_vocab"], Float[torch.Tensor, ""]],
    ]:
        """
        Forward pass with TransformerLens-compatible interface.
        
        Args:
            input: Token IDs or strings (will be tokenized)
            return_type: "logits", "loss", "both", or None
            stop_at_layer: If set, return residual stream at this layer (exclusive)
            prepend_bos: Whether to prepend BOS token (only for string input)
            
        Returns:
            Depends on return_type:
            - "logits": logits tensor
            - "loss": loss scalar (requires integer input as labels)
            - "both": (logits, loss) tuple
            - None: None (useful for just running hooks)
            
            If stop_at_layer is set, returns residual stream instead.
        """
        # Handle string input
        if isinstance(input, str) or isinstance(input, list):
            tokens = self.to_tokens(input, prepend_bos=prepend_bos)
        else:
            tokens = input
        
        if tokens.device != self.W_E.device:
            tokens = tokens.to(self.W_E.device)
        
        B, T = tokens.shape
        
        # Embedding
        x = self.embed(tokens)
        x = self.hook_embed(x)
        
        # Transformer blocks
        for i, block in enumerate(self.blocks):
            if stop_at_layer is not None and i >= stop_at_layer:
                break
            x = block(x)
        
        # If stopping early, return residual stream
        if stop_at_layer is not None:
            return x
        
        # Final layer norm
        x = self.ln_final(x)
        
        # Unembedding
        logits = self.unembed(x)
        
        # Return based on return_type
        if return_type is None:
            return None
        elif return_type == "logits":
            return logits
        elif return_type == "loss":
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = tokens[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            return loss
        elif return_type == "both":
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = tokens[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            return logits, loss
        else:
            raise ValueError(f"Invalid return_type: {return_type}")
    
    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "HookedSparseGPT":
        """
        Load a pretrained SparseGPT model as HookedSparseGPT.
        
        Args:
            repo_id: HuggingFace repo ID (e.g., "username/model-name")
            device: Device to load model on
            dtype: Data type for model weights
            
        Returns:
            HookedSparseGPT instance with loaded weights and tokenizer
        """
        import json
        from huggingface_hub import hf_hub_download
        
        # Download config and weights
        config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
        model_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin")
        
        # Load config
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        
        # Check bigram table is disabled
        model_cfg = config_dict["model_config"]
        if model_cfg.get("use_bigram_table", False):
            raise ValueError(
                "HookedSparseGPT does not support models with bigram_table enabled. "
                "The bigram table is not compatible with TransformerLens-style interpretability tools."
            )
        
        # Create ModelConfig
        sparse_config = ModelConfig(
            n_layer=model_cfg["n_layer"],
            d_model=model_cfg["d_model"],
            n_ctx=model_cfg["n_ctx"],
            d_head=model_cfg["d_head"],
            d_mlp=model_cfg["d_mlp"],
            vocab_size=model_cfg["vocab_size"],
            use_rms_norm=model_cfg["use_rms_norm"],
            tie_embeddings=model_cfg.get("tie_embeddings", False),
            use_positional_embeddings=model_cfg.get("use_positional_embeddings", False),
            use_bigram_table=False,  # Asserted above
            use_attention_sinks=model_cfg.get("use_attention_sinks", True),
            activation=model_cfg.get("activation", "gelu"),
            dropout=model_cfg.get("dropout", 0.0),
            use_bias=model_cfg.get("use_bias", True),
        )
        
        # Get tokenizer name
        tokenizer_name = config_dict.get("training_config", {}).get("tokenizer_name")
        
        # Create hooked config
        hooked_cfg = HookedSparseGPTConfig.from_sparse_gpt_config(
            sparse_config,
            tokenizer_name=tokenizer_name,
            device=device,
            dtype=dtype,
        )
        
        # Load tokenizer
        tokenizer = None
        if tokenizer_name:
            try:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
            except Exception as e:
                print(f"Warning: Could not load tokenizer '{tokenizer_name}': {e}")
        
        # Create model
        model = cls(hooked_cfg, tokenizer=tokenizer)
        
        # Load weights from SparseGPT format
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model._load_sparse_gpt_weights(state_dict)
        
        model.to(device)
        model.eval()
        
        return model
    
    def _load_sparse_gpt_weights(self, state_dict: Dict[str, torch.Tensor]):
        """
        Load weights from SparseGPT state dict into HookedSparseGPT.
        
        Maps SparseGPT weight names to HookedSparseGPT structure.
        """
        # Embedding
        self.embed.weight.data.copy_(state_dict["wte.weight"])
        
        # Unembedding (lm_head)
        self.unembed.weight.data.copy_(state_dict["lm_head.weight"])
        
        # Final layer norm
        if "ln_f.weight" in state_dict:
            self.ln_final.weight.data.copy_(state_dict["ln_f.weight"])
        
        # Transformer blocks
        for i, block in enumerate(self.blocks):
            prefix = f"blocks.{i}."
            
            # Layer norms (SparseGPT uses ln_1/ln_2, we use ln1/ln2)
            block.ln1.weight.data.copy_(state_dict[f"{prefix}ln_1.weight"])
            block.ln2.weight.data.copy_(state_dict[f"{prefix}ln_2.weight"])
            
            # LayerNorm also has bias
            if hasattr(block.ln1, 'bias') and block.ln1.bias is not None:
                if f"{prefix}ln_1.bias" in state_dict:
                    block.ln1.bias.data.copy_(state_dict[f"{prefix}ln_1.bias"])
            if hasattr(block.ln2, 'bias') and block.ln2.bias is not None:
                if f"{prefix}ln_2.bias" in state_dict:
                    block.ln2.bias.data.copy_(state_dict[f"{prefix}ln_2.bias"])
            
            # Attention
            # SparseGPT uses c_attn for combined QKV, c_proj for output
            attn_weight = state_dict[f"{prefix}attn.c_attn.weight"]
            if f"{prefix}attn.c_attn.bias" in state_dict:
                attn_bias = state_dict[f"{prefix}attn.c_attn.bias"]
            else:
                attn_bias = None
            
            # Transpose because nn.Linear weight is (out, in) but we use x @ W
            block.attn.W_QKV.data.copy_(attn_weight.T)
            if attn_bias is not None and block.attn.b_QKV is not None:
                block.attn.b_QKV.data.copy_(attn_bias)
            
            # Attention output projection
            out_weight = state_dict[f"{prefix}attn.c_proj.weight"]
            if f"{prefix}attn.c_proj.bias" in state_dict:
                out_bias = state_dict[f"{prefix}attn.c_proj.bias"]
            else:
                out_bias = None
            
            # Transpose because nn.Linear weight is (out, in) but we use x @ W
            block.attn.W_O.data.copy_(out_weight.T)
            if out_bias is not None and block.attn.b_O is not None:
                block.attn.b_O.data.copy_(out_bias)
            
            # Attention sink logit
            if block.attn.sink_logit is not None and f"{prefix}attn.attn_fn.sink_logit" in state_dict:
                block.attn.sink_logit.data.copy_(state_dict[f"{prefix}attn.attn_fn.sink_logit"])
            
            # MLP
            # SparseGPT uses c_fc for input, c_proj for output
            block.mlp.W_in.data.copy_(state_dict[f"{prefix}mlp.c_fc.weight"].T)
            block.mlp.W_out.data.copy_(state_dict[f"{prefix}mlp.c_proj.weight"].T)
            
            if block.mlp.b_in is not None and f"{prefix}mlp.c_fc.bias" in state_dict:
                block.mlp.b_in.data.copy_(state_dict[f"{prefix}mlp.c_fc.bias"])
            if block.mlp.b_out is not None and f"{prefix}mlp.c_proj.bias" in state_dict:
                block.mlp.b_out.data.copy_(state_dict[f"{prefix}mlp.c_proj.bias"])

