"""
Weight-sparse GPT model implementation.
Based on Gao et al. (2025) "Weight-sparse transformers have interpretable circuits"
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig, SparsityConfig


class AbsTopK(nn.Module):
    """
    Activation sparsity via absolute-value top-k.
    Zeros out all but the k largest values by magnitude.
    
    Paper Section 3.1: "At each designated location, zero out all but the k largest 
    values by magnitude"
    """
    
    def __init__(self, k: int):
        super().__init__()
        self.k = k
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.k >= x.shape[-1]:
            return x
        
        # Find top-k indices by absolute value
        _, topk_indices = torch.topk(x.abs(), self.k, dim=-1, sorted=False)
        
        # Create output with zeros
        result = torch.zeros_like(x)
        result.scatter_(-1, topk_indices, x.gather(-1, topk_indices))
        
        return result
    
    def extra_repr(self) -> str:
        return f"k={self.k}"


class SDPAWithSink(nn.Module):
    """
    Scaled dot-product attention with learnable attention sinks.
    
    Paper Section 1.6: "Per-head learnable attention denominator bias"
    Reference: Xiao et al. (2023)
    
    This adds a dummy KV slot whose logit is learnable (the "sink") and whose V is zero.
    This allows the attention to "dump" probability mass when no real key is relevant.
    """
    
    def __init__(self, n_heads: int, init_logit: float = 0.0):
        super().__init__()
        self.sink_logit = nn.Parameter(torch.full((n_heads,), init_logit))
    
    def forward(
        self,
        q: torch.Tensor,  # (B, H, Lq, D)
        k: torch.Tensor,  # (B, H, Lk, D)
        v: torch.Tensor,  # (B, H, Lk, Dv)
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: Optional[float] = None,
    ) -> torch.Tensor:
        B, H, Lq, D = q.shape
        _, _, Lk, Dv = k.shape[0], k.shape[1], k.shape[2], v.shape[-1]
        
        # Prepend a dummy KV slot (always visible)
        k_sink = torch.zeros((B, H, 1, D), dtype=q.dtype, device=q.device)
        v_sink = torch.zeros((B, H, 1, Dv), dtype=v.dtype, device=v.device)
        k_aug = torch.cat([k_sink, k], dim=2)  # (B, H, Lk+1, D)
        v_aug = torch.cat([v_sink, v], dim=2)  # (B, H, Lk+1, Dv)
        
        # Build causal mask: allow sink (col 0) always, lower-triangular for real keys
        allow = torch.zeros((Lq, Lk + 1), dtype=torch.bool, device=q.device)
        allow[:, 0] = True  # Sink column always visible
        # Lower-triangular for real keys (shifted by 1)
        real_mask = torch.ones((Lq, Lk), dtype=torch.bool, device=q.device).tril()
        allow[:, 1:] = real_mask
        
        # Convert to additive mask
        neg_inf = torch.finfo(q.dtype).min
        base_mask = torch.where(
            allow,
            torch.zeros((), dtype=q.dtype, device=q.device),
            torch.full((), neg_inf, dtype=q.dtype, device=q.device),
        )
        
        # Add learnable sink bias to column 0
        sink_bias = self.sink_logit.to(dtype=q.dtype, device=q.device).view(1, H, 1, 1)
        sink_bias_mask = torch.zeros((1, 1, 1, Lk + 1), dtype=q.dtype, device=q.device)
        sink_bias_mask[..., 0] = 1.0
        attn_mask = base_mask.unsqueeze(0).unsqueeze(0) + sink_bias_mask * sink_bias
        
        # SDPA with custom mask (is_causal=False to avoid double-masking)
        out = F.scaled_dot_product_attention(
            q, k_aug, v_aug,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=False,
            scale=scale,
        )
        return out


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with optional attention sinks.
    """
    
    def __init__(self, config: ModelConfig, use_sinks: bool = True):
        super().__init__()
        
        self.n_heads = config.n_heads
        self.d_head = config.d_head
        self.d_model = config.d_model
        self.dropout = config.dropout
        self.use_flash = config.use_flash_attention
        
        # QKV projection (combined for efficiency)
        self.c_attn = nn.Linear(
            config.d_model,
            3 * config.d_head * config.n_heads,
            bias=config.use_bias
        )
        
        # Output projection
        self.c_proj = nn.Linear(
            config.d_head * config.n_heads,
            config.d_model,
            bias=config.use_bias
        )
        
        # Dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Attention implementation
        if use_sinks and config.use_attention_sinks:
            self.attn_fn = SDPAWithSink(config.n_heads)
        else:
            self.attn_fn = None  # Use standard SDPA
        
        # For non-flash attention (fallback)
        if not self.use_flash:
            self.register_buffer(
                "causal_mask",
                torch.tril(torch.ones(config.n_ctx, config.n_ctx)).view(
                    1, 1, config.n_ctx, config.n_ctx
                )
            )
    
    def forward(
        self,
        x: torch.Tensor,
        activation_sparsity_fn: Optional[callable] = None,
    ) -> torch.Tensor:
        B, T, C = x.shape
        
        # QKV projection
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_heads * self.d_head, dim=-1)
        
        # Apply activation sparsity to Q, K, V if provided
        if activation_sparsity_fn is not None:
            q = activation_sparsity_fn(q, "attn_q")
            k = activation_sparsity_fn(k, "attn_k")
            v = activation_sparsity_fn(v, "attn_v")
        
        # Reshape for attention: (B, T, H*D) -> (B, H, T, D)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        
        scale = 1.0 / math.sqrt(self.d_head)
        
        # Attention
        if self.use_flash:
            if self.attn_fn is not None:
                # Use attention sinks
                y = self.attn_fn(
                    q, k, v,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=True,
                    scale=scale,
                )
            else:
                # Standard flash attention
                y = F.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=True,
                    scale=scale,
                )
        else:
            # Manual attention implementation
            att = (q @ k.transpose(-2, -1)) * scale
            att = att.masked_fill(
                self.causal_mask[:, :, :T, :T] == 0,
                float('-inf')
            )
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        
        # Reshape back: (B, H, T, D) -> (B, T, H*D)
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.d_head)
        
        # Output projection
        y = self.c_proj(y)
        y = self.resid_dropout(y)
        
        return y


class MLP(nn.Module):
    """
    Feed-forward MLP block.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        self.c_fc = nn.Linear(config.d_model, config.d_mlp, bias=config.use_bias)
        self.c_proj = nn.Linear(config.d_mlp, config.d_model, bias=config.use_bias)
        self.dropout = nn.Dropout(config.dropout)
        
        if config.activation == "gelu":
            self.act_fn = nn.GELU()
        else:
            self.act_fn = nn.ReLU()
    
    def forward(
        self,
        x: torch.Tensor,
        activation_sparsity_fn: Optional[callable] = None,
    ) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.act_fn(x)
        
        # Apply activation sparsity after activation (to neurons)
        if activation_sparsity_fn is not None:
            x = activation_sparsity_fn(x, "mlp_neuron")
        
        x = self.c_proj(x)
        x = self.dropout(x)
        
        return x


class TransformerBlock(nn.Module):
    """
    Single transformer block with pre-norm architecture.
    
    Paper uses RMSNorm and applies AbsTopK at specific locations within the block.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Normalization
        if config.use_rms_norm:
            self.ln_1 = nn.RMSNorm(config.d_model)
            self.ln_2 = nn.RMSNorm(config.d_model)
        else:
            self.ln_1 = nn.LayerNorm(config.d_model)
            self.ln_2 = nn.LayerNorm(config.d_model)
        
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
    
    def forward(
        self,
        x: torch.Tensor,
        activation_sparsity_fn: Optional[callable] = None,
    ) -> torch.Tensor:
        # Attention block
        if activation_sparsity_fn is not None:
            normed = self.ln_1(x)
            normed = activation_sparsity_fn(normed, "attn_in")
        else:
            normed = self.ln_1(x)
        
        attn_out = self.attn(normed, activation_sparsity_fn)
        
        if activation_sparsity_fn is not None:
            attn_out = activation_sparsity_fn(attn_out, "attn_out")
        
        x = x + attn_out
        
        # MLP block
        if activation_sparsity_fn is not None:
            normed = self.ln_2(x)
            normed = activation_sparsity_fn(normed, "mlp_in")
        else:
            normed = self.ln_2(x)
        
        mlp_out = self.mlp(normed, activation_sparsity_fn)
        
        if activation_sparsity_fn is not None:
            mlp_out = activation_sparsity_fn(mlp_out, "mlp_out")
        
        x = x + mlp_out
        
        return x


class SparseGPT(nn.Module):
    """
    GPT model with weight and activation sparsity support.
    
    Based on Gao et al. (2025) "Weight-sparse transformers have interpretable circuits"
    """
    
    def __init__(self, config: ModelConfig, sparsity_config: Optional[SparsityConfig] = None):
        super().__init__()
        
        self.config = config
        self.sparsity_config = sparsity_config or SparsityConfig()
        
        # Token embeddings
        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        
        # Positional embeddings (optional - paper doesn't use them by default)
        if config.use_positional_embeddings:
            self.wpe = nn.Embedding(config.n_ctx, config.d_model)
        else:
            self.wpe = None
        
        self.drop = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layer)
        ])
        
        # Final layer norm
        if config.use_rms_norm:
            self.ln_f = nn.RMSNorm(config.d_model)
        else:
            self.ln_f = nn.LayerNorm(config.d_model)
        
        # LM head (separate from embedding if not tied)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        if config.tie_embeddings:
            self.lm_head.weight = self.wte.weight
        
        # Bigram table (paper Section 1.5)
        # Dense d_vocab x d_vocab matrix added directly to final logits
        # Authors initialize with small random values (rand * 0.02), not zeros
        if config.use_bigram_table:
            self.bigram_table = nn.Parameter(
                torch.rand(config.vocab_size, config.vocab_size) * 0.02
            )
        else:
            self.bigram_table = None
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Special scaled init for residual projections (per GPT-2 paper)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )
        
        # Setup activation sparsity
        self._setup_activation_sparsity()
    
    def _init_weights(self, module: nn.Module):
        """Initialize weights with normal distribution."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def _setup_activation_sparsity(self):
        """Setup activation sparsity functions based on config."""
        if not self.sparsity_config.enable_activation_sparsity:
            self._activation_sparsity_fn = None
            return
        
        # Parse locations where activation sparsity is applied
        locations = set(self.sparsity_config.activation_sparsity_locations.split(","))
        k_fraction = self.sparsity_config.activation_topk_fraction
        
        # Create AbsTopK modules for different dimensions
        # We'll create them lazily in the forward pass based on tensor shapes
        self._activation_sparsity_locations = locations
        self._activation_sparsity_k_fraction = k_fraction
        self._activation_sparsity_cache = {}
    
    def _get_activation_sparsity_fn(self) -> Optional[callable]:
        """Get the activation sparsity function."""
        if not self.sparsity_config.enable_activation_sparsity:
            return None
        
        def apply_activation_sparsity(x: torch.Tensor, location: str) -> torch.Tensor:
            if location not in self._activation_sparsity_locations:
                return x
            
            dim = x.shape[-1]
            k = max(1, int(dim * self._activation_sparsity_k_fraction))
            
            # Cache AbsTopK modules by dimension
            if dim not in self._activation_sparsity_cache:
                self._activation_sparsity_cache[dim] = AbsTopK(k)
            
            return self._activation_sparsity_cache[dim](x)
        
        return apply_activation_sparsity
    
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[list]]:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            labels: Optional labels for computing loss (same shape as input_ids)
            return_hidden_states: Whether to return all hidden states
            
        Returns:
            logits: Logits of shape (batch_size, seq_len, vocab_size)
            loss: Cross-entropy loss if labels provided, else None
            hidden_states: List of hidden states if return_hidden_states=True, else None
        """
        B, T = input_ids.shape
        device = input_ids.device
        
        assert T <= self.config.n_ctx, f"Sequence length {T} exceeds context length {self.config.n_ctx}"
        
        # Token embeddings
        x = self.wte(input_ids)  # (B, T, d_model)
        
        # Cast to autocast dtype if autocast is enabled
        # This ensures residual connections stay in lower precision throughout the model
        if torch.is_autocast_enabled('cuda'):
            x = x.to(torch.get_autocast_dtype('cuda'))
        
        # Add positional embeddings if enabled
        if self.wpe is not None:
            pos = torch.arange(0, T, dtype=torch.long, device=device)
            x = x + self.wpe(pos)
        
        x = self.drop(x)
        
        # Get activation sparsity function
        act_sparsity_fn = self._get_activation_sparsity_fn()
        
        # Collect hidden states if requested
        hidden_states = [x] if return_hidden_states else None
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, act_sparsity_fn)
            if return_hidden_states:
                hidden_states.append(x)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # LM head
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        # Add bigram logits if enabled
        if self.bigram_table is not None:
            # Get bigram contribution for each position (based on previous token)
            bigram_logits = F.embedding(input_ids, self.bigram_table)
            logits = logits + bigram_logits
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        
        return logits, loss, hidden_states
    
    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Return the number of parameters in the model.
        
        Args:
            non_embedding: If True, exclude position embeddings from count
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and self.wpe is not None:
            n_params -= self.wpe.weight.numel()
        return n_params
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.
        
        Args:
            input_ids: Starting token IDs of shape (batch_size, seq_len)
            max_new_tokens: Number of new tokens to generate
            temperature: Sampling temperature
            top_k: If set, only sample from top-k logits
            
        Returns:
            Generated token IDs including the input
        """
        for _ in range(max_new_tokens):
            # Crop to context length if needed
            idx_cond = input_ids if input_ids.size(1) <= self.config.n_ctx else input_ids[:, -self.config.n_ctx:]
            
            # Forward pass
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Optional top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids


def create_model(config: ModelConfig, sparsity_config: Optional[SparsityConfig] = None) -> SparseGPT:
    """Create a SparseGPT model from configuration."""
    return SparseGPT(config, sparsity_config)

