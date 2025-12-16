"""
Bridge modules for coupling dense and sparse models.
Based on Gao et al. (2025) "Weight-sparse transformers have interpretable circuits"

Bridges are encoder/decoder pairs that translate between dense and sparse
residual stream activations at each sublayer location.
"""

import math
from typing import List, Tuple, Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import AbsTopK, SparseGPT


# =============================================================================
# Gradient Buffering Optimization
# =============================================================================

class GradientBuffer:
    """
    A gradient buffer that accumulates gradients and releases them on demand.
    
    This is used for efficient hybrid pass computation in bridges training.
    Instead of L separate backward passes, we get:
    - L small backwards: From each KL loss to the buffer checkpoint (accumulated)
    - 1 big backward: From checkpoint through the rest of the network
    
    Usage:
        buffer = GradientBuffer(h)
        
        # Use buffer.accumulator for all hybrid forward passes
        for i in range(n_sites):
            y_hybrid = model.forward_from_site(buffer.accumulator, ...)
            loss += kl_divergence(y_target, y_hybrid)
        
        # First backward accumulates gradients (must use retain_graph=True)
        loss.backward(retain_graph=True)
        
        # Then release gradients through the buffer
        buffer.release_gradients()
    """
    
    def __init__(self, x: torch.Tensor):
        """
        Create a gradient buffer for tensor x.
        
        Args:
            x: Tensor to buffer gradients for. Must require gradients.
        """
        self.original = x
        self.grad_buf = None
        self._released = False
        
        # Create the accumulator using a custom autograd function
        buffer_ref = self  # Reference to self for the inner class
        
        class _BufferedGradAccum(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                # No need to save_for_backward since we don't use it
                # No need to clone - we're not modifying x, just passing it through
                return x
            
            @staticmethod
            def backward(ctx, grad):
                if buffer_ref.grad_buf is None:
                    buffer_ref.grad_buf = grad.clone()
                else:
                    buffer_ref.grad_buf = buffer_ref.grad_buf + grad
                # Return None to stop gradient propagation
                return None
        
        self.accumulator = _BufferedGradAccum.apply(x)
        self._AccumClass = _BufferedGradAccum
    
    def release_gradients(self, retain_graph: bool = True):
        """
        Release accumulated gradients back through the original tensor.
        
        Call this after backward() on the loss that uses the accumulator.
        
        Args:
            retain_graph: Whether to keep the computation graph after backward.
                Use False for the last release to free memory.
        """
        if self._released:
            return
        self._released = True
        
        if self.grad_buf is None:
            return
        
        # Continue backprop through the computation graph of original
        # This propagates gradients to all tensors that original depends on
        if self.original.grad_fn is not None:
            self.original.backward(self.grad_buf, retain_graph=retain_graph)
        
        # Clear references to free memory
        self.grad_buf = None
        self.original = None
        self.accumulator = None


class BridgeEncoder(nn.Module):
    """
    Encoder that maps dense residual activations to sparse residual activations.
    
    Structure: Linear + AbsTopK
    
    The AbsTopK enforces activation sparsity on the encoded representation,
    matching the sparse model's residual stream characteristics.
    """
    
    def __init__(
        self,
        d_dense: int,
        d_sparse: int,
        afrac: float = 0.25,
    ):
        """
        Args:
            d_dense: Dimension of dense model's residual stream
            d_sparse: Dimension of sparse model's residual stream
            afrac: Fraction of activations to keep in AbsTopK
        """
        super().__init__()
        self.linear = nn.Linear(d_dense, d_sparse, bias=True)
        k = max(1, int(d_sparse * afrac))
        self.topk = AbsTopK(k)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Dense activations of shape (..., d_dense)
            
        Returns:
            Sparse-space activations of shape (..., d_sparse)
        """
        return self.topk(self.linear(x))


class BridgeDecoder(nn.Module):
    """
    Decoder that maps sparse residual activations to dense residual activations.
    
    Structure: Linear (no activation sparsity)
    """
    
    def __init__(
        self,
        d_sparse: int,
        d_dense: int,
    ):
        """
        Args:
            d_sparse: Dimension of sparse model's residual stream
            d_dense: Dimension of dense model's residual stream
        """
        super().__init__()
        self.linear = nn.Linear(d_sparse, d_dense, bias=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Sparse activations of shape (..., d_sparse)
            
        Returns:
            Dense-space activations of shape (..., d_dense)
        """
        return self.linear(x)


class Bridge(nn.Module):
    """
    A single bridge consisting of an encoder and decoder pair.
    
    Encoder: dense → sparse (Linear + AbsTopK)
    Decoder: sparse → dense (Linear)
    """
    
    def __init__(
        self,
        d_dense: int,
        d_sparse: int,
        encoder_afrac: float = 0.25,
    ):
        super().__init__()
        self.encoder = BridgeEncoder(d_dense, d_sparse, encoder_afrac)
        self.decoder = BridgeDecoder(d_sparse, d_dense)
        
    def encode(self, h_dense: torch.Tensor) -> torch.Tensor:
        """Map dense activations to sparse space."""
        return self.encoder(h_dense)
    
    def decode(self, h_sparse: torch.Tensor) -> torch.Tensor:
        """Map sparse activations to dense space."""
        return self.decoder(h_sparse)


class BridgeSet(nn.Module):
    """
    Collection of all bridges for coupling dense and sparse models.
    
    For an L-layer model, there are 2L+1 bridge sites:
    - Site 0: After embedding
    - Site 2i+1: After layer i's attention (before MLP), for i = 0..L-1
    - Site 2i+2: After layer i's MLP, for i = 0..L-1
    
    Each site has both an encoder (dense→sparse) and decoder (sparse→dense).
    """
    
    def __init__(
        self,
        n_layers: int,
        d_dense: int,
        d_sparse: int,
        encoder_afrac: float = 0.25,
    ):
        """
        Args:
            n_layers: Number of transformer layers
            d_dense: Dimension of dense model's residual stream
            d_sparse: Dimension of sparse model's residual stream
            encoder_afrac: Fraction of activations to keep in encoder's AbsTopK
        """
        super().__init__()
        
        self.n_layers = n_layers
        self.n_sites = 2 * n_layers + 1
        self.d_dense = d_dense
        self.d_sparse = d_sparse
        
        # Create bridges for each site
        self.bridges = nn.ModuleList([
            Bridge(d_dense, d_sparse, encoder_afrac)
            for _ in range(self.n_sites)
        ])
    
    def encode(self, site_idx: int, h_dense: torch.Tensor) -> torch.Tensor:
        """Encode dense activations at a specific site."""
        return self.bridges[site_idx].encode(h_dense)
    
    def decode(self, site_idx: int, h_sparse: torch.Tensor) -> torch.Tensor:
        """Decode sparse activations at a specific site."""
        return self.bridges[site_idx].decode(h_sparse)
    
    def encode_all(self, h_dense_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """Encode dense activations at all sites."""
        assert len(h_dense_list) == self.n_sites
        return [
            self.bridges[i].encode(h_dense_list[i])
            for i in range(self.n_sites)
        ]
    
    def decode_all(self, h_sparse_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """Decode sparse activations at all sites."""
        assert len(h_sparse_list) == self.n_sites
        return [
            self.bridges[i].decode(h_sparse_list[i])
            for i in range(self.n_sites)
        ]


def nmse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute Normalized Mean Squared Error.
    
    NMSE = MSE(pred, target) / Var(target).detach()
    
    The denominator is detached to prevent weird optimization effects where
    one model learns high-variance directions to make NMSE artificially small.
    
    Args:
        pred: Predicted activations
        target: Target activations
        eps: Small constant for numerical stability
        
    Returns:
        Scalar NMSE loss
    """
    mse = F.mse_loss(pred, target)
    # Compute variance across all dimensions except batch
    # Flatten to (batch, features) then compute variance
    target_flat = target.reshape(-1, target.shape[-1])
    var = target_flat.var(dim=0).mean()  # Mean variance across features
    
    return mse / (var.detach() + eps)


def compute_bridge_nmse_loss(
    h_dense_list: List[torch.Tensor],
    h_sparse_list: List[torch.Tensor],
    bridge_set: BridgeSet,
) -> torch.Tensor:
    """
    Compute the total NMSE loss for all bridge sites.
    
    L_NMSE = sum_i [ NMSE(encoder_i(h^d_i), h^s_i) + NMSE(decoder_i(h^s_i), h^d_i) ]
    
    Args:
        h_dense_list: List of dense activations at each bridge site
        h_sparse_list: List of sparse activations at each bridge site
        bridge_set: The bridge modules
        
    Returns:
        Total NMSE loss (scalar)
    """
    total_loss = 0.0
    n_sites = len(h_dense_list)
    
    for i in range(n_sites):
        h_d = h_dense_list[i]
        h_s = h_sparse_list[i]
        
        # Encoder loss: predict sparse from dense
        h_s_pred = bridge_set.encode(i, h_d)
        encoder_loss = nmse_loss(h_s_pred, h_s)
        
        # Decoder loss: predict dense from sparse
        h_d_pred = bridge_set.decode(i, h_s)
        decoder_loss = nmse_loss(h_d_pred, h_d)
        
        total_loss = total_loss + encoder_loss + decoder_loss
    
    return total_loss


class KLTargetCache:
    """
    Pre-computed values for efficient KL divergence computation.
    
    When computing multiple KL divergences against the same target distribution
    (e.g., y_dense for all hybrid passes), we can pre-compute the top-k indices
    and target softmax once, then reuse them for all source distributions.
    
    This saves ~4L+2 top-k operations per step for an L-layer model.
    """
    
    def __init__(
        self,
        logits_target: torch.Tensor,
        temperature: float = 1.0,
        topk: Optional[int] = 64,
    ):
        """
        Pre-compute target distribution values.
        
        Args:
            logits_target: Target logits, shape (batch, seq, vocab)
            temperature: Temperature for softmax
            topk: Number of top tokens to use (None for full vocab)
        """
        self.temperature = temperature
        self.topk = topk
        
        # Apply temperature scaling
        logits_scaled = logits_target / temperature
        
        # Flatten to (batch * seq, vocab) for easier processing
        self.orig_shape = logits_scaled.shape
        if logits_scaled.dim() == 3:
            logits_scaled = logits_scaled.reshape(-1, self.orig_shape[-1])
        
        if topk is not None and topk < logits_scaled.shape[-1]:
            # Pre-compute top-k indices (no gradient needed for indices)
            _, self.topk_indices = torch.topk(logits_scaled, topk, dim=-1)  # (N, k)
            
            # Pre-compute target softmax over top-k
            logits_target_topk = torch.gather(logits_scaled, dim=-1, index=self.topk_indices)
            self.p_target = F.softmax(logits_target_topk, dim=-1)  # (N, k)
            self.use_topk = True
        else:
            # Full vocabulary - pre-compute full softmax
            self.p_target = F.softmax(logits_scaled, dim=-1)
            self.topk_indices = None
            self.use_topk = False


def kl_divergence(
    logits_target: torch.Tensor,
    logits_source: torch.Tensor,
    temperature: float = 1.0,
    topk: Optional[int] = 64,
    target_cache: Optional[KLTargetCache] = None,
) -> torch.Tensor:
    """
    Compute KL divergence KL(target || source) with optional top-k approximation.
    
    When topk is specified, only computes KL over the top-k most probable tokens
    from the target distribution. This is much more efficient for large vocabularies
    since most probability mass is concentrated in a few tokens.
    
    The target distribution is what we're trying to match.
    The source distribution is what we're learning.
    
    Args:
        logits_target: Logits from the target distribution (e.g., dense model)
            Shape: (batch, seq, vocab) or (batch * seq, vocab)
            Can be None if target_cache is provided.
        logits_source: Logits from the source distribution (e.g., hybrid pass)
            Shape: same as logits_target
        temperature: Temperature for softmax (default 1.0)
        topk: If specified, only compute KL over top-k tokens from target.
            Set to None for exact KL (slower). Default 64.
        target_cache: Pre-computed target values for efficiency. If provided,
            logits_target is ignored and the cached values are used.
        
    Returns:
        KL divergence (scalar)
    """
    if target_cache is not None:
        # Use pre-computed target values
        temperature = target_cache.temperature
        
        # Apply temperature scaling to source
        logits_source = logits_source / temperature
        
        # Flatten source to match cache shape
        if logits_source.dim() == 3:
            logits_source = logits_source.reshape(-1, logits_source.shape[-1])
        
        if target_cache.use_topk:
            # Gather source logits at pre-computed top-k indices
            logits_source_topk = torch.gather(logits_source, dim=-1, index=target_cache.topk_indices)
            log_p_source = F.log_softmax(logits_source_topk, dim=-1)
            kl = F.kl_div(log_p_source, target_cache.p_target, reduction='batchmean')
        else:
            log_p_source = F.log_softmax(logits_source, dim=-1)
            kl = F.kl_div(log_p_source, target_cache.p_target, reduction='batchmean')
        
        return kl * (temperature ** 2)
    
    # Original implementation when no cache provided
    # Apply temperature scaling
    logits_target = logits_target / temperature
    logits_source = logits_source / temperature
    
    if topk is not None and topk < logits_target.shape[-1]:
        # Top-k approximation: only compute KL over the top-k tokens from target
        # This captures most of the probability mass efficiently
        
        # Flatten to (batch * seq, vocab) for easier processing
        orig_shape = logits_target.shape
        if logits_target.dim() == 3:
            logits_target = logits_target.reshape(-1, orig_shape[-1])
            logits_source = logits_source.reshape(-1, orig_shape[-1])
        
        # Get top-k indices from target distribution
        _, topk_indices = torch.topk(logits_target, topk, dim=-1)  # (N, k)
        
        # Gather the top-k logits from both distributions
        logits_target_topk = torch.gather(logits_target, dim=-1, index=topk_indices)  # (N, k)
        logits_source_topk = torch.gather(logits_source, dim=-1, index=topk_indices)  # (N, k)
        
        # Compute softmax only over the top-k (renormalized)
        p_target = F.softmax(logits_target_topk, dim=-1)
        log_p_source = F.log_softmax(logits_source_topk, dim=-1)
        
        # KL divergence over the top-k tokens
        kl = F.kl_div(log_p_source, p_target, reduction='batchmean')
    else:
        # Exact KL over full vocabulary
        p_target = F.softmax(logits_target, dim=-1)
        log_p_source = F.log_softmax(logits_source, dim=-1)
        kl = F.kl_div(log_p_source, p_target, reduction='batchmean')
    
    # Scale back by temperature^2 for proper gradient scaling
    return kl * (temperature ** 2)


class HybridKLResult:
    """
    Result container for hybrid KL loss computation.
    
    When using gradient buffering, this holds both the losses and the
    gradient buffers that must be released after backward().
    """
    
    def __init__(
        self,
        kl_d2s: torch.Tensor,
        kl_s2d: torch.Tensor,
        gradient_buffers: Optional[List[GradientBuffer]] = None,
    ):
        self.kl_d2s = kl_d2s
        self.kl_s2d = kl_s2d
        self.gradient_buffers = gradient_buffers or []
        self._released = False
    
    def release_gradients(self):
        """Release accumulated gradients from all buffers.
        
        Uses retain_graph=True for all but the last buffer to allow
        shared computation graphs, then retain_graph=False for the
        last buffer to free memory.
        """
        if self._released:
            return
        self._released = True
        
        n_buffers = len(self.gradient_buffers)
        for i, buffer in enumerate(self.gradient_buffers):
            # Only keep graph for all but the last buffer
            retain = (i < n_buffers - 1)
            buffer.release_gradients(retain_graph=retain)
        
        # Clear buffer list to free references
        self.gradient_buffers = []
    
    @property
    def total(self) -> torch.Tensor:
        """Sum of d2s and s2d losses."""
        return self.kl_d2s + self.kl_s2d


def compute_hybrid_kl_losses(
    dense_model: SparseGPT,
    sparse_model: SparseGPT,
    bridge_set: "BridgeSet",
    h_dense_list: List[torch.Tensor],
    h_sparse_list: List[torch.Tensor],
    y_dense: torch.Tensor,
    input_ids: torch.Tensor,
    kl_target_cache: Optional[KLTargetCache] = None,
) -> "HybridKLResult":
    """
    Compute KL losses for hybrid forward passes.
    
    This function computes two types of hybrid KL losses:
    1. d→s: Encode dense activations, run sparse model to end, KL to dense logits
    2. s→d: Decode sparse activations, run dense model to end, KL to dense logits
    
    Uses gradient buffering optimization to accumulate gradients efficiently.
    
    Args:
        dense_model: Frozen dense model
        sparse_model: Sparse model being trained
        bridge_set: Bridge modules
        h_dense_list: Dense activations at each bridge site
        h_sparse_list: Sparse activations at each bridge site
        y_dense: Dense model logits (target)
        input_ids: Input token IDs (for bigram table)
        kl_target_cache: Optional pre-computed KL target values for efficiency.
            If provided, reuses cached top-k indices and target softmax.
        
    Returns:
        HybridKLResult containing kl_d2s, kl_s2d losses and gradient buffers.
        IMPORTANT: Call result.release_gradients() after backward() to propagate
        gradients through the bridges.
    """
    kl_d2s, kl_s2d, buffers = _compute_hybrid_kl_losses_buffered(
        dense_model, sparse_model, bridge_set,
        h_dense_list, h_sparse_list, y_dense, input_ids,
        kl_target_cache
    )
    return HybridKLResult(kl_d2s, kl_s2d, buffers)


def _compute_hybrid_kl_losses_buffered(
    dense_model: SparseGPT,
    sparse_model: SparseGPT,
    bridge_set: "BridgeSet",
    h_dense_list: List[torch.Tensor],
    h_sparse_list: List[torch.Tensor],
    y_dense: torch.Tensor,
    input_ids: torch.Tensor,
    kl_target_cache: Optional[KLTargetCache] = None,
) -> Tuple[torch.Tensor, torch.Tensor, List["GradientBuffer"]]:
    """
    Optimized implementation using gradient buffering.
    
    This accumulates gradients at checkpoint locations, reducing the number
    of backward passes through the later layers of the model.
    
    IMPORTANT: This returns the gradient buffers that must be released after
    calling backward() on the total loss. Use the wrapper function
    `compute_hybrid_kl_losses` which handles this automatically.
    
    Returns:
        Tuple of (kl_d2s, kl_s2d, gradient_buffers)
    """
    n_sites = len(h_dense_list)
    gradient_buffers = []
    
    # =========================================================================
    # KL for dense→sparse hybrid passes (d2s)
    # =========================================================================
    kl_d2s_total = torch.tensor(0.0, device=y_dense.device, dtype=y_dense.dtype)
    
    for i in range(n_sites):
        # Encode dense activation to sparse space
        h_encoded = bridge_set.encode(i, h_dense_list[i])
        
        # Create gradient buffer
        buffer = GradientBuffer(h_encoded)
        gradient_buffers.append(buffer)
        
        # Run sparse model from site i to end using the accumulator
        y_hybrid = sparse_model.forward_from_site(buffer.accumulator, i, input_ids)
        
        # KL(dense || hybrid) - gradients will accumulate in buffer
        # Use cache if provided for efficiency
        kl_d2s_total = kl_d2s_total + kl_divergence(
            y_dense, y_hybrid, target_cache=kl_target_cache
        )
    
    # =========================================================================
    # KL for sparse→dense hybrid passes (s2d)
    # =========================================================================
    kl_s2d_total = torch.tensor(0.0, device=y_dense.device, dtype=y_dense.dtype)
    
    for i in range(n_sites):
        # Decode sparse activation to dense space
        h_decoded = bridge_set.decode(i, h_sparse_list[i])
        
        # Create gradient buffer
        buffer = GradientBuffer(h_decoded)
        gradient_buffers.append(buffer)
        
        # Run dense model from site i to end using the accumulator
        y_hybrid = dense_model.forward_from_site(buffer.accumulator, i, input_ids)
        
        # KL(dense || hybrid) - gradients will accumulate in buffer
        # Use cache if provided for efficiency
        kl_s2d_total = kl_s2d_total + kl_divergence(
            y_dense, y_hybrid, target_cache=kl_target_cache
        )
    
    return kl_d2s_total, kl_s2d_total, gradient_buffers


def verify_model_is_dense(model: nn.Module, model_name: str = "model") -> None:
    """
    Verify that a model has no weight or activation sparsity enabled.
    
    Raises AssertionError if the model has sparsity enabled.
    
    Args:
        model: The model to check
        model_name: Name for error messages
    """
    # Check sparsity config if available
    if hasattr(model, 'sparsity_config'):
        config = model.sparsity_config
        
        assert not config.enable_weight_sparsity, (
            f"{model_name} has weight sparsity enabled (enable_weight_sparsity=True). "
            "The dense model must have no weight sparsity."
        )
        
        assert not config.enable_activation_sparsity, (
            f"{model_name} has activation sparsity enabled (enable_activation_sparsity=True). "
            "The dense model must have no activation sparsity."
        )
    
    # Also check for actual sparse weights (in case loaded model has zeroed weights)
    total_params = 0
    nonzero_params = 0
    
    for name, param in model.named_parameters():
        if len(param.shape) >= 2:  # Only check weight matrices
            total_params += param.numel()
            nonzero_params += (param != 0).sum().item()
    
    if total_params > 0:
        sparsity = 1.0 - (nonzero_params / total_params)
        # Allow small sparsity from numerical zeros, but flag significant sparsity
        if sparsity > 0.01:  # More than 1% zeros is suspicious
            print(f"WARNING: {model_name} has {sparsity*100:.1f}% zero weights. "
                  "This may indicate weight sparsity was applied.")
