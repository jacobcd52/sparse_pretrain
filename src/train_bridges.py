"""
Training loop for bridges training.
Based on Gao et al. (2025) "Weight-sparse transformers have interpretable circuits"

Bridges couple a frozen dense model to a weight-sparse model trained from scratch.
The training objective includes:
- Standard LM loss on sparse model
- KL distillation from dense to sparse
- NMSE reconstruction losses for bridges
- KL losses for hybrid forward passes
"""

import os
import math
import time
import json
from pathlib import Path
from typing import Optional, List
from dataclasses import asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed

from .config import ModelConfig, SparsityConfig
from .config_bridges import FullBridgesConfig
from .model import SparseGPT, create_model
from .bridges import (
    BridgeSet,
    compute_bridge_nmse_loss,
    compute_hybrid_kl_losses,
    kl_divergence,
    verify_model_is_dense,
)
from .sparsity import WeightSparsifier, SharkfinScheduler, normalize_grad_rms_
from .data import create_dataloader, create_validation_data


def load_dense_model(config: FullBridgesConfig, device: str = "cpu") -> SparseGPT:
    """
    Load the frozen dense model.
    
    Args:
        config: Full bridges config
        device: Device to load the model on
        
    Returns:
        Loaded dense model in eval mode with requires_grad=False
    """
    if config.dense_model.local_path:
        # Load from local checkpoint
        local_path = Path(config.dense_model.local_path)
        
        # Load config
        config_path = local_path / "config.json"
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        
        model_config = ModelConfig(**config_dict["model_config"])
        sparsity_config = SparsityConfig(**config_dict["sparsity_config"])
        
        # Create and load model
        model = SparseGPT(model_config, sparsity_config)
        
        model_path = local_path / "pytorch_model.bin"
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        
    else:
        # Load from HuggingFace Hub
        model = SparseGPT.from_pretrained(config.dense_model.repo_id, device=device)
    
    # Freeze the model
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    return model


def count_parameters(model: nn.Module) -> dict:
    """Count various parameter statistics."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_params": total,
        "trainable_params": trainable,
    }


def compute_grad_stats(model: nn.Module) -> dict:
    """Compute gradient statistics for logging."""
    stats = {}
    
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2).item()
            grad_norms.append(grad_norm)
    
    if grad_norms:
        stats["grad/norm_mean"] = sum(grad_norms) / len(grad_norms)
        stats["grad/norm_max"] = max(grad_norms)
        stats["grad/norm_min"] = min(grad_norms)
    
    return stats


@torch.no_grad()
def evaluate_validation(
    dense_model: nn.Module,
    sparse_model: nn.Module,
    val_batches: list,
    accelerator: Accelerator,
    batch_size: int = 16,
    device: str = "cuda",
) -> dict:
    """
    Evaluate models on validation data.
    
    Args:
        dense_model: The frozen dense model
        sparse_model: The sparse model being trained
        val_batches: List of token tensors
        accelerator: Accelerator for mixed precision autocast
        batch_size: Batch size for evaluation
        device: Device to use
        
    Returns:
        Dictionary with validation metrics
    """
    sparse_model.eval()
    
    total_loss_sparse = 0.0
    total_loss_dense = 0.0
    total_tokens = 0
    
    for i in range(0, len(val_batches), batch_size):
        batch_tensors = val_batches[i:i+batch_size]
        input_ids = torch.stack(batch_tensors).to(device)
        
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            # Sparse model loss
            logits_sparse, _, _ = sparse_model(input_ids, labels=None)
            
            # Dense model loss
            logits_dense, _, _ = dense_model(input_ids, labels=None)
        
        # Compute losses
        shift_logits_sparse = logits_sparse[:, :-1, :].contiguous()
        shift_logits_dense = logits_dense[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        
        loss_sparse = F.cross_entropy(
            shift_logits_sparse.view(-1, shift_logits_sparse.size(-1)),
            shift_labels.view(-1),
            reduction='sum',
        )
        
        loss_dense = F.cross_entropy(
            shift_logits_dense.view(-1, shift_logits_dense.size(-1)),
            shift_labels.view(-1),
            reduction='sum',
        )
        
        total_loss_sparse += loss_sparse.item()
        total_loss_dense += loss_dense.item()
        total_tokens += shift_labels.numel()
    
    sparse_model.train()
    
    avg_loss_sparse = total_loss_sparse / total_tokens if total_tokens > 0 else 0.0
    avg_loss_dense = total_loss_dense / total_tokens if total_tokens > 0 else 0.0
    
    return {
        "val/loss_sparse": avg_loss_sparse,
        "val/loss_dense": avg_loss_dense,
        "val/perplexity_sparse": math.exp(min(avg_loss_sparse, 100)),
        "val/perplexity_dense": math.exp(min(avg_loss_dense, 100)),
        "val/tokens": total_tokens,
    }


def save_checkpoint(
    accelerator: Accelerator,
    sparse_model: nn.Module,
    bridge_set: BridgeSet,
    optimizer: torch.optim.Optimizer,
    scheduler: SharkfinScheduler,
    sparsifier: Optional[WeightSparsifier],
    step: int,
    loss: float,
    checkpoint_dir: str,
    keep_n: int = 5,
):
    """Save a training checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = checkpoint_dir / f"checkpoint-{step}"
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    if accelerator.is_main_process:
        # Save sparse model
        sparse_model_unwrapped = accelerator.unwrap_model(sparse_model)
        torch.save(
            sparse_model_unwrapped.state_dict(),
            checkpoint_path / "sparse_model.bin"
        )
        
        # Save bridges
        bridge_set_unwrapped = accelerator.unwrap_model(bridge_set)
        torch.save(
            bridge_set_unwrapped.state_dict(),
            checkpoint_path / "bridges.bin"
        )
        
        # Save optimizer
        torch.save(optimizer.state_dict(), checkpoint_path / "optimizer.bin")
        
        # Save extra state
        extra_state = {
            "step": step,
            "loss": loss,
            "sparsifier_state": {
                "current_step": sparsifier.state.current_step,
                "current_l0_fraction": sparsifier.state.current_l0_fraction,
            } if sparsifier is not None else None,
            "scheduler_state": {
                "current_step": scheduler.current_step,
            },
        }
        with open(checkpoint_path / "extra_state.json", "w") as f:
            json.dump(extra_state, f)
    
    # Clean up old checkpoints
    if accelerator.is_main_process:
        checkpoints = sorted(
            checkpoint_dir.glob("checkpoint-*"),
            key=lambda x: int(x.name.split("-")[1])
        )
        while len(checkpoints) > keep_n:
            oldest = checkpoints.pop(0)
            import shutil
            shutil.rmtree(oldest)


def train_bridges(config: FullBridgesConfig):
    """
    Main bridges training function.
    
    Args:
        config: Full bridges configuration object
    """
    # Initialize accelerator
    accelerator = Accelerator(
        mixed_precision=config.training.mixed_precision,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        log_with="wandb" if config.training.use_wandb else None,
    )
    
    # Set seed for reproducibility
    set_seed(config.training.seed)
    
    # Initialize W&B on main process
    wandb_run_url = None
    if accelerator.is_main_process and config.training.use_wandb:
        wandb.init(
            project=config.training.wandb_project,
            name=config.training.wandb_run_name,
            entity=config.training.wandb_entity,
            config=config.to_dict(),
        )
        wandb_run_url = wandb.run.url
        accelerator.print(f"W&B run: {wandb_run_url}")
        
        # Save config at the start
        config_path = Path(config.training.checkpoint_dir) / "config.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config.to_yaml(str(config_path))
        wandb.save(str(config_path))
    
    # ==========================================================================
    # Load and verify dense model
    # ==========================================================================
    accelerator.print("Loading dense model...")
    dense_model = load_dense_model(config, device=accelerator.device)
    
    # Verify it's actually dense
    verify_model_is_dense(dense_model, "dense_model")
    accelerator.print(f"  Dense model verified: no sparsity")
    
    d_dense = dense_model.config.d_model
    n_layers_dense = dense_model.config.n_layer
    accelerator.print(f"  d_model={d_dense}, n_layers={n_layers_dense}")
    
    # ==========================================================================
    # Create dataloader and tokenizer
    # ==========================================================================
    accelerator.print("Loading dataset and tokenizer...")
    dataloader, tokenizer = create_dataloader(
        dataset_name=config.training.dataset_name,
        tokenizer_name=config.training.tokenizer_name,
        seq_length=config.sparse_model.n_ctx,
        batch_size=config.training.batch_size,
        split=config.training.dataset_split,
        text_column=config.training.text_column,
        seed=config.training.seed,
        process_index=accelerator.process_index,
        num_processes=accelerator.num_processes,
    )
    
    # Update vocab size from tokenizer
    config.sparse_model.vocab_size = len(tokenizer)
    accelerator.print(f"Vocabulary size: {config.sparse_model.vocab_size}")
    
    # Create validation data
    accelerator.print("Loading validation data...")
    val_batches, val_desc = create_validation_data(
        dataset_name=config.training.dataset_name,
        tokenizer=tokenizer,
        seq_length=config.sparse_model.n_ctx,
        text_column=config.training.text_column,
        val_split=config.training.val_split,
        holdout_fraction=config.training.val_holdout_fraction,
        max_tokens=config.training.val_max_batches * config.sparse_model.n_ctx * 16,
        seed=config.training.seed + 1,
    )
    accelerator.print(f"  {val_desc}")
    
    # ==========================================================================
    # Create sparse model (randomly initialized)
    # ==========================================================================
    accelerator.print("Creating sparse model...")
    sparse_model = create_model(config.sparse_model, config.sparsity)
    
    d_sparse = config.sparse_model.d_model
    n_layers_sparse = config.sparse_model.n_layer
    accelerator.print(f"  d_model={d_sparse}, n_layers={n_layers_sparse}")
    
    # Verify layer counts match (required for bridges)
    assert n_layers_dense == n_layers_sparse, (
        f"Dense and sparse models must have the same number of layers. "
        f"Got {n_layers_dense} vs {n_layers_sparse}"
    )
    
    # Log parameter counts
    param_stats_sparse = count_parameters(sparse_model)
    param_stats_dense = count_parameters(dense_model)
    accelerator.print(f"  Sparse model parameters: {param_stats_sparse['total_params']:,}")
    accelerator.print(f"  Dense model parameters: {param_stats_dense['total_params']:,}")
    
    # ==========================================================================
    # Create bridges
    # ==========================================================================
    accelerator.print("Creating bridges...")
    bridge_set = BridgeSet(
        n_layers=n_layers_sparse,
        d_dense=d_dense,
        d_sparse=d_sparse,
        encoder_afrac=config.bridges.encoder_afrac,
    )
    
    param_stats_bridges = count_parameters(bridge_set)
    accelerator.print(f"  Bridge parameters: {param_stats_bridges['total_params']:,}")
    accelerator.print(f"  Number of bridge sites: {bridge_set.n_sites}")
    
    if accelerator.is_main_process and config.training.use_wandb:
        wandb.log({
            "model/sparse_params": param_stats_sparse["total_params"],
            "model/dense_params": param_stats_dense["total_params"],
            "model/bridge_params": param_stats_bridges["total_params"],
        }, step=0)
    
    # ==========================================================================
    # Calculate training steps
    # ==========================================================================
    tokens_per_step = (
        config.training.batch_size 
        * config.sparse_model.n_ctx 
        * config.training.gradient_accumulation_steps
        * accelerator.num_processes
    )
    total_steps = config.training.total_tokens // tokens_per_step
    accelerator.print(f"Total training steps: {total_steps:,}")
    accelerator.print(f"Tokens per step: {tokens_per_step:,}")
    
    # ==========================================================================
    # Create optimizer (only for sparse model and bridges)
    # ==========================================================================
    # Combine parameters from sparse model and bridges
    trainable_params = list(sparse_model.parameters()) + list(bridge_set.parameters())
    
    use_raw_adam = config.optimizer.optimizer_type == "adam"
    
    if use_raw_adam:
        try:
            optimizer = torch.optim.Adam(
                trainable_params,
                lr=config.optimizer.learning_rate,
                betas=(config.optimizer.beta1, config.optimizer.beta2),
                eps=config.optimizer.eps,
                fused=True,
            )
        except TypeError:
            optimizer = torch.optim.Adam(
                trainable_params,
                lr=config.optimizer.learning_rate,
                betas=(config.optimizer.beta1, config.optimizer.beta2),
                eps=config.optimizer.eps,
            )
        accelerator.print("Using raw Adam optimizer")
    else:
        try:
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=config.optimizer.learning_rate,
                betas=(config.optimizer.beta1, config.optimizer.beta2),
                eps=config.optimizer.eps,
                weight_decay=config.optimizer.weight_decay,
                fused=True,
            )
        except TypeError:
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=config.optimizer.learning_rate,
                betas=(config.optimizer.beta1, config.optimizer.beta2),
                eps=config.optimizer.eps,
                weight_decay=config.optimizer.weight_decay,
            )
    
    # ==========================================================================
    # Create weight sparsifier (for sparse model only, NOT bridges)
    # ==========================================================================
    sparsifier = WeightSparsifier(
        model=sparse_model,  # Only sparsify the sparse model
        target_l0_fraction=config.sparsity.target_l0_fraction,
        anneal_start_fraction=config.sparsity.sparsity_anneal_start_fraction,
        anneal_end_fraction=config.sparsity.sparsity_anneal_end_fraction,
        min_weights_per_neuron=config.sparsity.min_weights_per_neuron,
        total_steps=total_steps,
        anneal_type=config.sparsity.anneal_type,
    ) if config.sparsity.enable_weight_sparsity else None
    
    # ==========================================================================
    # Create learning rate scheduler
    # ==========================================================================
    scheduler = SharkfinScheduler(
        optimizer=optimizer,
        base_lr=config.optimizer.learning_rate,
        total_steps=total_steps,
        warmup_fraction=config.optimizer.warmup_fraction,
        enable_lr_decay=config.optimizer.enable_lr_decay,
        sparsifier=sparsifier,
        use_sharkfin=config.optimizer.use_sharkfin_schedule,
    )
    
    # ==========================================================================
    # Prepare for distributed training
    # ==========================================================================
    sparse_model, bridge_set, optimizer, dataloader = accelerator.prepare(
        sparse_model, bridge_set, optimizer, dataloader
    )
    
    # Move dense model to device (it's not optimized so not prepared)
    dense_model = dense_model.to(accelerator.device)
    
    # Cast all modules to the same dtype for mixed precision
    # This is needed because hybrid KL losses are computed outside autocast for stability
    if config.training.mixed_precision == "bf16":
        sparse_model = sparse_model.to(torch.bfloat16)
        dense_model = dense_model.to(torch.bfloat16)
        bridge_set = bridge_set.to(torch.bfloat16)
    elif config.training.mixed_precision == "fp16":
        sparse_model = sparse_model.to(torch.float16)
        dense_model = dense_model.to(torch.float16)
        bridge_set = bridge_set.to(torch.float16)
    
    # ==========================================================================
    # Training loop
    # ==========================================================================
    accelerator.print("Starting bridges training...")
    
    sparse_model.train()
    bridge_set.train()
    
    step = 0
    tokens_seen = 0
    running_loss = 0.0
    running_loss_components = {
        "ce_sparse": 0.0,
        "kl_sparse": 0.0,
        "nmse": 0.0,
        "kl_d2s": 0.0,
        "kl_s2d": 0.0,
    }
    start_time = time.time()
    
    progress_bar = tqdm(
        total=total_steps,
        desc="Training",
        disable=not accelerator.is_main_process,
    )
    
    data_iter = iter(dataloader)
    grad_accum_steps = config.training.gradient_accumulation_steps
    
    # Set initial LR
    scheduler.step()
    
    while step < total_steps:
        # Get batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        
        # Forward/backward pass
        with accelerator.accumulate(sparse_model):
            # Determine autocast dtype
            if config.training.mixed_precision == "bf16":
                autocast_dtype = torch.bfloat16
            elif config.training.mixed_precision == "fp16":
                autocast_dtype = torch.float16
            else:
                autocast_dtype = None
            
            # ==================================================================
            # Forward passes
            # ==================================================================
            if autocast_dtype is not None:
                with torch.amp.autocast(device_type="cuda", dtype=autocast_dtype):
                    # Dense model forward (no grad)
                    with torch.no_grad():
                        y_dense, h_dense_list = dense_model.forward_with_bridge_sites(input_ids)
                    
                    # Sparse model forward
                    y_sparse, h_sparse_list = sparse_model.forward_with_bridge_sites(input_ids)
            else:
                with torch.no_grad():
                    y_dense, h_dense_list = dense_model.forward_with_bridge_sites(input_ids)
                y_sparse, h_sparse_list = sparse_model.forward_with_bridge_sites(input_ids)
            
            # ==================================================================
            # Compute losses (outside autocast for numerical stability)
            # ==================================================================
            
            # Shift for next-token prediction
            shift_logits_sparse = y_sparse[:, :-1, :].contiguous()
            shift_logits_dense = y_dense[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            # 1. Cross-entropy on sparse model (standard LM loss)
            loss_ce_sparse = F.cross_entropy(
                shift_logits_sparse.view(-1, shift_logits_sparse.size(-1)),
                shift_labels.view(-1),
            )
            
            # 2. KL distillation from dense to sparse
            loss_kl_sparse = kl_divergence(shift_logits_dense, shift_logits_sparse)
            
            # 3. NMSE reconstruction loss
            loss_nmse = compute_bridge_nmse_loss(
                h_dense_list, h_sparse_list, bridge_set
            )
            
            # 4 & 5. Hybrid KL losses
            hybrid_result = compute_hybrid_kl_losses(
                dense_model=dense_model,
                sparse_model=sparse_model,
                bridge_set=bridge_set,
                h_dense_list=h_dense_list,
                h_sparse_list=h_sparse_list,
                y_dense=y_dense,
                input_ids=input_ids,
            )
            loss_kl_d2s = hybrid_result.kl_d2s
            loss_kl_s2d = hybrid_result.kl_s2d
            
            # Total loss with configurable coefficients
            total_loss = (
                config.bridges.coef_ce_sparse * loss_ce_sparse
                + config.bridges.coef_kl_sparse * loss_kl_sparse
                + config.bridges.coef_nmse * loss_nmse
                + config.bridges.coef_kl_d2s * loss_kl_d2s
                + config.bridges.coef_kl_s2d * loss_kl_s2d
            )
            
            # Track loss components
            running_loss_components["ce_sparse"] += loss_ce_sparse.detach().item() / grad_accum_steps
            running_loss_components["kl_sparse"] += loss_kl_sparse.detach().item() / grad_accum_steps
            running_loss_components["nmse"] += loss_nmse.detach().item() / grad_accum_steps
            running_loss_components["kl_d2s"] += loss_kl_d2s.detach().item() / grad_accum_steps
            running_loss_components["kl_s2d"] += loss_kl_s2d.detach().item() / grad_accum_steps
            
            # Backward pass
            # Use retain_graph=True to allow gradient buffer release after
            accelerator.backward(total_loss, retain_graph=True)
            hybrid_result.release_gradients()
            
            # Gradient normalization
            if accelerator.sync_gradients and config.optimizer.enable_grad_clip:
                # Normalize gradients for all trainable params
                all_params = list(sparse_model.parameters()) + list(bridge_set.parameters())
                grad_rms = normalize_grad_rms_(all_params)
            else:
                grad_rms = 0.0
            
            # Optimizer step
            if accelerator.sync_gradients:
                optimizer.step()
                optimizer.zero_grad()
        
        # Only update after gradient accumulation
        if accelerator.sync_gradients:
            # Apply weight sparsity (to sparse model only, NOT bridges)
            if sparsifier is not None:
                sparsifier.step()
            
            # Manual weight decay for raw Adam
            if use_raw_adam and config.optimizer.weight_decay > 0:
                current_lr = scheduler.get_lr()
                unwrapped_sparse = accelerator.unwrap_model(sparse_model)
                # Note: bridges don't get weight decay (they're relatively small)
                with torch.no_grad():
                    for name, param in unwrapped_sparse.named_parameters():
                        if len(param.shape) > 1 and "bigram_table" not in name:
                            param.data -= config.optimizer.weight_decay * current_lr * param.data
            
            # Update learning rate
            scheduler.step()
            
            step += 1
            tokens_seen += tokens_per_step
            running_loss += total_loss.detach().item()
            
            # Logging
            if step % config.training.log_every_n_steps == 0:
                avg_loss = running_loss / config.training.log_every_n_steps
                running_loss = 0.0
                
                elapsed = time.time() - start_time
                tokens_per_sec = tokens_seen / elapsed
                
                log_dict = {
                    "train/loss": avg_loss,
                    "train/loss_ce_sparse": running_loss_components["ce_sparse"] / config.training.log_every_n_steps,
                    "train/loss_kl_sparse": running_loss_components["kl_sparse"] / config.training.log_every_n_steps,
                    "train/loss_nmse": running_loss_components["nmse"] / config.training.log_every_n_steps,
                    "train/loss_kl_d2s": running_loss_components["kl_d2s"] / config.training.log_every_n_steps,
                    "train/loss_kl_s2d": running_loss_components["kl_s2d"] / config.training.log_every_n_steps,
                    "train/learning_rate": scheduler.get_lr(),
                    "train/tokens_seen": tokens_seen,
                    "train/tokens_per_sec": tokens_per_sec,
                    "train/step": step,
                    "train/grad_rms": grad_rms,
                }
                
                # Reset loss components
                for key in running_loss_components:
                    running_loss_components[key] = 0.0
                
                # Sparsity stats
                if sparsifier is not None and step % config.training.log_sparsity_every_n_steps == 0:
                    sparsity_stats = sparsifier.get_sparsity_stats()
                    log_dict.update({
                        "sparsity/" + k: v for k, v in sparsity_stats.items()
                    })
                
                # Gradient stats
                if step % config.training.log_gradients_every_n_steps == 0:
                    grad_stats = compute_grad_stats(accelerator.unwrap_model(sparse_model))
                    log_dict.update(grad_stats)
                
                # Validation
                if (
                    accelerator.is_main_process
                    and step % config.training.eval_every_n_steps == 0
                    and len(val_batches) > 0
                ):
                    val_stats = evaluate_validation(
                        dense_model=dense_model,
                        sparse_model=accelerator.unwrap_model(sparse_model),
                        val_batches=val_batches[:config.training.val_max_batches],
                        accelerator=accelerator,
                        batch_size=16,
                        device=accelerator.device,
                    )
                    log_dict.update(val_stats)
                    accelerator.print(
                        f"  Step {step}: sparse_loss={val_stats['val/loss_sparse']:.4f}, "
                        f"dense_loss={val_stats['val/loss_dense']:.4f}"
                    )
                
                if accelerator.is_main_process and config.training.use_wandb:
                    wandb.log(log_dict, step=step)
                
                progress_bar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "lr": f"{scheduler.get_lr():.2e}",
                    "L0": f"{sparsifier.get_current_l0_fraction():.4f}" if sparsifier else "N/A",
                })
            
            # Checkpointing
            if step % config.training.checkpoint_every_n_steps == 0:
                save_checkpoint(
                    accelerator=accelerator,
                    sparse_model=sparse_model,
                    bridge_set=bridge_set,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    sparsifier=sparsifier,
                    step=step,
                    loss=total_loss.item(),
                    checkpoint_dir=config.training.checkpoint_dir,
                    keep_n=config.training.keep_n_checkpoints,
                )
            
            progress_bar.update(1)
    
    # Final checkpoint
    save_checkpoint(
        accelerator=accelerator,
        sparse_model=sparse_model,
        bridge_set=bridge_set,
        optimizer=optimizer,
        scheduler=scheduler,
        sparsifier=sparsifier,
        step=step,
        loss=total_loss.item(),
        checkpoint_dir=config.training.checkpoint_dir,
        keep_n=config.training.keep_n_checkpoints,
    )
    
    accelerator.print("Bridges training complete!")
    
    if accelerator.is_main_process and config.training.use_wandb:
        wandb.finish()


def main():
    """Main entry point for bridges training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train weight-sparse transformer with bridges")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--override",
        type=str,
        nargs="*",
        default=[],
        help="Override config values, e.g., --override bridges.coef_nmse=2.0 training.batch_size=32",
    )
    
    args = parser.parse_args()
    
    # Load config
    config = FullBridgesConfig.from_yaml(args.config)
    
    # Apply overrides
    for override in args.override:
        key, value = override.split("=", 1)
        parts = key.split(".")
        
        # Navigate to the right config object
        obj = config
        for part in parts[:-1]:
            obj = getattr(obj, part)
        
        # Convert value to appropriate type
        attr_name = parts[-1]
        current_value = getattr(obj, attr_name)
        
        if isinstance(current_value, bool):
            value = value.lower() in ("true", "1", "yes")
        elif isinstance(current_value, int):
            value = int(value)
        elif isinstance(current_value, float):
            value = float(value)
        
        setattr(obj, attr_name, value)
    
    # Run training
    train_bridges(config)


if __name__ == "__main__":
    main()
