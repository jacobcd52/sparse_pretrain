"""
Training loop for weight-sparse transformer pretraining.
Based on Gao et al. (2025) "Weight-sparse transformers have interpretable circuits"

Uses HuggingFace Accelerate for multi-GPU training and W&B for logging.
"""

import os
import math
import time
import json
from pathlib import Path
from typing import Optional
from dataclasses import asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed

from .config import Config
from .model import SparseGPT, create_model
from .sparsity import WeightSparsifier, SharkfinScheduler, clip_grad_rms_, normalize_grad_rms_
from .data import create_dataloader


def count_parameters(model: nn.Module) -> dict:
    """Count various parameter statistics."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Count by component
    embedding = sum(p.numel() for n, p in model.named_parameters() if "wte" in n or "wpe" in n)
    attention = sum(p.numel() for n, p in model.named_parameters() if "attn" in n)
    mlp = sum(p.numel() for n, p in model.named_parameters() if "mlp" in n)
    
    return {
        "total_params": total,
        "trainable_params": trainable,
        "embedding_params": embedding,
        "attention_params": attention,
        "mlp_params": mlp,
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


def compute_weight_stats(model: nn.Module) -> dict:
    """Compute weight statistics for logging."""
    stats = {}
    
    weight_norms = []
    weight_means = []
    weight_stds = []
    
    for name, param in model.named_parameters():
        if param.requires_grad and len(param.shape) >= 2:
            weight_norms.append(param.data.norm(2).item())
            weight_means.append(param.data.mean().item())
            weight_stds.append(param.data.std().item())
    
    if weight_norms:
        stats["weights/norm_mean"] = sum(weight_norms) / len(weight_norms)
        stats["weights/mean_mean"] = sum(weight_means) / len(weight_means)
        stats["weights/std_mean"] = sum(weight_stds) / len(weight_stds)
    
    return stats


def compute_activation_stats(logits: torch.Tensor, loss: torch.Tensor) -> dict:
    """Compute activation/output statistics for logging."""
    stats = {}
    
    with torch.no_grad():
        # Logit statistics
        stats["logits/mean"] = logits.mean().item()
        stats["logits/std"] = logits.std().item()
        stats["logits/max"] = logits.max().item()
        stats["logits/min"] = logits.min().item()
        
        # Probability statistics
        probs = torch.softmax(logits, dim=-1)
        stats["probs/entropy"] = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean().item()
        stats["probs/max_mean"] = probs.max(dim=-1).values.mean().item()
        
        # Perplexity
        stats["perplexity"] = math.exp(min(loss.item(), 100))  # Cap to avoid overflow
    
    return stats


def save_checkpoint(
    accelerator: Accelerator,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: SharkfinScheduler,
    sparsifier: WeightSparsifier,
    step: int,
    loss: float,
    checkpoint_dir: str,
    keep_n: int = 5,
):
    """Save a training checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save checkpoint
    checkpoint_path = checkpoint_dir / f"checkpoint-{step}"
    
    accelerator.save_state(str(checkpoint_path))
    
    # Save additional state
    if accelerator.is_main_process:
        extra_state = {
            "step": step,
            "loss": loss,
            "sparsifier_state": {
                "current_step": sparsifier.state.current_step,
                "current_l0_fraction": sparsifier.state.current_l0_fraction,
            },
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


def train(config: Config):
    """
    Main training function.
    
    Args:
        config: Full configuration object
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
    if accelerator.is_main_process and config.training.use_wandb:
        wandb.init(
            project=config.training.wandb_project,
            name=config.training.wandb_run_name,
            entity=config.training.wandb_entity,
            config=config.to_dict(),
        )
        
        # Save config at the start
        config_path = Path(config.training.checkpoint_dir) / "config.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config.to_yaml(str(config_path))
        wandb.save(str(config_path))
    
    # Create dataloader and tokenizer
    accelerator.print("Loading dataset and tokenizer...")
    dataloader, tokenizer = create_dataloader(
        dataset_name=config.training.dataset_name,
        tokenizer_name=config.training.tokenizer_name,
        seq_length=config.model.n_ctx,
        batch_size=config.training.batch_size,
        split=config.training.dataset_split,
        text_column=config.training.text_column,
        seed=config.training.seed,
    )
    
    # Update vocab size from tokenizer
    config.model.vocab_size = len(tokenizer)
    accelerator.print(f"Vocabulary size: {config.model.vocab_size}")
    
    # Create model
    accelerator.print("Creating model...")
    model = create_model(config.model, config.sparsity)
    
    # Log parameter counts
    param_stats = count_parameters(model)
    accelerator.print(f"Model parameters: {param_stats['total_params']:,}")
    if accelerator.is_main_process and config.training.use_wandb:
        wandb.log({"model/" + k: v for k, v in param_stats.items()}, step=0)
    
    # Enable gradient checkpointing if requested
    if config.training.gradient_checkpointing:
        # For our custom model, we'd need to implement this
        # For now, we'll skip it
        pass
    
    # Calculate training steps
    tokens_per_step = (
        config.training.batch_size 
        * config.model.n_ctx 
        * config.training.gradient_accumulation_steps
        * accelerator.num_processes
    )
    total_steps = config.training.total_tokens // tokens_per_step
    accelerator.print(f"Total training steps: {total_steps:,}")
    accelerator.print(f"Tokens per step: {tokens_per_step:,}")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optimizer.learning_rate,
        betas=(config.optimizer.beta1, config.optimizer.beta2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
        fused=True,
    )
    
    # Create weight sparsifier
    sparsifier = WeightSparsifier(
        model=model,
        target_l0_fraction=config.sparsity.target_l0_fraction,
        anneal_start_fraction=config.sparsity.sparsity_anneal_start_fraction,
        anneal_end_fraction=config.sparsity.sparsity_anneal_end_fraction,
        min_weights_per_neuron=config.sparsity.min_weights_per_neuron,
        total_steps=total_steps,
    ) if config.sparsity.enable_weight_sparsity else None
    
    # Create learning rate scheduler
    scheduler = SharkfinScheduler(
        optimizer=optimizer,
        base_lr=config.optimizer.learning_rate,
        total_steps=total_steps,
        warmup_fraction=config.optimizer.warmup_fraction,
        enable_lr_decay=config.optimizer.enable_lr_decay,
        sparsifier=sparsifier,
        use_sharkfin=config.optimizer.use_sharkfin_schedule,
    )
    
    # Prepare for distributed training
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    
    # Training loop
    accelerator.print("Starting training...")
    
    model.train()
    step = 0
    tokens_seen = 0
    running_loss = 0.0
    start_time = time.time()
    
    progress_bar = tqdm(
        total=total_steps,
        desc="Training",
        disable=not accelerator.is_main_process,
    )
    
    data_iter = iter(dataloader)
    
    # Set initial LR BEFORE first step (authors do this - step 0 has warmup LR of 0)
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
        
        # Forward pass
        with accelerator.accumulate(model):
            # Forward pass - don't compute loss inside (to match authors who compute loss outside autocast)
            logits, _, _ = model(input_ids, labels=None)
            
            # Compute loss OUTSIDE autocast (full precision), matching authors' code
            # shift by 1: logits[:-1] predicts labels[1:]
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            
            # Backward pass
            accelerator.backward(loss)
            
            # Gradient normalization (NOT clipping!)
            # Authors ALWAYS normalize gradients to RMS=1, not just clip when > 1
            if accelerator.sync_gradients and config.optimizer.enable_grad_clip:
                grad_rms = normalize_grad_rms_(model.parameters())
            else:
                grad_rms = 0.0
            
            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()
        
        # Only update after gradient accumulation
        if accelerator.sync_gradients:
            # Apply weight sparsity (after optimizer step)
            if sparsifier is not None:
                sparsifier.step()
            
            # Update learning rate
            scheduler.step()
            
            step += 1
            tokens_seen += tokens_per_step
            running_loss += loss.item()
            
            # Logging
            if step % config.training.log_every_n_steps == 0:
                avg_loss = running_loss / config.training.log_every_n_steps
                running_loss = 0.0
                
                elapsed = time.time() - start_time
                tokens_per_sec = tokens_seen / elapsed
                
                # Compute various statistics
                log_dict = {
                    "train/loss": avg_loss,
                    "train/perplexity": math.exp(min(avg_loss, 100)),
                    "train/learning_rate": scheduler.get_lr(),
                    "train/tokens_seen": tokens_seen,
                    "train/tokens_per_sec": tokens_per_sec,
                    "train/step": step,
                    "train/epoch": tokens_seen / config.training.total_tokens,
                    "train/grad_rms": grad_rms,
                }
                
                # Activation stats
                act_stats = compute_activation_stats(logits, loss)
                log_dict.update({"activations/" + k.split("/")[-1]: v for k, v in act_stats.items()})
                
                # Sparsity stats (at separate frequency)
                if sparsifier is not None and step % config.training.log_sparsity_every_n_steps == 0:
                    sparsity_stats = sparsifier.get_sparsity_stats()
                    log_dict.update({
                        "sparsity/" + k: v for k, v in sparsity_stats.items()
                    })
                
                # Gradient stats (at separate frequency)
                if step % config.training.log_gradients_every_n_steps == 0:
                    grad_stats = compute_grad_stats(accelerator.unwrap_model(model))
                    log_dict.update(grad_stats)
                
                # Weight stats (at separate frequency)
                if step % config.training.log_weights_every_n_steps == 0:
                    weight_stats = compute_weight_stats(accelerator.unwrap_model(model))
                    log_dict.update(weight_stats)
                
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
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    sparsifier=sparsifier,
                    step=step,
                    loss=loss.item(),
                    checkpoint_dir=config.training.checkpoint_dir,
                    keep_n=config.training.keep_n_checkpoints,
                )
            
            progress_bar.update(1)
    
    # Final checkpoint
    save_checkpoint(
        accelerator=accelerator,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        sparsifier=sparsifier,
        step=step,
        loss=loss.item(),
        checkpoint_dir=config.training.checkpoint_dir,
        keep_n=config.training.keep_n_checkpoints,
    )
    
    accelerator.print("Training complete!")
    
    if accelerator.is_main_process and config.training.use_wandb:
        wandb.finish()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train weight-sparse transformer")
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
        help="Override config values, e.g., --override training.batch_size=32 model.n_layer=12",
    )
    
    args = parser.parse_args()
    
    # Load config
    config = Config.from_yaml(args.config)
    
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
    train(config)


if __name__ == "__main__":
    main()

