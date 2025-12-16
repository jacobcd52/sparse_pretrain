"""
Configuration dataclasses for bridges training.
Based on Gao et al. (2025) "Weight-sparse transformers have interpretable circuits"

Bridges couple a frozen dense model to a weight-sparse model via per-sublayer
linear maps (encoders and decoders).
"""

from dataclasses import dataclass, field
from typing import Optional, Literal
import yaml
from pathlib import Path

from .config import ModelConfig, SparsityConfig, OptimizerConfig, TrainingConfig


@dataclass
class DenseModelConfig:
    """Configuration for the frozen dense model."""
    
    # Path to load the dense model from (HuggingFace repo ID or local path)
    repo_id: str = ""
    
    # If loading from local checkpoint directory instead of HF
    local_path: Optional[str] = None
    

@dataclass
class BridgesConfig:
    """Configuration for bridge modules and losses."""
    
    # Encoder AbsTopK fraction (fraction of activations to keep)
    encoder_afrac: float = 0.25
    
    # Loss coefficients
    coef_nmse: float = 1.0        # Normalized MSE reconstruction loss
    coef_kl_d2s: float = 1.0      # KL for dense→sparse hybrid passes
    coef_kl_s2d: float = 1.0      # KL for sparse→dense hybrid passes
    coef_ce_sparse: float = 1.0   # Cross-entropy on sparse model (standard LM loss)
    coef_kl_sparse: float = 1.0   # KL(dense, sparse) distillation loss


@dataclass
class BridgesTrainingConfig(TrainingConfig):
    """Training configuration for bridges training.
    
    Extends the standard TrainingConfig with bridge-specific settings.
    """
    
    # Override some defaults that may differ for bridges training
    wandb_project: str = "bridges_training"


@dataclass 
class FullBridgesConfig:
    """Full configuration for bridges training.
    
    Contains all sub-configs needed for bridges training:
    - dense_model: Configuration for loading the frozen dense model
    - sparse_model: Architecture config for the sparse model (trained from scratch)
    - bridges: Bridge-specific settings (loss coefficients, encoder afrac)
    - sparsity: Sparsity settings for the sparse model
    - optimizer: Optimizer settings
    - training: Training settings
    """
    
    dense_model: DenseModelConfig = field(default_factory=DenseModelConfig)
    sparse_model: ModelConfig = field(default_factory=ModelConfig)
    bridges: BridgesConfig = field(default_factory=BridgesConfig)
    sparsity: SparsityConfig = field(default_factory=SparsityConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: BridgesTrainingConfig = field(default_factory=BridgesTrainingConfig)
    
    @classmethod
    def from_yaml(cls, path: str) -> "FullBridgesConfig":
        """Load config from YAML file."""
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        
        return cls(
            dense_model=DenseModelConfig(**raw.get("dense_model", {})),
            sparse_model=ModelConfig(**raw.get("sparse_model", {})),
            bridges=BridgesConfig(**raw.get("bridges", {})),
            sparsity=SparsityConfig(**raw.get("sparsity", {})),
            optimizer=OptimizerConfig(**raw.get("optimizer", {})),
            training=BridgesTrainingConfig(**raw.get("training", {})),
        )
    
    def to_yaml(self, path: str) -> None:
        """Save config to YAML file."""
        from dataclasses import asdict
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for W&B logging."""
        from dataclasses import asdict
        return asdict(self)
