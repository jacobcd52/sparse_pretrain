# Design Choices

This document explains design choices made in this implementation, particularly those that follow the Gao et al. (2025) paper "Weight-sparse transformers have interpretable circuits" or deviate from standard practice.

**Note**: Default values are taken from the authors' reference code where available, which sometimes differ from values mentioned in the paper text.

## Architecture Choices

### Model Dimensions
**Authors' code**: Uses d_model=1024 as default, while paper text mentions d_model=2048 for some experiments.

**Config**: `model.d_model` (default: `1024`)

### Residual Stream Activation
**From authors' code**: An optional activation can be applied to the residual stream after each sub-block.

**Options**: "identity" (default, no activation) or "relu"

**Config**: `model.residual_activation` (default: `"identity"`)


### RMSNorm Instead of LayerNorm
**Paper Section 1.3**: The paper uses RMSNorm instead of LayerNorm.

**Reason**: "Ensures zero values have privileged meaning in residual stream." This is important for weight sparsity because we want zeros to be semantically meaningful.

**Config**: `model.use_rms_norm` (default: `true`)

### Small Head Dimension (d_head=16)
**Paper Section 1.2**: The paper uses d_head=16, which is much smaller than the standard GPT-2 value of 64.

**Reason**: "Smaller than typical for monosemanticity." Smaller heads lead to more heads, which may help with finding interpretable circuits.

**Trade-off**: More heads means more attention patterns to analyze, but each head has less representational capacity.

**Config**: `model.d_head` (default: `16`)

### Untied Embeddings
**Paper Section 1.4**: Token and unembedding matrices are separate parameters.

**Reason**: This allows the embeddings to specialize - the input embedding learns good representations for understanding, while the output embedding learns good representations for prediction.

**Config**: `model.tie_embeddings` (default: `false`)

### No Positional Embeddings
**Paper Section 1.4**: "Positional embeddings: NOT USED in most experiments"

**Reason**: Reference to Haviv et al. (2022) shows models work without positional encodings. This is "roughly neutral on loss" (Figure 26).

**Trade-off**: The model must infer position from other cues. This may limit performance on tasks that require precise positional reasoning.

**Config**: `model.use_positional_embeddings` (default: `false`)

### Bigram Table
**Paper Section 1.5**: A dense d_vocab × d_vocab matrix is added directly to final logits.

**Reason**: "Avoids sparse parameters needing to memorize bigram frequencies." This lets the sparse weights focus on more complex patterns while bigram statistics are handled separately.

**Trade-off**: Adds O(vocab_size²) parameters, but these are dense and don't count toward sparsity. With vocab_size=2048, this is ~4M parameters.

**Config**: `model.use_bigram_table` (default: `true`)

### Attention Sinks
**Paper Section 1.6**: Per-head learnable attention denominator bias.

**Reason**: "Leads to cleaner attention circuits without impacting loss substantially" (Figure 18). This allows attention heads to "dump" probability mass when no key is relevant.

**Implementation**: We add a dummy KV slot with learnable logit bias and zero value.

**Config**: `model.use_attention_sinks` (default: `true`)

## Sparsity Choices

### Post-Optimizer Magnitude Pruning
**Paper Section 2.1**: After each optimizer step, zero out all but the largest magnitude entries.

**Why not learned masks?**: The paper tested L0 regularization (Louizos et al., 2018) but found it "performs consistently worse than TopK" (Figure 35).

### Same Sparsity Fraction for All Matrices
**Paper Section 2.1**: "Every matrix has the SAME fraction of nonzero elements."

**Reason**: Simplicity and interpretability. Different sparsity per layer would add hyperparameters and complicate analysis.

**Config**: `sparsity.target_l0_fraction` (default: `0.015625` = 1/64 from authors' code)

### Sparsity Annealing Schedule
**Paper Section 2.2**: "Linearly anneal L0 from dense → target L0 over first 50% of training."

**From authors' code**: Annealing starts at 1% of training (not 0%) and ends at 50%.

**Reason**: Gradual sparsification allows the model to first learn good representations, then prune. Starting sparse hurts convergence. We always start fully dense (L0=1.0) and always anneal.

**Config**:
- `sparsity.sparsity_anneal_start_fraction` (default: `0.01`) - When to START annealing
- `sparsity.sparsity_anneal_end_fraction` (default: `0.5`) - When to END annealing

### Minimum 4 Nonzero Weights Per Neuron
**Paper Section 2.3**: "Never zero out values that would cause a neuron or attention channel to have fewer than j nonzero values"

**Reason**: "Reduces chance of dead neurons."

**Trade-off**: "This may artificially keep some irrelevant neurons alive."

**Config**: `sparsity.min_weights_per_neuron` (default: `4`)

### The L0 Scheduling "Bug"
**Paper Section 2.4**: There was a bug where embeddings and biases went from dense to sparse abruptly mid-training.

**Our Choice**: We implement the CORRECT behavior (gradual annealing for all parameters). The paper kept the bug because "Fixing this bug slightly hurt model quality."

**Rationale**: The bug's benefit is likely specific to their setup. Correct behavior is more predictable.

## Activation Sparsity Choices

### AbsTopK with k = 1/4
**Paper Section 3.1**: "Default k = 1/4 of the dimension at each location"

**Reason**: Some activation sparsity helps, but too much hurts. 1/4 is a good balance (Figure 37).

**Config**: `sparsity.activation_topk_fraction` (default: `0.25`)

### Activation Sparsity Locations
**Paper Section 1.7 and 3.2**: AbsTopK is applied at:
- After each RMSNorm in attention and MLP blocks
- After Q, K, V projections (separately)
- After MLP activation (post-GELU)
- At the end of each attention and MLP block

**Config**: `sparsity.activation_sparsity_locations`

## Optimizer Choices (CRITICAL!)

### CRITICAL: Adam with Manual Weight Decay
**From authors' code**: They use vanilla `torch.optim.Adam` (NOT AdamW) and apply weight decay MANUALLY after the optimizer step: `p.data -= wd * lr * p.data`. This means weight decay scales with learning rate, unlike AdamW where it's decoupled.

### CRITICAL: Gradient Normalization (NOT Clipping)
**From authors' code**: They ALWAYS normalize gradients to RMS=1, not just clip when RMS > 1. This ensures gradients always have the same scale, which is fundamentally different from clipping.

### Sharkfin LR Schedule NOT Used by Default
**From authors' code**: The sharkfin schedule (LR × 1/√L0) is only used in their legacy "pfrac" interface, not their simplified "frac_nonzero" interface.

### Unusually Large Epsilon (ε = 0.1)
**Paper Section 4.1**: "ε = 0.1 (NOTE: unusually large epsilon!)"

**Standard value**: PyTorch default is 1e-8.

**Reason**: Not explicitly stated, but likely helps with sparse gradient updates where some parameters get very small gradients.

**Config**: `optimizer.eps` (default: `0.1`)

### RMS Gradient Clipping
**Paper Section 4.2**: "Clip root-mean-square of gradient to 1. This is ESSENTIAL for training stability."

**Note**: This is different from the more common L2 norm clipping. RMS clipping scales by the number of parameters.

**Config**: `optimizer.grad_clip_rms` (default: `1.0`)

### Sharkfin Learning Rate Schedule
**Paper Section 4.3**: LR = base_schedule × 1/√L0

**Reason**: "Smaller L0 requires larger learning rates." As we prune more weights, the remaining weights need larger updates to compensate.

**Config**: `optimizer.use_sharkfin_schedule` (default: `true`)

### 1% Warmup
**Paper Section 4.4**: "Warmup for first 1% of training"

**Reason**: "Critical for stability at higher learning rates."

**Config**: `optimizer.warmup_fraction` (default: `0.01`)

## Other Choices

### Multi-GPU Strategy: HuggingFace Accelerate
We use HuggingFace Accelerate for multi-GPU training because:
1. It's well-maintained and widely used
2. Handles DDP automatically
3. Integrates well with W&B
4. Supports mixed precision easily

### Streaming Dataset
We use HuggingFace datasets in streaming mode because:
1. Works with arbitrarily large datasets
2. No need to download entire dataset upfront
3. Memory efficient

### Token-level Chunking
We concatenate all text and chunk into fixed-length sequences. This is standard for language model pretraining and maximizes GPU utilization.

## Differences from Paper

### Not Implemented
1. **The L0 scheduling bug**: We implement correct behavior (see above)
2. **Circuit pruning**: This is for later
3. **Evaluation tasks**: The 20 hand-crafted Python tasks are for later
4. **Feature binarization**: Validation method for later
5. **Bridges methodology**: Excluded per user request

### Potentially Different
1. **Dataset**: Paper uses Python code from GPT-4. User will specify their own dataset.
2. **Tokenizer**: Paper uses custom 2048-vocab BPE. User will specify their own tokenizer.
3. **Exact LR values**: Paper says "swept for every experiment." We use a default that may need tuning.

## Logging Philosophy

We log extensively because:
1. ML training can fail silently
2. Debugging is easier with more data
3. Storage is cheap, re-running experiments is expensive

**From authors' code**: Separate configurable frequencies for different types of metrics.

**Config**:
- `training.log_every_n_steps` (default: `1`) - Basic metrics (loss, LR, tokens/sec)
- `training.log_gradients_every_n_steps` (default: `10`) - Gradient statistics
- `training.log_weights_every_n_steps` (default: `100`) - Weight statistics
- `training.log_sparsity_every_n_steps` (default: `50`) - Sparsity metrics

We log:
- **Every log step**: loss, learning rate, tokens seen, grad RMS, activation stats
- **At gradient frequency**: gradient norm mean/max/min
- **At weights frequency**: weight norm/mean/std
- **At sparsity frequency**: current L0, target L0, sharkfin multiplier, actual nonzero count
- **Performance**: tokens/second

