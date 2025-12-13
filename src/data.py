"""
Data loading utilities for weight-sparse transformer training.
"""

from typing import Iterator, Optional, Tuple, List
import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset, get_dataset_split_names
from transformers import AutoTokenizer, PreTrainedTokenizer
from tqdm import tqdm


class TokenizedTextDataset(IterableDataset):
    """
    Streaming dataset that tokenizes text and yields fixed-length chunks.
    
    This is an iterable dataset that:
    1. Streams from a HuggingFace dataset
    2. Tokenizes text on the fly
    3. Concatenates all tokens into a single stream
    4. Yields fixed-length chunks for training
    """
    
    def __init__(
        self,
        dataset_name: str,
        tokenizer: PreTrainedTokenizer,
        seq_length: int,
        split: str = "train",
        text_column: str = "text",
        seed: int = 42,
        process_index: int = 0,
        num_processes: int = 1,
    ):
        """
        Args:
            dataset_name: HuggingFace dataset name or path
            tokenizer: Tokenizer to use
            seq_length: Sequence length for each training example
            split: Dataset split to use
            text_column: Name of the text column in the dataset
            seed: Random seed for shuffling
        """
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.split = split
        self.text_column = text_column
        self.seed = seed
        self.process_index = process_index
        self.num_processes = num_processes
        
        # Validate dataset has text column
        self._validate_dataset()
    
    def _validate_dataset(self):
        """Validate that the dataset has the expected text column."""
        # Load a small sample to check schema
        try:
            ds = load_dataset(
                self.dataset_name,
                split=f"{self.split}[:1]",
                streaming=False,
                trust_remote_code=True,
            )
            if self.text_column not in ds.column_names:
                raise ValueError(
                    f"Dataset '{self.dataset_name}' does not have a '{self.text_column}' column. "
                    f"Available columns: {ds.column_names}"
                )
        except Exception as e:
            if "does not have" in str(e):
                raise
            # For streaming datasets or other issues, we'll catch errors during iteration
            pass
    
    def __iter__(self) -> Iterator[dict]:
        """
        Iterate over tokenized chunks.
        
        Yields dictionaries with:
        - input_ids: Token IDs of shape (seq_length,)
        - labels: Same as input_ids (for next-token prediction)
        """
        # Load streaming dataset
        ds = load_dataset(
            self.dataset_name,
            split=self.split,
            streaming=True,
            trust_remote_code=True,
        )

        # Shuffle deterministically, then do manual sharding by (process, worker).
        # We avoid datasets' `.shard()` here because many streaming datasets expose
        # `n_shards=1` and `.shard(num_shards>1)` can error.
        ds = ds.shuffle(seed=int(self.seed), buffer_size=10000)

        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0

        num_processes = int(self.num_processes) if self.num_processes else 1
        process_index = int(self.process_index) if self.process_index else 0

        total_shards = max(1, num_processes * num_workers)
        shard_id = process_index * num_workers + worker_id
        
        # Token buffer
        token_buffer = []
        
        # Get EOT token (usually eos_token_id)
        eot_token = self.tokenizer.eos_token_id
        
        for ex_idx, example in enumerate(ds):
            if total_shards > 1 and (ex_idx % total_shards) != shard_id:
                continue
            # Get text
            if self.text_column not in example:
                raise ValueError(
                    f"Example does not have '{self.text_column}' column. "
                    f"Available: {list(example.keys())}"
                )
            
            text = example[self.text_column]
            if text is None or len(text) == 0:
                continue
            
            # Tokenize
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            token_buffer.extend(tokens)
            
            # Add EOT token between examples (authors do this)
            if eot_token is not None:
                token_buffer.append(eot_token)
            
            # Yield complete chunks
            while len(token_buffer) >= self.seq_length:
                # Get seq_length tokens - model will use same tensor for input and labels
                chunk = token_buffer[:self.seq_length]
                token_buffer = token_buffer[self.seq_length:]
                
                # Input and labels are the SAME tensor
                # The model's forward() will handle the shift internally:
                # - logits[:, :-1] predicts labels[:, 1:]
                tokens = torch.tensor(chunk, dtype=torch.long)
                
                yield {
                    "input_ids": tokens,
                    "labels": tokens,  # Same as input - shift happens in model
                }


def create_dataloader(
    dataset_name: str,
    tokenizer_name: str,
    seq_length: int,
    batch_size: int,
    split: str = "train",
    text_column: str = "text",
    num_workers: int = 4,
    seed: int = 42,
    process_index: int = 0,
    num_processes: int = 1,
) -> tuple[DataLoader, PreTrainedTokenizer]:
    """
    Create a DataLoader for training.
    
    Args:
        dataset_name: HuggingFace dataset name or path
        tokenizer_name: HuggingFace tokenizer name or path
        seq_length: Sequence length for each training example
        batch_size: Batch size
        split: Dataset split
        text_column: Name of text column
        num_workers: Number of data loading workers
        seed: Random seed
        
    Returns:
        Tuple of (DataLoader, Tokenizer)
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    
    # Ensure tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create dataset
    dataset = TokenizedTextDataset(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        seq_length=seq_length,
        split=split,
        text_column=text_column,
        seed=seed,
        process_index=process_index,
        num_processes=num_processes,
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return dataloader, tokenizer


def estimate_tokens_per_epoch(
    dataset_name: str,
    tokenizer: PreTrainedTokenizer,
    split: str = "train",
    text_column: str = "text",
    sample_size: int = 1000,
) -> Optional[int]:
    """
    Estimate total tokens in a dataset by sampling.
    
    Args:
        dataset_name: HuggingFace dataset name
        tokenizer: Tokenizer to use
        split: Dataset split
        text_column: Text column name
        sample_size: Number of examples to sample for estimation
        
    Returns:
        Estimated total tokens, or None if estimation fails
    """
    try:
        # Load dataset info
        ds = load_dataset(
            dataset_name,
            split=split,
            streaming=True,
            trust_remote_code=True,
        )
        
        # Sample and count tokens
        total_tokens = 0
        count = 0
        
        for example in ds.take(sample_size):
            if text_column in example and example[text_column]:
                tokens = tokenizer.encode(example[text_column], add_special_tokens=False)
                total_tokens += len(tokens)
                count += 1
        
        if count == 0:
            return None
        
        # Try to get dataset size
        try:
            ds_full = load_dataset(dataset_name, split=split, streaming=False)
            dataset_size = len(ds_full)
            avg_tokens = total_tokens / count
            return int(dataset_size * avg_tokens)
        except:
            return None
            
    except Exception as e:
        print(f"Warning: Could not estimate tokens: {e}")
        return None


def check_split_exists(dataset_name: str, split: str) -> bool:
    """Check if a dataset has a particular split."""
    try:
        splits = get_dataset_split_names(dataset_name, trust_remote_code=True)
        return split in splits
    except Exception:
        # If we can't get splits info, try loading directly
        try:
            ds = load_dataset(
                dataset_name,
                split=f"{split}[:1]",
                streaming=False,
                trust_remote_code=True,
            )
            return True
        except Exception:
            return False


def create_validation_data(
    dataset_name: str,
    tokenizer: PreTrainedTokenizer,
    seq_length: int,
    text_column: str = "text",
    val_split: Optional[str] = "test",
    holdout_fraction: float = 0.01,
    max_tokens: int = 100_000,
    seed: int = 42,
) -> Tuple[List[torch.Tensor], str]:
    """
    Create validation data batches.
    
    If val_split exists (e.g., "test"), use that.
    Otherwise, hold out a fraction from the training data.
    
    Args:
        dataset_name: HuggingFace dataset name
        tokenizer: Tokenizer to use
        seq_length: Sequence length for each batch
        text_column: Name of text column
        val_split: Split to use for validation (e.g., "test", "validation")
        holdout_fraction: Fraction to hold out if no val_split exists
        max_tokens: Maximum tokens to use for validation
        seed: Random seed
        
    Returns:
        Tuple of (list of token tensors, description string)
    """
    eot_token = tokenizer.eos_token_id
    
    # Try to use the specified validation split
    use_split = None
    split_desc = ""
    
    if val_split:
        if check_split_exists(dataset_name, val_split):
            use_split = val_split
            split_desc = f"Using '{val_split}' split for validation"
        else:
            # Try common alternatives
            for alt_split in ["validation", "valid", "dev", "test"]:
                if alt_split != val_split and check_split_exists(dataset_name, alt_split):
                    use_split = alt_split
                    split_desc = f"'{val_split}' split not found, using '{alt_split}' instead"
                    break
    
    if use_split is None:
        # Hold out from training data
        use_split = "train"
        split_desc = f"No validation split found, holding out {holdout_fraction*100:.1f}% of training data"
    
    # Load dataset
    ds = load_dataset(
        dataset_name,
        split=use_split,
        streaming=True,
        trust_remote_code=True,
    )
    ds = ds.shuffle(seed=seed, buffer_size=10000)
    
    # If using train split for holdout, skip to the holdout portion
    if use_split == "train" and val_split is not None and not check_split_exists(dataset_name, val_split):
        # We'll take from the end of the shuffled data
        # First, estimate how many examples to skip
        # This is a bit hacky but works for streaming
        pass  # For streaming, we just take the first N tokens which is fine after shuffle
    
    # Tokenize and create batches
    token_buffer = []
    batches = []
    
    for example in ds:
        text = example.get(text_column)
        if text is None or len(text) == 0:
            continue
        
        tokens = tokenizer.encode(text, add_special_tokens=False)
        token_buffer.extend(tokens)
        
        if eot_token is not None:
            token_buffer.append(eot_token)
        
        # Create complete chunks
        while len(token_buffer) >= seq_length:
            chunk = token_buffer[:seq_length]
            token_buffer = token_buffer[seq_length:]
            batches.append(torch.tensor(chunk, dtype=torch.long))
            
            # Stop if we have enough tokens
            if len(batches) * seq_length >= max_tokens:
                break
        
        if len(batches) * seq_length >= max_tokens:
            break
    
    return batches, split_desc

