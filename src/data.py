"""
Data loading utilities for weight-sparse transformer training.
"""

from typing import Iterator, Optional
import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizer


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
        
        # Shuffle with seed
        ds = ds.shuffle(seed=self.seed, buffer_size=10000)
        
        # Token buffer
        token_buffer = []
        
        # Get EOT token (usually eos_token_id)
        eot_token = self.tokenizer.eos_token_id
        
        for example in ds:
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

