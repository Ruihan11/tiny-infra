"""
Naive Llama3 implementation - clean PyTorch code without optimizations
Simple wrapper around HuggingFace model for educational purposes
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class Llama3Naive:
    """
    Naive Llama3 wrapper - no optimizations, just clean PyTorch code

    This is a simple wrapper that uses HuggingFace model directly
    without any performance optimizations like KV cache, FlashAttention, etc.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16
    ):
        """
        Initialize naive Llama3 model

        Args:
            model_name: HuggingFace model name or path
            device: Device to load model on ('cuda' or 'cpu')
            dtype: Data type for model weights
        """
        self.model_name = model_name
        self.device = device
        self.dtype = dtype if device == "cuda" else torch.float32

        print(f"Loading {model_name} (naive implementation)...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model - simple, no optimizations
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            device_map=device
        )
        self.model.eval()

        self.vocab_size = self.model.config.vocab_size

        print(f"Model loaded on {device}")

    def generate(
        self,
        prompt: str = None,
        input_ids: torch.Tensor = None,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        do_sample: bool = True,
        top_k: int = 50,
        top_p: float = 0.9
    ) -> str:
        """
        Generate text from prompt

        Args:
            prompt: Input text prompt
            input_ids: Alternative to prompt - pre-tokenized input
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling or greedy decoding
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter

        Returns:
            Generated text string
        """
        # Prepare input
        if input_ids is None:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            input_ids = inputs.input_ids.to(self.device)

        # Simple generation loop - no KV cache optimization
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward pass through entire sequence each time (inefficient but simple)
                outputs = self.model(input_ids)
                logits = outputs.logits

                # Get next token logits
                next_token_logits = logits[:, -1, :]

                # Apply temperature
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature

                # Sample or pick greedily
                if do_sample:
                    # Apply top-k filtering
                    if top_k > 0:
                        indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                        next_token_logits[indices_to_remove] = float('-inf')

                    # Apply top-p filtering
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                        # Remove tokens with cumulative probability above the threshold
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0

                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        next_token_logits[indices_to_remove] = float('-inf')

                    # Sample from the filtered distribution
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy decoding
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=-1)

                # Check for EOS token
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

        # Decode and return
        output_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return output_text

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Simple forward pass

        Args:
            input_ids: Input token IDs [batch_size, seq_len]

        Returns:
            Logits [batch_size, seq_len, vocab_size]
        """
        with torch.no_grad():
            outputs = self.model(input_ids)
            return outputs.logits

    def get_memory_usage(self) -> float:
        """Get GPU memory usage in GB"""
        if self.device == "cuda" and torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1e9
        return 0.0

    def get_model_size(self) -> float:
        """Get model size in GB"""
        param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        return param_size / 1e9
