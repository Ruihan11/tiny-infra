"""
HuggingFace Llama3 wrapper for inference
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional


class Llama3HF:
    """HuggingFace wrapper for Llama-3-8B"""

    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16
    ):
        """
        Initialize model using HuggingFace transformers

        Args:
            model_name: HuggingFace model name
            device: 'cuda' or 'cpu'
            dtype: torch.float16 or torch.float32
        """
        self.model_name = model_name
        self.device = device
        self.dtype = dtype

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0
    ) -> str:
        """
        Generate text from prompt

        Args:
            prompt: Input text
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text (full output including prompt)
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        outputs = self.model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.pad_token_id
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def get_memory_usage(self) -> float:
        """
        Get current GPU memory usage in GB

        Returns:
            Memory usage in GB
        """
        if self.device == "cuda":
            return torch.cuda.memory_allocated() / (1024 ** 3)
        return 0.0

    def get_model_size(self) -> float:
        """
        Get model parameter size in GB

        Returns:
            Model size in GB
        """
        param_size = sum(
            p.numel() * p.element_size()
            for p in self.model.parameters()
        )
        return param_size / (1024 ** 3)
