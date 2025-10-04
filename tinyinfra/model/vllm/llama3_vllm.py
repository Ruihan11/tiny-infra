"""
vLLM Llama3 wrapper for high-performance inference
"""
import torch
from typing import Optional


class Llama3VLLM:
    """vLLM wrapper for Llama-3-8B"""

    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B",
        device: str = "cuda",
        dtype: str = "auto",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9
    ):
        """
        Initialize model using vLLM

        Args:
            model_name: HuggingFace model name or path
            device: 'cuda' or 'cpu' (vLLM primarily uses CUDA)
            dtype: 'auto', 'float16', 'bfloat16', or 'float32'
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: Fraction of GPU memory to use (0.0-1.0)
        """
        try:
            from vllm import LLM
        except ImportError:
            raise ImportError(
                "vLLM not installed. Install with: pip install vllm"
            )

        self.model_name = model_name
        self.device = device
        self.dtype = dtype

        # Initialize vLLM engine
        self.model = LLM(
            model=model_name,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True
        )

        # Get tokenizer from vLLM
        self.tokenizer = self.model.get_tokenizer()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

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
        # Get default sampling params and update
        sampling_params = self.model.get_default_sampling_params()
        sampling_params.max_tokens = max_new_tokens
        sampling_params.temperature = temperature

        outputs = self.model.generate([prompt], sampling_params)

        # vLLM returns RequestOutput objects
        # Get the full text (prompt + generated)
        generated_text = outputs[0].outputs[0].text

        # Return prompt + generated (to match HF behavior)
        return prompt + generated_text

    def generate_batch(
        self,
        prompts: list,
        max_new_tokens: int = 100,
        temperature: float = 1.0
    ) -> list:
        """
        Generate text from multiple prompts (batch processing)

        Args:
            prompts: List of input texts
            max_new_tokens: Number of tokens to generate per prompt
            temperature: Sampling temperature

        Returns:
            List of generated texts (full output including prompts)
        """
        # Get default sampling params and update
        sampling_params = self.model.get_default_sampling_params()
        sampling_params.max_tokens = max_new_tokens
        sampling_params.temperature = temperature

        outputs = self.model.generate(prompts, sampling_params)

        # Return full text (prompt + generated) for each
        results = []
        for output in outputs:
            generated_text = output.outputs[0].text
            results.append(output.prompt + generated_text)

        return results

    def get_memory_usage(self) -> float:
        """
        Get current GPU memory usage in GB

        Returns:
            Memory usage in GB
        """
        if self.device == "cuda" and torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 ** 3)
        return 0.0

    def get_model_size(self) -> float:
        """
        Get model parameter size in GB (estimated)

        Returns:
            Model size in GB
        """
        # vLLM doesn't expose model parameters directly
        # Estimate based on model name
        if "8B" in self.model_name or "7B" in self.model_name:
            # ~8B params * 2 bytes (fp16) = ~16GB
            return 16.0
        elif "13B" in self.model_name:
            return 26.0
        elif "70B" in self.model_name:
            return 140.0
        else:
            # Default estimate
            return 16.0
