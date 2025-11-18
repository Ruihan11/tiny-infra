"""
Tests for Llama3HF (HuggingFace wrapper)
Run: pytest tinyinfra/test/test_llama3_hf.py -v
"""
import pytest
import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tinyinfra.model.llama3_hf import Llama3HF


class TestLlama3HF:
    """Test Llama3HF HuggingFace wrapper"""

    @pytest.fixture(scope="class")
    def model(self):
        """
        Load model once for all tests
        """
        model_name = "meta-llama/Meta-Llama-3-8B"

        print(f"\nLoading Llama3HF model: {model_name}")
        model = Llama3HF(
            model_name=model_name,
            device="cuda" if torch.cuda.is_available() else "cpu",
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        print("Llama3HF model loaded successfully")

        return model

    def test_model_loads(self, model):
        """Test that model loads without errors"""
        assert model is not None
        assert model.model is not None
        assert model.tokenizer is not None

    def test_basic_generation(self, model):
        """Test basic text generation"""
        prompt = "Hello, my name is"
        output = model.generate(prompt, max_new_tokens=10)

        print(f"\nPrompt: {prompt}")
        print(f"Output: {output}")

        assert isinstance(output, str)
        assert prompt in output
        assert len(output) > len(prompt)

    def test_explain_ai(self, model):
        """Test inference with 'explain ai' prompt"""
        prompt = "explain ai"
        output = model.generate(prompt, max_new_tokens=100)

        print(f"\nPrompt: {prompt}")
        print(f"Output: {output}")

        assert isinstance(output, str)
        assert len(output) > len(prompt)
        assert prompt in output.lower()

    def test_longer_generation(self, model):
        """Test longer generation"""
        prompt = "Explain what is machine learning in one sentence:"
        output = model.generate(prompt, max_new_tokens=50)

        print(f"\nPrompt: {prompt}")
        print(f"Output: {output}")

        assert len(output) > len(prompt)

    def test_deterministic_generation(self, model):
        """Test deterministic generation (temperature=0)"""
        prompt = "2 + 2 ="

        # Note: HuggingFace doesn't support temperature=0 directly
        # Setting temperature=0 will use greedy decoding (do_sample=False)
        output1 = model.generate(prompt, max_new_tokens=5, temperature=0.0)
        output2 = model.generate(prompt, max_new_tokens=5, temperature=0.0)

        print(f"\nPrompt: {prompt}")
        print(f"Output 1: {output1}")
        print(f"Output 2: {output2}")

        # Should be identical with temperature=0 (greedy decoding)
        assert output1 == output2

    def test_temperature_sampling(self, model):
        """Test sampling with different temperatures"""
        prompt = "Once upon a time"

        # Low temperature (more deterministic)
        output_low = model.generate(prompt, max_new_tokens=20, temperature=0.5)

        # Higher temperature (more random)
        output_high = model.generate(prompt, max_new_tokens=20, temperature=1.5)

        print(f"\nPrompt: {prompt}")
        print(f"Low temp output: {output_low}")
        print(f"High temp output: {output_high}")

        assert prompt in output_low
        assert prompt in output_high

    def test_various_max_tokens(self, model):
        """Test generation with various max token lengths"""
        prompt = "The sky is"

        for max_tokens in [5, 10, 20]:
            output = model.generate(prompt, max_new_tokens=max_tokens)
            print(f"\nMax tokens: {max_tokens}")
            print(f"Output: {output}")
            assert prompt in output

    def test_memory_usage(self, model):
        """Test memory reporting"""
        memory_gb = model.get_memory_usage()

        print(f"\nGPU Memory Usage: {memory_gb:.2f} GB")

        if torch.cuda.is_available():
            assert memory_gb > 0
            assert memory_gb < 100  # Sanity check

    def test_model_size(self, model):
        """Test model size reporting"""
        size_gb = model.get_model_size()

        print(f"\nModel Size: {size_gb:.2f} GB")

        assert size_gb > 0
        assert size_gb < 50  # Sanity check for 8B model

    def test_special_characters(self, model):
        """Test generation with special characters in prompt"""
        prompt = "What is 1+1? Answer:"
        output = model.generate(prompt, max_new_tokens=10)

        print(f"\nPrompt: {prompt}")
        print(f"Output: {output}")

        assert prompt in output

    def test_empty_generation(self, model):
        """Test generation with very short max_new_tokens"""
        prompt = "Hello"
        output = model.generate(prompt, max_new_tokens=1)

        print(f"\nPrompt: {prompt}")
        print(f"Output: {output}")

        assert prompt in output


# Quick standalone test
if __name__ == "__main__":
    print("Running quick test for Llama3HF...")

    model = Llama3HF(
        model_name="meta-llama/Meta-Llama-3-8B",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"✅ Model loaded")
    print(f"   Size: {model.get_model_size():.2f} GB")
    print(f"   Memory: {model.get_memory_usage():.2f} GB")

    prompt = "Hello, world!"
    output = model.generate(prompt, max_new_tokens=20)

    print(f"\n✅ Generation test")
    print(f"   Prompt: {prompt}")
    print(f"   Output: {output}")

    print("\n✅ All tests passed!")
