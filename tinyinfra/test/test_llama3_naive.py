"""
Tests for Llama3Naive (naive PyTorch wrapper)
Run: pytest tinyinfra/test/test_llama3_naive.py -v
"""
import pytest
import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tinyinfra.model.llama3_naive import Llama3Naive


class TestLlama3Naive:
    """Test Llama3Naive implementation"""

    @pytest.fixture(scope="class")
    def model(self):
        """
        Load model once for all tests
        """
        model_name = "meta-llama/Meta-Llama-3-8B"

        print(f"\nLoading Llama3Naive model: {model_name}")
        model = Llama3Naive(
            model_name=model_name,
            device="cuda" if torch.cuda.is_available() else "cpu",
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        print("Llama3Naive model loaded successfully")

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

    def test_greedy_decoding(self, model):
        """Test greedy decoding (deterministic generation)"""
        prompt = "2 + 2 ="

        output1 = model.generate(prompt, max_new_tokens=5, do_sample=False)
        output2 = model.generate(prompt, max_new_tokens=5, do_sample=False)

        print(f"\nPrompt: {prompt}")
        print(f"Output 1: {output1}")
        print(f"Output 2: {output2}")

        # Greedy decoding should be deterministic
        assert output1 == output2

    def test_sampling(self, model):
        """Test sampling generation"""
        prompt = "Once upon a time"
        output = model.generate(prompt, max_new_tokens=20, temperature=0.8, do_sample=True)

        print(f"\nPrompt: {prompt}")
        print(f"Output: {output}")

        assert prompt in output
        assert len(output) > len(prompt)

    def test_forward_pass(self, model):
        """Test forward pass"""
        prompt = "Hello world"
        inputs = model.tokenizer(prompt, return_tensors="pt")
        tokens = inputs.input_ids.to(model.device)

        logits = model.forward(tokens)

        print(f"\nInput shape: {tokens.shape}")
        print(f"Logits shape: {logits.shape}")

        assert logits.shape[0] == tokens.shape[0]  # batch size
        assert logits.shape[1] == tokens.shape[1]  # sequence length
        assert logits.shape[2] == model.vocab_size  # vocabulary size

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


# Quick standalone test
if __name__ == "__main__":
    print("Running quick test for Llama3Naive...")

    model = Llama3Naive(
        model_name="meta-llama/Meta-Llama-3-8B",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"✅ Model loaded")
    print(f"   Size: {model.get_model_size():.2f} GB")
    print(f"   Memory: {model.get_memory_usage():.2f} GB")

    prompt = "explain ai"
    output = model.generate(prompt, max_new_tokens=50)

    print(f"\n✅ Generation test")
    print(f"   Prompt: {prompt}")
    print(f"   Output: {output}")

    print("\n✅ All tests passed!")
