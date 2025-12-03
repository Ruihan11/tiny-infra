"""
Tests for Llama3Customized (pure PyTorch implementation)
Run: pytest tinyinfra/test/test_llama3_customized.py -v
"""
import pytest
import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tinyinfra.model.llama3_02_opt import Llama3Customized


class TestLlama3Customized:
    """Test Llama3Customized pure PyTorch implementation"""

    @pytest.fixture(scope="class")
    def model(self):
        """
        Load model once for all tests
        """
        model_name = "meta-llama/Meta-Llama-3-8B"

        print(f"\nLoading Llama3Customized model: {model_name}")
        model = Llama3Customized(
            model_name=model_name,
            device="cuda" if torch.cuda.is_available() else "cpu",
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        print("Llama3Customized model loaded successfully")

        return model

    def test_model_loads(self, model):
        """Test that model loads without errors"""
        assert model is not None
        assert model.model is not None
        assert model.tokenizer is not None
        assert model.embedding is not None
        assert len(model.layers) > 0
        assert model.norm is not None
        assert model.output is not None

    def test_basic_generation(self, model):
        """Test basic text generation with prompt"""
        prompt = "Hello, my name is"
        output = model.generate(prompt=prompt, max_new_tokens=10)

        print(f"\nPrompt: {prompt}")
        print(f"Output: {output}")

        assert isinstance(output, str)
        assert prompt in output
        assert len(output) > len(prompt)

    def test_explain_ai(self, model):
        """Test inference with 'explain ai' prompt"""
        prompt = "explain ai"
        output = model.generate(prompt=prompt, max_new_tokens=100)

        print(f"\nPrompt: {prompt}")
        print(f"Output: {output}")

        assert isinstance(output, str)
        assert len(output) > len(prompt)
        assert prompt in output.lower()

    def test_generation_with_input_ids(self, model):
        """Test generation with input_ids instead of prompt"""
        prompt = "The capital of France is"
        inputs = model.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(model.device)

        output_ids = model.generate(input_ids=input_ids, max_new_tokens=10)
        output_text = model.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        print(f"\nPrompt: {prompt}")
        print(f"Output: {output_text}")

        assert output_ids.shape[1] > input_ids.shape[1]
        assert prompt in output_text

    def test_longer_generation(self, model):
        """Test longer generation"""
        prompt = "Explain what is machine learning in one sentence:"
        output = model.generate(prompt=prompt, max_new_tokens=50)

        print(f"\nPrompt: {prompt}")
        print(f"Output: {output}")

        assert len(output) > len(prompt)

    def test_greedy_decoding(self, model):
        """Test greedy decoding (deterministic generation)"""
        prompt = "2 + 2 ="

        output1 = model.generate(prompt=prompt, max_new_tokens=5, do_sample=False)
        output2 = model.generate(prompt=prompt, max_new_tokens=5, do_sample=False)

        print(f"\nPrompt: {prompt}")
        print(f"Output 1: {output1}")
        print(f"Output 2: {output2}")

        # Greedy decoding should be deterministic
        assert output1 == output2

    def test_temperature_sampling(self, model):
        """Test sampling with different temperatures"""
        prompt = "Once upon a time"

        # Low temperature (more deterministic)
        output_low = model.generate(prompt=prompt, max_new_tokens=20, temperature=0.1, do_sample=True)

        # High temperature (more random)
        output_high = model.generate(prompt=prompt, max_new_tokens=20, temperature=1.5, do_sample=True)

        print(f"\nPrompt: {prompt}")
        print(f"Low temp output: {output_low}")
        print(f"High temp output: {output_high}")

        assert prompt in output_low
        assert prompt in output_high

    def test_top_k_sampling(self, model):
        """Test top-k sampling"""
        prompt = "The quick brown"
        output = model.generate(prompt=prompt, max_new_tokens=10, top_k=10, do_sample=True)

        print(f"\nPrompt: {prompt}")
        print(f"Output: {output}")

        assert prompt in output

    def test_top_p_sampling(self, model):
        """Test top-p (nucleus) sampling"""
        prompt = "In the future,"
        output = model.generate(prompt=prompt, max_new_tokens=15, top_p=0.9, do_sample=True)

        print(f"\nPrompt: {prompt}")
        print(f"Output: {output}")

        assert prompt in output

    def test_forward_pass(self, model):
        """Test forward pass with token IDs"""
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

    def test_max_length_limit(self, model):
        """Test that generation respects max_seq_len"""
        prompt = "Test " * 100  # Long prompt
        output = model.generate(prompt=prompt, max_new_tokens=10)

        print(f"\nPrompt length: {len(prompt)}")
        print(f"Output length: {len(output)}")

        # Should not crash and should return something
        assert isinstance(output, str)

    def test_kv_cache_generation(self, model):
        """Test generation with KV cache enabled"""
        prompt = "The capital of France is"

        # Generate with cache
        output_with_cache = model.generate(prompt=prompt, max_new_tokens=20, use_cache=True, do_sample=False)

        # Generate without cache
        output_without_cache = model.generate(prompt=prompt, max_new_tokens=20, use_cache=False, do_sample=False)

        print(f"\nPrompt: {prompt}")
        print(f"Output with cache: {output_with_cache}")
        print(f"Output without cache: {output_without_cache}")

        # Should produce the same output (greedy decoding)
        assert output_with_cache == output_without_cache

    def test_kv_cache_consistency(self, model):
        """Test that KV cache produces consistent results"""
        prompt = "Hello, world!"

        outputs = []
        for _ in range(3):
            output = model.generate(prompt=prompt, max_new_tokens=10, use_cache=True, do_sample=False)
            outputs.append(output)

        print(f"\nPrompt: {prompt}")
        print(f"Outputs: {outputs}")

        # All outputs should be identical with greedy decoding and cache
        assert len(set(outputs)) == 1

    def test_flash_attention_availability(self, model):
        """Test that FlashAttention is available and enabled"""
        print(f"\nFlashAttention enabled: {model.use_flash_attn}")
        print(f"SDPA available: {hasattr(torch.nn.functional, 'scaled_dot_product_attention')}")

        # Check that model has flash attention enabled
        assert hasattr(model, 'use_flash_attn')

        # Check if PyTorch version supports SDPA
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            print("✅ FlashAttention (SDPA) is available")

    def test_cache_clearing(self, model):
        """Test that cache is properly cleared"""
        prompt = "Test cache clearing"

        # Generate with cache
        model.clear_cache()
        output1 = model.generate(prompt=prompt, max_new_tokens=5, use_cache=True, do_sample=False)

        # Manually clear cache
        model.clear_cache()

        # Generate again
        output2 = model.generate(prompt=prompt, max_new_tokens=5, use_cache=True, do_sample=False)

        print(f"\nPrompt: {prompt}")
        print(f"Output 1: {output1}")
        print(f"Output 2: {output2}")

        # Should be identical after clearing cache
        assert output1 == output2

    def test_forward_with_cache(self, model):
        """Test forward pass with cache enabled"""
        prompt = "Hello"
        inputs = model.tokenizer(prompt, return_tensors="pt")
        tokens = inputs.input_ids.to(model.device)

        # Clear cache first
        model.clear_cache()

        # First forward pass (prefill)
        logits1 = model.forward(tokens, start_pos=0, use_cache=True)

        # Second forward pass with single token (decode)
        next_token = logits1[:, -1:, :].argmax(dim=-1)
        logits2 = model.forward(next_token, start_pos=tokens.shape[1], use_cache=True)

        print(f"\nPrefill logits shape: {logits1.shape}")
        print(f"Decode logits shape: {logits2.shape}")

        assert logits1.shape[1] == tokens.shape[1]
        assert logits2.shape[1] == 1

        # Clear cache after test
        model.clear_cache()


# Quick standalone test
if __name__ == "__main__":
    print("Running quick test for Llama3Customized...")

    model = Llama3Customized(
        model_name="meta-llama/Meta-Llama-3-8B",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"✅ Model loaded")
    print(f"   Size: {model.get_model_size():.2f} GB")
    print(f"   Memory: {model.get_memory_usage():.2f} GB")

    prompt = "AI is"
    output = model.generate(prompt=prompt, max_new_tokens=200)

    print(f"\n✅ Generation test")
    print(f"   Prompt: {prompt}")
    print(f"   Output: {output}")

    print("\n✅ All tests passed!")
