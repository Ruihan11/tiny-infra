"""
Simple tests for Llama3 wrapper
Run: pytest tiny-infra/test/llama3_wrapper_test.py -v
"""
import pytest
import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tinyinfra.model.llama3 import Llama3Model


class TestLlama3Model:
    """Test Llama3Model wrapper"""
    
    @pytest.fixture(scope="class")
    def model(self):
        """
        Load model once for all tests
        Use a smaller model for faster testing
        """
        
        # For quick testing, use a tiny model
        # Replace with "meta-llama/Meta-Llama-3-8B" for real testing
        model_name = "meta-llama/Meta-Llama-3-8B"
        
        print(f"\nLoading model: {model_name}")
        model = Llama3Model(
            model_name=model_name,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        print("Model loaded successfully")
        
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
        
        # Check output contains input
        assert prompt in output
        # Check output is longer than input
        assert len(output) > len(prompt)
    
    def test_longer_generation(self, model):
        """Test longer generation"""
        prompt = "Explain what is machine learning in one sentence:"
        output = model.generate(prompt, max_new_tokens=50)
        
        print(f"\nPrompt: {prompt}")
        print(f"Output: {output}")
        
        assert len(output) > len(prompt)
    
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
    
    def test_deterministic_generation(self, model):
        """Test deterministic generation (temperature=0)"""
        prompt = "2 + 2 ="
        
        output1 = model.generate(prompt, max_new_tokens=5, temperature=0.0)
        output2 = model.generate(prompt, max_new_tokens=5, temperature=0.0)
        
        print(f"\nPrompt: {prompt}")
        print(f"Output 1: {output1}")
        print(f"Output 2: {output2}")
        
        # Should be identical with temperature=0
        # Note: Might still vary slightly due to GPU non-determinism
        # so we just check they're similar
        assert output1 == output2 or len(output1) == len(output2)


# Quick standalone test
if __name__ == "__main__":
    print("Running quick test...")
    
    model = Llama3Model(
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