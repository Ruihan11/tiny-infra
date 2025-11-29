"""
Nsight Compute Profiling Test for TinyInfra
This file is designed to be used with nvidia-nsight-compute for profiling CUDA kernels
during LLM inference operations.

To run with nsight compute:
1. ncu --target-processes all --set full --export profile_report python test/test_nsight_profiling.py

Or for a shorter session:
2. ncu --target-processes all --set full --duration 30 python test/test_nsight_profiling.py

For system-wide profiling:
3. nsys profile --trace=cuda,nvtx --output=nsys_profile python test/test_nsight_profiling.py
"""

import torch
from tinyinfra.model.llama3_hf import Llama3HF
import time


def benchmark_model():
    """
    Run a simple benchmark to profile with Nsight Compute
    """
    print("Loading model for profiling...")
    model = Llama3HF(
        model_name='meta-llama/Meta-Llama-3-8B',
        device='cuda',
        dtype=torch.float16
    )
    
    print("Starting profiling benchmark...")
    
    # Warmup run
    for i in range(3):
        _ = model.generate(
            prompt="Explain artificial intelligence",
            max_new_tokens=64,
            temperature=0.7
        )
        print(f"Warmup run {i+1}/3 completed")
    
    # Actual profiling run - this is what you'll want to profile with nsight
    print("Starting generation for profiling...")
    start_time = time.time()
    
    output = model.generate(
        prompt="Explain artificial intelligence in detail",
        max_new_tokens=256,
        temperature=0.7
    )
    
    end_time = time.time()
    
    print(f"Generation completed in {end_time - start_time:.2f} seconds")
    print(f"Output length: {len(output)} characters")
    print("Sample output:", output[:200] + "..." if len(output) > 200 else output)
    
    return output


def profile_single_batch():
    """
    Profile a single batch generation to analyze CUDA kernels
    """
    print("Loading model for single batch profiling...")
    model = Llama3HF(
        model_name='meta-llama/Meta-Llama-3-8B',
        device='cuda',
        dtype=torch.float16
    )
    
    # Batch generation (as used in the throughput benchmark)
    prompts = ["Explain artificial intelligence"] * 8  # batch size of 8
    
    # Tokenize the batch
    inputs = model.tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(model.device)
    
    print("Starting batch generation for profiling...")
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=256,
            pad_token_id=model.tokenizer.pad_token_id,
            temperature=0.7
        )
    
    end_time = time.time()
    
    results = model.tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    print(f"Batch generation completed in {end_time - start_time:.2f} seconds")
    print(f"Generated {len(results)} outputs")
    
    return results


if __name__ == "__main__":
    print("Nsight Compute Profiling Test")
    print("="*50)
    
    # Run the model benchmark
    result = benchmark_model()
    
    print("\n" + "="*50)
    print("Batch profiling test:")
    batch_results = profile_single_batch()
    
    print("\nProfiling test completed successfully!")