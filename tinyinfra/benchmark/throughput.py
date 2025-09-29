"""
Throughput benchmark: measure tokens/second
"""
import torch
import time
from typing import Dict, List

class ThroughputBenchmark:
    """Measure generation throughput (tokens/second)"""
    
    def __init__(self, model):
        """
        Args:
            model: Llama3Model instance
        """
        self.model = model
    
    def run(
        self,
        prompt: str = "Explain artificial intelligence",
        batch_size: int = 1,
        num_tokens: int = 128,
        num_runs: int = 20,
        warmup_runs: int = 3
    ) -> Dict:
        """
        Run throughput benchmark
        
        Args:
            prompt: Input text
            batch_size: Number of prompts to process simultaneously
            num_tokens: Number of tokens to generate per prompt
            num_runs: Number of benchmark iterations
            warmup_runs: Number of warmup iterations (not counted)
            
        Returns:
            dict with throughput metrics
        """
        print(f"\n🔥 Throughput Benchmark")
        print(f"   Batch size: {batch_size}")
        print(f"   Tokens per run: {num_tokens}")
        print(f"   Runs: {num_runs} (+ {warmup_runs} warmup)")
        
        # Prepare prompts
        if batch_size == 1:
            prompts = [prompt]
        else:
            prompts = [prompt] * batch_size
        
        # Warmup
        print(f"\n⏳ Warming up ({warmup_runs} runs)...")
        for i in range(warmup_runs):
            _ = self._generate_batch(prompts, num_tokens)
            print(f"   Warmup {i+1}/{warmup_runs}")
        
        # Benchmark
        print(f"\n⚡ Running benchmark ({num_runs} runs)...")
        times = []
        total_tokens = 0
        
        for i in range(num_runs):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.perf_counter()
            
            outputs = self._generate_batch(prompts, num_tokens)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            elapsed = time.perf_counter() - start
            
            times.append(elapsed)
            tokens_generated = num_tokens * batch_size
            total_tokens += tokens_generated
            
            throughput = tokens_generated / elapsed
            print(f"   Run {i+1}/{num_runs}: {throughput:.2f} tok/s")
        
        # Calculate metrics
        total_time = sum(times)
        mean_time = total_time / num_runs
        tokens_per_run = num_tokens * batch_size
        
        results = {
            "throughput_tokens_per_sec": total_tokens / total_time,
            "mean_latency_sec": mean_time,
            "total_tokens": total_tokens,
            "total_time_sec": total_time,
            "tokens_per_run": tokens_per_run,
            "runs": num_runs,
            "batch_size": batch_size,
            "all_times": times
        }
        
        return results
    
    def _generate_batch(self, prompts: List[str], num_tokens: int) -> List[str]:
        """Generate for a batch of prompts"""
        if len(prompts) == 1:
            # Single prompt
            output = self.model.generate(
                prompts[0],
                max_new_tokens=num_tokens,
                temperature=1.0
            )
            return [output]
        else:
            # Batch generation
            inputs = self.model.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=num_tokens,
                    pad_token_id=self.model.tokenizer.pad_token_id
                )
            
            return self.model.tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    def print_results(self, results: Dict):
        """Pretty print results"""
        print(f"\n{'='*50}")
        print(f"📊 THROUGHPUT RESULTS")
        print(f"{'='*50}")
        print(f"Throughput:      {results['throughput_tokens_per_sec']:.2f} tokens/sec")
        print(f"Mean Latency:    {results['mean_latency_sec']*1000:.2f} ms")
        print(f"Total Tokens:    {results['total_tokens']:,}")
        print(f"Total Time:      {results['total_time_sec']:.2f} sec")
        print(f"Tokens/Run:      {results['tokens_per_run']}")
        print(f"Batch Size:      {results['batch_size']}")
        print(f"{'='*50}\n")