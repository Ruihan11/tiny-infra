# tiny-infra

```bash
git clone git@github.com:Ruihan11/tiny-infra.git
cd tiny-infra
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install -r requirements.txt 
```

hf auth login with your token

```python
python model.py
```

```bash
tinyinfra quantize awq --model meta-llama/Meta-Llama-3-8B --bits 4

tinyinfra quantize bnb --model meta-llama/Meta-Llama-3-8B --bits 4

tinyinfra quantize bnb --model meta-llama/Meta-Llama-3-8B --bits 8

tinyinfra benchmark throughput --model meta-llama/Meta-Llama-3-8B --batch-size 8 --num-tokens 256 --num-runs 20 

tinyinfra benchmark throughput --model models/quantized/Meta-Llama-3-8B-awq-int4 --batch-size 8 --num-tokens 256 --num-runs 20 

tinyinfra benchmark throughput --model models/quantized/Meta-Llama-3-8B-bnb-int4 --batch-size 8 --num-tokens 256 --num-runs 20 

tinyinfra benchmark throughput --model models/quantized/Meta-Llama-3-8B-bnb-int8 --batch-size 8 --num-tokens 256 --num-runs 20 
```
> some quick test on A40

| | llama3-8B | awq-int4 | bnb-int4 | bnb-int8 |
|-|-|-|-|-|
|Memory (GB)            |15.0|5.3|5.3|8.5|
|Throughput(token/sec)  |229.91|152.68|112.59|66.09|
|Mean Latency(ms)       |8907.83|13413.59|18190.15|30988.62|

## roadmap
Llama-3-8B End-to-End Inference Optimization
> Understand bottlenecks → Targeted optimization → Validate results  

Phase 1: Performance Profiling (Week 1-2)

- [x] Establish baseline metrics (latency, throughput, memory, accuracy)
- [x] PyTorch Profiler for operator-level analysis
- [ ] Nsight Systems for GPU kernel analysis (Maybe for future reference)
- [x] Custom vs HuggingFace implementation comparison

Phase 2: Model Quantization (Week 3-5)

- [x] AWQ INT4 quantization implementation
- [x] BitsAndBytes INT4/INT8 quantization
- [x] Performance benchmarking (A40 GPU)
- [x] Accuracy evaluation metrics
- [ ] Mixed precision strategies (sensitive layers in FP16)
- [ ] Per-channel vs per-tensor quantization comparison
- [ ] GPTQ quantization integration
- [ ] Quantization-aware training experiments

Phase 3: Custom Implementation Optimization (Week 6-7)

- [x] Separate HuggingFace and custom implementations
- [x] Pure PyTorch Llama3 with RoPE and GQA
- [ ] Implement KV cache for inference
- [ ] Flash Attention integration
- [ ] Fused kernel operations (RMSNorm + Attention)
- [ ] Memory-efficient attention patterns
- [ ] Benchmark custom vs HuggingFace

Phase 4: TensorRT Deployment (Week 8-9)

- [ ] PyTorch → ONNX → TensorRT conversion pipeline
- [ ] Handle dynamic shapes and KV cache
- [ ] FP16/INT8 TensorRT precision tuning
- [ ] TensorRT layer fusion optimization
- [ ] Profile with TensorRT Profiler
- [ ] Identify TensorRT limitations

Phase 5: Kernel Optimization (Week 10-11)

- [ ] Learn Triton programming (Vector Add → GEMM → Softmax)
- [ ] Flash Attention 2/3 integration
- [ ] Custom operator fusion (SwiGLU, RMSNorm, QKV projection)
- [ ] Roofline model analysis
- [ ] Performance tuning (BLOCK_SIZE, num_warps)
- [ ] Quantized GEMM kernels
