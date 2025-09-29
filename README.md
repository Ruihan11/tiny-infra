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

## roadmap
Llama-3-8B End-to-End Inference Optimization
> Understand bottlenecks → Targeted optimization → Validate results  

Phase 1: Performance Profiling (Week 1-2)  
- [x] Establish performance baseline (latency, throughput, memory, accuracy)
- [ ] PyTorch Profiler for operator-level analysis
- [ ] Nsight Systems for GPU activity analysis

Deliverables: Bottleneck diagnosis report

Phase 2: Model Quantization (Week 3-5)

- [ ] Learn quantization principles (INT8/INT4)
- [ ] AutoGPTQ quantization practice
- [ ] Compare AWQ/bitsandbytes approaches
- [ ] Diagnose and optimize accuracy loss
- [ ] Mixed precision strategy (sensitive layers in FP16)
- [ ] Quantization scheme performance comparison

Deliverables: Optimized quantized model + Quantization engineering report

Phase 3: TensorRT Deployment (Week 6-7)

- [ ] Convert PyTorch → ONNX → TensorRT
- [ ] Handle dynamic shapes and KV Cache
- [ ] Tune FP16/INT8 precision
- [ ] TensorRT Profiler analysis
- [ ] Identify TensorRT limitations

Deliverables: TensorRT engine + Performance improvement report

Phase 4: Kernel Optimization (Week 8-9)

- [ ] Learn Triton programming (Vector Add → GEMM → Softmax)
- [ ] Flash Attention integration and optimization
- [ ] Custom operator fusion (SwiGLU/RMSNorm)
- [ ] Roofline model analysis
- [ ] Performance tuning (BLOCK_SIZE, num_warps)

Deliverables: Optimized kernel library + Performance analysis report

Phase 5: Production Service (Week 10-11)

- [ ] FastAPI service (with streaming support)
- [ ] Dynamic batching implementation
- [ ] Prometheus + Grafana monitoring
- [ ] Load testing (locust)
- [ ] Stability guarantees (rate limiting, circuit breaking, graceful degradation)

Deliverables: Deployable inference service + Monitoring system

Phase 6: Documentation & Open Source (Week 12)

- [ ] Generate final performance comparison table
- [ ] Complete technical documentation (README + detailed guides)
- [ ] technical blog posts
