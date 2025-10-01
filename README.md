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


