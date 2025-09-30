"""
Model quantization tools - AWQ and BitsAndBytes
"""
import torch
from pathlib import Path
from typing import List
import json


class QuantizationError(Exception):
    """Quantization error"""
    pass


class BaseQuantizer:
    """Base quantizer"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    def _get_output_path(self, output_dir: str, method: str, bits: int) -> Path:
        model_name = Path(self.model_name).name
        return Path(output_dir) / f"{model_name}-{method}-int{bits}"
    
    def _get_dir_size(self, path: Path) -> float:
        total = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
        return total / (1024 ** 3)
    
    def _save_metadata(self, output_path: Path, method: str, bits: int, extra: dict):
        metadata = {
            "original_model": self.model_name,
            "method": method,
            "bits": bits,
            "pytorch_version": torch.__version__,
            **extra
        }
        with open(output_path / "quantization_info.json", 'w') as f:
            json.dump(metadata, f, indent=2)


class AWQQuantizer(BaseQuantizer):
    """AWQ quantization (Activation-aware Weight Quantization)"""
    
    def quantize(
        self,
        bits: int = 4,
        output_dir: str = "models/quantized",
        group_size: int = 128,
        **kwargs
    ) -> str:
        try:
            from awq import AutoAWQForCausalLM
            from transformers import AutoTokenizer
        except ImportError:
            raise QuantizationError(
                "\n❌ autoawq not installed!\n"
                "   Install: uv pip install autoawq"
            )
        
        print(f"\n🔧 AWQ Quantization")
        print(f"   Model: {self.model_name}")
        print(f"   Bits: {bits}")
        print(f"   Group size: {group_size}")
        
        # Load
        print(f"\n⏳ Loading model...")
        model = AutoAWQForCausalLM.from_pretrained(self.model_name)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Quantize
        print(f"\n⚙️  Quantizing...")
        quant_config = {
            "zero_point": True,
            "q_group_size": group_size,
            "w_bit": bits,
            "version": "GEMM"
        }
        
        model.quantize(tokenizer, quant_config=quant_config)
        
        # Save
        output_path = self._get_output_path(output_dir, "awq", bits)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n💾 Saving...")
        model.save_quantized(str(output_path))
        tokenizer.save_pretrained(str(output_path))
        
        self._save_metadata(output_path, "awq", bits, {"group_size": group_size})
        
        size = self._get_dir_size(output_path)
        print(f"\n✅ Complete!")
        print(f"   Output: {output_path}")
        print(f"   Size: {size:.2f} GB")
        
        return str(output_path)


class BitsAndBytesQuantizer(BaseQuantizer):
    """BitsAndBytes quantization (NF4/INT8)"""
    
    def quantize(
        self,
        bits: int = 8,
        output_dir: str = "models/quantized",
        **kwargs
    ) -> str:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError:
            raise QuantizationError(
                "\n❌ bitsandbytes not installed!\n"
                "   Install: uv pip install bitsandbytes"
            )
        
        print(f"\n🔧 BitsAndBytes Quantization")
        print(f"   Model: {self.model_name}")
        print(f"   Bits: {bits}")
        
        # Config
        if bits == 4:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif bits == 8:
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            raise ValueError(f"bits must be 4 or 8, got {bits}")
        
        # Load and quantize
        print(f"\n⏳ Loading and quantizing...")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quant_config,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Save
        output_path = self._get_output_path(output_dir, "bnb", bits)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n💾 Saving...")
        model.save_pretrained(str(output_path))
        tokenizer.save_pretrained(str(output_path))
        
        self._save_metadata(output_path, "bitsandbytes", bits, {
            "quant_type": "nf4" if bits == 4 else "int8",
        })
        
        size = self._get_dir_size(output_path)
        print(f"\n✅ Complete!")
        print(f"   Output: {output_path}")
        print(f"   Size: {size:.2f} GB")
        
        return str(output_path)