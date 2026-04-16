"""
Quantization configurations for Llama-2 13B PTQ sweep.

Each config defines the loading method, precision, and HuggingFace model ID.
Update model IDs here if using different quantized checkpoints.
"""

MODEL_REGISTRY = {
    "llama2-13b-fp16": {
        "hf_id": "meta-llama/Llama-2-13b-hf",
        "quant_type": "fp16",
        "bits": 16,
        "description": "FP16 Baseline",
    },
    "llama2-13b-gptq-int4": {
        "hf_id": "TheBloke/Llama-2-13B-GPTQ",
        "quant_type": "gptq",
        "bits": 4,
        "gptq_revision": "gptq-4bit-128g-actorder_True",  # ← add revision
        "description": "GPTQ INT4",
        },
    "llama2-13b-gptq-int8": {
        "hf_id": "TheBloke/Llama-2-13B-GPTQ",
        "quant_type": "gptq",
        "bits": 8,
        "gptq_revision": "gptq-8bit-128g-actorder_True",
        "description": "GPTQ INT8",
    },
    "llama2-13b-nf4": {
        "hf_id": "meta-llama/Llama-2-13b-hf",
        "quant_type": "nf4",
        "bits": 4,
        "description": "NF4 (bitsandbytes 4-bit)",
    },
    "llama2-13b-awq-int4": {
        "hf_id": "TheBloke/Llama-2-13B-AWQ",
        "quant_type": "awq",
        "bits": 4,
        "description": "AWQ INT4",
    },

}