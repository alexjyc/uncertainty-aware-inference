"""
Full evaluation — Llama-2 7B NF4 (bitsandbytes 4-bit).
Requires: Meta HF gated access approved.

Usage:
    HF_TOKEN=<token> python TeamA/eval_nf4.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from TeamA.configs import MODEL_REGISTRY
from shared.model_loader import load_model, free_model
from shared.eval_utils import run_eval

CONFIG_KEY  = "llama2-7b-nf4"
OUTPUT_DIR  = "TeamA/results/nf4"
MAX_SAMPLES = 100

model, tokenizer = load_model(CONFIG_KEY, MODEL_REGISTRY)

try:
    run_eval(
        model=model,
        tokenizer=tokenizer,
        datasets_to_run=["hellaswag", "triviaqa", "pubmedqa"],
        max_samples=MAX_SAMPLES,
        output_dir=OUTPUT_DIR,
        model_tag="llama2-7b",
        precision="int4",
        quant_method="nf4",
        seed=42,
    )
finally:
    free_model(model)
