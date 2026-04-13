"""
Full evaluation — Llama-2 7B GPTQ INT8.
Requires: gptqmodel installed (see below).

Fix GPTQ dependency before running:
    pip uninstall auto-gptq -y && pip install gptqmodel

Usage:
    HF_TOKEN=<token> python TeamA/eval_gptq_int8.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from TeamA.configs import MODEL_REGISTRY
from shared.model_loader import load_model, free_model
from shared.eval_utils import run_eval

CONFIG_KEY  = "llama2-7b-gptq-int8"
OUTPUT_DIR  = "TeamA/results/gptq_int8"
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
        precision="int8",
        quant_method="gptq",
        seed=42,
    )
finally:
    free_model(model)
