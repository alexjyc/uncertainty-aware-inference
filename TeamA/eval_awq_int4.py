"""
Full evaluation — Llama-2 7B AWQ INT4.
No gated access required (TheBloke public repo).

Requirements:
    conda activate uncertainty_aware_env
    python TeamA/eval_awq_int4.py

No HF_TOKEN needed.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from TeamA.configs import MODEL_REGISTRY
from shared.model_loader import load_model, free_model
from shared.eval_utils import run_eval

CONFIG_KEY  = "llama2-7b-awq-int4"
OUTPUT_DIR  = "TeamA/results/awq_int4"
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
        quant_method="awq",
        seed=42,
    )
finally:
    free_model(model)
