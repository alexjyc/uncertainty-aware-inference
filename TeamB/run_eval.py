# Team B — Calibration Evaluation
# Runs one quantization config across hellaswag, triviaqa, pubmedqa.
#
# Usage:
#   python TeamB/run_eval.py --config mistral-7b-fp16
#   python TeamB/run_eval.py --config mistral-7b-awq-int4 --samples 100
#   python TeamB/run_eval.py --config mistral-7b-gptq-int4 --datasets hellaswag triviaqa
#
import os
import sys
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wandb
from TeamB.configs import MODEL_REGISTRY
from shared.model_loader import load_model, free_model
from shared.eval_utils import run_eval

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, choices=list(MODEL_REGISTRY.keys()),
                    help="config to run")
parser.add_argument("--samples", type=int, default=None,
                    help="samples per dataset (default: full dataset)")
parser.add_argument("--datasets", nargs="+", default=["hellaswag", "triviaqa", "pubmedqa"],
                    help="datasets to evaluate on")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

config_key = args.config
print(f"\n{'-'*60}")
print(f"Running: {config_key}")
print(f"Datasets: {args.datasets}  |  Samples: {args.samples}")
print(f"{'-'*60}\n")

output_dir = f"./updated_results/{config_key}"
os.makedirs(output_dir, exist_ok=True)

run = wandb.init(
    entity="Uncertainty_Aware_Inference_Lab",
    project="UAI_Project",
    name=f"team-b_{config_key}",
    config={
        "model":        "mistral-7b",
        "team":         "team-b",
        "quant_method": MODEL_REGISTRY[config_key]["quant_type"],
        "precision":    str(MODEL_REGISTRY[config_key]["bits"]) + "bit",
        "dataset":      args.datasets,
        "seed":         args.seed,
    },
)

model, tokenizer = load_model(config_key, MODEL_REGISTRY)
tokenizer.pad_token = tokenizer.eos_token
try:
    run_eval(
        model=model,
        tokenizer=tokenizer,
        datasets_to_run=args.datasets,
        max_samples=args.samples,
        output_dir=output_dir,
        model_tag="mistral-7b",
        precision=MODEL_REGISTRY[config_key]["quant_type"],
        quant_method=str(MODEL_REGISTRY[config_key]["bits"]) + "bit",
        seed=args.seed,
        wandb=wandb,
    )
finally:
    free_model(model)
    run.finish()

print(f"\nResults saved to {output_dir}/")
