# Uncertainty-Aware Inference — Team B Systems Toolkit

This directory contains Team B's **reusable benchmarking toolkit** for the PTQ sweep.  
Teams A and C can use these scripts directly for their own models by changing a small set of config values — everything else stays the same.

---

## What's in this folder

```
TeamB/
├── configs.py                  ← Model registry (edit this for your model)
├── run_eval.py                 ← Calibration evaluation (HellaSwag, TriviaQA, PubMedQA)
├── run_eval_with_ckpt.py       ← Same, but with crash-safe checkpointing + W&B resume
├── run_vllm.py                 ← vLLM throughput benchmarking script
├── run_profiler.py             ← PyTorch Profiler + Roofline analysis script
├── pytorch_profiler.py         ← Lower-level profiler harness (used by run_profiler.py)
├── mistral_7b.ipynb            ← Calibration eval notebook (Mistral-7B)
├── teamb_vllm_only.ipynb       ← vLLM throughput notebook (standalone)
└── teamb_profiler_only.ipynb   ← PyTorch Profiler notebook (standalone)
```

---

## Quick-start for Teams A and C

### Step 1 — Edit `configs.py`

This is the **only file you need to change** for your model. Add your model entries following the same pattern:

```python
MODEL_REGISTRY = {
    "llama2-7b-fp16": {
        "hf_id":      "meta-llama/Llama-2-7b-chat-hf",  # HuggingFace model ID
        "quant_type": "fp16",
        "bits":       16,
        "description": "LLaMA-2 7B FP16 Baseline",
    },
    "llama2-7b-gptq-int4": {
        "hf_id":         "TheBloke/Llama-2-7B-Chat-GPTQ",
        "quant_type":    "gptq",
        "bits":          4,
        "gptq_revision": "main",   # use "main" for INT4; name the branch only for INT8
        "description":   "LLaMA-2 7B GPTQ INT4",
    },
    "llama2-7b-gptq-int8": {
        "hf_id":         "TheBloke/Llama-2-7B-Chat-GPTQ",
        "quant_type":    "gptq",
        "bits":          8,
        "gptq_revision": "gptq-8bit-128g-actorder_True",
        "description":   "LLaMA-2 7B GPTQ INT8",
    },
    "llama2-7b-awq-int4": {
        "hf_id":      "TheBloke/Llama-2-7B-Chat-AWQ",
        "quant_type": "awq",
        "bits":       4,
        "description": "LLaMA-2 7B AWQ INT4",
    },
    "llama2-7b-nf4": {
        "hf_id":      "meta-llama/Llama-2-7b-chat-hf",  # base model, quantized on-the-fly
        "quant_type": "nf4",
        "bits":       4,
        "description": "LLaMA-2 7B NF4 (bitsandbytes)",
    },
}
```

**Rules for `gptq_revision`:**
- INT4 → use `"main"`. On TheBloke repos the main branch is always the 4-bit model.  
  The branch name `gptq-4bit-128g-actorder_True` does **not** exist and will cause a 404.
- INT8 → use the explicit branch name, e.g. `"gptq-8bit-128g-actorder_True"`.

### Step 2 — Update `run_vllm.py` config table

`run_vllm.py` has its own `VLLM_CONFIGS` dict that maps config keys to vLLM load params. Add entries for your model:

```python
VLLM_CONFIGS = {
    # ... existing entries ...
    "llama2-7b-fp16": {
        "model":        "meta-llama/Llama-2-7b-chat-hf",
        "quantization": None,
        "revision":     None,
        "dtype":        "float16",
    },
    "llama2-7b-gptq-int4": {
        "model":        "TheBloke/Llama-2-7B-Chat-GPTQ",
        "quantization": "marlin",   # Marlin kernel for INT4 — fastest
        "revision":     "main",
        "dtype":        "float16",
    },
    "llama2-7b-awq-int4": {
        "model":        "TheBloke/Llama-2-7B-Chat-AWQ",
        "quantization": "awq_marlin",   # awq_marlin is ~4x faster than "awq"
        "revision":     None,
        "dtype":        "float16",
    },
    # NF4 falls back to HF batched generation automatically — no entry needed
}
```

**vLLM quantization key rules:**
| Precision | `quantization` value | Notes |
|-----------|---------------------|-------|
| FP16 | `None` | cuBLAS |
| GPTQ INT4 | `"marlin"` | Fastest; requires `revision="main"` |
| GPTQ INT8 | `"gptq"` | No INT8 Marlin; exllama path |
| AWQ INT4 | `"awq_marlin"` | Use this, not `"awq"` — 4-5× faster |
| NF4 | handled via HF fallback | Not supported by vLLM |

### Step 3 — Update `run_profiler.py` model registry

`run_profiler.py` has its own `MODEL_REGISTRY` dict (used for model loading and roofline). Add your entries mirroring `configs.py`:

```python
MODEL_REGISTRY = {
    # ... existing entries ...
    "llama2-7b-fp16": {
        "hf_id":      "meta-llama/Llama-2-7b-chat-hf",
        "quant_type": "fp16",
        "bits":       16,
    },
    "llama2-7b-awq-int4": {
        "hf_id":      "TheBloke/Llama-2-7B-Chat-AWQ",
        "quant_type": "awq",
        "bits":       4,
    },
    # ...
}
```

Also add your AWQ config key to `CUSTOM_KERNEL_CONFIGS` if it uses AWQ:

```python
CUSTOM_KERNEL_CONFIGS = {"mistral-7b-awq-int4", "llama2-7b-awq-int4"}
```

### Step 4 — Update the notebooks

In the three notebooks, change the `ALL_CONFIGS` list and `REPO_DIR` to point at your team folder:

```python
# In teamb_vllm_only.ipynb or teamb_profiler_only.ipynb:
REPO_DIR    = "/content/uncertainty-aware-inference/TeamA"   # ← your folder
ALL_CONFIGS = [
    "llama2-7b-fp16",
    "llama2-7b-gptq-int4",
    "llama2-7b-gptq-int8",
    "llama2-7b-awq-int4",
    "llama2-7b-nf4",
]
```

---

## Running calibration evaluation

### Single config (recommended — reload model fresh each time)
```bash
python run_eval.py --config mistral-7b-fp16
python run_eval.py --config mistral-7b-gptq-int4
python run_eval.py --config mistral-7b-awq-int4 --samples 200
```

### With crash-safe checkpointing (recommended for Colab)
```bash
# Saves every 50 examples. If Colab disconnects, rerun the same command
# and it will resume from the last checkpoint.
python run_eval_with_ckpt.py --config mistral-7b-fp16 \
    --output-root /content/drive/MyDrive/uai_results \
    --save-every 50
```

### Using notebooks
Open `mistral_7b.ipynb` and run cells sequentially. Each config section sets `CONFIG_KEY` and calls `run_eval.py`.

**Output structure:**
```
updated_results/
└── mistral-7b-fp16/
    ├── hellaswag_results.json
    ├── triviaqa_results.json
    └── pubmedqa_results.json
```

---

## Running vLLM throughput benchmark

### Single config
```bash
python run_vllm.py --config mistral-7b-fp16    --output-dir ./vllm_results
python run_vllm.py --config mistral-7b-nf4     --output-dir ./vllm_results
```

### Full sweep via notebook
Open `teamb_vllm_only.ipynb`. Section 5 loops over all configs automatically.

**Output:** `vllm_results/{config_key}_vllm.json` per config.

---

## Running PyTorch Profiler

### Single config
```bash
python run_profiler.py --config mistral-7b-fp16    --output-dir ./profiler_results
python run_profiler.py --config mistral-7b-awq-int4 --output-dir ./profiler_results
```

### Full sweep via notebook
Open `teamb_profiler_only.ipynb`. Section 5 loops over all configs automatically.

**Output per config:**
- `profiler_results/{config_key}_profile.json` — timing, memory, kernel breakdown, roofline
- `profiler_results/{config_key}_chrome.json` — Chrome trace (open at [perfetto.dev](https://ui.perfetto.dev))

**Note on AWQ:** AWQ uses custom C++ CUDA extensions that bypass the PyTorch dispatcher. The profiler will not capture individual kernel names for AWQ — this is expected and documented in the output JSON. Timing and memory numbers are still accurate.

**GPU requirement:** A100 (40 GB)

---

## W&B logging

All results are logged to the **`UAI_Project`** project under the **`Uncertainty_Aware_Inference_Lab`** entity.

Each script creates a run named `teamB_{config_key}_{experiment}` where experiment is one of:
- `vllm_throughput` — from `run_vllm.py`
- `pytorch_profiler` — from `run_profiler.py`
- `team-b_{config_key}` — from `run_eval.py`

---

## File-by-file reference

### `configs.py`
Central model registry. Each entry defines `hf_id`, `quant_type`, `bits`, and optionally `gptq_revision`. This is the single source of truth for model loading in `run_eval.py` and `run_eval_with_ckpt.py`.

### `run_eval.py`
Runs calibration evaluation (HellaSwag, TriviaQA, PubMedQA) for one config. Calls `shared/eval_utils.run_eval()`. Accepts `--config`, `--samples`, `--datasets`, `--seed`.

### `run_eval_with_ckpt.py`
Same as `run_eval.py` but adds:
- `--output-root` for Drive-mounted output (survives Colab disconnects)
- `--save-every N` for checkpoint frequency
- Deterministic W&B run ID so a restart rejoins the same run automatically

### `run_vllm.py`
Benchmarks throughput via vLLM serving engine. Each config runs in a subprocess for clean GPU state. NF4 falls back to HF `model.generate()` automatically. Uses `nvidia-smi` for GPU memory (PyTorch allocator is invisible to vLLM's memory pool).

### `run_profiler.py`
Two-phase profiler:
1. **Timing phase** — `PROFILE_STEPS` runs outside profiler, no measurement overhead
2. **Kernel phase** — 1 run inside `torch.profiler` context, exports Chrome trace

### `mistral_7b.ipynb`
Calibration eval notebook. Runs `run_eval.py` for each quantization config sequentially.

### `teamb_vllm_only.ipynb`
Dedicated vLLM throughput notebook. Runs `run_vllm.py` for all configs, produces summary table and W&B logs. Does not run the profiler.

### `teamb_profiler_only.ipynb`
Dedicated profiler notebook. Runs `run_profiler.py` for all configs, produces kernel breakdown tables, roofline plot, and W&B logs. Does not run vLLM.
