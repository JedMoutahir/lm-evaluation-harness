# Jed’s LM-Evaluation-Harness Mini Eval

This folder adds a **minimal reproducible evaluation setup** for Hugging Face models using `lm-evaluation-harness`.

It runs:
- **MMLU (5-shot)**
- **GSM8K (5-shot)**
- **TruthfulQA (0-shot)**

All results saved as JSON + CSV.

> Works on CPU for smoke tests, GPU recommended for full runs.

---

## Quickstart

```bash
conda env create -f env.yml
conda activate lm-eval
```

Run a model:
```bash
python run_eval.py   --model-name meta-llama/Llama-3.1-8B-Instruct   --tasks mmlu,gsm8k,truthfulqa   --shots 5   --out runs/llama-8b
```

Results written to:
```
runs/<model>/
  - scores.json
  - scores.csv
```

Edit tasks in `run_eval.py` to add more benchmarks.
