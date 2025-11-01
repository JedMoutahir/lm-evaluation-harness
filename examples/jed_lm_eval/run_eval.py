import argparse, json
from pathlib import Path
import subprocess
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--tasks", default="mmlu,gsm8k,truthfulqa")
    ap.add_argument("--shots", type=int, default=5)
    ap.add_argument("--out", default="runs/model_eval")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "scores.json"
    csv_path = out_dir / "scores.csv"

    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={args.model_name},dtype=bfloat16",
        "--tasks", args.tasks,
        "--num_fewshot", str(args.shots),
        "--output_path", str(json_path)
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # Convert to CSV
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        rows = []
        for task, metrics in data.get("results", {}).items():
            for metric, val in metrics.items():
                rows.append({"task": task, "metric": metric, "value": val})
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"Saved CSV to {csv_path}")

if __name__ == "__main__":
    main()
