import pandas as pd
import os

old_file = "/home/rpisanu/MMHS150K/mmhs_qwen_mixed_results.csv"


if not os.path.exists(old_file):
    exit(1)

df = pd.read_csv(old_file)

df = df.rename(
    columns={"tweet_id": "id", "prompt_type": "prompt", "pred_label": "pred"}
)

out_path = "results_benchmarks/results_qwen_MMHS150K.csv"

os.makedirs(os.path.dirname(out_path), exist_ok=True)

df.to_csv(out_path, index=False)
