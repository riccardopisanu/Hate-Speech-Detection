import pandas as pd
import os

INPUT_FILES = {
    "MAMI": "/home/rpisanu/tesi_repo/experts/results_llm/results_InternVL_MAMI.csv",
    "Hateful Memes": "/home/rpisanu/tesi_repo/experts/results_llm/results_InternVL_HatefulMemes.csv",
    "MultiOFF": "/home/rpisanu/tesi_repo/experts/results_llm/results_InternVL_MultiOFF.csv",
}

PROMPTS = {
    "Demo": "P_DEMOCRACY",
    "CoT": "P_COT",
    "Const": "P_HEURISTIC",  # O 'P_CONSTRAINT' a seconda del tuo file
}


def generate_latex_table(dataset_name, csv_path):
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)

    try:
        pivot = df.pivot_table(
            index=["image", "label"],
            columns="prompt",
            values="final_pred",
            aggfunc="first",
        ).reset_index()
    except KeyError as e:
        return

    for p_key, p_col in PROMPTS.items():
        if p_col not in pivot.columns:
            return

    correct_demo = pivot[PROMPTS["Demo"]] == pivot["label"]
    correct_cot = pivot[PROMPTS["CoT"]] == pivot["label"]
    correct_const = pivot[PROMPTS["Const"]] == pivot["label"]

    wins_const = correct_const & (~correct_demo) & (~correct_cot)

    wins_cot = correct_cot & (~correct_demo) & (~correct_const)

    wins_demo = correct_demo & (~correct_cot) & (~correct_const)

    total_failures = (~correct_demo) & (~correct_cot) & (~correct_const)

    count_const = wins_const.sum()
    count_cot = wins_cot.sum()
    count_demo = wins_demo.sum()
    count_fail = total_failures.sum()

    label_tag = f"tab:const_{dataset_name.lower().replace(' ', '_')}"


for name, path in INPUT_FILES.items():
    generate_latex_table(name, path)
