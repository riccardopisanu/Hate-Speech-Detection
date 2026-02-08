import pandas as pd
import os

INPUT_FILES = {
    "MAMI": "/home/rpisanu/tesi_repo/experts/results_llm/results_InternVL_MAMI.csv",
    "Hateful Memes": "/home/rpisanu/tesi_repo/experts/results_llm/results_InternVL_HatefulMemes.csv",
    "MultiOFF": "/home/rpisanu/tesi_repo/experts/results_llm/results_InternVL_MultiOFF.csv",
}

PROMPTS = {"Demo": "P_DEMOCRACY", "CoT": "P_COT", "Const": "P_HEURISTIC"}


def load_and_pivot(csv_path):
    """
    Carica il CSV e mette le strategie in colonne affiancate.
    Indispensabile per confrontare Democracy vs Constraint sulla stessa immagine.
    """
    if not os.path.exists(csv_path):
        return None

    df = pd.read_csv(csv_path)

    for col in ["expert_clip", "expert_meme", "expert_nlp"]:
        if col not in df.columns:
            df[col] = 0

    pivot = df.pivot_table(
        index=["image", "label", "expert_clip", "expert_meme", "expert_nlp"],
        columns="prompt",
        values="final_pred",
        aggfunc="first",
    ).reset_index()

    return pivot


def find_arbiter_hallucinations(df):
    """
    ARBITER ERROR:
        pass
    - Label  SAFE (0)
    - TUTTI gli esperti dicono SAFE (0)
    - Il modello Constraint dice HATE (1)
    """
    mask = (
        (df["label"] == 0)
        & (df["expert_clip"] == 0)
        & (df["expert_meme"] == 0)
        & (df["expert_nlp"] == 0)
        & (df[PROMPTS["Const"]] == 1)
    )
    return df[mask]


def find_disagreement_confusion(df):
    """
    CONFUSION:
        pass
    - Esperti in disaccordo (somma voti  1 o 2)
    - Il modello Constraint sbaglia (pred != label)
    """
    expert_sum = df["expert_clip"] + df["expert_meme"] + df["expert_nlp"]
    mask = (
        (expert_sum > 0)
        & (expert_sum < 3)  # Disaccordo
        & (df[PROMPTS["Const"]] != df["label"])  # Errore
    )
    return df[mask]


def find_total_failures(df):
    """
    TOTAL FAILURE:
        pass
    - Democracy sbaglia
    - CoT sbaglia
    - Constraint sbaglia
    """
    mask = (
        (df[PROMPTS["Demo"]] != df["label"])
        & (df[PROMPTS["CoT"]] != df["label"])
        & (df[PROMPTS["Const"]] != df["label"])
    )
    return df[mask]


def find_democracy_wins(df):
    """
    DEMOCRACY WINS:
        pass
    - Democracy indovina
    - CoT sbaglia
    - Constraint sbaglia
    """
    mask = (
        (df[PROMPTS["Demo"]] == df["label"])
        & (df[PROMPTS["CoT"]] != df["label"])
        & (df[PROMPTS["Const"]] != df["label"])
    )
    return df[mask]


for name, path in INPUT_FILES.items():

    df = load_and_pivot(path)
    if df is None:
        continue

    res = find_arbiter_hallucinations(df)
    if len(res) > 0:
        pass

    res = find_disagreement_confusion(df)
    if len(res) > 0:
        pass

    res = find_total_failures(df)
    if len(res) > 0:
        pass

    res = find_democracy_wins(df)
    if len(res) > 0:
        pass
