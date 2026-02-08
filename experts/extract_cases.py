import pandas as pd
import os

INPUT_FILES = {
    "MAMI": "/home/rpisanu/tesi_repo/experts/results_llm/results_InternVL_MAMI.csv",
    "Hateful Memes": "/home/rpisanu/tesi_repo/experts/results_llm/results_InternVL_HatefulMemes.csv",
    "MultiOFF": "/home/rpisanu/tesi_repo/experts/results_llm/results_InternVL_MultiOFF.csv",
}

PROMPTS = {"Demo": "P_DEMOCRACY", "CoT": "P_COT", "Const": "P_HEURISTIC"}


def load_and_pivot(csv_path):
    """Trasforma il CSV da formato lungo a largo (una riga per immagine)."""
    if not os.path.exists(csv_path):
        return None

    df = pd.read_csv(csv_path)

    pivot = df.pivot_table(
        index=["image", "label", "expert_clip", "expert_meme", "expert_nlp"],
        columns="prompt",
        values="final_pred",
        aggfunc="first",  # Prende il primo valore se ci sono duplicati
    ).reset_index()

    return pivot


def find_mami_success(df):
    """
    CASO 1: MAMI (Successo del Constraint)
    - Label  SAFE (0)
    - Esperti Visivi dicono HATE (1) -> Bias
    - Esperto Testo dice SAFE (0) -> Safe Anchor
    - Democracy sbaglia (1) -> Segue il bias visivo
    - Constraint indovina (0) -> Segue la regola "Only classify as 1 if..."
    """
    mask = (
        (df["label"] == 0)
        & (df["expert_nlp"] == 0)
        & ((df["expert_clip"] == 1) | (df["expert_meme"] == 1))  # Visual Bias
        & (df[PROMPTS["Demo"]] == 1)  # Baseline allucina
        & (df[PROMPTS["Const"]] == 0)  # Ours corregge
    )
    return df[mask]


def find_hm_success(df):
    """
    CASO 2: Hateful Memes (Successo del CoT o Constraint Broad)
    - Label  HATE (1)
    - Esperto Testo dice SAFE (0) -> Benign Confounder
    - Constraint/CoT indovinano (1) -> Hanno capito l'ironia
    - Democracy sbaglia (0) -> Confusa dal voto Safe del testo
    """
    mask = (
        (df["label"] == 1)
        & (df["expert_nlp"] == 0)  # Testo ingannevole
        & (df[PROMPTS["Demo"]] == 0)  # Democracy fallisce
        & (
            (df[PROMPTS["Const"]] == 1) | (df[PROMPTS["CoT"]] == 1)
        )  # Uno dei nostri metodi capisce
    )
    return df[mask]


def find_failure_sarcasm(df):
    """
    CASO 3: Failure (Sarcasmo Testuale)
    - Label  HATE (1)
    - Esperto Testo dice SAFE (0) -> Non capisce il sarcasmo
    - Constraint sbaglia (0) -> Si fida troppo del testo ("Safe Anchor")
    """
    mask = (
        (df["label"] == 1)
        & (df["expert_nlp"] == 0)  # RoBERTa non vede sarcasmo
        & (df[PROMPTS["Const"]] == 0)  # Il sistema si blocca
    )
    return df[mask]


df_mami = load_and_pivot(INPUT_FILES["MAMI"])
if df_mami is not None:
    candidates = find_mami_success(df_mami)
    if len(candidates) > 0:
        pass

df_hm = load_and_pivot(INPUT_FILES["Hateful Memes"])
if df_hm is not None:
    candidates = find_hm_success(df_hm)
    if len(candidates) > 0:
        pass

df_moff = load_and_pivot(INPUT_FILES["MultiOFF"])
if df_moff is not None:
    candidates = find_failure_sarcasm(df_moff)
    if len(candidates) > 0:
        pass
