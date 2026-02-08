import pandas as pd
import os
import shutil
import zipfile

INPUT_FILES = {
    "MAMI": "/home/rpisanu/tesi_repo/experts/results_llm/results_InternVL_MAMI.csv",
    "Hateful Memes": "/home/rpisanu/tesi_repo/experts/results_llm/results_InternVL_HatefulMemes.csv",
    "MultiOFF": "/home/rpisanu/tesi_repo/experts/results_llm/results_InternVL_MultiOFF.csv",
}

SEARCH_ROOTS = ["/beegfs-scratch/rpisanu", "/home/rpisanu"]

OUTPUT_DIR = "TESI_FINAL_CASES"
ZIP_NAME = "tesi_final_cases.zip"

PROMPTS = {"Demo": "P_DEMOCRACY", "CoT": "P_COT", "Const": "P_HEURISTIC"}


def load_and_pivot(csv_path):
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    for col in ["expert_clip", "expert_meme", "expert_nlp"]:
        if col not in df.columns:
            df[col] = 0
    return df.pivot_table(
        index=["image", "label", "expert_clip", "expert_meme", "expert_nlp"],
        columns="prompt",
        values="final_pred",
        aggfunc="first",
    ).reset_index()


def find_mami_success(df):
    return df[
        (df["label"] == 0)
        & (df["expert_nlp"] == 0)
        & ((df["expert_clip"] == 1) | (df["expert_meme"] == 1))
        & (df[PROMPTS["Demo"]] == 1)
        & (df[PROMPTS["Const"]] == 0)
    ]


def find_hm_success(df):
    return df[
        (df["label"] == 1)
        & (df["expert_nlp"] == 0)
        & (df[PROMPTS["Demo"]] == 0)
        & ((df[PROMPTS["Const"]] == 1) | (df[PROMPTS["CoT"]] == 1))
    ]


def find_arbiter_err(df):
    return df[
        (df["label"] == 0)
        & (df["expert_clip"] == 0)
        & (df["expert_meme"] == 0)
        & (df["expert_nlp"] == 0)
        & (df[PROMPTS["Const"]] == 1)
    ]


def find_confusion(df):
    s = df["expert_clip"] + df["expert_meme"] + df["expert_nlp"]
    return df[(s > 0) & (s < 3) & (df[PROMPTS["Const"]] != df["label"])]


def find_total_fail(df):
    return df[
        (df[PROMPTS["Demo"]] != df["label"])
        & (df[PROMPTS["CoT"]] != df["label"])
        & (df[PROMPTS["Const"]] != df["label"])
    ]


def find_demo_wins(df):
    return df[
        (df[PROMPTS["Demo"]] == df["label"])
        & (df[PROMPTS["CoT"]] != df["label"])
        & (df[PROMPTS["Const"]] != df["label"])
    ]


def find_sarcasm_fail(df):
    return df[
        (df["label"] == 1) & (df["expert_nlp"] == 0) & (df[PROMPTS["Const"]] == 0)
    ]  # Specifico per MultiOFF


if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR)


tasks = []  # (id_immagine, cartella_destinazione)

df_mami = load_and_pivot(INPUT_FILES["MAMI"])
if df_mami is not None:
    ids = (
        find_mami_success(df_mami)["image"].head(10).tolist()
    )  # Ne prendiamo fino a 10
    for i in ids:
        tasks.append((i, "Success_Stories/MAMI_Visual_Hallucination"))

df_hm = load_and_pivot(INPUT_FILES["Hateful Memes"])
if df_hm is not None:
    ids = find_hm_success(df_hm)["image"].head(10).tolist()
    for i in ids:
        tasks.append((i, "Success_Stories/HM_Implicit_Hate"))

for name, path in INPUT_FILES.items():
    df = load_and_pivot(path)
    if df is None:
        continue

    if name != "MAMI":
        ids = find_sarcasm_fail(df)["image"].head(5).tolist()
        for i in ids:
            tasks.append((i, f"Failures/Type_C_Blind_Anchor/{name}"))

    ids = find_arbiter_err(df)["image"].head(5).tolist()
    for i in ids:
        tasks.append((i, f"Failures/Type_A_Arbiter_Error/{name}"))

    ids = find_confusion(df)["image"].head(5).tolist()
    for i in ids:
        tasks.append((i, f"Failures/Type_B_Confusion/{name}"))

    ids = find_total_fail(df)["image"].head(5).tolist()
    for i in ids:
        tasks.append((i, f"Failures/Type_D_Total_Failure/{name}"))

    ids = find_demo_wins(df)["image"].head(5).tolist()
    for i in ids:
        tasks.append((i, f"Failures/Type_E_Democracy_Wins/{name}"))

unique_ids = set([t[0] for t in tasks])

found_map = {}
for base_dir in SEARCH_ROOTS:
    if not os.path.exists(base_dir):
        continue
    for root, dirs, files in os.walk(base_dir):
        if ".git" in root or "__pycache__" in root:
            continue
        for file in files:
            if file in [os.path.basename(u) for u in unique_ids]:
                found_map[file] = os.path.join(root, file)

for img_id, dest_folder in tasks:
    filename = os.path.basename(img_id)
    if filename in found_map:
        full_dest = os.path.join(OUTPUT_DIR, dest_folder)
        os.makedirs(full_dest, exist_ok=True)

        shutil.copy(found_map[filename], os.path.join(full_dest, filename))
    else:
        pass

shutil.make_archive(ZIP_NAME.replace(".zip", ""), "zip", OUTPUT_DIR)
full_zip = os.path.abspath(ZIP_NAME)
