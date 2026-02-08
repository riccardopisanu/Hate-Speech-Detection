import pandas as pd
import os

FILES = {
    "MAMI": "/home/rpisanu/tesi_repo/experts/results_llm/results_InternVL_MAMI.csv",
    "HM": "/home/rpisanu/tesi_repo/experts/results_llm/results_InternVL_HatefulMemes.csv",
    "MOFF": "/home/rpisanu/tesi_repo/experts/results_llm/results_InternVL_MultiOFF.csv",
}

IDS_TO_CHECK = [
    "15026.jpg",
    "15068.jpg",  # MAMI Success
    "15210.jpg",  # MAMI Arbiter
    "01796.png",
    "01726.png",  # HM Success
    "12XLnzK.png",  # MOFF Failure
]


for name, path in FILES.items():
    if not os.path.exists(path):
        continue

    df = pd.read_csv(path)

    for target_id in IDS_TO_CHECK:
        clean_target = os.path.basename(target_id)

        col_img = "image" if "image" in df.columns else "image_name"

        match = df[df[col_img].astype(str).apply(os.path.basename) == clean_target]

        if not match.empty:
            row = match.iloc[0]
            pred = row.get("pred_constraint", row.get("pred_heuristic", "N/A"))
            label = row["label"]
        else:
            if name == "MAMI" and "jpg" in clean_target:
                pass
