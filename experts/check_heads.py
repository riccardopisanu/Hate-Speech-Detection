import pandas as pd
import os

files = ["final_table_mami.csv", "final_table_multioff.csv", "final_table_hateful.csv"]


for f in files:

    if os.path.exists(f):
        df = pd.read_csv(f)

        target_cols = ["image_name", "true_label", "PRED_NLP", "PRED_CLIP", "PRED_MEME"]

        show_cols = [c for c in target_cols if c in df.columns]

        if "PRED_NLP" in df.columns:
            pass

    else:
        pass
