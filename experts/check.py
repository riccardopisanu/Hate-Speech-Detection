import pandas as pd
import os

# --- I TUOI CSV (Quelli che hai generato tu) ---
FILES = {
    "MAMI": "/home/rpisanu/tesi_repo/experts/results_llm/results_InternVL_MAMI.csv",
    "HM":   "/home/rpisanu/tesi_repo/experts/results_llm/results_InternVL_HatefulMemes.csv",
    "MOFF": "/home/rpisanu/tesi_repo/experts/results_llm/results_InternVL_MultiOFF.csv"
}

# GLI ID CHE ABBIAMO SCARICATO (Quelli che ti sembrano "sbagliati")
IDS_TO_CHECK = [
    "15026.jpg", "15068.jpg", # MAMI Success
    "15210.jpg", # MAMI Arbiter
    "01796.png", "01726.png", # HM Success
    "12XLnzK.png" # MOFF Failure
]

print(f"{'DATASET':<10} | {'FILE NAME (CSV)':<20} | {'PRED_CONST':<10} | {'LABEL':<5} | {'MATCH?'}")
print("-" * 80)

for name, path in FILES.items():
    if not os.path.exists(path): continue
    
    df = pd.read_csv(path)
    
    # Cerchiamo se questi ID esistono davvero nel tuo CSV
    for target_id in IDS_TO_CHECK:
        # Pulizia nome (toglie 'img/' o path se presenti nel CSV o nel target)
        clean_target = os.path.basename(target_id)
        
        # Cerca nel dataframe (nella colonna 'image' o 'image_name')
        col_img = 'image' if 'image' in df.columns else 'image_name'
        
        # Filtra la riga
        match = df[df[col_img].astype(str).apply(os.path.basename) == clean_target]
        
        if not match.empty:
            # Trovato!
            row = match.iloc[0]
            pred = row.get('pred_constraint', row.get('pred_heuristic', 'N/A'))
            label = row['label']
            print(f"{name:<10} | {row[col_img]:<20} | {pred:<10} | {label:<5} | ✅ SI, ESISTE")
        else:
            # Se stiamo cercando un ID MAMI nel dataset HM, è normale non trovarlo
            # Ma se non lo trova nel dataset giusto, c'è un problema
            if name == "MAMI" and "jpg" in clean_target:
                 print(f"{name:<10} | {clean_target:<20} | {'-':<10} | {'-':<5} | ❌ NON TROVATO NEL CSV")

print("-" * 80)
