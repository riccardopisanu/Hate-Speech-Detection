import pandas as pd
import os
import shutil
import zipfile

# ==============================================================================
# 1. CONFIGURAZIONE
# ==============================================================================
CSV_FILES = {
    "HM": "/home/rpisanu/tesi_repo/experts/results_llm/results_InternVL_HatefulMemes.csv",
    "MOFF": "/home/rpisanu/tesi_repo/experts/results_llm/results_InternVL_MultiOFF.csv"
}

SEARCH_ROOTS = ["/beegfs-scratch/rpisanu", "/home/rpisanu"]
OUTPUT_DIR = "TESI_ABSOLUTE_WINS"
ZIP_NAME = "tesi_absolute_wins.zip"

PROMPTS = {"Demo": "P_DEMOCRACY", "CoT": "P_COT", "Const": "P_HEURISTIC"}

# ==============================================================================
# 2. CARICAMENTO E PIVOT (Essenziale per i tuoi CSV)
# ==============================================================================
def load_and_pivot(csv_path):
    if not os.path.exists(csv_path):
        print(f"⚠️ File mancante: {csv_path}"); return None
    
    df = pd.read_csv(csv_path)
    if 'image_name' in df.columns: df.rename(columns={'image_name': 'image'}, inplace=True)
    
    # Se è già pivotato, ritorna
    if PROMPTS['CoT'] in df.columns: return df
    
    print(f"   -> Pivot in corso per {os.path.basename(csv_path)}...")
    cols = ['image', 'label'] + [c for c in ['text', 'caption'] if c in df.columns]
    valid_cols = [c for c in cols if c in df.columns]
    
    try:
        return df.pivot_table(index=valid_cols, columns='prompt', values='final_pred', aggfunc='first').reset_index()
    except Exception as e:
        print(f"❌ Errore pivot: {e}"); return None

# ==============================================================================
# 3. LOGICA "VINCITORE ASSOLUTO" (Uno contro Tutti)
# ==============================================================================
def find_absolute_wins():
    tasks = [] 
    
    # --- A. HATEFUL MEMES: SOLO CoT VINCE ---
    # CoT == Label  AND  Const != Label  AND  Demo != Label
    df = load_and_pivot(CSV_FILES["HM"])
    if df is not None and PROMPTS['CoT'] in df.columns:
        wins = df[
            (df[PROMPTS['CoT']] == df['label']) & 
            (df[PROMPTS['Const']] != df['label']) & 
            (df[PROMPTS['Demo']] != df['label'])
        ]
        print(f"📊 HM (Solo CoT vince): Trovati {len(wins)} casi.")
        
        # Prendiamo i primi 5 casi
        for _, row in wins.head(5).iterrows():
            tasks.append((str(row['image']), "HM_Only_CoT_Wins"))
            print(f"   -> {row['image']} (Label: {row['label']})")

    # --- B. MULTIOFF: SOLO DEMOCRACY VINCE ---
    # Demo == Label  AND  CoT != Label  AND  Const != Label
    df = load_and_pivot(CSV_FILES["MOFF"])
    if df is not None and PROMPTS['Demo'] in df.columns:
        wins = df[
            (df[PROMPTS['Demo']] == df['label']) & 
            (df[PROMPTS['CoT']] != df['label']) & 
            (df[PROMPTS['Const']] != df['label'])
        ]
        print(f"\n📊 MOFF (Solo Demo vince): Trovati {len(wins)} casi.")
        
        for _, row in wins.head(5).iterrows():
            tasks.append((str(row['image']), "MOFF_Only_Demo_Wins"))
            print(f"   -> {row['image']} (Label: {row['label']})")

    return tasks

# ==============================================================================
# 4. SCANSIONE FISICA E ZIP
# ==============================================================================
def run():
    tasks = find_absolute_wins()
    if not tasks: print("⚠️ Nessun caso trovato."); return

    if os.path.exists(OUTPUT_DIR): shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)
    
    needed = set([os.path.basename(t[0]) for t in tasks])
    print(f"\n🕵️‍♂️ Cerco {len(needed)} file sul disco...")
    
    found_map = {}
    for base in SEARCH_ROOTS:
        if not os.path.exists(base): continue
        for root, _, files in os.walk(base):
            for f in files:
                if f in needed: found_map[f] = os.path.join(root, f)
    
    count = 0
    for fname, folder in tasks:
        clean = os.path.basename(fname)
        if clean in found_map:
            dest = os.path.join(OUTPUT_DIR, folder)
            os.makedirs(dest, exist_ok=True)
            # Prefisso per evitare confusione
            prefix = "HM_" if "HM" in folder else "MOFF_"
            shutil.copy(found_map[clean], os.path.join(dest, f"{prefix}{clean}"))
            count += 1
        else:
            print(f"❌ File non trovato su disco: {clean}")

    print(f"\n✅ Copiati {count}/{len(tasks)} file.")
    shutil.make_archive(ZIP_NAME.replace('.zip', ''), 'zip', OUTPUT_DIR)
    
    print("\n" + "="*80)
    print("COMMANDO DOWNLOAD (Windows):")
    print(f'scp -o "ProxyJump cagliero_thesis_students@mp4.polito.it" -o "UserKnownHostsFile=/dev/null" -o "StrictHostKeyChecking=no" rpisanu@hpc-legionlogin.polito.it:{os.path.abspath(ZIP_NAME)} "C:\\Users\\ricca\\OneDrive\\Desktop\\immagini"')
    print("="*80)

if __name__ == "__main__":
    run()
