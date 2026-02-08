import pandas as pd
import os

# ==============================================================================
# 1. CONFIGURAZIONE PATHS (CORRETTI)
# ==============================================================================
INPUT_FILES = {
    "MAMI": "/home/rpisanu/tesi_repo/experts/results_llm/results_InternVL_MAMI.csv",
    "Hateful Memes": "/home/rpisanu/tesi_repo/experts/results_llm/results_InternVL_HatefulMemes.csv",
    "MultiOFF": "/home/rpisanu/tesi_repo/experts/results_llm/results_InternVL_MultiOFF.csv"
}

# Mapping dei nomi prompt nel CSV alle colonne che useremo
PROMPTS = {
    "Demo": "P_DEMOCRACY",
    "CoT": "P_COT",
    "Const": "P_HEURISTIC"
}

# ==============================================================================
# 2. FUNZIONI DI UTILITÀ (PIVOTING)
# ==============================================================================

def load_and_pivot(csv_path):
    """
    Carica il CSV e mette le strategie in colonne affiancate.
    Indispensabile per confrontare Democracy vs Constraint sulla stessa immagine.
    """
    if not os.path.exists(csv_path):
        print(f"❌ File non trovato: {csv_path}")
        return None
        
    df = pd.read_csv(csv_path)
    
    # Controlliamo se le colonne degli esperti esistono, altrimenti mettiamo 0 di default
    for col in ['expert_clip', 'expert_meme', 'expert_nlp']:
        if col not in df.columns:
            df[col] = 0

    # Pivot table: Index=image, Columns=prompt, Values=final_pred
    pivot = df.pivot_table(
        index=['image', 'label', 'expert_clip', 'expert_meme', 'expert_nlp'], 
        columns='prompt', 
        values='final_pred',
        aggfunc='first'
    ).reset_index()
    
    return pivot

# ==============================================================================
# 3. FUNZIONI DI FILTRAGGIO ERRORI
# ==============================================================================

def find_arbiter_hallucinations(df):
    """
    ARBITER ERROR:
    - Label è SAFE (0)
    - TUTTI gli esperti dicono SAFE (0)
    - Il modello Constraint dice HATE (1)
    """
    mask = (
        (df['label'] == 0) &
        (df['expert_clip'] == 0) & (df['expert_meme'] == 0) & (df['expert_nlp'] == 0) &
        (df[PROMPTS['Const']] == 1)
    )
    return df[mask]

def find_disagreement_confusion(df):
    """
    CONFUSION:
    - Esperti in disaccordo (somma voti è 1 o 2)
    - Il modello Constraint sbaglia (pred != label)
    """
    expert_sum = df['expert_clip'] + df['expert_meme'] + df['expert_nlp']
    mask = (
        (expert_sum > 0) & (expert_sum < 3) & # Disaccordo
        (df[PROMPTS['Const']] != df['label']) # Errore
    )
    return df[mask]

def find_total_failures(df):
    """
    TOTAL FAILURE:
    - Democracy sbaglia
    - CoT sbaglia
    - Constraint sbaglia
    """
    mask = (
        (df[PROMPTS['Demo']] != df['label']) &
        (df[PROMPTS['CoT']] != df['label']) &
        (df[PROMPTS['Const']] != df['label'])
    )
    return df[mask]

def find_democracy_wins(df):
    """
    DEMOCRACY WINS:
    - Democracy indovina
    - CoT sbaglia
    - Constraint sbaglia
    """
    mask = (
        (df[PROMPTS['Demo']] == df['label']) &
        (df[PROMPTS['CoT']] != df['label']) &
        (df[PROMPTS['Const']] != df['label'])
    )
    return df[mask]

# ==============================================================================
# 4. ESECUZIONE MAIN
# ==============================================================================

print("🚀 STARTING FAILURE ANALYSIS...\n")

for name, path in INPUT_FILES.items():
    print(f"📊 ANALYZING DATASET: {name}")
    
    # 1. Carica e Pivota
    df = load_and_pivot(path)
    if df is None: continue

    # 2. Arbiter Errors
    res = find_arbiter_hallucinations(df)
    print(f"   🔴 ARBITER ERRORS (Hallucinations): {len(res)}")
    if len(res) > 0:
        print(f"      Esempio ID: {res['image'].iloc[0]}")

    # 3. Disagreement Confusion
    res = find_disagreement_confusion(df)
    print(f"   🟠 CONFUSION (Expert Disagreement): {len(res)}")
    if len(res) > 0:
        print(f"      Esempio ID: {res['image'].iloc[0]}")

    # 4. Total Failures
    res = find_total_failures(df)
    print(f"   ⚫ TOTAL FAILURES (Hard Samples): {len(res)}")
    if len(res) > 0:
        print(f"      Esempio ID: {res['image'].iloc[0]}")

    # 5. Democracy Wins
    res = find_democracy_wins(df)
    print(f"   🟢 DEMOCRACY WINS (Simpler is better): {len(res)}")
    if len(res) > 0:
        print(f"      Esempio ID: {res['image'].iloc[0]}")
    
    print("-" * 40)

print("\n✅ Analysis Complete.")
