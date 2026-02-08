import pandas as pd
import os

# ==============================================================================
# 1. CONFIGURAZIONE PATHS
# ==============================================================================
# Inserisci i percorsi ai file generati dal tuo script di inferenza
INPUT_FILES = {
    "MAMI": "/home/rpisanu/tesi_repo/experts/results_llm/results_InternVL_MAMI.csv",
    "Hateful Memes": "/home/rpisanu/tesi_repo/experts/results_llm/results_InternVL_HatefulMemes.csv",
    "MultiOFF": "/home/rpisanu/tesi_repo/experts/results_llm/results_InternVL_MultiOFF.csv"
}

# Nomi dei prompt nel CSV (devono coincidere con quelli del tuo codice)
PROMPTS = {
    "Demo": "P_DEMOCRACY",
    "CoT": "P_COT",
    "Const": "P_HEURISTIC"
}

# ==============================================================================
# 2. FUNZIONI DI FILTRAGGIO
# ==============================================================================

def load_and_pivot(csv_path):
    """Trasforma il CSV da formato lungo a largo (una riga per immagine)."""
    if not os.path.exists(csv_path):
        print(f"❌ File non trovato: {csv_path}")
        return None
        
    df = pd.read_csv(csv_path)
    
    # Pivot table: Index=image, Columns=prompt, Values=final_pred
    pivot = df.pivot_table(
        index=['image', 'label', 'expert_clip', 'expert_meme', 'expert_nlp'], 
        columns='prompt', 
        values='final_pred',
        aggfunc='first' # Prende il primo valore se ci sono duplicati
    ).reset_index()
    
    return pivot

def find_mami_success(df):
    """
    CASO 1: MAMI (Successo del Constraint)
    - Label è SAFE (0)
    - Esperti Visivi dicono HATE (1) -> Bias
    - Esperto Testo dice SAFE (0) -> Safe Anchor
    - Democracy sbaglia (1) -> Segue il bias visivo
    - Constraint indovina (0) -> Segue la regola "Only classify as 1 if..."
    """
    mask = (
        (df['label'] == 0) & 
        (df['expert_nlp'] == 0) & 
        ((df['expert_clip'] == 1) | (df['expert_meme'] == 1)) & # Visual Bias
        (df[PROMPTS['Demo']] == 1) &  # Baseline allucina
        (df[PROMPTS['Const']] == 0)   # Ours corregge
    )
    return df[mask]

def find_hm_success(df):
    """
    CASO 2: Hateful Memes (Successo del CoT o Constraint Broad)
    - Label è HATE (1)
    - Esperto Testo dice SAFE (0) -> Benign Confounder
    - Constraint/CoT indovinano (1) -> Hanno capito l'ironia
    - Democracy sbaglia (0) -> Confusa dal voto Safe del testo
    """
    mask = (
        (df['label'] == 1) & 
        (df['expert_nlp'] == 0) & # Testo ingannevole
        (df[PROMPTS['Demo']] == 0) & # Democracy fallisce
        ((df[PROMPTS['Const']] == 1) | (df[PROMPTS['CoT']] == 1)) # Uno dei nostri metodi capisce
    )
    return df[mask]

def find_failure_sarcasm(df):
    """
    CASO 3: Failure (Sarcasmo Testuale)
    - Label è HATE (1)
    - Esperto Testo dice SAFE (0) -> Non capisce il sarcasmo
    - Constraint sbaglia (0) -> Si fida troppo del testo ("Safe Anchor")
    """
    mask = (
        (df['label'] == 1) & 
        (df['expert_nlp'] == 0) & # RoBERTa non vede sarcasmo
        (df[PROMPTS['Const']] == 0) # Il sistema si blocca
    )
    return df[mask]

# ==============================================================================
# 3. ESECUZIONE
# ==============================================================================

print("🔍 RICERCA CANDIDATI PER CASE STUDIES...\n")

# --- 1. MAMI SUCCESS ---
print("--- CASO 1: MITIGATING VISUAL HALLUCINATION (MAMI) ---")
df_mami = load_and_pivot(INPUT_FILES["MAMI"])
if df_mami is not None:
    candidates = find_mami_success(df_mami)
    print(f"Trovati {len(candidates)} candidati perfetti.")
    if len(candidates) > 0:
        print("Ecco i primi 5 ID immagine da controllare:")
        print(candidates['image'].head(5).tolist())
        # Opzionale: salva CSV per ispezione
        # candidates.to_csv("candidates_mami.csv")

# --- 2. HATEFUL MEMES SUCCESS ---
print("\n--- CASO 2: BRIDGING IMPLICIT HATE (HATEFUL MEMES) ---")
df_hm = load_and_pivot(INPUT_FILES["Hateful Memes"])
if df_hm is not None:
    candidates = find_hm_success(df_hm)
    print(f"Trovati {len(candidates)} candidati perfetti.")
    if len(candidates) > 0:
        print("Ecco i primi 5 ID immagine (scegli quello dove testo e img contrastano):")
        print(candidates['image'].head(5).tolist())

# --- 3. FAILURE ANALYSIS (MULTIOFF o MAMI) ---
print("\n--- CASO 3: FAILURE / SARCASM (MultiOFF) ---")
df_moff = load_and_pivot(INPUT_FILES["MultiOFF"])
if df_moff is not None:
    candidates = find_failure_sarcasm(df_moff)
    print(f"Trovati {len(candidates)} fallimenti dovuti al Safe Anchor.")
    if len(candidates) > 0:
        print("Ecco i primi 5 ID immagine (cerca testo sarcastico):")
        print(candidates['image'].head(5).tolist())

print("\n✅ Fatto. Usa questi ID per trovare le immagini e inserirle nella tesi.")
