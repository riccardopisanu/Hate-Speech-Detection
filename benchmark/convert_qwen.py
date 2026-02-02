import pandas as pd
import os

# File di input (Percorso assoluto, corretto)
old_file = "/home/rpisanu/MMHS150K/mmhs_qwen_mixed_results.csv"

print(f"📖 Leggo {old_file}...")

if not os.path.exists(old_file):
    print(f"❌ ERRORE: Il file non esiste: {old_file}")
    exit(1)

df = pd.read_csv(old_file)

# Rinomina colonne
df = df.rename(columns={
    "tweet_id": "id",
    "prompt_type": "prompt",
    "pred_label": "pred"
})

# --- CORREZIONE QUI ---
# Siccome sei già nella cartella 'benchmark', punta direttamente alla sottocartella
out_path = "results_benchmarks/results_qwen_MMHS150K.csv"

# Creiamo la cartella se non esiste (sicurezza extra)
os.makedirs(os.path.dirname(out_path), exist_ok=True)

df.to_csv(out_path, index=False)

print(f"✅ Convertito e salvato in: {out_path}")
print(f"📊 Totale righe convertite: {len(df)}")
