import pandas as pd
import os

# I nomi dei file generati da prepare_tables.py
files = [
    "final_table_mami.csv", 
    "final_table_multioff.csv", 
    "final_table_hateful.csv"
]

print("🔍 CONTROLLO HEADER CSV PER INTERNVL")

for f in files:
    print(f"\n{'='*60}")
    print(f"📄 FILE: {f}")
    
    if os.path.exists(f):
        df = pd.read_csv(f)
        print(f"   Dimensioni: {df.shape} (Righe, Colonne)")
        print(f"   Colonne presenti: {list(df.columns)}")
        
        # Selezioniamo solo le colonne cruciali per vedere se il merge è ok
        target_cols = ['image_name', 'true_label', 'PRED_NLP', 'PRED_CLIP', 'PRED_MEME']
        
        # Filtra solo quelle che esistono davvero nel file
        show_cols = [c for c in target_cols if c in df.columns]
        
        print("\n   --- ANTEPRIMA (Prime 5 righe) ---")
        print(df[show_cols].head(5).to_string(index=False))
        
        # Check rapido sui valori
        if 'PRED_NLP' in df.columns:
            print(f"\n   Stats PRED_NLP (RoBERTa Base): {df['PRED_NLP'].unique()}")
        
    else:
        print("   ❌ FILE NON TROVATO! Hai lanciato prepare_tables.py?")

print(f"\n{'='*60}")
