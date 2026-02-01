import pandas as pd
import os

# --- 1. CONFIGURAZIONE FILE RISULTATI (Pre-trained + Experts) ---
CONFIG = {
    "MAMI": {
        "nlp":  "results_roberta_MAMI_Base.csv",
        "clip": "results_clipmlp_MAMI.csv",
        "meme": "results_memeclip_MAMI.csv",
        "out":  "final_table_mami.csv"
    },
    "MultiOFF": {
        "nlp":  "results_roberta_MultiOFF_Base.csv",
        "clip": "results_clipmlp_MultiOFF.csv",
        "meme": "results_memeclip_MultiOFF.csv",
        "out":  "final_table_multioff.csv"
    },
    "HatefulMemes": {
        "nlp":  "results_roberta_HatefulMemes_Base.csv",
        "clip": "results_clipmlp_HatefulMemes.csv",
        "meme": "results_memeclip_HatefulMemes.csv",
        "out":  "final_table_hateful.csv"
    }
}

# --- 2. CONFIGURAZIONE DATASET ORIGINALI (Per recuperare il TESTO) ---
ORIGINAL_DATASETS = {
    "MAMI": {
        "path": "/home/rpisanu/MAMI/MAMI_DATASET/test.tsv",
        "format": "tsv",
        "text_col": "Text Transcription",
        "img_col": "file_name"
    },
    "MultiOFF": {
        "path": "/home/rpisanu/MultiOFF/MultiOFF_Dataset/Split Dataset/Testing_meme_dataset.csv",
        "format": "csv",
        "text_col": "sentence",
        "img_col": "image_name"
    },
    "HatefulMemes": {
        "path": "/home/rpisanu/Hateful/hateful_memes/data/dev.jsonl",
        "format": "jsonl",
        "text_col": "text",
        "img_col": "img"
    }
}

def load_original_text(dataset_name):
    """Carica il dataset originale e restituisce un DataFrame con [image_name, text]"""
    cfg = ORIGINAL_DATASETS[dataset_name]
    print(f"   📖 Recupero testo da: {cfg['path']}")
    
    if cfg['format'] == 'jsonl':
        df = pd.read_json(cfg['path'], lines=True)
    elif cfg['format'] == 'tsv':
        df = pd.read_csv(cfg['path'], sep='\t')
    else:
        df = pd.read_csv(cfg['path'])
        
    # Standardizza nomi colonne
    df = df.rename(columns={cfg['img_col']: 'image_name', cfg['text_col']: 'text'})
    
    # Assicurati che image_name sia stringa
    df['image_name'] = df['image_name'].astype(str)
    df['text'] = df['text'].fillna("").astype(str)
    
    return df[['image_name', 'text']]

def find_pred_col(df, model_type):
    possibilities = [f'{model_type}_pred', 'pred', 'prediction', 'roberta_pred', 'clipmlp_pred', 'memeclip_pred']
    for col in df.columns:
        if col in possibilities: return col
    for col in df.columns:
        if col.endswith('_pred'): return col
    return None

def merge_dataset(name, files):
    print(f"\n🔄 Elaborazione {name}...")
    
    # 1. Carica RoBERTa BASE (Master DataFrame)
    if not os.path.exists(files['nlp']):
        print(f"❌ File mancante: {files['nlp']}")
        return
    
    df_nlp = pd.read_csv(files['nlp'])
    
    # Standardizza image_name
    img_col = 'image_name' if 'image_name' in df_nlp.columns else 'img'
    if img_col != 'image_name': df_nlp.rename(columns={img_col: 'image_name'}, inplace=True)
    df_nlp['image_name'] = df_nlp['image_name'].astype(str)
    
    # Trova predizione NLP
    pred_col = find_pred_col(df_nlp, 'roberta')
    if pred_col:
        df_nlp.rename(columns={pred_col: 'PRED_NLP'}, inplace=True)
    else:
        print(f"⚠️ Colonna predizioni non trovata in {files['nlp']}")
        return

    # 2. Carica CLIP+MLP
    if os.path.exists(files['clip']):
        df_clip = pd.read_csv(files['clip'])
        df_clip['image_name'] = df_clip['image_name'].astype(str)
        pred_col = find_pred_col(df_clip, 'clipmlp')
        if pred_col:
            df_clip = df_clip[['image_name', pred_col]].rename(columns={pred_col: 'PRED_CLIP'})
            df_nlp = df_nlp.merge(df_clip, on='image_name', how='left')
    
    # 3. Carica MemeCLIP
    if os.path.exists(files['meme']):
        df_meme = pd.read_csv(files['meme'])
        df_meme['image_name'] = df_meme['image_name'].astype(str)
        pred_col = find_pred_col(df_meme, 'memeclip')
        if pred_col:
            df_meme = df_meme[['image_name', pred_col]].rename(columns={pred_col: 'PRED_MEME'})
            df_nlp = df_nlp.merge(df_meme, on='image_name', how='left')

    # 4. MERGE DEL TESTO ORIGINALE (FIX CRITICO)
    df_text = load_original_text(name)
    # Rimuovi colonna text se esiste già ma è vuota/sbagliata per sovrascriverla
    if 'text' in df_nlp.columns:
        df_nlp = df_nlp.drop(columns=['text'])
    
    df_nlp = df_nlp.merge(df_text, on='image_name', how='left')

    # Pulizia e Salvataggio
    cols_to_fill = ['PRED_NLP', 'PRED_CLIP', 'PRED_MEME']
    for c in cols_to_fill:
        if c in df_nlp.columns: df_nlp[c] = df_nlp[c].fillna(-1).astype(int)
    
    if 'label_true' in df_nlp.columns: df_nlp.rename(columns={'label_true': 'true_label'}, inplace=True)
    elif 'label' in df_nlp.columns: df_nlp.rename(columns={'label': 'true_label'}, inplace=True)

    # Verifica finale
    if 'text' not in df_nlp.columns:
        print("❌ ERRORE GRAVE: Colonna 'text' ancora mancante!")
    else:
        print(f"✅ Testo recuperato. Esempi: {df_nlp['text'].head(2).values}")

    df_nlp.to_csv(files['out'], index=False)
    print(f"✅ Salvato: {files['out']} ({len(df_nlp)} righe)")

if __name__ == "__main__":
    for dataset_name, file_paths in CONFIG.items():
        merge_dataset(dataset_name, file_paths)
