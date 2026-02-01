import os
import torch
import pandas as pd
import argparse
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import clip

# --- 1. CONFIGURAZIONE MINIMA ---
class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def get_dummy_config():
    return DotDict({
        'model': DotDict({'clip_model': "ViT-L/14", 'class_names': ['no', 'yes']}),
        'data': DotDict({'num_classes': 2}),
        'map_dim': 1024
    })

# --- 2. DEFINIZIONE CLASSI (Per caricare lo scheletro) ---
class CosineClassifier(nn.Module):
    def __init__(self, feat_dim, num_classes=2):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_classes, feat_dim))

class MemeCLIP_Skeleton(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = CosineClassifier(1024, 2)
        # Dummy layers per simulare la struttura
        self.image_map = nn.Sequential(nn.Linear(10, 10)) 
        self.text_map = nn.Sequential(nn.Linear(10, 10))
        self.pre_output = nn.Sequential(nn.Linear(10, 10))

# --- 3. DIAGNOSTICA DATI ---
def check_data(test_file, image_root):
    print("\n" + "="*40)
    print("🕵️‍♂️ DIAGNOSTICA DATI")
    print("="*40)
    
    if not os.path.exists(test_file):
        print(f"❌ ERRORE: File CSV non trovato: {test_file}")
        return

    try:
        if test_file.endswith('.jsonl'): df = pd.read_json(test_file, lines=True)
        else: 
            try: df = pd.read_csv(test_file, sep=',')
            except: df = pd.read_csv(test_file, sep='\t')
    except Exception as e:
        print(f"❌ ERRORE lettura CSV: {e}")
        return

    print(f"✅ CSV caricato: {len(df)} righe.")
    
    # Trova colonna immagini
    cols = {c.lower().strip(): c for c in df.columns}
    img_col = next((cols[c] for c in ['img', 'image_name', 'id'] if c in cols), None)
    
    if not img_col:
        print(f"❌ ERRORE: Colonna immagine non trovata. Colonne disponibili: {list(df.columns)}")
        return

    print(f"📷 Colonna immagine rilevata: '{img_col}'")
    
    # Test path prima immagine
    first_img_name = str(df.iloc[0][img_col])
    print(f"   Esempio nome file nel CSV: '{first_img_name}'")
    
    path_1 = os.path.join(image_root, os.path.basename(first_img_name))
    path_2 = os.path.join(image_root, first_img_name)
    
    print(f"   Tentativo 1 (Basename): {path_1}")
    if os.path.exists(path_1): print("   ✅ FILE TROVATO!")
    else: print("   ❌ FILE NON TROVATO")
    
    print(f"   Tentativo 2 (Direct):   {path_2}")
    if os.path.exists(path_2): print("   ✅ FILE TROVATO!")
    else: print("   ❌ FILE NON TROVATO")

    if not os.path.exists(path_1) and not os.path.exists(path_2):
        print("\n⚠️ ATTENZIONE: Le immagini non vengono trovate.")
        print(f"   Controlla che '{image_root}' sia la cartella giusta.")
        print(f"   Contenuto cartella root (primi 5 file):")
        try:
            print(os.listdir(image_root)[:5])
        except:
            print("   (Impossibile leggere la cartella)")

# --- 4. DIAGNOSTICA PESI ---
def check_weights(checkpoint_path):
    print("\n" + "="*40)
    print("⚖️ DIAGNOSTICA PESI")
    print("="*40)
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint non trovato: {checkpoint_path}")
        return

    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        state_dict = ckpt['state_dict']
        print(f"✅ Checkpoint caricato. Chiavi totali: {len(state_dict)}")
        
        keys = list(state_dict.keys())
        print("\n🔍 Esempio chiavi nel Checkpoint (prime 10):")
        for k in keys[:10]:
            print(f"   - {k}")
            
        print("\n🔍 Cerca chiavi 'classifier':")
        class_keys = [k for k in keys if 'classifier' in k]
        if class_keys:
            for k in class_keys: print(f"   FOUND: {k}  <-- IMPORTANTE")
        else:
            print("   ❌ NESSUNA chiave 'classifier' trovata!")

        print("\n🔍 Cerca chiavi 'map':")
        map_keys = [k for k in keys if 'map' in k][:5]
        for k in map_keys: print(f"   FOUND: {k}")

    except Exception as e:
        print(f"❌ Errore lettura checkpoint: {e}")

# --- MAIN ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)
    # Argomenti dummy per compatibilità con lo script slurm esistente
    parser.add_argument("--dataset_name", type=str, default="")
    parser.add_argument("--model_type", type=str, default="")
    args = parser.parse_args()

    check_data(args.test_file, args.image_root)
    check_weights(args.checkpoint_path)
