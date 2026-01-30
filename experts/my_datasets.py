import os
import pandas as pd
import torch
import clip
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# --- CONFIGURAZIONE CLIP ---
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

GLOBAL_ERROR_COUNT = 0

def get_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD)
    ])

class Custom_Dataset(Dataset):
    def __init__(self, cfg, root_folder, dataset, label, split='train', image_size=224, info_file=None):
        super(Custom_Dataset, self).__init__()
        self.cfg = cfg
        self.root_folder = root_folder
        self.dataset = dataset
        self.split = split
        self.label = label
        self.image_size = image_size
        self.transform = get_transform(self.image_size)

        # Risoluzione del file di annotazioni
        if info_file is not None:
            self.info_file = info_file
        else:
            if split == 'test': self.info_file = cfg.data.test_file
            elif split == 'dev': self.info_file = cfg.data.val_file
            else: self.info_file = cfg.data.train_file

        print(f"\n[DATASET DEBUG] Inizializzazione Dataset '{split}'")
        print(f"[DATASET DEBUG] File: {self.info_file}")

        # --- CARICAMENTO DATI ---
        try:
            if self.info_file.endswith('.jsonl'):
                self.df = pd.read_json(self.info_file, lines=True)
            else:
                # --- LOGICA SPECIALE PER MAMI ---
                if self.dataset == 'MAMI':
                    # MAMI è un .csv ma separato da TAB (\t)
                    # on_bad_lines='skip' evita crash su righe corrotte
                    self.df = pd.read_csv(self.info_file, sep='\t', on_bad_lines='skip', engine='python')
                else:
                    # Logica standard per altri dataset
                    sep = '\t' if self.info_file.endswith('.tsv') else ','
                    self.df = pd.read_csv(self.info_file, sep=sep)

            # Pulizia nomi colonne
            self.df.columns = [c.strip() for c in self.df.columns]
            print(f"[DATASET DEBUG] Righe caricate: {len(self.df)}")
            print(f"[DATASET DEBUG] Colonne: {list(self.df.columns)}") # Utile per debug
        except Exception as e:
            print(f"[DATASET FATAL ERROR] Errore caricamento file: {e}")
            raise e

        # Tokenizer CLIP
        try:
            _, self.clip_preprocess = clip.load(cfg.model.clip_model, device="cpu", jit=False)
        except:
            pass 
        self.clip_tokenizer = clip.tokenize

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        global GLOBAL_ERROR_COUNT
        row = self.df.iloc[idx]

        # 1. RECUPERO TESTO (Con fix per MAMI)
        text_col = 'text' # Default
        if 'sentence' in row: text_col = 'sentence'
        elif 'Text Transcription' in row: text_col = 'Text Transcription'
        elif 'violenceText Transcription' in row: text_col = 'violenceText Transcription' # Fix per header rotto
        
        txt = str(row[text_col]) if text_col in row and pd.notna(row[text_col]) else "null"

        # 2. RECUPERO NOME FILE IMMAGINE
        image_fn = ""
        if 'img' in row: image_fn = str(row['img']).strip()
        elif 'image_name' in row: image_fn = str(row['image_name']).strip()
        elif 'file_name' in row: image_fn = str(row['file_name']).strip()

        # 3. COSTRUZIONE PERCORSO E CARICAMENTO
        image_path = os.path.join(self.root_folder, image_fn)

        image_tensor = None
        try:
            if not os.path.exists(image_path):
                candidates = [
                    image_path + '.jpg', image_path + '.png', image_path + '.jpeg',
                    image_path.replace('.jpg', '.png'), image_path.replace('.png', '.jpg')
                ]
                found = False
                for c in candidates:
                    if os.path.exists(c):
                        image_path = c
                        found = True
                        break
                if not found:
                     raise FileNotFoundError(f"File non trovato: {image_path}")

            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image)

        except Exception as e:
            if GLOBAL_ERROR_COUNT < 10:
                print(f"[IMAGE ERROR] Riga {idx} | Path: {image_path} | Err: {e}", flush=True)
                GLOBAL_ERROR_COUNT += 1
            # SAFE MODE
            image_tensor = torch.zeros((3, self.image_size, self.image_size))

        # 4. TOKENIZZAZIONE TESTO
        try:
            text_tensor = self.clip_tokenizer(txt, truncate=True).squeeze(0)
        except:
            text_tensor = self.clip_tokenizer("null", truncate=True).squeeze(0)

        # 5. RECUPERO LABEL
        raw_label = row['label'] if 'label' in row else row.get(self.label, 0)

        try:
            label_value = int(raw_label)
        except ValueError:
            txt_label = str(raw_label).lower().strip()
            if txt_label in ['offensive', 'hateful', 'misogynous', 'yes', '1']:
                label_value = 1
            else:
                label_value = 0

        return {
            'image': image_tensor,
            'text': text_tensor,
            'label': label_value,
            'idx_meme': image_fn,
            'origin_text': txt
        }

def load_dataset(cfg, split='train'):
    info_file = None
    if split == 'train': info_file = cfg.data.train_file
    elif split in ['val', 'dev']: info_file = cfg.data.val_file
    elif split == 'test': info_file = cfg.data.test_file

    return Custom_Dataset(
        cfg=cfg,
        root_folder=cfg.data.root_dir,
        dataset=cfg.data.dataset_name,
        label=cfg.data.label,
        split=split,
        info_file=info_file,
        image_size=cfg.data.image_size
    )
