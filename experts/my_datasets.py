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

# Contatore globale per limitare lo spam nel log
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

        # Logica prioritaria per il file CSV
        if info_file is not None:
            self.info_file = info_file
        else:
            # Fallback (non dovrebbe servire se load_dataset è corretto)
            if split == 'test':
                self.info_file = cfg.data.test_file
            elif split == 'dev':
                self.info_file = cfg.data.val_file
            else:
                self.info_file = cfg.data.train_file

        # --- DEBUG INIZIALE ---
        print(f"\n[DATASET DEBUG] Inizializzazione Dataset '{split}'")
        print(f"[DATASET DEBUG] CSV File: {self.info_file}")
        print(f"[DATASET DEBUG] Root Folder: {self.root_folder}")

        # Caricamento CSV con gestione separatori
        try:
            sep = '\t' if self.info_file.endswith('.tsv') else ','
            self.df = pd.read_csv(self.info_file, sep=sep)
            # Rimuove spazi dai nomi delle colonne
            self.df.columns = [c.strip() for c in self.df.columns]
            print(f"[DATASET DEBUG] Righe nel CSV: {len(self.df)}")
            print(f"[DATASET DEBUG] Colonne trovate: {list(self.df.columns)}")
        except Exception as e:
            print(f"[DATASET FATAL ERROR] Impossibile leggere il CSV: {e}")
            raise e

        # Tokenizer CLIP
        _, self.clip_preprocess = clip.load(cfg.model.clip_model, device="cpu", jit=False)
        self.clip_tokenizer = clip.tokenize

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        global GLOBAL_ERROR_COUNT
        row = self.df.iloc[idx]

        # 1. Recupero testo
        txt = str(row['text']) if 'text' in row and pd.notna(row['text']) else 'null'

        # 2. Recupero nome file immagine
        # Prova colonne comuni se 'img' non esiste
        if 'img' in row:
            image_fn = str(row['img']).strip()
        elif 'file_name' in row:
            image_fn = str(row['file_name']).strip()
        else:
            print(f"[DATASET ERROR] Colonna immagine mancante alla riga {idx}")
            return None

        # 3. Costruzione Percorso
        image_path = os.path.join(self.root_folder, image_fn)

        # 4. Check Esistenza File (Logica Robusta)
        # Se il file non c'è, prova ad aggiungere l'estensione o a cercarlo senza
        if not os.path.exists(image_path):
            # Tentativo 1: Forse manca .jpg?
            if os.path.exists(image_path + '.jpg'):
                image_path = image_path + '.jpg'
            # Tentativo 2: Forse ha .jpg nel CSV ma il file è .png?
            elif os.path.exists(image_path.replace('.jpg', '.png')):
                image_path = image_path.replace('.jpg', '.png')

        # 5. Caricamento Immagine
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image)
        except Exception as e:
            # --- QUI AVVIENE LA MAGIA DEL DEBUG ---
            if GLOBAL_ERROR_COUNT < 10: # Stampa solo i primi 10 errori
                print(f"\n[MISSING FILE ERROR] Riga {idx}")
                print(f" -> Cercato in: '{image_path}'")
                print(f" -> Root Folder: '{self.root_folder}'")
                print(f" -> Nome da CSV: '{image_fn}'")
                print(f" -> Errore sistema: {e}")
                GLOBAL_ERROR_COUNT += 1
            return None

        # 6. Tokenizzazione
        try:
            text_tensor = self.clip_tokenizer(txt, truncate=True).squeeze(0)
        except:
            text_tensor = self.clip_tokenizer("null", truncate=True).squeeze(0)

        # 7. Label (Gestione Test Set Cieco)
        label_value = -1
        if 'label' in row:
            label_value = row['label']
        elif self.label in row:
            label_value = row[self.label]

        return {
            'image': image_tensor,
            'text': text_tensor,
            'label': label_value,
            'idx_meme': image_fn,
            'origin_text': txt
        }

# Manteniamo il Collator originale se presente nel file, o uno semplice
class Custom_Collator(object):
    def __init__(self, cfg):
        self.cfg = cfg
    def __call__(self, batch):
        # Filtra i None
        batch = list(filter(lambda x: x is not None, batch))
        if not batch:
            return None
        return torch.utils.data.dataloader.default_collate(batch)

# Funzione load_dataset semplificata
def load_dataset(cfg, split='train'):
    info_file = None
    if split == 'train': info_file = cfg.data.train_file
    elif split == 'dev': info_file = cfg.data.val_file
    elif split == 'test': info_file = cfg.data.test_file

    dataset = Custom_Dataset(
        cfg=cfg,
        root_folder=cfg.data.root_dir,
        dataset=cfg.data.dataset_name,
        label=cfg.data.label,
        split=split,
        info_file=info_file
    )
    return dataset
