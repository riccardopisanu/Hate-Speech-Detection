import os
import torch
import torch.nn.functional as F
import pandas as pd
import argparse
import clip
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import numpy as np

# ==============================================================================
# CONFIGURAZIONE DEI PROMPT (LE DOMANDE CHE FACCIAMO A CLIP)
# ==============================================================================
# CLIP Zero-Shot funziona confrontando l'immagine con queste descrizioni.
# Più frasi usiamo, più robusta è la predizione.

CLASS_0_PROMPTS = [ # NON ODIO
    "a normal image",
    "a harmless meme",
    "a funny meme",
    "a benign image",
    "a photo of something non-offensive",
    "a clean meme"
]

CLASS_1_PROMPTS = [ # ODIO
    "a hateful image",
    "an offensive meme",
    "a racist or sexist meme",
    "hate speech",
    "a photo of something toxic",
    "a disgusting meme"
]

# ==============================================================================
# DATASET & UTILS (Uguale al robust loader che abbiamo già)
# ==============================================================================
class SimpleDataset(Dataset):
    def __init__(self, df, img_dir, preprocess):
        self.df = df
        self.img_dir = img_dir
        self.preprocess = preprocess
        self._check_first_image()

    def _check_first_image(self):
        if len(self.df) > 0:
            first_row = self.df.iloc[0]
            img_name = str(first_row['image_name'])
            print(f"🔍 DEBUG PATH: {os.path.join(self.img_dir, img_name)}")

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = str(row['image_name'])
        
        candidates = [
            os.path.join(self.img_dir, img_name),
            os.path.join(self.img_dir, os.path.basename(img_name)),
            os.path.join(self.img_dir, img_name + ".jpg"),
            os.path.join(self.img_dir, img_name + ".png")
        ]
        path = next((p for p in candidates if os.path.exists(p)), None)
        if not path: return None 

        try: image = self.preprocess(Image.open(path).convert("RGB"))
        except: return None 
        
        return {'image': image, 'label': int(row['label']), 'image_name': img_name}

def load_data(file_path):
    print(f"📂 Reading: {file_path}")
    if file_path.endswith('.jsonl'): df = pd.read_json(file_path, lines=True)
    elif file_path.endswith('.tsv'): df = pd.read_csv(file_path, sep='\t')
    else: df = pd.read_csv(file_path)
    
    cols = {c.lower().strip(): c for c in df.columns}
    img = next((cols[c] for c in ['file_name', 'filename', 'img', 'image_name', 'id'] if c in cols), 'img')
    lbl = next((cols[c] for c in ['label', 'misogynous'] if c in cols), 'label')
    
    df['image_name'] = df[img].astype(str)
    
    def clean_lbl(x):
        try: return int(x)
        except: return 1 if str(x).lower().strip() in ['yes', '1', 'offensive', 'misogynous'] else 0
    df['label'] = df[lbl].apply(clean_lbl)
    return df

# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="Generic")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 ZERO-SHOT Testing on {args.dataset_name}")
    
    # 1. Carica CLIP Originale (Pretrained)
    model, preprocess = clip.load("ViT-L/14", device=device)
    model.eval()

    # 2. Prepara il Dataset
    df = load_data(args.test_file)
    loader = DataLoader(SimpleDataset(df, args.image_root, preprocess), 
                        batch_size=32, num_workers=4, shuffle=False, 
                        collate_fn=lambda x: torch.utils.data.dataloader.default_collate([i for i in x if i]))

    # 3. Encoding dei Text Prompts (Classi)
    print("📝 Encoding Zero-Shot Prompts...")
    with torch.no_grad():
        tok_0 = clip.tokenize(CLASS_0_PROMPTS).to(device)
        tok_1 = clip.tokenize(CLASS_1_PROMPTS).to(device)
        
        # Facciamo la media degli embedding per avere un prototipo robusto
        emb_0 = model.encode_text(tok_0).mean(dim=0, keepdim=True)
        emb_1 = model.encode_text(tok_1).mean(dim=0, keepdim=True)
        
        # Normalizzazione
        emb_0 /= emb_0.norm(dim=-1, keepdim=True)
        emb_1 /= emb_1.norm(dim=-1, keepdim=True)
        
        # Stack delle classi [2, 768]
        text_features = torch.cat([emb_0, emb_1], dim=0)

    # 4. Inference Loop
    print("🧪 Inference Loop...")
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(loader):
            images = batch['image'].to(device)
            labels = batch['label']
            
            # Encode Image
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            # Similarità (Image @ Text.T) -> [Batch, 2]
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            
            preds = similarity.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    # 5. Risultati
    acc = accuracy_score(all_labels, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
    
    print("\n" + "="*60)
    print(f"📊 RESULTS: CLIP ZERO-SHOT (Pretrained) on {args.dataset_name}")
    print("="*60)
    print(f"   Accuracy:          {acc:.4f}")
    print(f"   Binary Recall:     {r:.4f}")
    print(f"   Binary Precision:  {p:.4f}")
    print(f"   F1 Score:          {f1:.4f}")
    print("="*60 + "\n")
    
    pd.DataFrame({'label_true': all_labels, 'zeroshot_pred': all_preds}).to_csv(f"results_zeroshot_{args.dataset_name}.csv", index=False)
