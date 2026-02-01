import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import argparse
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import clip 
import numpy as np

# ==============================================================================
# 0. CONFIGURAZIONE & UTILS
# ==============================================================================
class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def get_dummy_config():
    return DotDict({
        'model': DotDict({
            'clip_model': "ViT-L/14",
            'drop_probs': [0.1, 0.1, 0.1],
            'unmapped_dim': 1024,
            'num_mapping_layers': 1,
            'num_pre_output_layers': 3,
            'class_names': ['non_misogynous', 'misogynous']
        }),
        'data': DotDict({'num_classes': 2}),
        'map_dim': 1024,
        'clip_variant': "ViT-L/14"
    })

# ==============================================================================
# 1. CLASSI DEL MODELLO
# ==============================================================================
class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.fc(x) + x

class LinearProjection(nn.Module):
    def __init__(self, dim_in, dim_out, num_layers=2, drop_probs=[0.1]*3):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.extend([nn.Linear(dim_in, dim_out), nn.ReLU(), nn.Dropout(drop_probs[0])])
            dim_in = dim_out 
        self.proj = nn.Sequential(*layers)
    def forward(self, x): return self.proj(x)

class CLIP_Text(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model
    def forward(self, x): return self.clip_model.encode_text(x)

class MemeCLIP(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(ignore=['cfg']) 
        
        self.map_dim = cfg.map_dim
        num_classes = cfg.data.num_classes
        
        self.clip_model, _ = clip.load(cfg.model.clip_model, device="cpu", jit=False)
        self.clip_model.float()
        self.clip_model.visual.proj = None 
        visual_dim = 1024 

        self.text_encoder = CLIP_Text(self.clip_model)

        self.img_adapter = Adapter(self.map_dim, 4).to(self.clip_model.dtype)
        self.text_adapter = Adapter(self.map_dim, 4).to(self.clip_model.dtype)
        
        self.image_map = LinearProjection(visual_dim, self.map_dim, 1, [0.1]*3)
        self.text_map = LinearProjection(768, self.map_dim, 1, [0.1]*3)
        
        pre_output_layers = [nn.Dropout(0.1)]
        for _ in range(cfg.model.num_pre_output_layers):
            pre_output_layers.extend([nn.Linear(self.map_dim, self.map_dim), nn.ReLU(), nn.Dropout(0.1)])
        self.pre_output = nn.Sequential(*pre_output_layers)
        
        self.classifier = nn.Linear(self.map_dim, num_classes)

    def forward(self, image, text):
        with torch.no_grad():
            img_feat = self.clip_model.encode_image(image).float()
            txt_feat = self.text_encoder(text).float()

        img_feat = self.image_map(img_feat)
        txt_feat = self.text_map(txt_feat)
        img_feat = self.img_adapter(img_feat)
        txt_feat = self.text_adapter(txt_feat)

        combined_feat = (img_feat + txt_feat) / 2.0
        combined_feat = self.pre_output(combined_feat)
        return self.classifier(combined_feat)

class CLIP_MLP_Classifier(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(ignore=['cfg']) 
        self.clip_model, _ = clip.load(self.cfg.model.clip_model, device="cpu", jit=False)
        self.embed_dim = self.clip_model.visual.output_dim
        self.mlp = nn.Sequential(
            nn.Linear(self.embed_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, self.cfg.data.num_classes)
        )
    def forward(self, images, texts):
        with torch.no_grad():
            i = self.clip_model.encode_image(images.float())
            t = self.clip_model.encode_text(texts)
        i = i / i.norm(dim=-1, keepdim=True)
        t = t / t.norm(dim=-1, keepdim=True)
        return self.mlp(torch.cat((i, t), dim=1).float())

# ==============================================================================
# 2. CARICAMENTO DATI (DIAGNOSTICO)
# ==============================================================================
class MultimodalDataset(Dataset):
    def __init__(self, df, img_dir, preprocess):
        self.df = df
        self.img_dir = img_dir
        self.preprocess = preprocess
        self.tokenizer = clip.tokenize
        self._check_first_image()

    def _check_first_image(self):
        if len(self.df) > 0:
            first_row = self.df.iloc[0]
            img_name = str(first_row['image_name'])
            print(f"\n🔍 DEBUG IMAGE PATH:")
            print(f"   Root: {self.img_dir}")
            print(f"   Filename: {img_name}")
            cand = os.path.join(self.img_dir, img_name)
            if os.path.exists(cand): print(f"   ✅ FOUND AT: {cand}")
            else: print(f"   ❌ NOT FOUND AT: {cand}")
            print("-" * 30)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = str(row['image_name'])
        
        candidates = [
            os.path.join(self.img_dir, img_name),
            os.path.join(self.img_dir, os.path.basename(img_name)),
            os.path.join(self.img_dir, img_name + ".png"),
            os.path.join(self.img_dir, img_name + ".jpg")
        ]
        final_path = None
        for p in candidates:
            if os.path.exists(p):
                final_path = p
                break
        
        if final_path is None: return None 

        try: image = self.preprocess(Image.open(final_path).convert("RGB"))
        except: return None 
        
        # Gestione robusta del testo mancante
        if 'text' in row:
            txt_raw = str(row['text'])
        else:
            txt_raw = " " # Empty string fallback
            
        text_token = self.tokenizer(txt_raw[:77], truncate=True).squeeze(0)
        return {'image': image, 'text': text_token, 'labels': torch.tensor(int(row['label'])), 'image_name': img_name}

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch) if batch else None

def load_data_robust(file_path):
    print(f"📂 Reading: {file_path}")
    if file_path.endswith('.jsonl'): df = pd.read_json(file_path, lines=True)
    elif file_path.endswith('.tsv'): df = pd.read_csv(file_path, sep='\t')
    else: df = pd.read_csv(file_path)
        
    cols = {c.lower().strip(): c for c in df.columns}
    
    img_cands = ['file_name', 'filename', 'img', 'image_name', 'id']
    img = next((cols[c] for c in img_cands if c in cols), None)
    
    txt_cands = ['text', 'sentence', 'text transcription', 'caption', 'clean_text', 'content', 'full_text']
    txt = next((cols[c] for c in txt_cands if c in cols), None)
    
    lbl_cands = ['label', 'misogynous']
    lbl = next((cols[c] for c in lbl_cands if c in cols), None)
    
    if img is None:
        print("❌ CRITICAL ERROR: Image column not found!")
        exit(1)
    if txt is None:
        print("⚠️ WARNING: Text column not found! Will use empty strings.")
        df['text'] = " "
    else:
        df['text'] = df[txt].fillna("").astype(str)

    if lbl is None:
        print("⚠️ WARNING: Label column not found. Assuming dummy labels (0).")
        df['label'] = 0
    else:
        def clean_lbl(x):
            try: return int(x)
            except: return 1 if str(x).lower().strip() in ['yes', '1', 'offensive', 'misogynous'] else 0
        df['label'] = df[lbl].apply(clean_lbl)

    df['image_name'] = df[img].astype(str)
    
    print(f"   ✅ Using: Image='{img}', Text='{txt if txt else 'DUMMY'}', Label='{lbl if lbl else 'DUMMY'}'")
    return df

# ==============================================================================
# 3. SMART LOADER (FIXED: weights_only=False)
# ==============================================================================
def smart_load_weights(model, checkpoint_path, device):
    print(f"🔧 Smart Loading from: {checkpoint_path}")
    
    # FIX: Aggiunto weights_only=False per evitare errore UnpicklingError con YACS/Config
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        # Fallback per vecchie versioni di PyTorch che non hanno weights_only
        ckpt = torch.load(checkpoint_path, map_location=device)

    state_dict = {k.replace('model.', ''): v for k, v in ckpt['state_dict'].items()}
    new_state_dict = {}
    for k, v in state_dict.items():
        if k in model.state_dict(): new_state_dict[k] = v
        else:
            k_fix = k.replace('text_encoder.', 'text_encoder.clip_model.')
            if k_fix in model.state_dict(): new_state_dict[k_fix] = v
    msg = model.load_state_dict(new_state_dict, strict=False)
    print(f"✅ Loaded weights. Missing: {len(msg.missing_keys)}")

# ==============================================================================
# 4. MAIN
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="Generic")
    parser.add_argument("--model_type", type=str, required=True)
    args = parser.parse_args()

    print(f"🚀 Testing {args.model_type} on {args.dataset_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = get_dummy_config()

    if "memeclip" in args.model_type.lower(): model = MemeCLIP(cfg)
    else: model = CLIP_MLP_Classifier(cfg)

    smart_load_weights(model, args.checkpoint_path, device)
    model.to(device)
    model.eval()
    
    _, preprocess = clip.load("ViT-L/14", device=device, jit=False)
    df = load_data_robust(args.test_file)
    loader = DataLoader(MultimodalDataset(df, args.image_root, preprocess), 
                        batch_size=32, num_workers=4, shuffle=False, collate_fn=collate_fn)

    all_preds, all_probs, all_labels, all_names = [], [], [], []
    print("🧪 Inference Loop...")
    with torch.no_grad():
        for batch in tqdm(loader):
            if batch is None: continue
            images, texts = batch['image'].to(device), batch['text'].to(device)
            logits = model(images, texts)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(batch['labels'].numpy())
            all_names.extend(batch['image_name'])

    if len(all_labels) == 0:
        print("❌ ERROR: No valid predictions.")
        exit(1)

    acc = accuracy_score(all_labels, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
    pm, rm, f1m, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)

    print("\n" + "="*60)
    print(f"📊 RESULTS: {args.model_type} on {args.dataset_name}")
    print("="*60)
    print(f"   Accuracy:          {acc:.4f}")
    print("-" * 30)
    print("   >>> BINARY METRICS (Class 1) <<<")
    print(f"   Precision:         {p:.4f}")
    print(f"   Recall:            {r:.4f}")
    print(f"   F1 Score:          {f1:.4f}")
    print("-" * 30)
    print("   >>> MACRO METRICS <<<")
    print(f"   Macro F1:          {f1m:.4f}")
    print("="*60 + "\n")

    pd.DataFrame({
        'image_name': all_names, 'label_true': all_labels,
        f'{args.model_type}_pred': all_preds, f'{args.model_type}_prob': all_probs
    }).to_csv(f"results_{args.model_type}_{args.dataset_name}.csv", index=False)
    print("💾 Saved CSV.")
