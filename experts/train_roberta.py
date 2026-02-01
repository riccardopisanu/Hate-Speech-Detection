import os, sys
import pandas as pd
import numpy as np
import torch
import argparse
from torch.utils.data import DataLoader
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from tqdm.auto import tqdm

# --- ARGUMENT PARSER ---
parser = argparse.ArgumentParser()
parser.add_argument("--train_file", type=str, required=True, help="Path to training file")
parser.add_argument("--test_file", type=str, required=True, help="Path to test file")
parser.add_argument("--dataset_name", type=str, default="Generic", help="Name of the dataset")
parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
parser.add_argument("--do_train", type=str, default="True", help="Set to False to skip training")
args = parser.parse_args()

DO_TRAIN = args.do_train.lower() in ('true', '1', 't', 'yes')
MODEL_NAME = "cardiffnlp/twitter-roberta-base-hate-latest"
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print(f"🚀 Job: {args.dataset_name} | Mode: {'FINE-TUNING' if DO_TRAIN else 'BASELINE (No Train)'}")

# --- 1. FUNZIONI DI CARICAMENTO ---
def load_data_robust(file_path):
    print(f"📂 Reading: {file_path}")
    if not os.path.exists(file_path):
        sys.exit(f"❌ File not found: {file_path}")

    try:
        if file_path.endswith('.jsonl'):
            df_raw = pd.read_json(file_path, lines=True)
        else:
            df_raw = pd.read_csv(file_path, sep='\t')
            if len(df_raw.columns) < 2:
                df_raw = pd.read_csv(file_path, sep=',')
    except Exception as e:
        sys.exit(f"❌ Read error: {e}")

    cols_map = {c.lower().strip(): c for c in df_raw.columns}
    df_clean = pd.DataFrame()

    img_col = None
    for cand in ['img', 'image_name', 'file_name', 'id']:
        if cand in cols_map: img_col = cols_map[cand]; break
    
    if img_col: df_clean['image_name'] = df_raw[img_col].astype(str)
    else: df_clean['image_name'] = df_raw.index.astype(str)

    text_col = None
    for cand in ['text transcription', 'text', 'caption', 'sentence']:
        if cand in cols_map: text_col = cols_map[cand]; break
    if not text_col: sys.exit(f"❌ Text column missing")
    df_clean['text'] = df_raw[text_col].fillna("").astype(str)

    label_col = None
    for cand in ['misogynous', 'label']:
        if cand in cols_map: label_col = cols_map[cand]; break
    if not label_col: sys.exit(f"❌ Label column missing")

    def clean_label(x):
        try: return int(x)
        except: return 1 if str(x).lower().strip() in ['offensive', 'hateful', 'misogynous', 'yes', '1'] else 0

    df_clean['label'] = df_raw[label_col].apply(clean_label)
    return df_clean

# --- 2. DATASET ---
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self): return len(self.labels)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.to(DEVICE)

def tokenize_data(df):
    return tokenizer(df['text'].tolist(), padding=True, truncation=True, max_length=128)

# --- 3. TRAINING ---
if DO_TRAIN:
    print("⚙️ Training...")
    df_train_full = load_data_robust(args.train_file)
    train_df, val_df = train_test_split(df_train_full, test_size=0.1, random_state=42, stratify=df_train_full['label'])
    
    train_dataset = CustomDataset(tokenize_data(train_df), train_df['label'].tolist())
    val_dataset = CustomDataset(tokenize_data(val_df), val_df['label'].tolist())
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    optimizer = AdamW(model.parameters(), lr=2e-5)
    num_steps = args.epochs * len(train_loader)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_steps)

    for epoch in range(args.epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            outputs.loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

# --- 4. TESTING ---
print("\n🧪 Testing...")
df_test = load_data_robust(args.test_file)
test_dataset = CustomDataset(tokenize_data(df_test), df_test['label'].tolist())
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

model.eval()
test_preds, test_probs, test_labels = [], [], []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Inference"):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        outputs = model(**batch)
        probs = torch.softmax(outputs.logits, dim=-1)
        test_preds.extend(torch.argmax(outputs.logits, dim=-1).cpu().numpy())
        test_probs.extend(probs[:, 1].cpu().numpy())
        test_labels.extend(batch['labels'].cpu().numpy())

# --- METRICHE BINARIE vs MACRO ---
acc = accuracy_score(test_labels, test_preds)

# Binary: Focus sulla classe 1 (Hate/Misogyny) - QUELLI CHE SERVONO A TE
p_bin, r_bin, f1_bin, _ = precision_recall_fscore_support(test_labels, test_preds, average='binary')

# Macro: Media tra le due classi
p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(test_labels, test_preds, average='macro')

print("\n" + "="*60)
print(f"📊 RESULTS: {args.dataset_name} ({'Fine-Tuned' if DO_TRAIN else 'Baseline'})")
print("="*60)
print(f"   Accuracy:          {acc:.4f}")
print("-" * 30)
print("   >>> BINARY METRICS (Class 1 - Hate/Misogyny) <<<")
print(f"   Precision (Bin):   {p_bin:.4f}  <-- Guarda questo per la tua tesi")
print(f"   Recall (Bin):      {r_bin:.4f}  <-- E questo")
print(f"   F1 Score (Bin):    {f1_bin:.4f}")
print("-" * 30)
print("   >>> MACRO METRICS (Average) <<<")
print(f"   Macro F1:          {f1_macro:.4f}")
print(f"   Macro Precision:   {p_macro:.4f}")
print(f"   Macro Recall:      {r_macro:.4f}")
print("="*60 + "\n")

# SALVA CSV
output_filename = f"results_roberta_{args.dataset_name}_{'FT' if DO_TRAIN else 'Base'}.csv"
pd.DataFrame({
    'image_name': df_test['image_name'],
    'label_true': test_labels,
    'roberta_pred': test_preds,
    'roberta_prob': test_probs
}).to_csv(output_filename, index=False)
