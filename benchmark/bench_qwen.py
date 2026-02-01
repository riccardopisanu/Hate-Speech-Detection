import os, sys, argparse, json, time
import pandas as pd
from collections import defaultdict
import torch
import re
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- CONFIGURAZIONE MODELLO ---
MODEL_PATH = "/home/rpisanu/models/Qwen-VL-Chat"

# --- CONFIGURAZIONE DATASET E PROMPT ---
CONFIGS = {
    "MAMI": {
        "base_dir": "/home/rpisanu/MAMI/MAMI_DATASET",
        "file_path": "test.tsv",
        "img_dir": "test_images_clean/test",
        "type": "tsv",
        "prompts": [
            "Considering the visual and textual content together, is this offensive or discriminatory? Answer 1 or 0.",
            "Evaluate the meme’s image and caption together. If it expresses sexist or hateful content, output 1. If not, output 0.",
            "Analyze the meme as a whole (image + text). If it contains misogynistic or sexist stereotypes, output 1. If it does not, output 0. Respond with only one character.",
            "Using the combination of image and text, decide if the meme promotes hate or discrimination. Reply ONLY with 1 or 0.",
            "Classify this meme: 1 = hateful or offensive toward women, 0 = not hateful. Output only the digit.",
            "When interpreting the meme’s image and text jointly, does it demean or stereotype women? Answer 1 if yes, 0 if no."
        ]
    },
    "MultiOFF": {
        "base_dir": "/home/rpisanu/MultiOFF/MultiOFF_Dataset",
        "file_path": "Split Dataset/Testing_meme_dataset.csv", # Testing set come da tuo script
        "img_dir": "Labelled Images",
        "type": "csv",
        "prompts": [
            "Considering both the image and the text of this meme, is it offensive? Answer only 1 for offensive, 0 for non-offensive.",
            "Look at the meme's visual and textual elements together. Does it contain offensive content? Output ONLY 1 if offensive, 0 if not.",
            "Analyze this meme using both its image and textual content. Determine if it is hateful: reply with 1 for hateful, 0 for not hateful.",
            "Using the combination of image and text, classify this meme as offensive or not: output strictly 1 for offensive, 0 for non-offensive.",
            "Evaluate this meme by considering BOTH the image and the text. Respond with a single digit: 1 if offensive, 0 if not offensive.",
            "Assess the meme holistically (image + text). If any element is offensive return 1; if neither is offensive return 0. Reply with only the digit.",
            "Determine whether this meme is offensive using its visual and textual content. Answer EXACTLY '1' for offensive or '0' for non-offensive, nothing else.",
            "Does the meme contain offensive or hateful content when considering both the picture and text? Output only 1 (offensive) or 0 (not offensive)."
        ]
    },
    "MMHS150K": {
        "base_dir": "/home/rpisanu/MMHS150K",
        "file_path": "MMHS150K_GT.json",
        "split_file": "splits/test_ids.txt",
        "img_dir": "img_resized",
        "type": "mmhs_json",
        "prompts": [
            "Analyze both the image and the text. Does this express any form of hate speech? Reply with 1=yes or 0=no.",
            "Considering the visual and textual content together, is this offensive or discriminatory? Answer 1 or 0.",
            "Does this post promote violence or dehumanize a specific group of people? Answer 1 (Yes) or 0 (No). Do not flag general offensive language.",
            "Classify as 1 (Hate) ONLY if the content is clearly hateful targeting a protected group. If it is just rude, sarcastic, or uses slang, classify as 0 (Safe)."
        ]
    },
    "HatefulMemes": {
        "base_dir": "/home/rpisanu/Hateful/hateful_memes/data",
        "file_path": "dev.jsonl",
        "img_dir": "img",
        "type": "jsonl",
        "prompts": [
            "Is this meme hateful? Answer 1 for hateful, 0 for not hateful.",
            "Considering both the image and the text, is this meme offensive? 1=yes, 0=no.",
            "Analyze the meme and decide: 1=hateful, 0=not hateful.",
            "Does this meme contain hate speech or offensive content? Answer 1=yes, 0=no.",
            "Evaluate the meme’s image and text together. Is it hateful? Output only 1 for hateful, 0 for not.",
            "Look at both the image and the caption. If the meme expresses hate or stereotypes, reply 1. Otherwise reply 0.",
            "Does the meme promote hate, discrimination, or offensive stereotypes? Respond with 1 if yes, 0 if no.",
            "Is the meme offensive or hateful when considering both modalities? Return only 1 or 0.",
            "Classify this meme: 1 = hateful or offensive, 0 = not hateful. Output only the digit.",
            "Does the meme contain harmful or derogatory content? Answer with a single digit: 1 for yes, 0 for no."
        ]
    }
}

# ─────────────── UTILS ───────────────
def find_image_robust(base_img_dir, img_identifier):
    # Logica robusta per trovare immagini
    path = os.path.join(base_img_dir, img_identifier)
    if os.path.exists(path): return path
    
    # Caso sottocartella img/ già presente nel nome (HatefulMemes)
    if "img/" in img_identifier:
        basename = os.path.basename(img_identifier)
        path = os.path.join(base_img_dir, basename)
        if os.path.exists(path): return path
        
    # Estensioni
    name_no_ext = os.path.splitext(os.path.basename(img_identifier))[0]
    for ext in [".jpg", ".JPG", ".png", ".PNG", ".jpeg"]:
        path = os.path.join(base_img_dir, name_no_ext + ext)
        if os.path.exists(path): return path
    
    return None

def map_prediction(pred_text: str) -> int:
    s = str(pred_text).lower().strip()
    
    # Qwen Refusals
    if "cannot" in s or "sorry" in s: return 0
    if "not offensive" in s or "non-offensive" in s or "non-hateful" in s: return 0
    if "offensive" in s or "hateful" in s: return 1
    
    nums = re.findall(r"\b(0|1)\b", s)
    if nums: return int(nums[-1])
    
    if "yes" in s: return 1
    if "no" in s: return 0
    
    return -1

# ─────────────── DATA LOADER ───────────────
def load_dataset_data(name):
    cfg = CONFIGS[name]
    full_path = os.path.join(cfg["base_dir"], cfg["file_path"])
    print(f"📂 Loading {name} from {full_path}...")
    data_items = []

    if cfg["type"] == "tsv": # MAMI
        df = pd.read_csv(full_path, sep="\t")
        # Normalize columns
        cols = {c.lower().strip(): c for c in df.columns}
        img_col = cols.get("file_name") or cols.get("image_name")
        lbl_col = cols.get("misogynous") or cols.get("label")
        txt_col = cols.get("text_transcription") or cols.get("text")
        
        for _, row in df.iterrows():
            data_items.append({
                "image_name": str(row[img_col]).strip(),
                "text": str(row[txt_col]),
                "label": int(row[lbl_col])
            })

    elif cfg["type"] == "csv": # MultiOFF
        df = pd.read_csv(full_path)
        for _, row in df.iterrows():
            if pd.isna(row.get("image_name")): continue
            lbl_str = str(row.get("label", "")).strip().lower()
            label = 1 if lbl_str == "offensive" else 0
            data_items.append({
                "image_name": str(row.get("image_name")),
                "text": str(row.get("sentence")),
                "label": label
            })

    elif cfg["type"] == "jsonl": # Hateful Memes
        with open(full_path, 'r') as f:
            for line in f:
                obj = json.loads(line)
                data_items.append({
                    "image_name": obj["img"],
                    "text": obj["text"],
                    "label": int(obj["label"])
                })

    elif cfg["type"] == "mmhs_json": # MMHS150K
        with open(full_path, "r") as f: gt_data = json.load(f)
        split_path = os.path.join(cfg["base_dir"], cfg["split_file"])
        with open(split_path, "r") as f: test_ids = set(line.strip() for line in f if line.strip())
        
        for tweet_id, entry in gt_data.items():
            if tweet_id not in test_ids: continue
            labels = entry.get("labels", [])
            if not labels: continue
            maj_label = max(set(labels), key=labels.count)
            label = 1 if maj_label == 1 else 0 # 1=Hate in MMHS raw
            
            data_items.append({
                "image_name": f"{tweet_id}",
                "text": entry.get("tweet_text", ""),
                "label": label
            })

    return data_items, os.path.join(cfg["base_dir"], cfg["img_dir"])

# ─────────────── MAIN ───────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=CONFIGS.keys())
    parser.add_argument("--out_dir", type=str, default="results_benchmarks")
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cfg = CONFIGS[args.dataset]

    # 1. LOAD MODEL (QWEN SPECIFIC)
    print(f"🚀 Loading Qwen-VL-Chat from {MODEL_PATH}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
        if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.eod_id
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, device_map="auto", trust_remote_code=True, bf16=True, local_files_only=True
        ).eval()

        # FIX QWEN GENERATION CONFIG
        model.generation_config.chat_format = "chatml"
        model.generation_config.max_window_size = 4096
        model.generation_config.max_length = 4096
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        
        print("✅ Qwen-VL Loaded & Patched.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)

    # 2. LOAD DATA
    dataset, img_root = load_dataset_data(args.dataset)
    if args.sample: dataset = dataset[:args.sample]
    print(f"📊 Dataset loaded: {len(dataset)} samples. Img Root: {img_root}")

    # 3. INFERENCE
    results = []
    metrics = defaultdict(lambda: {"y_true": [], "y_pred": []})
    start_time = time.time()
    
    for idx, item in enumerate(dataset):
        if idx % 50 == 0: print(f"   [{idx}/{len(dataset)}] Processing...")
        
        img_path = find_image_robust(img_root, item["image_name"])
        if not img_path: continue

        for p in cfg["prompts"]:
            try:
                # Qwen Specific Query Format
                query = tokenizer.from_list_format([
                    {"image": img_path},
                    {"text": f"Text: {item['text']}\n\n{p}"}
                ])
                
                with torch.no_grad():
                    response, _ = model.chat(
                        tokenizer, query=query, history=None,
                        max_new_tokens=50, do_sample=False
                    )
                
                pred = map_prediction(response)
                
                results.append({
                    "id": item["image_name"],
                    "true_label": item["label"],
                    "pred": pred,
                    "response": response,
                    "prompt": p
                })
                
                if pred != -1:
                    metrics[p]["y_true"].append(item["label"])
                    metrics[p]["y_pred"].append(pred)
                    
            except Exception as e:
                print(f"❌ Error {item['image_name']}: {e}")

    # 4. SAVE
    out_csv = os.path.join(args.out_dir, f"results_qwen_{args.dataset}.csv")
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"✅ Saved results to {out_csv}")

    # 5. METRICS
    print(f"\n🏆 RESULTS: {args.dataset} (Qwen-VL)")
    print("="*100)
    print(f"{'PROMPT (First 40 chars)':<40} | {'ACC':<6} | {'F1 MAC':<6} | {'F1 BIN':<6} | {'REC':<6} | {'PREC':<6}")
    print("-" * 100)

    for p, d in metrics.items():
        if d["y_pred"]:
            acc = accuracy_score(d["y_true"], d["y_pred"])
            p_bin = precision_score(d["y_true"], d["y_pred"], average='binary', zero_division=0)
            r_bin = recall_score(d["y_true"], d["y_pred"], average='binary', zero_division=0)
            f1_bin = f1_score(d["y_true"], d["y_pred"], average='binary', zero_division=0)
            f1_mac = f1_score(d["y_true"], d["y_pred"], average='macro', zero_division=0)
            
            print(f"{p[:40]:<40} | {acc:.4f} | {f1_mac:.4f} | {f1_bin:.4f} | {r_bin:.4f} | {p_bin:.4f}")
    print("="*100 + "\n")

if __name__ == "__main__":
    main()
