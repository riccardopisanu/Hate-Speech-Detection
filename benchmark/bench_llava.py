import os, sys, argparse, json, time, re
import pandas as pd
from collections import defaultdict
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from PIL import Image
from transformers import AutoProcessor, LlavaNextForConditionalGeneration

# --- CONFIGURAZIONE MODELLO ---
MODEL_PATH = "/home/rpisanu/models/llava-v1.6-mistral-7b"

# --- CONFIGURAZIONE DATASET E PROMPT ZERO-SHOT (NO CoT) ---
CONFIGS = {
    "MAMI": {
        "base_dir": "/home/rpisanu/MAMI/MAMI_DATASET",
        "file_path": "test.tsv",
        "img_dir": "test_images_clean/test",
        "type": "tsv",
        "prompts": [
            "Is this meme misogynistic? Answer 1 (Yes) or 0 (No).",
            "Does this image and text combination express hate against women? Answer 1 or 0.",
            "Classify this meme: 1 = misogynous, 0 = not misogynous. Output only the digit.",
            "Is this content offensive toward women? Answer 1 or 0.",
            "Look at the meme. Is it sexist? Answer only 1 (Yes) or 0 (No)."
        ]
    },
    "MultiOFF": {
        "base_dir": "/home/rpisanu/MultiOFF/MultiOFF_Dataset",
        "file_path": "Split Dataset/Testing_meme_dataset.csv",
        "img_dir": "Labelled Images",
        "type": "csv",
        "prompts": [
            "Is this meme offensive? Answer 1 (Yes) or 0 (No).",
            "Classify this content: 1 = Offensive, 0 = Non-offensive.",
            "Does this meme contain hateful or inappropriate content? Answer 1 or 0.",
            "Is this meme safe or offensive? Answer 1 for offensive, 0 for safe.",
            "Analyze the image and text. Is it offensive? Output strictly 1 or 0."
        ]
    },
    "MMHS150K": {
        "base_dir": "/home/rpisanu/MMHS150K",
        "file_path": "MMHS150K_GT.json",
        "split_file": "splits/test_ids.txt",
        "img_dir": "img_resized",
        "type": "mmhs_json",
        "prompts": [
            "Is this tweet image hateful? Answer 1 (Yes) or 0 (No).",
            "Does this content constitute Hate Speech? Answer 1 or 0.",
            "Classify this post: 1 = Hate Speech, 0 = Not Hate Speech.",
            "Is this offensive or racist? Answer 1 (Hate) or 0 (Safe).",
            "Detect hate speech in this image and text. Output 1 for Hate, 0 for Safe."
        ]
    },
    "HatefulMemes": {
        "base_dir": "/home/rpisanu/Hateful/hateful_memes/data",
        "file_path": "dev.jsonl",
        "img_dir": "img",
        "type": "jsonl",
        "prompts": [
            "Is this meme hateful? Answer 1 (Yes) or 0 (No).",
            "Classify this meme: 1 = Hateful, 0 = Not Hateful.",
            "Does this meme contain hate speech? Answer 1 or 0.",
            "Is this content offensive or hateful? Output only the label 1 or 0.",
            "Is this a hateful meme targeting a protected group? Answer 1 or 0."
        ]
    }
}

# ─────────────── UTILS ───────────────
def find_image_robust(base_img_dir, img_identifier):
    path = os.path.join(base_img_dir, img_identifier)
    if os.path.exists(path): return path
    if "img/" in img_identifier:
        basename = os.path.basename(img_identifier)
        path = os.path.join(base_img_dir, basename)
        if os.path.exists(path): return path
    name_no_ext = os.path.splitext(os.path.basename(img_identifier))[0]
    for ext in [".jpg", ".JPG", ".png", ".PNG", ".jpeg"]:
        path = os.path.join(base_img_dir, name_no_ext + ext)
        if os.path.exists(path): return path
    return None

def map_prediction(pred_text: str) -> int:
    s = str(pred_text).lower().strip()
    
    # Keyword priority
    if "not offensive" in s or "non-offensive" in s or "not hateful" in s or "not misogynous" in s: return 0
    if "offensive" in s or "hateful" in s or "misogynous" in s or "sexist" in s: return 1
    
    # LLaVA chat format specific
    if "label: 0" in s: return 0
    if "label: 1" in s: return 1

    # Numeric
    nums = re.findall(r"\b(0|1)\b", s)
    if nums: return int(nums[-1])
    
    # Yes/No
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
        cols = {c.lower().strip(): c for c in df.columns}
        img_col = cols.get("file_name") or cols.get("image_name")
        lbl_col = cols.get("misogynous") or cols.get("label")
        txt_col = cols.get("text_transcription") or cols.get("text")
        for _, row in df.iterrows():
            data_items.append({"image_name": str(row[img_col]).strip(), "text": str(row[txt_col]), "label": int(row[lbl_col])})

    elif cfg["type"] == "csv": # MultiOFF
        df = pd.read_csv(full_path)
        for _, row in df.iterrows():
            if pd.isna(row.get("image_name")): continue
            lbl_str = str(row.get("label", "")).strip().lower()
            label = 1 if lbl_str == "offensive" else 0
            data_items.append({"image_name": str(row.get("image_name")), "text": str(row.get("sentence")), "label": label})

    elif cfg["type"] == "jsonl": # Hateful Memes
        with open(full_path, 'r') as f:
            for line in f:
                obj = json.loads(line)
                data_items.append({"image_name": obj["img"], "text": obj["text"], "label": int(obj["label"])})

    elif cfg["type"] == "mmhs_json": # MMHS150K
        with open(full_path, "r") as f: gt_data = json.load(f)
        split_path = os.path.join(cfg["base_dir"], cfg["split_file"])
        with open(split_path, "r") as f: test_ids = set(line.strip() for line in f if line.strip())
        for tweet_id, entry in gt_data.items():
            if tweet_id not in test_ids: continue
            labels = entry.get("labels", [])
            if not labels: continue
            maj_label = max(set(labels), key=labels.count)
            label = 1 if maj_label == 1 else 0
            data_items.append({"image_name": f"{tweet_id}", "text": entry.get("tweet_text", ""), "label": label})

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

    # 1. LOAD MODEL (LLaVA-v1.6)
    print(f"🚀 Loading LLaVA from {MODEL_PATH}...")
    try:
        processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
        # Patch pad token
        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
            
        model = LlavaNextForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            local_files_only=True,
            device_map="auto"
        ).eval()
        print("✅ LLaVA Loaded.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)

    # 2. LOAD DATA
    dataset, img_root = load_dataset_data(args.dataset)
    if args.sample: dataset = dataset[:args.sample]
    print(f"📊 Dataset loaded: {len(dataset)} samples. Img Root: {img_root}")

    # 3. INFERENCE LOOP
    results = []
    metrics = defaultdict(lambda: {"y_true": [], "y_pred": []})
    
    for idx, item in enumerate(dataset):
        if idx % 50 == 0: print(f"   [{idx}/{len(dataset)}] Processing...")
        
        img_path = find_image_robust(img_root, item["image_name"])
        if not img_path: continue

        try:
            image = Image.open(img_path).convert("RGB")
            
            for p in cfg["prompts"]:
                # Formato Prompt LLaVA v1.6 (Chat Template standard)
                # Template: [INST] <image>\nQUESTION [/INST]
                
                # Sostituzione placeholder se il prompt è un template
                # Ma qui i prompt sono secchi, quindi aggiungiamo solo il testo del meme
                full_text = f"{p}\nMeme Text: '{item['text']}'"
                
                final_prompt = f"[INST] <image>\n{full_text} [/INST]"
                
                inputs = processor(text=final_prompt, images=image, return_tensors="pt").to(model.device)
                
                with torch.no_grad():
                    output = model.generate(
                        **inputs,
                        max_new_tokens=50, # Pochi token, ci aspettiamo 1 o 0
                        do_sample=False,
                        pad_token_id=processor.tokenizer.pad_token_id
                    )
                
                generated_text = processor.decode(output[0], skip_special_tokens=True)
                
                if "[/INST]" in generated_text:
                    response = generated_text.split("[/INST]")[-1].strip()
                else:
                    response = generated_text

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

    # 4. SAVE & METRICS
    out_csv = os.path.join(args.out_dir, f"results_llava_{args.dataset}.csv")
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"✅ Saved results to {out_csv}")

    # 5. METRICS PRINT
    print(f"\n🏆 RESULTS: {args.dataset} (LLaVA Zero-Shot)")
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
