import os, sys, argparse, json, time, re
import pandas as pd
from collections import defaultdict
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from PIL import Image
from transformers import AutoProcessor, LlavaNextForConditionalGeneration

# --- CONFIGURAZIONE MODELLO ---
MODEL_PATH = "/home/rpisanu/models/llava-v1.6-mistral-7b"
SAVE_EVERY = 50 # Salva ogni 50 immagini

# --- CONFIGURAZIONE DATASET ---
CONFIGS = {
    "MAMI": {
        "base_dir": "/home/rpisanu/MAMI/MAMI_DATASET",
        "file_path": "test.tsv",
        "img_dir": "test_images_clean/test",
        "type": "tsv",
        "prompts": ["Is this meme misogynistic? Answer 1 (Yes) or 0 (No)."]
    },
    "MultiOFF": {
        "base_dir": "/home/rpisanu/MultiOFF/MultiOFF_Dataset",
        "file_path": "Split Dataset/Testing_meme_dataset.csv",
        "img_dir": "Labelled Images",
        "type": "csv",
        "prompts": ["Is this meme offensive? Answer 1 (Yes) or 0 (No)."]
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
            "Classify this post: 1 = Hate Speech, 0 = Not Hate Speech."
        ]
    },
    "HatefulMemes": {
        "base_dir": "/home/rpisanu/Hateful/hateful_memes/data",
        "file_path": "dev.jsonl",
        "img_dir": "img",
        "type": "jsonl",
        "prompts": ["Is this meme hateful? Answer 1 (Yes) or 0 (No)."]
    }
}

# ─────────────── UTILS ───────────────
def find_image_robust(base_img_dir, img_identifier):
    path = os.path.join(base_img_dir, img_identifier)
    if os.path.exists(path): return path
    if "img/" in img_identifier:
        path = os.path.join(base_img_dir, os.path.basename(img_identifier))
        if os.path.exists(path): return path
    name_no_ext = os.path.splitext(os.path.basename(img_identifier))[0]
    for ext in [".jpg", ".JPG", ".png", ".PNG", ".jpeg"]:
        path = os.path.join(base_img_dir, name_no_ext + ext)
        if os.path.exists(path): return path
    return None

def map_prediction(pred_text: str) -> int:
    s = str(pred_text).lower().strip()
    if "label: 0" in s: return 0
    if "label: 1" in s: return 1
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
            label = 1 if lbl_str in ["offensive", "1", "1.0"] else 0
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
    out_csv = os.path.join(args.out_dir, f"results_llava_{args.dataset}.csv")

    # 1. LOAD DATA & SMART RESUME
    dataset, img_root = load_dataset_data(args.dataset)
    if args.sample: dataset = dataset[:args.sample]
    
    # --- LOGICA DI RESUME ---
    processed_ids = set()
    if os.path.exists(out_csv):
        print(f"🔄 Found existing results file: {out_csv}")
        try:
            df_done = pd.read_csv(out_csv)
            # Raccoglie ID univoci processati
            processed_ids = set(df_done["id"].astype(str))
            print(f"⏩ Already processed: {len(processed_ids)} items.")
        except:
            print("⚠️ Error reading existing CSV, starting from scratch.")
    
    # Filtra il dataset: tieni solo quelli il cui 'image_name' NON è in processed_ids
    dataset_to_do = [d for d in dataset if d["image_name"] not in processed_ids]
    print(f"📉 Remaining to process: {len(dataset_to_do)} / {len(dataset)}")
    
    if not dataset_to_do:
        print("🎉 Dataset fully processed!")
        sys.exit(0)

    # 2. LOAD MODEL (Solo se c'è roba da fare)
    print(f"🚀 Loading LLaVA from {MODEL_PATH}...")
    try:
        processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
        model = LlavaNextForConditionalGeneration.from_pretrained(
            MODEL_PATH, torch_dtype=torch.float16, low_cpu_mem_usage=True, local_files_only=True, device_map="auto"
        ).eval()
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)

    # 3. INFERENCE LOOP
    buffer = []
    processed_count = 0
    start_time = time.time()
    
    for idx, item in enumerate(dataset_to_do):
        processed_count += 1
        if processed_count % 10 == 0: 
             elapsed = time.time() - start_time
             print(f"   Processed {processed_count}/{len(dataset_to_do)} (Speed: {processed_count/(elapsed+0.1):.2f} it/s)")
        
        img_path = find_image_robust(img_root, item["image_name"])
        if not img_path: continue

        try:
            image = Image.open(img_path).convert("RGB")
            
            for p in cfg["prompts"]:
                full_text = f"{p}\nMeme Text: '{item['text']}'"
                final_prompt = f"[INST] <image>\n{full_text} [/INST]"
                
                inputs = processor(text=final_prompt, images=image, return_tensors="pt").to(model.device)
                
                with torch.no_grad():
                    output = model.generate(
                        **inputs, max_new_tokens=50, do_sample=False, pad_token_id=processor.tokenizer.pad_token_id
                    )
                
                generated_text = processor.decode(output[0], skip_special_tokens=True)
                response = generated_text.split("[/INST]")[-1].strip() if "[/INST]" in generated_text else generated_text
                pred = map_prediction(response)
                
                buffer.append({
                    "id": item["image_name"],
                    "true_label": item["label"],
                    "pred": pred,
                    "response": response,
                    "prompt": p
                })
        
        except Exception as e:
            print(f"❌ Error {item['image_name']}: {e}")

        # 4. INCREMENTAL SAVE
        if len(buffer) >= SAVE_EVERY:
            write_header = not os.path.exists(out_csv)
            pd.DataFrame(buffer).to_csv(out_csv, mode='a', index=False, header=write_header)
            print(f"💾 Checkpoint saved ({len(buffer)} items).")
            buffer = []

    # Final Save
    if buffer:
        write_header = not os.path.exists(out_csv)
        pd.DataFrame(buffer).to_csv(out_csv, mode='a', index=False, header=write_header)
        print(f"✅ Final batch saved.")

if __name__ == "__main__":
    main()
