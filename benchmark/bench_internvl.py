import os, sys, argparse, json, time
import pandas as pd
from PIL import Image
from collections import defaultdict
import torch
import re
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer

# --- CONFIGURAZIONE MODELLO ---
MODEL_PATH = "/home/rpisanu/InternVL2_5-8B"
sys.path.append(MODEL_PATH)

try:
    from modeling_internvl_chat import InternVLChatModel
except ImportError:
    print("⚠️ Warning: Import diretto fallito, provo AutoModel.")
    from transformers import AutoModelForCausalLM as InternVLChatModel

# --- CONFIGURAZIONE DATASET E PROMPT ---
CONFIGS = {
    "MAMI": {
        "base_dir": "/home/rpisanu/MAMI/MAMI_DATASET",
        "file_path": "test.tsv",
        "img_dir": "test_images_clean/test",
        "type": "tsv",
        "prompts": [
            "Analyze the meme as a whole (image + text). If it contains misogynistic or sexist stereotypes, output 1. If it does not, output 0. Respond with only one character.",
            "Does this meme, when considering both the image and the text, express offensive or hateful content toward women? Answer with 1 for yes, 0 for no.",
            "Look carefully at the visual elements AND the written caption. If the meme insults, demeans, or stereotypes women when combining image and text, reply with 1. Otherwise reply with 0.",
            "Does the meme promote stereotypes, discrimination, or hate toward women? Respond ONLY with 1 for yes, 0 for no.",
            "Classify this meme: 1 = hateful or offensive toward women, 0 = not hateful. Output only the digit."
        ]
    },
    "MultiOFF": {
        "base_dir": "/home/rpisanu/MultiOFF/MultiOFF_Dataset",
        "file_path": "Split Dataset/Validation_meme_dataset.csv",
        "img_dir": "Labelled Images",
        "type": "csv",
        "prompts": [
            "Considering both the image and the text of this meme, is it offensive? Answer 1 for offensive, 0 for non-offensive.",
            "Analyze this meme using both its image and textual content. Determine if it is hateful: 1=hateful, 0=not hateful.",
            "Using the combination of image and text, classify this meme as offensive or not: 1=offensive, 0=non-offensive.",
            "Look at the meme's visual and textual elements together. Does it contain offensive content? Answer 1=yes, 0=no."
        ]
    },
    "MMHS150K": {
        "base_dir": "/home/rpisanu/MMHS150K",
        "file_path": "MMHS150K_GT.json",
        "split_file": "splits/test_ids.txt",
        "img_dir": "img_resized",
        "type": "mmhs_json",
        "prompts": [
            "Analyze the tweet (image and text). Does it contain Hate Speech targeting a protected group (race, religion, gender)? Answer 0 for Non-Hate (including mere profanity/slang) or 1 for Hate Speech.",
            "Does this post promote violence or dehumanize a specific group of people? Answer 1 (Yes) or 0 (No). Do not flag general offensive language.",
            "Classify as 1 (Hate) ONLY if the content is clearly hateful. If it is just rude, sarcastic, or uses slang, classify as 0 (Safe)."
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
            "Evaluate the image and text together. Is it hateful? Output only 1 for hateful, 0 for not.",
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

def preprocess_image(image):
    transform = T.Compose([
        T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    return transform(image).unsqueeze(0).to(torch.bfloat16).cuda()

def map_prediction(pred_text: str) -> int:
    s = str(pred_text).lower().strip()
    if "non-offensive" in s or "not offensive" in s or "not hateful" in s: return 0
    if "offensive" in s or "hateful" in s: return 1
    if "yes" in s: return 1
    if "no" in s: return 0
    nums = re.findall(r"\b(0|1)\b", s)
    if nums: return int(nums[-1])
    return -1

# ─────────────── DATA LOADER ───────────────
def load_dataset_data(name):
    cfg = CONFIGS[name]
    full_path = os.path.join(cfg["base_dir"], cfg["file_path"])
    print(f"📂 Loading {name} from {full_path}...")
    data_items = []

    if cfg["type"] == "tsv": # MAMI
        df = pd.read_csv(full_path, sep="\t")
        img_col = "file_name" if "file_name" in df.columns else "image_name"
        lbl_col = "misogynous" if "misogynous" in df.columns else "label"
        txt_col = "Text Transcription" if "Text Transcription" in df.columns else "text"
        for _, row in df.iterrows():
            data_items.append({"image_name": str(row[img_col]).strip(), "text": str(row[txt_col]), "label": int(row[lbl_col])})

    elif cfg["type"] == "csv": # MultiOFF
        df = pd.read_csv(full_path)
        for _, row in df.iterrows():
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

    print(f"🚀 Loading InternVL from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = InternVLChatModel.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    ).cuda().eval()

    dataset, img_root = load_dataset_data(args.dataset)
    if args.sample: dataset = dataset[:args.sample]
    print(f"📊 Dataset loaded: {len(dataset)} samples.")

    results = []
    metrics = defaultdict(lambda: {"y_true": [], "y_pred": []})
    start_time = time.time()
    
    for idx, item in enumerate(dataset):
        if idx % 50 == 0: print(f"   [{idx}/{len(dataset)}] Processing...")
        img_path = find_image_robust(img_root, item["image_name"])
        if not img_path: continue

        try:
            image = Image.open(img_path).convert("RGB")
            pixel_values = preprocess_image(image)
            for p in cfg["prompts"]:
                full_prompt = f"Text: {item['text']}\n\n{p}"
                with torch.no_grad():
                    response = model.chat(tokenizer, pixel_values, full_prompt, generation_config=dict(max_new_tokens=50, do_sample=False))
                resp_str = response[0] if isinstance(response, tuple) else response
                pred = map_prediction(resp_str)
                results.append({"id": item["image_name"], "true_label": item["label"], "pred": pred, "response": resp_str, "prompt": p})
                if pred != -1:
                    metrics[p]["y_true"].append(item["label"])
                    metrics[p]["y_pred"].append(pred)
        except Exception as e:
            print(f"❌ Error {item['image_name']}: {e}")

    out_csv = os.path.join(args.out_dir, f"results_internvl_{args.dataset}.csv")
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"✅ Saved results to {out_csv}")

    # --- CALCOLO METRICHE ESTESO ---
    print(f"\n🏆 RESULTS: {args.dataset}")
    print("="*100)
    # Header della tabella
    print(f"{'PROMPT (First 40 chars)':<40} | {'ACC':<6} | {'F1 MAC':<6} | {'F1 BIN':<6} | {'REC':<6} | {'PREC':<6}")
    print("-" * 100)

    for p, d in metrics.items():
        if d["y_pred"]:
            # Calcolo di tutte le metriche
            acc = accuracy_score(d["y_true"], d["y_pred"])
            
            # Binary (Focus su classe 1 = Hate)
            p_bin = precision_score(d["y_true"], d["y_pred"], average='binary', zero_division=0)
            r_bin = recall_score(d["y_true"], d["y_pred"], average='binary', zero_division=0)
            f1_bin = f1_score(d["y_true"], d["y_pred"], average='binary', zero_division=0)
            
            # Macro (Equilibrato)
            f1_mac = f1_score(d["y_true"], d["y_pred"], average='macro', zero_division=0)
            
            # Stampa formattata
            print(f"{p[:40]:<40} | {acc:.4f} | {f1_mac:.4f} | {f1_bin:.4f} | {r_bin:.4f} | {p_bin:.4f}")
            
    print("="*100 + "\n")

if __name__ == "__main__":
    main()
