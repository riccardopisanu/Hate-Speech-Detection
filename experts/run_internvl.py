import os, sys, argparse, time
from collections import defaultdict
import pandas as pd
import torch
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
from transformers import AutoTokenizer

# -------------------------------------------------------------------
# 1. CONFIGURAZIONE & IMPORT MODELLO
# -------------------------------------------------------------------
DEFAULT_MODEL_PATH = "/home/rpisanu/InternVL2_5-8B"

def load_model_class(model_path):
    sys.path.append(model_path)
    try:
        from modeling_internvl_chat import InternVLChatModel
        return InternVLChatModel
    except ImportError:
        print(f"❌ ERRORE: Non trovo 'modeling_internvl_chat.py' in {model_path}")
        sys.exit(1)

# -------------------------------------------------------------------
# 2. UTILS PREPROCESSING
# -------------------------------------------------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def preprocess_image(image: Image.Image):
    transform = T.Compose([
        T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])
    return transform(image).unsqueeze(0).to(torch.bfloat16).cuda()

def find_image_path(root_dir, img_name):
    candidates = [
        os.path.join(root_dir, img_name),
        os.path.join(root_dir, str(img_name)),
        os.path.join(root_dir, str(img_name).split('.')[0] + ".png"),
        os.path.join(root_dir, str(img_name).split('.')[0] + ".jpg"),
        os.path.join(root_dir, "img", str(img_name))
    ]
    for p in candidates:
        if os.path.exists(p): return p
    return None

def map_prediction(pred_text: str) -> int:
    s = str(pred_text).lower().strip()
    if s.startswith("1") or "yes" in s or "misogynistic" in s or "hateful" in s: return 1
    if s.startswith("0") or "no" in s or "safe" in s or "not offensive" in s: return 0
    import re
    match = re.search(r'\b(0|1)\b', s)
    return int(match.group(1)) if match else 0

# -------------------------------------------------------------------
# 3. PROMPT ADATTIVI (DOMAIN ADAPTATION)
# -------------------------------------------------------------------
def get_prompts_dict(text_ocr, p_clip, p_meme, p_nlp, dataset_name):
    """
    Seleziona la strategia euristica in base al dataset.
    """
    def decode(val): return "HATEFUL" if val == 1 else "SAFE"
    
    context = (
        f"OPINIONS:\n"
        f"- Visual (CLIP): {decode(p_clip)}\n"
        f"- Context (MemeCLIP): {decode(p_meme)}\n"
        f"- Text (RoBERTa): {decode(p_nlp)}\n"
    )

    prompts = {}

    # 1. DEMOCRACY
    prompts["P_DEMOCRACY"] = (
        f"You are a content moderator. TEXT: \"{text_ocr}\".\n"
        f"{context}\n"
        f"Task: Based on the experts and your visual analysis, contains misogyny or hate? "
        f"Output 1 (Yes) or 0 (No). Return ONLY the digit."
    )

    # 2. CHAIN OF THOUGHT (CoT)
    prompts["P_COT"] = (
        f"You are an expert linguist. TEXT: \"{text_ocr}\".\n"
        f"{context}\n"
        f"Step 1: Analyze the text for hidden irony.\n"
        f"Step 2: Analyze the image for stereotypes.\n"
        f"Step 3: Compare with expert opinions.\n"
        f"Step 4: Decide.\n"
        f"Is this hateful? Output 1 or 0. Respond ONLY with the final digit."
    )

    # 3. HEURISTIC (ADATTIVA) 
    rule = ""
    
    # LOGICA ADATTIVA:
    # MAMI (Misoginia) -> Spesso esplicita, violenta o disumanizzante. Serve filtro stretto.
    if "MAMI" in dataset_name:
        if p_nlp == 0 and (p_clip == 1 or p_meme == 1):
            rule = "WARNING: Text is safe. Only classify as 1 if the IMAGE contains violence or dehumanization."
            
    # Hateful Memes / MultiOFF -> Spesso ironica, simbolica, "benign confounders". Serve filtro ampio.
    else: 
        if p_nlp == 0 and (p_meme == 1 or p_clip == 1):
            rule = "WARNING: Text seems safe, but visual experts detected danger. Look for implicit hate or visual stereotypes."
    
    prompts["P_HEURISTIC"] = (
        f"Arbiter Task. TEXT: \"{text_ocr}\".\n"
        f"{context}\n"
        f"RULE: {rule}\n"
        f"Classify as Hateful (1) or Safe (0). Return ONLY the digit."
    )

    return prompts

# -------------------------------------------------------------------
# 4. MAIN LOOP
# -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str, required=True)
    parser.add_argument("--img_root", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="Dataset")
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"🔄 Caricamento InternVL da: {DEFAULT_MODEL_PATH}")
    InternVLChatModel = load_model_class(DEFAULT_MODEL_PATH)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_PATH, trust_remote_code=True, local_files_only=True)
        model = InternVLChatModel.from_pretrained(
            DEFAULT_MODEL_PATH,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).cuda().eval()
        print("✅ InternVL caricato.")
    except Exception as e:
        print(f"❌ ERRORE Caricamento Modello: {e}")
        sys.exit(1)

    print(f"📂 Loading CSV: {args.csv_file}")
    df = pd.read_csv(args.csv_file)
    if 'true_label' not in df.columns and 'label' in df.columns: 
        df.rename(columns={'label': 'true_label'}, inplace=True)
    
    if args.sample: df = df.head(args.sample)

    print(f"📊 Dataset: {args.dataset_name} | Samples: {len(df)}")
    
    results = []
    metrics = defaultdict(lambda: {"y_true": [], "y_pred": []})

    print("🚀 Starting Inference...")
    for idx, row in df.iterrows():
        img_name = str(row['image_name'])
        text = str(row.get('text', ''))
        label = int(row['true_label'])
        
        p_clip = int(row.get('PRED_CLIP', -1))
        p_meme = int(row.get('PRED_MEME', -1))
        p_nlp  = int(row.get('PRED_NLP', -1))

        img_path = find_image_path(args.img_root, img_name)
        if not img_path: continue

        # Passiamo il nome del dataset per attivare l'euristica giusta
        prompts = get_prompts_dict(text, p_clip, p_meme, p_nlp, args.dataset_name)

        try:
            pil_img = Image.open(img_path).convert("RGB")
            pixel_values = preprocess_image(pil_img)
            
            for p_name, p_text in prompts.items():
                
                with torch.no_grad():
                    # 300 Token per Chain of Thought
                    response = model.chat(
                        tokenizer, 
                        pixel_values, 
                        p_text, 
                        history=None, 
                        generation_config=dict(max_new_tokens=1024, do_sample=False)
                    )
                
                pred = map_prediction(str(response))
                
                results.append({
                    "image": img_name,
                    "label": label,
                    "prompt": p_name,
                    "response": str(response),
                    "final_pred": pred,
                    "expert_clip": p_clip,
                    "expert_meme": p_meme,
                    "expert_nlp": p_nlp
                })
                
                metrics[p_name]["y_true"].append(label)
                metrics[p_name]["y_pred"].append(pred)

        except Exception as e:
            print(f"❌ Error {img_name}: {e}")

        if (idx+1) % 50 == 0 or (idx+1) == len(df): 
            print(f"   Processed {idx+1}/{len(df)}...")

    # Salvataggio
    out_csv = os.path.join(args.out_dir, f"results_InternVL_{args.dataset_name}.csv")
    pd.DataFrame(results).to_csv(out_csv, index=False)
    
    print("\n" + "="*85)
    print(f"🏆 RESULTS: {args.dataset_name}")
    print("="*85)
    print(f"{'PROMPT':<15} | {'ACC':<6} | {'F1 MACRO':<8} | {'F1 BINARY':<8} | {'REC (Bin)':<8} | {'PREC (Bin)':<8}")
    print("-" * 85)
    
    for p_name, data in metrics.items():
        if len(data['y_true']) > 0:
            y_true = data['y_true']
            y_pred = data['y_pred']
            acc = accuracy_score(y_true, y_pred)
            p_bin, r_bin, f1_bin, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
            f1_macro = f1_score(y_true, y_pred, average='macro')
            print(f"{p_name:<15} | {acc:.4f} | {f1_macro:.4f}   | {f1_bin:.4f}    | {r_bin:.4f}     | {p_bin:.4f}")
    print("="*85 + "\n")

if __name__ == "__main__":
    main()
