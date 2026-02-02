import os
import glob
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

# Cartella dove hai salvato i CSV
RESULTS_DIR = "results_benchmarks"
OUTPUT_FILE = "final_benchmark_summary.csv"

def clean_labels(series):
    """
    Pulisce le label: converte a numero e forza tutto ciò che non è 1 a diventare 0.
    Gestisce i -1 (errori di parsing) trattandoli come predizioni 'safe' (0).
    """
    # Converte a numerico, i non-numeri diventano NaN
    s = pd.to_numeric(series, errors='coerce').fillna(0)
    # Arrotonda (nel caso ci siano float tipo 1.0) e converte a int
    s = s.round().astype(int)
    # Mappa qualsiasi cosa diversa da 1 (es: -1, 2, 99) a 0
    # Questo risolve l'errore "Target is multiclass"
    s = s.apply(lambda x: 1 if x == 1 else 0)
    return s

def get_best_metrics():
    files = glob.glob(os.path.join(RESULTS_DIR, "*.csv"))
    
    if not files:
        print(f"❌ Nessun file CSV trovato in {RESULTS_DIR}")
        return

    print(f"🔍 Trovati {len(files)} file. Analisi e pulizia in corso...\n")

    summary_data = []

    for filepath in files:
        filename = os.path.basename(filepath)
        try:
            # Parsing nome file
            parts = filename.replace("results_", "").replace(".csv", "").split("_")
            model_name = parts[0].upper()
            dataset_name = "_".join(parts[1:])
        except:
            continue

        try:
            df = pd.read_csv(filepath)
            if df.empty: continue

            # Trova colonne
            prompt_col = next((c for c in ["prompt", "prompt_text", "prompt_type"] if c in df.columns), None)
            true_col = next((c for c in ["true_label", "label", "true", "ground_truth"] if c in df.columns), None)
            pred_col = next((c for c in ["pred", "pred_label", "prediction"] if c in df.columns), None)

            if not (prompt_col and true_col and pred_col):
                # Tentativo disperato: se mancano le intestazioni standard, prova a indovinare
                if "true_label" not in df.columns and "label" in df.columns: true_col = "label"
                if "pred_label" not in df.columns and "pred" in df.columns: pred_col = "pred"
                
                if not (true_col and pred_col):
                    print(f"⚠️  Saltato {filename}: colonne non trovate.")
                    continue

            best_f1 = -1
            best_row = None

            for prompt_val in df[prompt_col].unique():
                sub_df = df[df[prompt_col] == prompt_val].copy()
                
                # --- PULIZIA DATI ---
                y_true = clean_labels(sub_df[true_col])
                y_pred = clean_labels(sub_df[pred_col])

                # Se dopo la pulizia una colonna è vuota o ha solo una classe, gestiamo l'eccezione
                if len(y_true) == 0: continue

                try:
                    # Calcolo metriche
                    f1_mac = f1_score(y_true, y_pred, average='macro', zero_division=0)
                    f1_bin = f1_score(y_true, y_pred, average='binary', zero_division=0)
                    acc = accuracy_score(y_true, y_pred)
                    
                    if f1_mac > best_f1:
                        best_f1 = f1_mac
                        best_row = {
                            "Dataset": dataset_name,
                            "Model": model_name,
                            "Macro F1": round(f1_mac, 4),
                            "Binary F1": round(f1_bin, 4),
                            "Accuracy": round(acc, 4),
                            "Samples": len(sub_df),
                            "Best Prompt": str(prompt_val).replace("\n", " ")[:40] + "..."
                        }
                except Exception as e:
                    # Se fallisce ancora, salta silenziosamente questo prompt
                    pass
            
            if best_row:
                summary_data.append(best_row)

        except Exception as e:
            print(f"❌ Errore leggendo {filename}: {e}")

    # --- TABELLA FINALE ---
    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        # Ordina per Dataset e poi per F1 Score
        df_summary = df_summary.sort_values(by=["Dataset", "Macro F1"], ascending=[True, False])
        
        print("\n" + "="*115)
        print(f"🏆 CLASSIFICA FINALE BENCHMARK (Ordinata per Macro F1)")
        print("="*115)
        print(df_summary.to_string(index=False))
        print("="*115)
        
        df_summary.to_csv(OUTPUT_FILE, index=False)
        print(f"\n✅ Salvato in '{OUTPUT_FILE}'")
    else:
        print("❌ Nessun risultato valido.")

if __name__ == "__main__":
    get_best_metrics()
