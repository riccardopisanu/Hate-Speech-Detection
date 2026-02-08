import pandas as pd
import os

# ==============================================================================
# 1. CONFIGURAZIONE PATHS
# ==============================================================================
# Assicurati che questi percorsi siano corretti
INPUT_FILES = {
    "MAMI": "/home/rpisanu/tesi_repo/experts/results_llm/results_InternVL_MAMI.csv",
    "Hateful Memes": "/home/rpisanu/tesi_repo/experts/results_llm/results_InternVL_HatefulMemes.csv",
    "MultiOFF": "/home/rpisanu/tesi_repo/experts/results_llm/results_InternVL_MultiOFF.csv"
}

# Mapping: Come si chiamano i prompt nel CSV -> Come li chiamiamo nella logica
# (Adatta se i tuoi nomi nel CSV sono diversi, es. 'pred_constraint')
PROMPTS = {
    "Demo": "P_DEMOCRACY",
    "CoT": "P_COT",
    "Const": "P_HEURISTIC" # O 'P_CONSTRAINT' a seconda del tuo file
}

# ==============================================================================
# 2. FUNZIONE DI CALCOLO
# ==============================================================================
def generate_latex_table(dataset_name, csv_path):
    if not os.path.exists(csv_path):
        print(f"⚠️  File non trovato: {csv_path}")
        return

    # Carica
    df = pd.read_csv(csv_path)

    # Pivot (Trasforma da lungo a largo: una riga per immagine)
    # Crea colonne separate per ogni strategia
    try:
        pivot = df.pivot_table(
            index=['image', 'label'], 
            columns='prompt', 
            values='final_pred',
            aggfunc='first'
        ).reset_index()
    except KeyError as e:
        print(f"❌ Errore colonne nel dataset {dataset_name}: {e}")
        return

    # Controlla se le colonne esistono (gestione errori)
    for p_key, p_col in PROMPTS.items():
        if p_col not in pivot.columns:
            print(f"❌ Colonna '{p_col}' non trovata in {dataset_name}. Colonne presenti: {pivot.columns.tolist()}")
            return

    # --- LOGICA BOOLEANA ---
    # 1. Chi ha indovinato?
    correct_demo = pivot[PROMPTS['Demo']] == pivot['label']
    correct_cot = pivot[PROMPTS['CoT']] == pivot['label']
    correct_const = pivot[PROMPTS['Const']] == pivot['label']

    # 2. UNIQUE WINS (Solo lui ha indovinato, gli altri hanno sbagliato)
    # Constraint Wins: (Const OK) AND (Demo KO) AND (CoT KO)
    wins_const = correct_const & (~correct_demo) & (~correct_cot)
    
    # CoT Wins: (CoT OK) AND (Demo KO) AND (Const KO)
    wins_cot = correct_cot & (~correct_demo) & (~correct_const)
    
    # Demo Wins: (Demo OK) AND (CoT KO) AND (Const KO)
    wins_demo = correct_demo & (~correct_cot) & (~correct_const)

    # 3. TOTAL FAILURES (Tutti hanno sbagliato)
    total_failures = (~correct_demo) & (~correct_cot) & (~correct_const)

    # Conteggi
    count_const = wins_const.sum()
    count_cot = wins_cot.sum()
    count_demo = wins_demo.sum()
    count_fail = total_failures.sum()

    # ==========================================================================
    # 3. GENERAZIONE LATEX
    # ==========================================================================
    label_tag = f"tab:const_{dataset_name.lower().replace(' ', '_')}"
    
    print(f"\n% --- TABELLA GENERATA PER: {dataset_name} ---")
    print(r"\begin{table}[h!]")
    print(r"    \centering")
    print(r"    \begin{tabular}{lc}")
    print(r"        \toprule")
    print(r"        \textbf{Outcome Category} & \textbf{Count} \\")
    print(r"        \midrule")
    print(f"        Conditional Constraint wins & ${count_const}$ \\\\")
    print(f"        Chain-of-Thought (CoT) wins & ${count_cot}$ \\\\")
    print(f"        Naive Aggregation (Democracy) wins & ${count_demo}$ \\\\")
    print(r"        \midrule")
    print(f"        \\textit{{Total Failures (All strategies failed)}} & ${count_fail}$ \\\\")
    print(r"        \bottomrule")
    print(r"    \end{tabular}")
    print(f"    \\caption{{Breakdown of successful predictions for \\textbf{{{dataset_name}}}. The table shows how many times a specific strategy was the \\textit{{only}} one to correctly classify a sample.}}")
    print(f"    \\label{{{label_tag}}}")
    print(r"\end{table}")
    print("% -----------------------------------------------")

# ==============================================================================
# 4. ESECUZIONE
# ==============================================================================
print("Generazione tabelle LaTeX in corso...")
for name, path in INPUT_FILES.items():
    generate_latex_table(name, path)
