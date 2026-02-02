import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Leggi i dati
df = pd.read_csv("final_benchmark_summary.csv")

# Imposta stile
sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 6))

# Crea il grafico a barre raggruppato
chart = sns.barplot(
    data=df,
    x="Dataset",
    y="Macro F1",
    hue="Model",
    palette="viridis",
    edgecolor="black"
)

# Aggiungi i valori sopra le barre
for container in chart.containers:
    chart.bar_label(container, fmt='%.2f', padding=3, fontsize=10)

plt.title("Confronto Modelli Multimodali (Macro F1 Score)", fontsize=16, fontweight='bold')
plt.ylabel("Macro F1 Score", fontsize=12)
plt.xlabel("Dataset", fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.tight_layout()

# Salva
plt.savefig("benchmark_chart.png", dpi=300)
print("✅ Grafico salvato come 'benchmark_chart.png'")
