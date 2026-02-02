from graphviz import Digraph
import os

# Configurazione del grafico
dot = Digraph(comment='Framework Architecture', format='png')
dot.attr(rankdir='LR', size='10,8', dpi='300') # Left to Right, Alta risoluzione

# Stile dei nodi
dot.attr('node', shape='box', style='filled', fontname='Helvetica', fontsize='12')

# --- 1. INPUT ---
dot.attr('node', fillcolor='#E3F2FD', color='#1565C0') # Blu chiaro
dot.node('IMG', 'Input Image\n(Meme)')
dot.node('TXT', 'Input Text\n(Caption/OCR)')

# --- 2. EXPERTS LAYER ---
dot.attr('node', fillcolor='#FFF3E0', color='#EF6C00') # Arancione
with dot.subgraph(name='cluster_experts') as c:
    c.attr(label='Expert Models Pool', color='lightgrey', style='dashed')
    c.node('EXP1', 'Demographic Expert\n(Race/Gender)')
    c.node('EXP2', 'Object Detection\n(Scene Objects)')
    c.node('EXP3', 'Sentiment Expert\n(Text Tone)')
    c.node('EXP4', 'External Knowledge\n(Symbolism)')

# --- 3. INJECTION MECHANISM ---
dot.attr('node', fillcolor='#E8F5E9', color='#2E7D32', shape='component') # Verde
dot.node('PROMPT', 'Prompt Injection Strategy\n\n1. Simple Context\n2. Chain-of-Thought\n3. Conflict Handling')

# --- 4. MAIN LLM ---
dot.attr('node', fillcolor='#F3E5F5', color='#7B1FA2', shape='box3d') # Viola
dot.node('LLM', 'Multimodal LLM\n(Reasoning Core)\n\n[Qwen / InternVL / Phi-4]')

# --- 5. OUTPUT ---
dot.attr('node', fillcolor='#FFEBEE', color='#C62828', shape='ellipse') # Rosso
dot.node('OUT', 'Final Prediction\n(Label + Rationale)')

# --- COLLEGAMENTI (EDGES) ---
# Input verso Esperti
dot.edge('IMG', 'EXP1')
dot.edge('IMG', 'EXP2')
dot.edge('TXT', 'EXP3')
dot.edge('IMG', 'EXP4')

# Input verso LLM (L'LLM vede anche l'immagine originale)
dot.edge('IMG', 'LLM', color='grey', style='dashed', label=' Visual Context')

# Esperti verso Prompt Injection
dot.edge('EXP1', 'PROMPT', label=' Tags')
dot.edge('EXP2', 'PROMPT')
dot.edge('EXP3', 'PROMPT')
dot.edge('EXP4', 'PROMPT')

# Injection verso LLM
dot.edge('PROMPT', 'LLM', label=' Enriched Prompt')

# LLM verso Output
dot.edge('LLM', 'OUT')

# Salvataggio
output_path = 'framework_schema'
dot.render(output_path, view=False)
print(f"✅ Schema salvato come {output_path}.png")
