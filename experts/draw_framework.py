from graphviz import Digraph

# Configurazione Grafico
dot = Digraph(comment='Proposed Framework', format='png')
dot.attr(rankdir='LR', size='12,8', dpi='300', splines='polyline') 
# Usa 'polyline' per evitare curve strane con tanti collegamenti
dot.attr('node', shape='box', style='filled', fontname='Helvetica', fontsize='11')

# --- 1. INPUT ---
with dot.subgraph(name='cluster_input') as c:
    c.attr(color='white')
    c.node('IMG', 'Meme Image\n(I)', fillcolor='#E3F2FD', color='#1565C0', shape='folder')
    c.node('TXT', 'Meme Text\n(T)', fillcolor='#E3F2FD', color='#1565C0', shape='note')

# --- 2. EXPERT ENSEMBLE ---
with dot.subgraph(name='cluster_experts') as c:
    c.attr(label='Phase 2: Domain Experts', color='lightgrey', style='dashed', fontname='Helvetica-Bold')
    
    # RoBERTa (Solo Testo)
    c.node('ROBERTA', 'RoBERTa\n(Textual Expert)\n"Safe Anchor"', fillcolor='#FFF3E0', color='#EF6C00')
    
    # CLIP+MLP (Multimodale)
    c.node('CLIP', 'CLIP + MLP\n(Generalist VLM)\n"Explicit Hate"', fillcolor='#FFF3E0', color='#EF6C00')
    
    # MemeCLIP (Multimodale)
    c.node('MEME', 'MemeCLIP\n(Specialist VLM)\n"Implicit/Context"', fillcolor='#FFE0B2', color='#E65100', penwidth='2.0')

# --- 3. VOTING & STRATEGIES ---
with dot.subgraph(name='cluster_strategies') as c:
    c.attr(label='Phase 3: Knowledge Injection', color='#E8F5E9', style='rounded')
    c.node('VOTES', 'Expert Voting\n{V_text, V_gen, V_spec}', shape='component', fillcolor='#C8E6C9', color='#2E7D32')
    c.node('LOGIC', 'Conditional Constraint\n(Strategy C)\n\nIF (Text==Safe) & (Spec==Hate):\nInject "Strict Rule"', 
           shape='diamond', fillcolor='#A5D6A7', color='#1B5E20', fontsize='10')

# --- 4. BACKBONE ---
dot.node('LLM', 'InternVL 2.5\n(Backbone LLM)', shape='box3d', fillcolor='#F3E5F5', color='#7B1FA2', fontsize='12', fontname='Helvetica-Bold')

# --- 5. OUTPUT ---
dot.node('OUT', 'Final Prediction\n(0 / 1)', shape='ellipse', fillcolor='#FFEBEE', color='#C62828')

# --- COLLEGAMENTI ---

# Il TESTO va a TUTTI gli esperti
dot.edge('TXT', 'ROBERTA', color='#1565C0')
dot.edge('TXT', 'CLIP', color='#1565C0')
dot.edge('TXT', 'MEME', color='#1565C0')

# L'IMMAGINE va ai due Visual Experts (e non a RoBERTa)
dot.edge('IMG', 'CLIP', color='#1565C0')
dot.edge('IMG', 'MEME', color='#1565C0')

# Experts -> Voting
dot.edge('ROBERTA', 'VOTES')
dot.edge('CLIP', 'VOTES')
dot.edge('MEME', 'VOTES')

# Voting -> Logic -> LLM
dot.edge('VOTES', 'LOGIC', label=' Votes')
dot.edge('LOGIC', 'LLM', label=' Dynamic Prompt')

# Pass-through dei dati grezzi al LLM
dot.edge('IMG', 'LLM', color='grey', style='dotted')
dot.edge('TXT', 'LLM', color='grey', style='dotted')

# Output
dot.edge('LLM', 'OUT')

# Salvataggio
output_path = 'framework_schema_v2'
dot.render(output_path, view=False)
print(f"✅ Schema aggiornato salvato come {output_path}.png")
