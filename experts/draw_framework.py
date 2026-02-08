from graphviz import Digraph

dot = Digraph(comment="Proposed Framework", format="png")
dot.attr(rankdir="LR", size="12,8", dpi="300", splines="polyline")
dot.attr("node", shape="box", style="filled", fontname="Helvetica", fontsize="11")

with dot.subgraph(name="cluster_input") as c:
    c.attr(color="white")
    c.node(
        "IMG", "Meme Image\n(I)", fillcolor="#E3F2FD", color="#1565C0", shape="folder"
    )
    c.node("TXT", "Meme Text\n(T)", fillcolor="#E3F2FD", color="#1565C0", shape="note")

with dot.subgraph(name="cluster_experts") as c:
    c.attr(
        label="Phase 2: Domain Experts",
        color="lightgrey",
        style="dashed",
        fontname="Helvetica-Bold",
    )

    c.node(
        "ROBERTA",
        'RoBERTa\n(Textual Expert)\n"Safe Anchor"',
        fillcolor="#FFF3E0",
        color="#EF6C00",
    )

    c.node(
        "CLIP",
        'CLIP + MLP\n(Generalist VLM)\n"Explicit Hate"',
        fillcolor="#FFF3E0",
        color="#EF6C00",
    )

    c.node(
        "MEME",
        'MemeCLIP\n(Specialist VLM)\n"Implicit/Context"',
        fillcolor="#FFE0B2",
        color="#E65100",
        penwidth="2.0",
    )

with dot.subgraph(name="cluster_strategies") as c:
    c.attr(label="Phase 3: Knowledge Injection", color="#E8F5E9", style="rounded")
    c.node(
        "VOTES",
        "Expert Voting\n{V_text, V_gen, V_spec}",
        shape="component",
        fillcolor="#C8E6C9",
        color="#2E7D32",
    )
    c.node(
        "LOGIC",
        'Conditional Constraint\n(Strategy C)\n\nIF (Text==Safe) & (Spec==Hate):\nInject "Strict Rule"',
        shape="diamond",
        fillcolor="#A5D6A7",
        color="#1B5E20",
        fontsize="10",
    )

dot.node(
    "LLM",
    "InternVL 2.5\n(Backbone LLM)",
    shape="box3d",
    fillcolor="#F3E5F5",
    color="#7B1FA2",
    fontsize="12",
    fontname="Helvetica-Bold",
)

dot.node(
    "OUT",
    "Final Prediction\n(0 / 1)",
    shape="ellipse",
    fillcolor="#FFEBEE",
    color="#C62828",
)


dot.edge("TXT", "ROBERTA", color="#1565C0")
dot.edge("TXT", "CLIP", color="#1565C0")
dot.edge("TXT", "MEME", color="#1565C0")

dot.edge("IMG", "CLIP", color="#1565C0")
dot.edge("IMG", "MEME", color="#1565C0")

dot.edge("ROBERTA", "VOTES")
dot.edge("CLIP", "VOTES")
dot.edge("MEME", "VOTES")

dot.edge("VOTES", "LOGIC", label=" Votes")
dot.edge("LOGIC", "LLM", label=" Dynamic Prompt")

dot.edge("IMG", "LLM", color="grey", style="dotted")
dot.edge("TXT", "LLM", color="grey", style="dotted")

dot.edge("LLM", "OUT")

output_path = "framework_schema_v2"
dot.render(output_path, view=False)
