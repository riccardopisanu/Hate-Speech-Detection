# Hateful Meme Detection via Large Language Models: A Comparative Analysis.

## Abstract
The proliferation of Internet memes has transformed social media into a highly visual landscape, concurrently creating a sophisticated vehicle for multimodal hate speech. 
Detecting such content is notoriously difficult due to the "semantic gap", where hateful intent arises solely from the intersection of benign text and neutral imagery.

Current State-of-the-Art Large Language Models (LLMs), while possessing advanced reasoning capabilities, suffer from significant limitations: they function as "black boxes," are prone to hallucinations, and exhibit structural biases induced by safety training.

To address these challenges, this thesis proposes a novel architecture based on Knowledge Injection. 
Instead of relying on a single end-to-end model, we introduce a modular framework where a Generative "Meta-Reasoner" arbitrates conflicting signals provided by specialized "Discriminative Experts". By translating numerical expert predictions into textual context, we enable the Large Language Model to ground its reasoning in domain-specific signals.

We evaluate the proposed architecture across diverse benchmarks covering distinct typologies of hate, including misogyny, implicit hate, and sarcasm.
The results demonstrate that our Knowledge-Injected strategies consistently outperform zero-shot baselines and rival fully fine-tuned models. 

Crucially, through a rigorous error analysis, we identify and formalize the "Arbiter Error": a unidirectional failure mode where the Generative Model systematically overrides a correct "Safe" consensus from experts to hallucinate "Hate" (False Positives), driven by an over-sensitive Safety Alignment. Conversely, the system exhibits zero blindness when experts unanimously flag hate.

This work empirically proves that "raw" Generative AI is structurally biased towards over-censorship and demonstrates that constraining LLMs with discriminative expert knowledge is a viable path to achieve robust, explainable, and balanced automated content moderation.

## Repository Structure

The codebase is divided into two main modules:

1. **`benchmark/`**: Scripts and SLURM jobs to evaluate base LMMs (InternVL 2.5, QwenVL, Phi-4, LLaVA) in a Zero-Shot setting across MAMI, Hateful Memes, and MultiOFF datasets.
2. **`experts/`**: The core framework implementing our Knowledge Injection. It includes the fine-tuning of discriminative experts (RoBERTa, MemeCLIP, CLIP) and the prompting strategies (Chain-of-Thought, Conditional Constraints) applied to InternVL 2.5 acting as the meta-reasoner.

---
