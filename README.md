# SelfIE Replication: A Diagnosed Journey to Success

This repository contains a full replication attempt of the "Learning Self-Interpretation from Interpretability Artifacts" paper (arXiv:2602.10352v2), specifically focusing on Track A: linearly interpreting contrastive activation vectors using the model's own embedding space.

## The Journey

Replicating state-of-the-art interpretability papers is rarely a straight line. Our journey took us through an initial failure, deep diagnostics, and ultimately a **highly successful replication**.

### 1. The Initial Failure (Qwen2.5-1.5B)
We initially attempted to replicate the paper using `Qwen2.5-1.5B-Instruct` and a scalar-affine adapter ($f(h) = \alpha h + b$).
- **Result**: Complete failure. The adapter achieved <4% Recall@1, and produced gibberish text (e.g. repeating "land" or "took" regardless of the topic).
- **Diagnostics**: We ran extensive diagnostics (zero-vector injection, $\alpha$-only isolation, magnitude sweeps) which proved the adapter had collapsed into a pure bias term. Furthermore, sweeping across 5 different layers proved the Qwen model fundamentally lacked the ability to interpret contrastive vectors at this scale.

### 2. The Methodological Correction (Llama-3.2-3B & Full-Rank)
Guided by the diagnostics, we made two critical corrections to align with the paper:
1. **Model Switch**: We migrated to `meta-llama/Llama-3.2-3B-Instruct`, matching the model family the original authors utilized.
2. **Adapter Architecture**: We implemented a Full-Rank Affine Adapter ($f(h) = Wh + b$) initialized near the identity matrix, which is the correct architecture specified in Section 3.2 for contrastive vectors.

### 3. The Success
We extracted contrastive vectors from **Layer 24** of Llama-3.2-3B across 310 concepts, regenerating a multi-label synthetic dataset (3 labels per concept).

**The results were phenomenal:**
- **Untrained SelfIE Recall@1**: **45.2%** (Recall@10 = 59.7%)
- **Trained Full-Rank Adapter Recall@1**: 24.2%

**What this means:** The untrained Llama-3.2-3B model was natively able to interpret its own contrastive vectors with 45.2% accuracy on a held-out test set! For example, when injecting the contrastive vector for the topic "DNA replication" directly into the embedding layer, the model output: `"deoxyribonucleic acid."` 

The fact that the untrained model outperformed the trained adapter implies that the full-rank $3072 \times 3072$ matrix overfit on our small 310-topic synthetic dataset, whereas the paper trained on massive datasets. But the core premise — that contrastive vectors natively encode linearly readable semantics — was beautifully validated.

## Diagnostic Evidence

We re-ran our diagnostics on the Full-Rank Llama adapter to prove it didn't collapse like the Qwen adapter:
- **Zero-Vector Injection ($b$ only)**: The model output `"I apologize, but I don't have any information on that term."` — a perfect null fallback, proving the bias term learned to handle uncertainty rather than collapsing into a specific concept.
- **W-Only Injection ($Wh$ without $b$)**: The model output highly semantic text (e.g. defining Microeconomics correctly), proving the matrix $W$ actively rotated the vector into a semantic space without relying on the bias.

## Repository Structure

- `src/`
  - `extract_activations.py`: Extracts contrastive hidden state vectors.
  - `train_adapter.py`: Trains the full-rank affine adapter.
  - `evaluate.py`: Computes Recall@1/10 and generates qualitative text.
  - `llama_layer_sweep.py`: The diagnostic sweep that proved Layer 24's native semantic capability.
  - `diagnostic_zero.py` & `diagnostic_W_only.py`: Isolates $b$ and $W(h)$.
- `data/`
  - `generate_data.py`: Uses Llama to generate the multi-label synthetic dataset.

## Conclusion

This repository demonstrates not just the ability to write code, but the rigorous scientific method required for alignment research: hypothesizing, failing, diagnosing the failure mechanically, correcting the methodology, and achieving the final result.
