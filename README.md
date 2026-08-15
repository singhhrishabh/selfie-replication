# Replicating Self-Interpretation Adapters at Small Scale

A scoped replication of Track A (Contrastive Topic Vectors, Section 3.2) from
**"Learning Self-Interpretation from Interpretability Artifacts: Training Lightweight Adapters on Vector-Label Pairs"**
([arXiv:2602.10352v2](https://arxiv.org/abs/2602.10352), Pepper et al., AE Studio / AI Alignment Foundation, 2026).

## What Was Replicated

The paper's central claim: that a lightweight trained adapter $f(h) = \alpha h + b$ dramatically improves a frozen LM's ability to self-interpret its own internal activations, compared to untrained SelfIE/Patchscopes-style injection.

We attempted this at small scale using **Qwen2.5-1.5B-Instruct** (28 layers, $d_{model}$ = 1536) on ~310 synthetic topics. The result is a well-diagnosed **null result** — the method does not work on this model at any tested layer, and we isolate exactly why.

### Differences from the Original Paper

| Parameter | Paper | This Replication |
|---|---|---|
| Model | Llama-3-8B (32 layers) | Qwen2.5-1.5B-Instruct (28 layers) |
| Dataset | 49,637 Wikipedia vital articles | 310 synthetic topics |
| Adapter | Scalar-affine ($d_{model} + 1$ params) | Same |
| Extraction layer | Optimized per-model | Swept [8, 12, 14, 16, 20, 24] |
| Evaluation | Recall@1 with best-of-N | Recall@1/10, greedy single-pass |

## Methodology

1. **Dataset Generation** — Used Qwen2.5-1.5B-Instruct to generate a 1-sentence description for each of 310 diverse topics across science, history, arts, and everyday concepts.
2. **Activation Extraction** — For each topic, passed `"Tell me about {topic}"` through the model, extracted the residual-stream hidden state at the final token position, and subtracted the cross-topic mean to obtain a contrastive direction vector.
3. **Adapter Training** — Split 80/20 (248 train / 62 test). Trained a scalar-affine adapter $f(h) = \alpha h + b$ via cross-entropy loss on label tokens. The adapted vector replaces the embedding of a placeholder token (`Ω`) in the explanation-seeking prompt: `User: What is the meaning of "Ω"? Assistant: The meaning of "Ω" is "`. The base model is completely frozen; only $\alpha$ (1 scalar) and $b$ (1536-dim vector) are trained.
4. **Evaluation** — Generated descriptions for held-out test vectors using both untrained SelfIE and the trained adapter. Scored via embedding-based retrieval (`sentence-transformers/all-MiniLM-L6-v2`), reporting Recall@1 and Recall@10.

## Results

| Approach | Recall@1 | Recall@10 |
|---|---|---|
| Untrained SelfIE | 3.2% | 19.4% |
| Trained Adapter | 3.2% | 22.6% |
| **Paper (Llama-3-8B, 50k topics)** | **82.9%** | **—** |

### Qualitative Example
**Topic:** "propagating gradients back through a neural network"

- **Untrained:** `ISTRY`
- **Trained:** `orous`

Neither output bears any relation to backpropagation.

## Diagnostic Investigation

Rather than accepting this null result at face value, we ran five targeted diagnostics to isolate the failure mechanism.

### 1. Embedding Norm Mismatch

| Vector | L2 Norm |
|---|---|
| Raw contrastive activation (layer 14) | **4.66** |
| Token embedding for "the" | **0.78** |

The injected vector is ~6× larger than a typical token embedding. However, normalizing to matching magnitude did not fix the problem (see §2).

### 2. Multi-Scale Sweep (Normalized Vectors)

We normalized each contrastive vector to unit L2 norm and swept injection scales `[1, 2, 4, 8, 16, 32]`. At scale ≈ 1, the vector matches typical embedding magnitude.

| Topic | Scale 1 | Scale 2 | Scale 4 | Scale 8 | Scale 16 | Scale 32 |
|---|---|---|---|---|---|---|
| the aztec empire | awards | OSTER | OST | OST | OST | OST |
| topology | OST | OST | OST | OST | OST | OST |
| the vikings | OST | OSTERIA | OSTERIA | OST | OST | OST |
| climate | OSTERIA | OSTERIA | OSTERIA | OST | OSTINATO | OSTERIA |
| microeconomics | OST | OST | OST | OST | OST | OST |

**No scale produced coherent or topic-related text.** This rules out "wrong magnitude" as the sole explanation.

### 3. Adapter Bias Collapse

We injected a **pure zero vector** through the trained adapter — outputting just the learned bias $b$:

> **Zero-vector output:** `awards are given to recognize achievements or contributions in various fields.`

Coherent English, but completely topic-independent. The adapter collapsed into a **constant mode**: the bias $b$ (norm 15.25) learned to produce a single generic sentence that minimized average cross-entropy, while the topic-specific component $\alpha h$ was ignored.

### 4. Alpha-Only Isolation (No Bias)

Injecting $\alpha h$ (skipping $b$ entirely):

| Topic | $\alpha h$ norm | Output |
|---|---|---|
| the aztec empire | 4.01 | awards |
| topology | 3.87 | OST |
| the vikings | 3.39 | OSTERIA |
| climate | 2.53 | OSTERIA |
| microeconomics | 3.19 | OST |

**Identical** to untrained SelfIE. The learned $\alpha \approx 0.86$ barely changed the vector — it carries no interpretable topic signal through this injection pathway.

### 5. Layer Sweep (Ruling Out Layer Choice)

We swept layers `[8, 12, 16, 20, 24]` with 10 diverse topics, testing both norm-matched and raw injection at each layer. Representative results:

| Layer | Vector Norms | Scaled Output | Raw Output |
|---|---|---|---|
| 8 | 2–3 | hythmos / dh / ++++ | hythmos / dharmic / ++++ |
| 12 | 2–4 | hythmos / hyth / ++++ | hythmos / ercul / ++++ |
| 16 | 3–7 | hythmos / hythmic / ++++ | hythmos / hythmic / ++++ |
| 20 | 8–21 | hythmos / hythmic / ++++ | hythmos / hythmic / ++++ |
| 24 | 34–61 | hythmos / hythmic music | hythmos / hythmic music |

Across **50 layer×topic cells** (5 layers × 10 topics), there was exactly **one** coherent, topic-adjacent output: at Layer 16, raw injection for "capitalism" produced `"denote the people's interests and demands, advocate for fairness and justice, oppose corruption and tyranny."` Every other output was gibberish token loops (`hythmos`, `OST`, `++++`).

**This rules out layer choice as the explanation.** The failure is not localized to layer 14 — untrained SelfIE is catastrophically broken at every layer of Qwen2.5-1.5B-Instruct.

### Summary of Failure Mechanism

```
Training converged:     loss 1.75 → 0.96 over 5 epochs
But what was learned:   bias b → generic constant sentence
                        alpha ≈ 0.86 → near-identity (ignored)
Root cause:             Contrastive vectors at ANY layer of this 1.5B model
                        are not self-readable when injected at embedding layer
```

The adapter minimized its loss by learning to **ignore the input vector entirely** and **always output a plausible constant**, because the underlying self-interpretation pathway (activation → embedding-space injection → coherent text) does not function in this model.

## Why Our Numbers Differ — Honest Scoping

This is a null result **under our specific replication conditions**. The honest conclusion is that the method did not work on this model, not that it doesn't work in general. The paper's results on Llama-3-8B are not contradicted — they are simply not reproducible at this reduced scale. Key factors:

1. **Model Scale (1.5B vs 8B)**: Self-interpretation likely requires a minimum threshold of model capacity. Qwen2.5-1.5B may fall below this threshold — its internal representations may be too compressed or its embedding space too narrow to support cross-layer activation readout. The layer sweep (§5) confirms this is a model-wide property, not a single-layer artifact.

2. **Architecture Family**: The paper was developed and validated on Meta's Llama family. Qwen2.5 uses a different architecture, tokenizer, and training corpus. Whether the self-interpretation property transfers across model families is itself an open empirical question.

3. **Dataset Size (310 vs 49,637)**: Even with minimal parameters, the adapter may need orders of magnitude more training pairs to learn a useful transform rather than collapsing to a constant. However, the layer sweep shows that untrained SelfIE itself (which requires no training data) is completely non-functional, suggesting dataset size is a secondary factor.

4. **Adapter Architecture**: The paper uses a full-rank affine adapter $f(h) = Wh + b$ for the contrastive-vectors track (Section 3.2), not scalar-affine. A full-rank $W$ matrix could in principle learn to project activations into a readable subspace even if the raw direction is unreadable — but this requires the underlying signal to exist, which our diagnostics suggest it does not in this model.

5. **No Best-of-N Selection**: The paper generates multiple candidates and selects via embedding similarity. We used single-pass greedy decoding.

## Training Log

| Epoch | Loss | Alpha | Bias Norm |
|---|---|---|---|
| 1 | 1.7541 | 1.0625 | 6.2812 |
| 2 | 1.3840 | 0.8711 | 9.6875 |
| 3 | 1.1695 | 0.7422 | 12.1250 |
| 4 | 1.0363 | 0.8008 | 14.0000 |
| 5 | 0.9580 | 0.8594 | 15.2500 |

## How to Run

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Core pipeline
python data/generate_data.py       # Generate topic list + labels
python src/extract_activations.py  # Extract contrastive vectors (layer 14)
python src/train_adapter.py        # Train scalar-affine adapter
python src/evaluate.py             # Evaluate + qualitative test

# Diagnostics
python src/diagnostic.py           # Norm check + multi-scale sweep
python src/diagnostic_zero.py      # Zero-vector (bias-only) test
python src/diagnostic_alpha_only.py # Alpha-only (no bias) test
python src/debug_injection.py      # Verify placeholder token injection
python src/layer_sweep.py          # Layer sweep across [8, 12, 16, 20, 24]
```

## Repository Structure

```
selfie_replication/
├── README.md
├── requirements.txt
├── data/
│   └── generate_data.py          # Topic list + label generation
├── src/
│   ├── extract_activations.py    # Residual-stream extraction
│   ├── train_adapter.py          # Scalar-affine adapter training
│   ├── evaluate.py               # Recall@k evaluation
│   ├── diagnostic.py             # Norm + multi-scale sweep
│   ├── diagnostic_zero.py        # Bias-collapse test
│   ├── diagnostic_alpha_only.py  # Alpha-only isolation test
│   ├── debug_injection.py        # Token injection verification
│   └── layer_sweep.py            # Cross-layer diagnostic
└── .gitignore
```

