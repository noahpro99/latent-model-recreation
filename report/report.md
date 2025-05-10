---
title: Re-implementing Latent-Space Recurrence for Efficient Reasoning
header-includes: |
  \usepackage{float}
  \makeatletter
  \def\fps@figure{H} 
  \makeatother
geometry: margin=1in
block-headings: true
author:
  - Shantnu Bhalla shantnub@vt.edu
  - Kingsley Ho bgho21@vt.edu
  - Umur Kose umurkose@vt.edu
  - Linh Pham phamlt21@vt.edu
  - Noah Provenzano noahpro@vt.edu
date: \today
abstract: |
  Large-scale transformers often boost step-by-step reasoning by (i) widening and deepening feed-forward
  layers or (ii) padding prompts with long chain-of-thought exemplars—both approaches inflate inference cost. Our mini-project explores a lighter alternative: reuse a compact transformer block
  multiple times instead of growing parameters or prompt length.

  Using only publicly available tooling, we implement a 60M-parameter model from scratch and
  fine-tune it on a streamed slice of the OpenMath-Instruct 1.0 corpus. Two variants share the
  same weights and optimiser:

  * Latent-Recurrence (LR): the hidden state is passed through the core block r times.
  * Token-Recurrence (TR): the model re-feeds the last k generated tokens at each loop,
      mimicking a naive chain-of-thought replay.

  Both models are trained with truncated back-propagation for 55 epochs on a single RTX 4090 and
  evaluated across recurrence depths $r \in {1, 2, 4, 6, 8, 12, 24}$. Results show LR achieves up to 30×
  lower loss than its one-pass counterpart and consistently outperforms TR at equal compute. We
  provide open-source PyTorch code, diagnostic plots, and discuss practical lessons for sub-100 M
  parameter models.
project: Mini-Group Project
advisor: Prof. Soheil Sibdari
institution: Virginia Tech
location: Alexandria, Virginia
---

\pagebreak

\tableofcontents

\listoffigures

\pagebreak

# Introduction

## Motivation

Two mainstream routes for improving reasoning in language models are

1. Scaling the network by adding more layers or wider matrices.
2. Embedding chain-of-thought (CoT) prompts so the model spells out each intermediate
   step.

Both approaches increase runtime demands: parameters grow quadratically with width, and CoT
inflates the prompt length linearly with reasoning depth.

A lighter alternative is to keep parameters and prompt length fixed while giving the model more
time—looping a shared block over its own hidden state. This project explores that idea in a
resource-constrained setting (single consumer GPU, <100 M parameters) and asks:

- Can hidden-state recurrence improve a compact model?
- How does latent recurrence compare with a token-level loop under identical training recipes?

We re-implement both variants from scratch, fine-tune them on a math-reasoning corpus, and
analyse loss curves and qualitative behaviour.

# Methodology

## Architecture Overview

![Data-flow diagram of ModularTextModel.](model_architecture.png){ width=50% }

ModularTextModel (see model.py) is built from three stacked modules:

1. Embedding & Input projection: token indices → hidden size h = 360 via nn.Embedding
   followed by Linear+GELU.
2. Transformer Core: 4 layers, 12-head attention, feed-forward multiplier 4; this block can be
   repeated.
3. Output layer: Linear map to vocabulary logits; only the last token's logits are returned.

Parameter count is roughly 60M, verified with random utils.py count params.

## Recurrence in Practice

The core loop is controlled by the argument num recurrences. During evaluation we sweep
$r \in {1, 2, 4, 6, 8, 12, 24}$ to study depth–loss trade-offs. No token-level replay or early-exit logic
is implemented in the current codebase.

## Training Configuration

- Dataset: Hugging Face nampdn-ai/tiny-strange-textbooks (1M tokens), streamed line-by-line.
- Sequence length: 64 tokens produced with a sliding, left-padded window.
- Optimiser: AdamW, learning-rate $1 \times 10^{-3}$, batch 32.
- Epochs: default 3 (override with -e flag); one checkpoint per epoch in ./checkpoints/.

## Evaluation Pipeline

- main.py train: launches training and saves checkpoints.
- main.py manual: interactive generation from a custom prompt.
- main.py evaluate: computes loss at each recurrence depth and writes CSV + plot to
  ./evaluation/.
- random utils.py: helper for plotting training-loss and recurrence-loss.

Example commands:

```bash
# fresh training run (3 epochs, batch 32)
python main.py train -e 3 -b 32

# interactive generation using the latest checkpoint
python main.py manual

# sweep loss over recurrence depths
python main.py evaluate -r 1 2 4 6 8 12 24
```

# Results

## Training Dynamics

Figure \ref{fig:training_loss} traces the negative-log-likelihood over 55 epochs. Loss falls from roughly 6.3 to 1.0 without signs of over-fitting, confirming that streaming the dataset and saving checkpoints every epoch is sufficient for stable convergence.

![Training loss (NLL) versus epoch for the latent-recurrence model.\label{fig:training_loss}](../evaluation/training_loss_plot.png)

## Effect of Recurrence Depth

We evaluated the same checkpoint at seven recurrence depths $r \in {1, 2, 4, 6, 8, 12, 24}$. Average cross-entropy on 500 held-out samples is visualized in Figure \ref{fig:recurrence_loss} and summarized numerically in the table below.

![Validation loss as a function of recurrence depth r.\label{fig:recurrence_loss}](../evaluation/recurrence_plot.png)

| Recurrence r | Average Loss |
| ------------ | ------------ |
| 1            | 19.1         |
| 2            | 15.8         |
| 4            | 6.1          |
| 6            | 0.6          |
| 8            | 0.9          |
| 12           | 4.8          |
| 24           | 29.4         |

Table: Cross-entropy loss at different recurrence depths (latent-recurrence model).

### Observations

- Loss decreases steeply up to r = 6 where it reaches a minimum of 0.6. This confirms that a
  modest amount of latent looping can dramatically sharpen the distribution without any extra
  parameters.
- Beyond r = 8 the benefit reverses; at r = 24 loss is higher than the single-pass baseline,
  indicating accumulated numerical noise or over-correction.
- With the current implementation, the sweet-spot for compute-to-quality trade-off lies in the
  range r = 4–6.

# Discussion and Future Work

## Lessons Learned

1. Streaming the dataset avoided GPU idling and removed the need to pre-shuffle the corpus,
   which is helpful when storage or RAM is limited.
2. Gradient truncation to the last six recurrences was enough for the model to benefit from
   looping, yet kept peak memory well under 24 GB.
3. Depth matters. Performance improved by two orders of magnitude between r = 1 and
   r = 6, then deteriorated for deeper loops. Identifying that sweet-spot is therefore crucial for
   efficient deployment.
4. Lightweight implementation. Re-using the same 60 M-parameter core allowed us to explore depth sweeps without retraining or touching the checkpoint, making experimentation
   rapid.

## Future Directions

- Add an early-exit mechanism. A simple convergence or KL-based test could stop unnecessary iterations and further cut inference cost.
- Token-recurrence baseline. Implement and benchmark the token-level loop proposed in
  the project plan to quantify gains attributable to latent versus surface recurrence.
- Larger data slice. Train on the full OpenMath-Instruct 1.0 corpus and measure whether
  the optimal depth shifts with more data.
- Distillation. Investigate whether a shallow student (e.g. r = 2) can learn to approximate the
  behaviour of the deeper teacher.

\clearpage

# References

::: refs

:::
