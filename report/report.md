---

title: Re‑implementing Latent‑Space Recurrence for Efficient Reasoning
bibliography: bib.bib
header-includes: |
  \usepackage{float}
  \makeatletter
  \def\fps@figure{H} 
  \makeatother
geometry: margin=1in
block-headings: true
author:
  - Shantnu Bhalla  <shantnub@vt.edu>
  - Kingsley Ho  <bgho21@vt.edu>
  - Umur Kose  <umurkose@vt.edu>
  - Linh Pham  <phamlt21@vt.edu>
  - Noah Provenzano  <noahpro@vt.edu>
date: \today
abstract: |
  Large scale transformers often boost step‑by‑step reasoning by (i) widening and deepening feed‑forward layers or (ii) padding prompts with long chain‑of‑thought 
  exemplars both strategies inflate inference cost.
  Our project explores a lighter alternative: **latent recurrence**, where a compact transformer block is
  reused several times instead of growing parameters or prompt length.

  Using only publicly available tooling, we implement a **117M param**
  recurrent language model. The hidden state is passed through the core block $r$ times, and the same checkpoint is evaluated at depths
  $r \in \{1, 2, 4, 6, 8, 12, 24\}$.  
  Training runs for a handful of epochs; loss drops by roughly 30× between one pass and the
  optimal $r=6$, after which further looping degrades performance revealing a clear efficiency sweet spot for this model size.

  **Code**: <https://github.com/noahpro99/latent-model-recreation>

project: Mini‑Group Project
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
2. Embedding chain-of-thought prompts so the model spells out each intermediate step.

Both approaches increase runtime demands: parameters grow quadratically
with width, and CoT inflates the prompt length linearly with reasoning
depth.

A lighter alternative is to keep parameters and prompt length fixed while
giving the model more time looping over its own hidden state. This idea
was explored in *Scaling up Test-Time Compute with Latent Reasoning*,
showing strong gains at billion parameter scale. We replicate the concept
in a resource‑constrained setting (~100M params) and
ask:

* Can hidden state recurrence improve a compact model?  
* Where is the sweet spot depth for compute‑to‑quality trade‑off?

# Methodology

## Architecture Overview

![Data‑flow diagram of RecurrentTransformerModel](model_architecture.png){ width=50% }

`RecurrentTransformerModel` (see `model.py`) stacks:

1. **Embedding & positional encoding**: token IDs plus learned position embeddings $\to$ hidden size $h=768$.  
2. **Input transformer block**: single self‑attention + FF pass.  
3. **Recurrent core**: list of 8 transformer blocks, the *entire list* is repeated $r$ times.  
4. **Output transformer block**: one final refinement pass.  
5. **LayerNorm + Linear head**: maps to vocabulary logits for *all* sequence positions.

Total size ~117M parameters (`random_utils.py count_params`).

## Recurrence Sweep

Depth is controlled by `num_recurrences`. After training with `num_recurrences = 6`, we reload the checkpoint and re‑run inference at
$r \in {1,2,4,6,8,12,24}` to measure sensitivity.

## Training Configuration

* **Dataset**: Hugging Face `nampdn-ai/tiny-strange-textbooks` (streaming).  
* **Sequence length**: 128 tokens, sliding left‑padded window.  
* **Batch size**: 32 during training, 4 in the DataLoader defaults.  
* **Optimizer**: AdamW, LR $1\times10^{-3}$.  
* **Epochs**: 3 (default), checkpoint saved each epoch.

## Evaluation Pipeline

* `main.py train`: start training / resume from checkpoint.  
* `main.py evaluate`: compute loss at chosen depths, save CSV + plot.  
* `random_utils.py`: generate `training_loss_plot.png` and `recurrence_plot.png`.

```bash
# fresh 3‑epoch run
python src/main.py train -e 3 -b 32

# sweep recurrence depths
python src/main.py evaluate -r 1 2 4 6 8 12 24

```
# Results

## Training Dynamics

Figure \ref{fig:training_loss} traces the negative-log-likelihood over 55 epochs. Loss falls from roughly 6.3 to 1.0 without signs of over-fitting, confirming that streaming the dataset and saving checkpoints every epoch is sufficient for stable convergence.

![Training loss (NLL) versus epoch for the latent-recurrence model\label{fig:training_loss}](../evaluation/training_loss_plot.png)

## Effect of Recurrence Depth

After training the model at a recurrence depth of 6, we evaluated this checkpoint at seven other recurrence depths $r \in {1, 2, 4, 6, 8, 12, 24}$. Average cross-entropy on 500 held-out samples is visualized in Figure \ref{fig:recurrence_loss} and summarized numerically in the table below.

![Validation loss as a function of recurrence depth r\label{fig:recurrence_loss}](../evaluation/recurrence_plot.png)

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

- Loss decreases steeply up to r = 6 where it reaches a minimum of 0.6. This makes sense as the model was trained at a recurrence depth of 6. This shows that latent recurrence has a significant impact on model performance.
- Beyond r = 8 the benefit reverses; at r = 24 loss is higher than the single-pass baseline, showing how sensitive the model is to its training recurrence depth.
- With the current implementation, the sweet-spot for compute-to-quality trade-off lies in the range r = 4-6.

# Discussion and Future Work

## Lessons Learned

1. Streaming the dataset avoided GPU idling and removed the need to pre-shuffle the corpus, which is helpful when storage or RAM is limited.
2. Qualitatively, the model was not big enough to produce intelligible text. The output was often gibberish, although noticeably less so as we made the model deeper. This is likely due to the limited size of the training set and the model itself.
3. Depth matters. Performance improved by two orders of magnitude between r = 1 and r = 6, then deteriorated for deeper loops. Identifying that sweet-spot is therefore crucial for efficient deployment.
4. Lightweight implementation. Re-using the same 28M parameter core allowed us to explore depth sweeps without retraining or touching the checkpoint, making experimentation rapid.

## Future Directions

- Token-recurrence baseline. Train and benchmark the model at separate recurrence levels proposed in the project plan to quantify gains attributable to latent versus surface recurrence.
- Better benchmark. Train using reinforcement learning to better benchmark the performance instead of just loss on the full corpus to measure whether the optimal depth shifts.
- Distillation. Investigate whether a shallow student (e.g. r = 2) can learn to approximate the behaviour of the deeper teacher.

\clearpage

# References

::: refs

:::
