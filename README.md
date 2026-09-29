# Subjective Q&A Labeling with ModernBERT

A multi-target regression system for the Google QUEST Q&A Labeling task. The model predicts 30 subjective question and answer attributes from Stack Exchange text and is evaluated by mean Spearman rank correlation.

The final system reached a mean Spearman correlation of **0.422**, a reported **20.57% relative improvement** over the BERT baseline.

## Approach

- Cleans HTML, normalizes LaTeX as `[MATH]`, masks digits, and expands contractions.
- Preserves both ends of long question bodies with head-tail truncation.
- Encodes category, host, title, body, and answer in one structured sequence.
- Uses ModernBERT with learnable weighted pooling across layers 4-12.
- Concatenates CLS, mean, and max pooled representations.
- Applies multi-sample dropout to reduce training variance.
- Trains with `0.9 * BCE + 0.1 * differentiable Spearman loss`.
- Aligns predictions with the empirical label distribution through quantile snapping.

## Selected Results

| Experiment | Mean Spearman |
| --- | ---: |
| Baseline model | 0.37 |
| No post-processing | 0.3858 |
| Quantile snapping | 0.4026 |
| Sentiment features | 0.4184 |
| Final ModernBERT system | 0.422 |

The report also documents negative results, including adversarial weight perturbation and XGBoost feature fusion. Keeping these experiments makes the repository useful as an honest record of model development.

## Repository Guide

```text
src/preprocess.py    text cleaning and structured input construction
src/loss.py          hybrid BCE and correlation objective
src/postprocess.py   distribution-aware prediction adjustment
train_v2.ipynb       main training workflow
inference.ipynb      inference and submission workflow
docs/                final report and presentation
```

Additional upstream branches preserve the AWP, XGBoost, DeBERTa, and early ModernBERT experiments.

## Setup

Python 3.10 or newer is required. With `uv` installed:

```bash
uv sync
```

The competition dataset and trained model weights are not committed. Download the Google QUEST Q&A Labeling data from Kaggle and follow [KAGGLE_GUIDE.md](KAGGLE_GUIDE.md) to configure the notebook paths.

## Team and Attribution

This is a four-person NTHU course project by Shaochieh Huang, An Li, Weifan Ching, and Yuchen Li. The code history is preserved from the [original team repository](https://github.com/Windmill10/NLP_final).

See the [final report](docs/final-report.pdf) for the complete architecture, ablations, and references, and the [presentation](docs/presentation.pptx) for a concise overview.

No open-source license is granted for reuse.
