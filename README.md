# Subjective Q&A Labeling with ModernBERT

A portfolio overview of an NTHU natural language processing course project for the Google QUEST Q&A Labeling task. The system predicts 30 subjective question and answer attributes from Stack Exchange text and is evaluated with mean Spearman rank correlation.

The final system reached a mean Spearman correlation of **0.422**, a reported **20.57% relative improvement** over the BERT baseline.

## Approach

- Clean HTML, normalize LaTeX as `[MATH]`, mask digits, and expand contractions.
- Preserve both ends of long question bodies with head-tail truncation.
- Encode category, host, title, body, and answer in one structured sequence.
- Use ModernBERT with learnable weighted pooling across layers 4-12.
- Concatenate CLS, mean, and max pooled representations.
- Train with multi-sample dropout and a hybrid BCE/correlation objective.
- Align predictions with the empirical label distribution through quantile snapping.

## Selected Results

| Experiment | Mean Spearman |
| --- | ---: |
| Baseline model | 0.37 |
| No post-processing | 0.3858 |
| Quantile snapping | 0.4026 |
| Sentiment features | 0.4184 |
| Final ModernBERT system | 0.422 |

## Portfolio Note

This repository contains a public project overview only. The implementation and course deliverables remain in the private team repository so collaborators' work is not republished without their permission.

No source code or reusable assets are distributed from this repository.
