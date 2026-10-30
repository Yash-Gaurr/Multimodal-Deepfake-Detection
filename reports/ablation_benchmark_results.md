# Multimodal Deepfake Detection - Comparative Ablation Results

| Architecture / Mode | Accuracy (%) | AUC-ROC | EER (%) | F1-Score | Latency (ms/sample) |
| --- | --- | --- | --- | --- | --- |
| 1. Visual-Only (Video Baseline) | 25.00% | 0.1667 | 83.33% | 0.0000 | 48.1 |
| 2. Audio-Only (Audio Baseline) | 75.00% | 0.7500 | 25.00% | 0.8571 | 2.5 |
| 3. Late Fusion (Score Concat) | 25.00% | 1.0000 | 0.00% | 0.0000 | 16.2 |
| 4. Multimodal Cross-Attention (Proposed) | 25.00% | 0.1667 | 91.67% | 0.0000 | 32.1 |

