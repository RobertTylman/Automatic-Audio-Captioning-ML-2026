# Scaffold Pipeline

Right now the repo includes: 

- BEATs-style encoder adapter with multi-layer aggregation
- ConvNeXt dummy encoder adapter
- configurable encoder fusion
- Conformer post-encoder
- BART-style decoder with cross-attention
- PyTorch Lightning training loop
- wandb logging


The following parts are not implemented yet:

- CLAP filtering
- hybrid reranking
- LLM summarization
- real ConvNeXt checkpoint loading
- real BEATs checkpoint loading

## Project Layout

```text
Automatic-Audio-Captioning-ML-2026/
├── configs/
│   └── baseline_dcase24.yaml
├── scripts/
│   └── train.py
├── src/
│   └── aac_baseline/
│       ├── data/
│       ├── models/
│       │   └── encoders/
│       └── training/
└── requirements.txt
```

## Train

```bash
python3 scripts/train.py --config configs/baseline_dcase24.yaml
```