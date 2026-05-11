# Automatic Audio Captioning ML 2026

This repo trains and evaluates audio captioning models built from BEATs, ConvNeXt, and AST encoders with three fusion options:

## Environment

```bash
conda create -n aac python=3.10
conda activate aac
pip install -r requirements.txt
```

If you want OpenAI-based post-processing during inference, you can set the key in a repo-local `.env` file:

```bash
OPENAI_API_KEY=your_key_here
```

## Config Guide
We provide the configuration files used to train our different model configurations.

One can use the first six configs for standard Clotho training:

- `beats_convnext_feature_fusion`
- `beats_convnext_learned_resampling_fusion`
- `beats_convnext_sequence_fusion`
- `beats_convnext_ast_feature_fusion`
- `beats_convnext_ast_learned_resampling_fusion`
- `beats_convnext_ast_sequence_fusion`

The four `pretrain_*.yaml` configs can be used when one wants to pretrain on AudioCaps + Clotho mixup captions before moving back to Clotho-only training.

## Training

Basic training:

```bash
python scripts/train.py --config configs/beats_convnext_ast_learned_resampling_fusion.yaml
```

Override hardware from the command line when needed:

```bash
python scripts/train.py \
  --config configs/beats_convnext_ast_learned_resampling_fusion.yaml \
  --devices 2 \
  --strategy ddp
```

Resume the exact same run from a Lightning checkpoint:

```bash
python scripts/train.py \
  --config configs/beats_convnext_ast_learned_resampling_fusion.yaml \
  --ckpt-path path/to/checkpoints/last.ckpt
```

`--ckpt-path` restores model weights, optimizer state, and trainer progress.

## Pretraining Then Finetuning

Pretraining example:

```bash
python scripts/train.py \
  --config configs/pretrain_beats_convnext_ast_learned_resampling_audiocaps_mixup.yaml \
  --devices 2 \
  --strategy ddp
```

Finetuning is the easy part: switch to the matching Clotho-only config and initialize from the pretrain checkpoint.

- `pretrain_beats_convnext_learned_resampling_audiocaps_mixup.yaml` -> `beats_convnext_learned_resampling_fusion.yaml`
- `pretrain_beats_convnext_sequence_audiocaps_mixup.yaml` -> `beats_convnext_sequence_fusion.yaml`
- `pretrain_beats_convnext_ast_learned_resampling_audiocaps_mixup.yaml` -> `beats_convnext_ast_learned_resampling_fusion.yaml`
- `pretrain_beats_convnext_ast_sequence_audiocaps_mixup.yaml` -> `beats_convnext_ast_sequence_fusion.yaml`

Example finetune command:

```bash
python scripts/train.py \
  --config configs/beats_convnext_ast_learned_resampling_fusion.yaml \
  --init-from-checkpoint path/to/pretrain.ckpt
```

`--init-from-checkpoint` loads weights only. It does not resume optimizer state or epoch counters, which makes it the right option for finetuning.

## Inference

### Greedy decoding

```bash
python scripts/evaluation/generate_outputs.py \
  --config configs/beats_convnext_ast_learned_resampling_fusion.yaml \
  --checkpoint path/to/model.ckpt \
  --output-dir outputs/inference/greedy \
  --split eval \
  --audio-dir path/to/clotho/evaluation \
  --caption-csv path/to/clotho/clotho_captions_evaluation.csv \
  --mode greedy
```

### Reranking without LLM post-processing

This runs nucleus sampling plus CLAP-based reranking, then keeps the best reranked caption without calling OpenAI.

```bash
python scripts/evaluation/generate_outputs.py \
  --config configs/beats_convnext_ast_learned_resampling_fusion.yaml \
  --checkpoint path/to/model.ckpt \
  --output-dir outputs/inference/rerank_no_llm \
  --split eval \
  --audio-dir path/to/clotho/evaluation \
  --caption-csv path/to/clotho/clotho_captions_evaluation.csv \
  --mode rerank \
  --top-p 0.95 \
  --temperature 0.5 \
  --num-return-sequences 64 \
  --no-llm
```

### Reranking with ChatGPT-style post-processing

This uses the same reranking pipeline, then sends the top candidates through the SAP summarization prompt using the OpenAI API.

```bash
python scripts/evaluation/generate_outputs.py \
  --config configs/beats_convnext_ast_learned_resampling_fusion.yaml \
  --checkpoint path/to/model.ckpt \
  --output-dir outputs/inference/rerank_llm \
  --split eval \
  --audio-dir path/to/clotho/evaluation \
  --caption-csv path/to/clotho/clotho_captions_evaluation.csv \
  --mode rerank \
  --top-p 0.95 \
  --temperature 0.5 \
  --num-return-sequences 64 \
  --openai-model gpt-4.1-mini
```

## Notes

- The training script supports `--devices`, `--strategy`, `--ckpt-path`, and `--init-from-checkpoint`.
- The inference script supports both plain greedy decoding and rerank mode.
- The rerank mode can run with or without OpenAI post-processing depending on whether you pass `--no-llm`.