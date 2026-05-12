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

### Gradio Demo

Install the environment:

```bash
conda create -n aac python=3.10
conda activate aac
pip install -r requirements.txt
```

Make a trained checkpoint available somewhere on the machine running the demo.
Then pass its path with `--checkpoint`:

```bash
python -u demo.py \
  --config configs/beats_convnext_ast_sequence_fusion.yaml \
  --checkpoint "insert/checkpoint/filepath/here.ckpt" \
  --device cpu
```

For example, if you want to download the checkpoint from Google Drive first:

```bash
mkdir -p checkpoints
gdown "https://drive.google.com/uc?id=1rBX1yiRJbqVtOOuOKF7RhwZSqoixKu8U" \
  -O checkpoints/aac-epochepoch=02-vallossval_loss=3.5995.ckpt
```

Then launch the local upload-and-caption UI:

```bash
python -u demo.py \
  --config configs/beats_convnext_ast_sequence_fusion.yaml \
  --checkpoint checkpoints/aac-epochepoch=02-vallossval_loss=3.5995.ckpt \
  --device cpu
```

Open the printed local URL, upload an audio file, and the app will return one
greedy-decoded caption from the trained audio captioning model. By default,
`demo.py` skips the placeholder upstream encoder checkpoint paths in the YAML
and loads weights from the trained Lightning checkpoint instead. If you want to
force the YAML bootstrap weights to load first, pass
`--no-skip-pretraining-bootstrap`.
If you need a specific port, add `--server-port 7860`; otherwise Gradio will
choose an open port automatically.

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
