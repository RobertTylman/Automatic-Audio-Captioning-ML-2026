# Evaluation Scripts

These scripts are intentionally separate from training config. Pass the model
YAML/checkpoint plus the generation options you want for a particular report.

## Generate Captions

Greedy/model-only output:

```bash
python scripts/evaluation/generate_outputs.py \
  --config configs/all_together.yaml \
  --checkpoint /path/to/checkpoint.ckpt \
  --output-dir /path/to/report/greedy \
  --split eval \
  --audio-dir /path/to/clotho_audio_evaluation \
  --caption-csv /path/to/clotho_captions_evaluation.csv \
  --mode greedy
```

Nucleus sampling + CLAP/decoder reranking + optional ChatGPT SAP summary:

```bash
python scripts/evaluation/generate_outputs.py \
  --config configs/all_together.yaml \
  --checkpoint /path/to/checkpoint.ckpt \
  --output-dir /path/to/report/rerank \
  --split eval \
  --audio-dir /path/to/clotho_audio_evaluation \
  --caption-csv /path/to/clotho_captions_evaluation.csv \
  --mode rerank \
  --top-p 0.95 \
  --temperature 0.5 \
  --num-return-sequences 64
```

Use `--no-llm` to keep the best reranked candidate without calling the OpenAI
API. If the output CSV already exists, generation resumes by skipping completed
`file_name` rows. Checkpoint loading is strict by default; use
`--allow-partial-checkpoint` only when you intentionally want to inspect a
partially compatible checkpoint.

## Score Saved Captions

```bash
python scripts/evaluation/score_outputs.py \
  --predictions-csv /path/to/report/greedy/greedy_output.csv \
  --references-csv /path/to/clotho_captions_evaluation.csv \
  --output-dir /path/to/report/greedy/scores \
  --metrics dcase2024
```

`dcase2024` reports METEOR, CIDEr-D, SPICE, SPIDEr, SPIDEr-FL, FENSE, FER,
SBERT similarity, and Vocabulary through `aac-metrics`.
