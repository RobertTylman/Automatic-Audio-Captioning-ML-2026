#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Load OPENAI_API_KEY from .env.local if present.
if [ -f "$SCRIPT_DIR/.env.local" ]; then
    set -a; . "$SCRIPT_DIR/.env.local"; set +a
fi

if [ -z "$OPENAI_API_KEY" ]; then
    echo "OPENAI_API_KEY is not set"
    exit 1
fi

# macOS + conda Python doesn't ship a usable CA bundle for urllib/requests,
# so OpenAI calls fail with CERTIFICATE_VERIFY_FAILED. Point SSL at certifi.
CERTIFI_PATH="$(python3 -c 'import certifi; print(certifi.where())' 2>/dev/null || true)"
if [ -n "$CERTIFI_PATH" ]; then
    export SSL_CERT_FILE="$CERTIFI_PATH"
    export REQUESTS_CA_BUNDLE="$CERTIFI_PATH"
fi

# Avoid numba/librosa cache locator issues in some macOS envs.
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_cache"
mkdir -p "$NUMBA_CACHE_DIR"

# 1) Generate 64 candidate captions per audio sample via nucleus sampling.
python3 -u inference_sampling.py \
    ./exp \
    config/nucleus_t0.5_p95.json \
    evaluation \
    True \
    ./config/conformer_config.json

# 2) Rerank candidates by audio-text semantic similarity.
python3 -u reranking/encoder_rerank_sampling_outputs.py \
    ./exp \
    ./exp/inference_evaluation_nucleus_t0.5_p95 \
    ./config/conformer_config.json \
    evaluation

# 3) Keep top 32 by audio-text similarity, compute pairwise cosine among them,
# then prompt GPT with SAP to synthesize one final caption per audio.
# Optional args: <openai_model> <top_audio_keep> <top_pairwise_keep>
python3 -u reranking/llm_summarize_sap.py \
    ./exp/inference_evaluation_nucleus_t0.5_p95 \
    gpt-4.1-mini \
    32 \
    12

# 4) Evaluate final selected caption per file against references.
python3 -u evaluate.py \
    "./exp/inference_evaluation_nucleus_t0.5_p95/llm_sap_summary_output.csv" \
    evaluation
