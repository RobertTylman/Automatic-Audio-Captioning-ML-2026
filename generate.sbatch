#!/bin/bash
#SBATCH --job-name=aac_generate
#SBATCH --account=torch_pr_1003_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/yd3245/logs/aac_generate-%j.out
#SBATCH --error=/scratch/yd3245/logs/aac_generate-%j.err

set -eo pipefail

mkdir -p /scratch/yd3245/logs

REPO_ROOT="/home/yd3245/machine_listening/Automatic-Audio-Captioning-ML-2026"
CONFIG_PATH="configs/all_together_learned_resampling_no_scheduler.yaml"
CHECKPOINT_PATH="/scratch/rht9410/clotho/run_outputs/all_together_learned_resampling_no_scheduler/checkpoints/aac-epochepoch=07-vallossval_loss=3.6350.ckpt"
OUTPUT_DIR="/home/yd3245/machine_listening/Automatic-Audio-Captioning-ML-2026/predictions_scores/all_together_learned_resampling_no_scheduler/greedy"
AUDIO_DIR="/scratch/rht9410/clotho/evaluation"
CAPTION_CSV="/scratch/rht9410/clotho/clotho_captions_evaluation.csv"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

srun singularity exec --nv --fakeroot \
  --bind /home:/home \
  --bind /scratch:/scratch \
  --bind /projects:/projects \
  --overlay /scratch/rht9410/clotho/images/overlay-15GB-500K-aac.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -lc "
source /ext3/env.sh
eval \"\$(conda shell.bash hook)\"
conda activate aac-dcase24

export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONNOUSERSITE=1

cd \${REPO_ROOT}

python -u scripts/evaluation/generate_outputs.py \
  --config \${CONFIG_PATH} \
  --checkpoint \${CHECKPOINT_PATH} \
  --output-dir \${OUTPUT_DIR} \
  --audio-dir \${AUDIO_DIR} \
  --caption-csv \${CAPTION_CSV} \
  --mode greedy

echo "Generation complete. Starting scoring..."

python -u scripts/evaluation/score_outputs.py \
  --predictions-csv \${OUTPUT_DIR}/greedy_output.csv \
  --references-csv \${CAPTION_CSV} \
  --output-dir \${OUTPUT_DIR}/scores
"
