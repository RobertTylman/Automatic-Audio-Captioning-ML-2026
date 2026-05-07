#!/bin/bash
#SBATCH --job-name=aac_train_1gpu
#SBATCH --account=torch_pr_1003_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/yd3245/logs/aac_train_1gpu-%j.out
#SBATCH --error=/scratch/yd3245/logs/aac_train_1gpu-%j.err

set -eo pipefail

mkdir -p /scratch/yd3245/logs

REPO_ROOT="/home/yd3245/machine_listening/Automatic-Audio-Captioning-ML-2026"
CONFIG_PATH="/home/yd3245/machine_listening/Automatic-Audio-Captioning-ML-2026/configs/all_together_learned_resampling_no_scheduler.yaml"
WANDB_MODE_VALUE="online"

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
export WANDB_MODE=${WANDB_MODE_VALUE}

cd ${REPO_ROOT}

python -u scripts/train.py \
  --config ${CONFIG_PATH}
"
