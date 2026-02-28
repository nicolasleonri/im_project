#!/bin/bash
#SBATCH --job-name=mlm_pretrain_beto
#SBATCH --output=logs/mlm/mlm_pretrain_beto_%j.out
#SBATCH --partition=scavenger
#SBATCH --account=agfritz
#SBATCH --qos=prio
#SBATCH --nodes=1
#SBATCH --ntasks=1

#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=5G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=01:00:00

# Load necessary modules
module purge
module load CUDA/12.4.0
# module load cuDNN/8.9.2.26-CUDA-12.1.1
module load virtualenv/20.26.2-GCCcore-13.3.0

# Environment variables
export HF_HOME=/scratch/nicolasal97/.cache/huggingface

# Activate virtual environment
source venv/mlm/bin/activate

echo "Setup done. Running python script..."
####### FIRST TASK: PREPROCESS CORPUS #######
## Needs 1x1GB, 1 CPU and around 10 minutes
# python3 -u src/mlm/preprocess_corpus.py \
#     --input_dir data/mlm \
#     --output_file data/mlm/corpus_clean.txt

###### SECOND TASK: PRETRAIN BETO #######
## Needs 1x5GB, 1xH100 and around 1 hour
python3 -u src/mlm/pretrain_beto.py \
    --corpus_file data/mlm/corpus_clean.txt \
    --output_dir /scratch/nicolasal97/im_project/beto-cgec \
    --per_device_train_batch_size 64 \
    --bf16

deactivate
module purge
echo "Script finished"

# sbatch --dependency=afterok:JOBID preprocessing.bash