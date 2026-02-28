#!/bin/bash
#SBATCH --job-name=mlm_pretrain_beto
#SBATCH --output=logs/mlm/mlm_pretrain_beto_%j.out
#SBATCH --partition=scavenger
#SBATCH --account=agfritz
#SBATCH --qos=prio
#SBATCH --nodes=1
#SBATCH --ntasks=1

#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=10G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=01:00:00

# Load necessary modules
module purge
module load CUDA/12.4.0
# module load cuDNN/8.9.2.26-CUDA-12.1.1
module load virtualenv/20.26.2-GCCcore-13.3.0

# Environment variables
export HF_HOME=/scratch/nicolasal97/.cache/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

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
# python3 -u src/mlm/pretrain_beto.py \
#     --corpus_file data/mlm/corpus_clean.txt \
#     --output_dir /scratch/nicolasal97/im_project/beto-cgec \
#     --per_device_train_batch_size 64 \
#     --bf16

##### THIRD TASK: EVALUATE BETO #######
## Needs 1x5GB, 1xH100 and around 30 minutes
python -u src/mlm/finetune_beto.py \
    --model_name_or_path dccuchile/bert-base-spanish-wwm-uncased \
    --source_dataset data/preprocessing/datasets/gorrostieta_lopezlopez_2019.csv \
    --test_dataset data/preprocessing/test_set_final.csv \
    --task binary \
    --output_dir results/beto-base_guzman_binary \
    --n_trials 20 \
    --bf16

deactivate
module purge
echo "Script finished"

# sbatch --dependency=afterok:JOBID preprocessing.bash