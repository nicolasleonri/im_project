#!/bin/bash
#SBATCH --job-name=characterize_models
#SBATCH --output=logs/evaluating/characterize_models_%j.out
#SBATCH --partition=scavenger
#SBATCH --account=agfritz
#SBATCH --qos=prio
#SBATCH --nodes=1
#SBATCH --ntasks=1

#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=01:00:00

# Load necessary modules
module purge
module load CUDA/12.4.0
module load virtualenv/20.26.2-GCCcore-13.3.0

# Environment variables
export HF_HOME=/scratch/nicolasal97/.cache/huggingface
# export TRANSFORMERS_OFFLINE=1
# export HF_DATASETS_OFFLINE=1

# Activate virtual environment
source venv/mlm/bin/activate 

echo "Setup done. Running python script..."

python3 scripts/evaluating/characterize_datasets.py \
    --datasets_dir       data/preprocessing/datasets/ \
    --target_corpus      data/mlm/corpus_clean.txt \
    --target_test_set    data/preprocessing/test_set_final.csv \
    --xlmr_base          xlm-roberta-base \
    --xlmr_cgec          /scratch/nicolasal97/im_project/xlm-roberta-cgec \
    --output             results/characterization4.csv

deactivate
module purge
echo "Script finished"

    # --roberta_base       skimai/spanberta-base-cased \
    # --roberta_cgec       /scratch/nicolasal97/im_project/spanberta-cgec \
    
        # --beto_base          dccuchile/bert-base-spanish-wwm-uncased \
    # --beto_cgec          /scratch/nicolasal97/im_project/beto-cgec \

# sbatch --dependency=afterok:JOBID preprocessing.bash