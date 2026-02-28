#!/bin/bash
#SBATCH --job-name=mlm_pretrain_beto
#SBATCH --output=logs/mlm/mlm_pretrain_beto_%j.out
#SBATCH --partition=scavenger
#SBATCH --account=agfritz
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1

#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=5G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=01:00:00

# Load necessary modules
module purge
module load virtualenv/20.26.2-GCCcore-13.3.0

# Activate virtual environment
source venv/mlm/bin/activate

echo "Setup done. Running python script..."
####### FIRST TASK: PREPROCESS CORPUS #######
## Needs 1x1GB, 1 CPU and around 10 minutes
# python3 -u src/mlm/preprocess_corpus.py \
#     --input_dir data/mlm \
#     --output_file data/mlm/corpus_clean.txt

python pretrain_beto.py \
    --corpus_file data/mlm/corpus_clean.txt \
    --output_dir /scratch/nicolasal97/im_project/beto-cgec \
    --bf16

deactivate
module purge
echo "Script finished"

# sbatch --dependency=afterok:JOBID preprocessing.bash