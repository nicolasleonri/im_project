#!/bin/bash
#SBATCH --job-name=mlm_pretrain_models
#SBATCH --output=logs/mlm/mlm_pretrain_models_%j.out
#SBATCH --partition=scavenger
#SBATCH --account=agfritz
#SBATCH --qos=prio
#SBATCH --nodes=1
#SBATCH --ntasks=1

#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=7G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=12:00:00

# Load necessary modules
module purge
module load CUDA/12.4.0
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
## Needs 2x7GB, 1xH100 and around 1 hour
python3 -u src/mlm/pretrain_beto.py \
    --model_name dccuchile/bert-base-spanish-wwm-uncased \
    --corpus_file data/mlm/corpus_clean.txt \
    --output_dir /scratch/nicolasal97/im_project/beto-cgec \
    --per_device_train_batch_size 64 \
    --bf16

python3 -u src/mlm/pretrain_beto.py \
    --model_name skimai/spanberta-base-cased \
    --corpus_file data/mlm/corpus_clean.txt \
    --output_dir /scratch/nicolasal97/im_project/spanberta-cgec \
    --per_device_train_batch_size 64 \
    --bf16

python3 -u src/mlm/pretrain_beto.py \
    --model_name PeterPanecillo/PlanTL-GOB-ES-roberta-base-bne-copy \
    --corpus_file data/mlm/corpus_clean.txt \
    --output_dir /scratch/nicolasal97/im_project/roberta-cgec \
    --per_device_train_batch_size 64 \
    --bf16

python3 -u src/mlm/pretrain_beto.py \
    --model_name FacebookAI/xlm-roberta-base \
    --corpus_file data/mlm/corpus_clean.txt \
    --output_dir /scratch/nicolasal97/im_project/xlm-roberta-cgec \
    --per_device_train_batch_size 64 \
    --bf16

##### THIRD TASK: EVALUATE BETO #######
## Needs 1x2GB, 1xH100 and around 5 minutes per task
# python3 -u src/mlm/finetune_beto.py \
#     --model_name_or_path dccuchile/bert-base-spanish-wwm-uncased \
#     --source_dataset data/preprocessing/datasets/furman_2023.csv \
#     --test_dataset data/preprocessing/test_set_final.csv \
#     --task component \
#     --output_dir results/beto-base_furman_2023_component \
#     --n_trials 20 \
#     --bf16

# python3 -u src/mlm/finetune_beto.py \
#     --model_name_or_path dccuchile/bert-base-spanish-wwm-uncased \
#     --source_dataset data/preprocessing/datasets/gorrostieta_lopezlopez_2019.csv \
#     --test_dataset data/preprocessing/test_set_final.csv \
#     --task binary \
#     --output_dir results/beto-base_gorrostieta_lopezlopez_2019_binary \
#     --n_trials 20 \
#     --bf16

# python3 -u src/mlm/finetune_beto.py \
#     --model_name_or_path dccuchile/bert-base-spanish-wwm-uncased \
#     --source_dataset data/preprocessing/datasets/gorrostieta_lopezlopez_2019.csv \
#     --test_dataset data/preprocessing/test_set_final.csv \
#     --task component \
#     --output_dir results/beto-base_gorrostieta_lopezlopez_2019_component \
#     --n_trials 20 \
#     --bf16

# python3 -u src/mlm/finetune_beto.py \
#     --model_name_or_path dccuchile/bert-base-spanish-wwm-uncased \
#     --source_dataset data/preprocessing/datasets/guzman_monteza_2023.csv \
#     --test_dataset data/preprocessing/test_set_final.csv \
#     --task binary \
#     --output_dir results/beto-base_guzman_monteza_2023_binary \
#     --n_trials 20 \
#     --bf16

# python3 -u src/mlm/finetune_beto.py \
#     --model_name_or_path dccuchile/bert-base-spanish-wwm-uncased \
#     --source_dataset data/preprocessing/datasets/kovatchev_taule_2022.csv \
#     --test_dataset data/preprocessing/test_set_final.csv \
#     --task binary \
#     --output_dir results/beto-base_kovatchev_taule_2022_binary \
#     --n_trials 20 \
#     --bf16

# python3 -u src/mlm/finetune_beto.py \
#     --model_name_or_path dccuchile/bert-base-spanish-wwm-uncased \
#     --source_dataset data/preprocessing/datasets/kovatchev_taule_2022.csv \
#     --test_dataset data/preprocessing/test_set_final.csv \
#     --task component \
#     --output_dir results/beto-base_kovatchev_taule_2022_component \
#     --n_trials 20 \
#     --bf16

# python3 -u src/mlm/finetune_beto.py \
#     --model_name_or_path dccuchile/bert-base-spanish-wwm-uncased \
#     --source_dataset data/preprocessing/datasets/ruizdolz_2021.csv \
#     --test_dataset data/preprocessing/test_set_final.csv \
#     --task binary \
#     --output_dir results/beto-base_ruizdolz_2021_binary \
#     --n_trials 20 \
#     --bf16

# python3 -u src/mlm/finetune_beto.py \
#     --model_name_or_path dccuchile/bert-base-spanish-wwm-uncased \
#     --source_dataset data/preprocessing/datasets/ruizdolz_2021.csv \
#     --test_dataset data/preprocessing/test_set_final.csv \
#     --task component \
#     --output_dir results/beto-base_ruizdolz_2021_component \
#     --n_trials 20 \
#     --bf16

# python3 -u src/mlm/finetune_beto.py \
#     --model_name_or_path dccuchile/bert-base-spanish-wwm-uncased \
#     --source_dataset data/preprocessing/datasets/ruizdolz_2024.csv \
#     --test_dataset data/preprocessing/test_set_final.csv \
#     --task component \
#     --output_dir results/beto-base_ruizdolz_2024_component \
#     --n_trials 20 \
#     --bf16

# python3 -u src/mlm/finetune_beto.py \
#     --model_name_or_path dccuchile/bert-base-spanish-wwm-uncased \
#     --source_dataset data/preprocessing/datasets/segura_tinoco_2022.csv \
#     --test_dataset data/preprocessing/test_set_final.csv \
#     --task component \
#     --output_dir results/beto-base_segura_tinoco_2022_component \
#     --n_trials 20 \
#     --bf16

# python3 -u src/mlm/finetune_beto.py \
#     --model_name_or_path dccuchile/bert-base-spanish-wwm-uncased \
#     --source_dataset data/preprocessing/datasets/yeginbergen_2024.csv \
#     --test_dataset data/preprocessing/test_set_final.csv \
#     --task component \
#     --output_dir results/beto-base_yeginbergen_2024_component \
#     --n_trials 20 \
#     --bf16

deactivate
module purge
echo "Script finished"

# sbatch --dependency=afterok:JOBID preprocessing.bash