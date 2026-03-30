#!/bin/bash
#SBATCH --job-name=finetune_and_eval_models
#SBATCH --output=logs/evaluating/test/finetune_and_eval_models_%j.out
#SBATCH --partition=scavenger
#SBATCH --account=agfritz
#SBATCH --qos=prio
#SBATCH --nodes=1
#SBATCH --ntasks=1

#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=24:00:00

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

# Set common parameters
TEST_DATASET="data/preprocessing/test_set_final.csv"
N_TRIALS=20

echo "========================================="
echo "Starting All Source Dataset Experiments"
echo "========================================="
echo "Test dataset: $TEST_DATASET"
echo "Number of trials: $N_TRIALS"
echo "========================================="

# ##### 1) Furman 2023
# echo ""
# echo "=== Dataset 1/8: Furman 2023 ==="
# echo "----------------------------------------"

# # BETO Base
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path dccuchile/bert-base-spanish-wwm-uncased \
#     --source_dataset data/preprocessing/datasets/furman_2023.csv \
#     --test_dataset $TEST_DATASET \
#     --task component \
#     --output_dir results/beto-base_furman_2023_component \
#     --n_trials $N_TRIALS \
#     --bf16

# # BETO-CGEC
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path /scratch/nicolasal97/im_project/beto-cgec \
#     --source_dataset data/preprocessing/datasets/furman_2023.csv \
#     --test_dataset $TEST_DATASET \
#     --task component \
#     --output_dir results/beto-cgec_furman_2023_component \
#     --n_trials $N_TRIALS \
#     --bf16

# # SpanBERTa Base
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path skimai/spanberta-base-cased \
#     --source_dataset data/preprocessing/datasets/furman_2023.csv \
#     --test_dataset $TEST_DATASET \
#     --task component \
#     --output_dir results/spanberta-base_furman_2023_component \
#     --n_trials $N_TRIALS \
#     --bf16

# # SpanBERTa-CGEC
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path /scratch/nicolasal97/im_project/spanberta-cgec \
#     --source_dataset data/preprocessing/datasets/furman_2023.csv \
#     --test_dataset $TEST_DATASET \
#     --task component \
#     --output_dir results/spanberta-cgec_furman_2023_component \
#     --n_trials $N_TRIALS \
#     --bf16

# # XLM-RoBERTa Base
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path FacebookAI/xlm-roberta-base \
#     --source_dataset data/preprocessing/datasets/furman_2023.csv \
#     --test_dataset $TEST_DATASET \
#     --task component \
#     --output_dir results/xlm-roberta-base_furman_2023_component \
#     --n_trials $N_TRIALS \
#     --bf16

# # XLM-RoBERTa-CGEC
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path /scratch/nicolasal97/im_project/xlm-roberta-cgec \
#     --source_dataset data/preprocessing/datasets/furman_2023.csv \
#     --test_dataset $TEST_DATASET \
#     --task component \
#     --output_dir results/xlm-roberta-cgec_furman_2023_component \
#     --n_trials $N_TRIALS \
#     --bf16

# ##### 2) Gorrostieta Lopez-Lopez 2019
# echo ""
# echo "=== Dataset 2/8: Gorrostieta Lopez-Lopez 2019 ==="
# echo "----------------------------------------"

# # BETO Base
# # Binary task
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path dccuchile/bert-base-spanish-wwm-uncased \
#     --source_dataset data/preprocessing/datasets/gorrostieta_lopezlopez_2019.csv \
#     --test_dataset $TEST_DATASET \
#     --task binary \
#     --output_dir results/beto-base_gorrostieta_lopezlopez_2019_binary \
#     --n_trials $N_TRIALS \
#     --bf16

# # Component task
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path dccuchile/bert-base-spanish-wwm-uncased \
#     --source_dataset data/preprocessing/datasets/gorrostieta_lopezlopez_2019.csv \
#     --test_dataset $TEST_DATASET \
#     --task component \
#     --output_dir results/beto-base_gorrostieta_lopezlopez_2019_component \
#     --n_trials $N_TRIALS \
#     --bf16

# # BETO-CGEC
# # Binary task
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path /scratch/nicolasal97/im_project/beto-cgec \
#     --source_dataset data/preprocessing/datasets/gorrostieta_lopezlopez_2019.csv \
#     --test_dataset $TEST_DATASET \
#     --task binary \
#     --output_dir results/beto-cgec_gorrostieta_lopezlopez_2019_binary \
#     --n_trials $N_TRIALS \
#     --bf16

# # Component task
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path /scratch/nicolasal97/im_project/beto-cgec \
#     --source_dataset data/preprocessing/datasets/gorrostieta_lopezlopez_2019.csv \
#     --test_dataset $TEST_DATASET \
#     --task component \
#     --output_dir results/beto-cgec_gorrostieta_lopezlopez_2019_component \
#     --n_trials $N_TRIALS \
#     --bf16

# # SpanBERTa Base
# # Binary task
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path skimai/spanberta-base-cased \
#     --source_dataset data/preprocessing/datasets/gorrostieta_lopezlopez_2019.csv \
#     --test_dataset $TEST_DATASET \
#     --task binary \
#     --output_dir results/spanberta-base_gorrostieta_lopezlopez_2019_binary \
#     --n_trials $N_TRIALS \
#     --bf16

# # Component task
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path skimai/spanberta-base-cased \
#     --source_dataset data/preprocessing/datasets/gorrostieta_lopezlopez_2019.csv \
#     --test_dataset $TEST_DATASET \
#     --task component \
#     --output_dir results/spanberta-base_gorrostieta_lopezlopez_2019_component \
#     --n_trials $N_TRIALS \
#     --bf16

# # SpanBERTa-CGEC
# # Binary task
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path /scratch/nicolasal97/im_project/spanberta-cgec \
#     --source_dataset data/preprocessing/datasets/gorrostieta_lopezlopez_2019.csv \
#     --test_dataset $TEST_DATASET \
#     --task binary \
#     --output_dir results/spanberta-cgec_gorrostieta_lopezlopez_2019_binary \
#     --n_trials $N_TRIALS \
#     --bf16

# # Component task
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path /scratch/nicolasal97/im_project/spanberta-cgec \
#     --source_dataset data/preprocessing/datasets/gorrostieta_lopezlopez_2019.csv \
#     --test_dataset $TEST_DATASET \
#     --task component \
#     --output_dir results/spanberta-cgec_gorrostieta_lopezlopez_2019_component \
#     --n_trials $N_TRIALS \
#     --bf16

# # XLM-RoBERTa Base
# # Binary task
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path FacebookAI/xlm-roberta-base \
#     --source_dataset data/preprocessing/datasets/gorrostieta_lopezlopez_2019.csv \
#     --test_dataset $TEST_DATASET \
#     --task binary \
#     --output_dir results/xlm-roberta-base_gorrostieta_lopezlopez_2019_binary \
#     --n_trials $N_TRIALS \
#     --bf16

# # Component task
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path FacebookAI/xlm-roberta-base \
#     --source_dataset data/preprocessing/datasets/gorrostieta_lopezlopez_2019.csv \
#     --test_dataset $TEST_DATASET \
#     --task component \
#     --output_dir results/xlm-roberta-base_gorrostieta_lopezlopez_2019_component \
#     --n_trials $N_TRIALS \
#     --bf16

# # XLM-RoBERTa-CGEC
# # Binary task
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path /scratch/nicolasal97/im_project/xlm-roberta-cgec \
#     --source_dataset data/preprocessing/datasets/gorrostieta_lopezlopez_2019.csv \
#     --test_dataset $TEST_DATASET \
#     --task binary \
#     --output_dir results/xlm-roberta-cgec_gorrostieta_lopezlopez_2019_binary \
#     --n_trials $N_TRIALS \
#     --bf16

# # Component task
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path /scratch/nicolasal97/im_project/xlm-roberta-cgec \
#     --source_dataset data/preprocessing/datasets/gorrostieta_lopezlopez_2019.csv \
#     --test_dataset $TEST_DATASET \
#     --task component \
#     --output_dir results/xlm-roberta-cgec_gorrostieta_lopezlopez_2019_component \
#     --n_trials $N_TRIALS \
#     --bf16

# ##### 3) Guzman Monteza 2023
# echo ""
# echo "=== Dataset 3/8: Guzman Monteza 2023 ==="
# echo "----------------------------------------"

# # BETO Base
# # Binary task
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path dccuchile/bert-base-spanish-wwm-uncased \
#     --source_dataset data/preprocessing/datasets/guzman_monteza_2023.csv \
#     --test_dataset $TEST_DATASET \
#     --task binary \
#     --output_dir results/beto-base_guzman_monteza_2023_binary \
#     --n_trials $N_TRIALS \
#     --bf16

# # Component task [Only UNK info]
# # python3 -u src/evaluating/finetune_and_eval_model.py \
# #     --model_name_or_path dccuchile/bert-base-spanish-wwm-uncased \
# #     --source_dataset data/preprocessing/datasets/guzman_monteza_2023.csv \
# #     --test_dataset $TEST_DATASET \
# #     --task component \
# #     --output_dir results/beto-base_guzman_monteza_2023_component \
# #     --n_trials $N_TRIALS \
# #     --bf16

# # BETO-CGEC
# # Binary task
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path /scratch/nicolasal97/im_project/beto-cgec \
#     --source_dataset data/preprocessing/datasets/guzman_monteza_2023.csv \
#     --test_dataset $TEST_DATASET \
#     --task binary \
#     --output_dir results/beto-cgec_guzman_monteza_2023_binary \
#     --n_trials $N_TRIALS \
#     --bf16

# # Component task [Only UNK info]
# # python3 -u src/evaluating/finetune_and_eval_model.py \
# #     --model_name_or_path /scratch/nicolasal97/im_project/beto-cgec \
# #     --source_dataset data/preprocessing/datasets/guzman_monteza_2023.csv \
# #     --test_dataset $TEST_DATASET \
# #     --task component \
# #     --output_dir results/beto-cgec_guzman_monteza_2023_component \
# #     --n_trials $N_TRIALS \
# #     --bf16

# # SpanBERTa Base
# # Binary task
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path skimai/spanberta-base-cased \
#     --source_dataset data/preprocessing/datasets/guzman_monteza_2023.csv \
#     --test_dataset $TEST_DATASET \
#     --task binary \
#     --output_dir results/spanberta-base_guzman_monteza_2023_binary \
#     --n_trials $N_TRIALS \
#     --bf16

# # Component task [Only UNK info]
# # python3 -u src/evaluating/finetune_and_eval_model.py \
# #     --model_name_or_path skimai/spanberta-base-cased \
# #     --source_dataset data/preprocessing/datasets/guzman_monteza_2023.csv \
# #     --test_dataset $TEST_DATASET \
# #     --task component \
# #     --output_dir results/spanberta-base_guzman_monteza_2023_component \
# #     --n_trials $N_TRIALS \
# #     --bf16

# # SpanBERTa-CGEC
# # Binary task
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path /scratch/nicolasal97/im_project/spanberta-cgec \
#     --source_dataset data/preprocessing/datasets/guzman_monteza_2023.csv \
#     --test_dataset $TEST_DATASET \
#     --task binary \
#     --output_dir results/spanberta-cgec_guzman_monteza_2023_binary \
#     --n_trials $N_TRIALS \
#     --bf16

# # Component task [Only UNK info]
# # python3 -u src/evaluating/finetune_and_eval_model.py \
# #     --model_name_or_path /scratch/nicolasal97/im_project/spanberta-cgec \
# #     --source_dataset data/preprocessing/datasets/guzman_monteza_2023.csv \
# #     --test_dataset $TEST_DATASET \
# #     --task component \
# #     --output_dir results/spanberta-cgec_guzman_monteza_2023_component \
# #     --n_trials $N_TRIALS \
# #     --bf16

# # XLM-RoBERTa Base
# # Binary task
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path FacebookAI/xlm-roberta-base \
#     --source_dataset data/preprocessing/datasets/guzman_monteza_2023.csv \
#     --test_dataset $TEST_DATASET \
#     --task binary \
#     --output_dir results/xlm-roberta-base_guzman_monteza_2023_binary \
#     --n_trials $N_TRIALS \
#     --bf16

# # Component task [Only UNK info]
# # python3 -u src/evaluating/finetune_and_eval_model.py \
# #     --model_name_or_path FacebookAI/xlm-roberta-base \
# #     --source_dataset data/preprocessing/datasets/guzman_monteza_2023.csv \
# #     --test_dataset $TEST_DATASET \
# #     --task component \
# #     --output_dir results/xlm-roberta-base_guzman_monteza_2023_component \
# #     --n_trials $N_TRIALS \
# #     --bf16

# # XLM-RoBERTa-CGEC
# # Binary task
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path /scratch/nicolasal97/im_project/xlm-roberta-cgec \
#     --source_dataset data/preprocessing/datasets/guzman_monteza_2023.csv \
#     --test_dataset $TEST_DATASET \
#     --task binary \
#     --output_dir results/xlm-roberta-cgec_guzman_monteza_2023_binary \
#     --n_trials $N_TRIALS \
#     --bf16

# # Component task [Only UNK info]
# # python3 -u src/evaluating/finetune_and_eval_model.py \
# #     --model_name_or_path /scratch/nicolasal97/im_project/xlm-roberta-cgec \
# #     --source_dataset data/preprocessing/datasets/guzman_monteza_2023.csv \
# #     --test_dataset $TEST_DATASET \
# #     --task component \
# #     --output_dir results/xlm-roberta-cgec_guzman_monteza_2023_component \
# #     --n_trials $N_TRIALS \
# #     --bf16

# ##### 4) Kovatchev Taule 2022
# echo ""
# echo "=== Dataset 4/8: Kovatchev Taule 2022 ==="
# echo "----------------------------------------"

# # BETO Base
# # Binary task
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path dccuchile/bert-base-spanish-wwm-uncased \
#     --source_dataset data/preprocessing/datasets/kovatchev_taule_2022.csv \
#     --test_dataset $TEST_DATASET \
#     --task binary \
#     --output_dir results/beto-base_kovatchev_taule_2022_binary \
#     --n_trials $N_TRIALS \
#     --bf16

# # Component task
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path dccuchile/bert-base-spanish-wwm-uncased \
#     --source_dataset data/preprocessing/datasets/kovatchev_taule_2022.csv \
#     --test_dataset $TEST_DATASET \
#     --task component \
#     --output_dir results/beto-base_kovatchev_taule_2022_component \
#     --n_trials $N_TRIALS \
#     --bf16

# # BETO-CGEC
# # Binary task
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path /scratch/nicolasal97/im_project/beto-cgec \
#     --source_dataset data/preprocessing/datasets/kovatchev_taule_2022.csv \
#     --test_dataset $TEST_DATASET \
#     --task binary \
#     --output_dir results/beto-cgec_kovatchev_taule_2022_binary \
#     --n_trials $N_TRIALS \
#     --bf16

# # Component task
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path /scratch/nicolasal97/im_project/beto-cgec \
#     --source_dataset data/preprocessing/datasets/kovatchev_taule_2022.csv \
#     --test_dataset $TEST_DATASET \
#     --task component \
#     --output_dir results/beto-cgec_kovatchev_taule_2022_component \
#     --n_trials $N_TRIALS \
#     --bf16

# # SpanBERTa Base
# # Binary task
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path skimai/spanberta-base-cased \
#     --source_dataset data/preprocessing/datasets/kovatchev_taule_2022.csv \
#     --test_dataset $TEST_DATASET \
#     --task binary \
#     --output_dir results/spanberta-base_kovatchev_taule_2022_binary \
#     --n_trials $N_TRIALS \
#     --bf16

# # Component task
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path skimai/spanberta-base-cased \
#     --source_dataset data/preprocessing/datasets/kovatchev_taule_2022.csv \
#     --test_dataset $TEST_DATASET \
#     --task component \
#     --output_dir results/spanberta-base_kovatchev_taule_2022_component \
#     --n_trials $N_TRIALS \
#     --bf16

# # SpanBERTa-CGEC
# # Binary task
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path /scratch/nicolasal97/im_project/spanberta-cgec \
#     --source_dataset data/preprocessing/datasets/kovatchev_taule_2022.csv \
#     --test_dataset $TEST_DATASET \
#     --task binary \
#     --output_dir results/spanberta-cgec_kovatchev_taule_2022_binary \
#     --n_trials $N_TRIALS \
#     --bf16

# # Component task
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path /scratch/nicolasal97/im_project/spanberta-cgec \
#     --source_dataset data/preprocessing/datasets/kovatchev_taule_2022.csv \
#     --test_dataset $TEST_DATASET \
#     --task component \
#     --output_dir results/spanberta-cgec_kovatchev_taule_2022_component \
#     --n_trials $N_TRIALS \
#     --bf16

# # XLM-RoBERTa Base
# # Binary task
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path FacebookAI/xlm-roberta-base \
#     --source_dataset data/preprocessing/datasets/kovatchev_taule_2022.csv \
#     --test_dataset $TEST_DATASET \
#     --task binary \
#     --output_dir results/xlm-roberta-base_kovatchev_taule_2022_binary \
#     --n_trials $N_TRIALS \
#     --bf16

# # Component task
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path FacebookAI/xlm-roberta-base \
#     --source_dataset data/preprocessing/datasets/kovatchev_taule_2022.csv \
#     --test_dataset $TEST_DATASET \
#     --task component \
#     --output_dir results/xlm-roberta-base_kovatchev_taule_2022_component \
#     --n_trials $N_TRIALS \
#     --bf16

# # XLM-RoBERTa-CGEC
# # Binary task
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path /scratch/nicolasal97/im_project/xlm-roberta-cgec \
#     --source_dataset data/preprocessing/datasets/kovatchev_taule_2022.csv \
#     --test_dataset $TEST_DATASET \
#     --task binary \
#     --output_dir results/xlm-roberta-cgec_kovatchev_taule_2022_binary \
#     --n_trials $N_TRIALS \
#     --bf16

# # Component task
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path /scratch/nicolasal97/im_project/xlm-roberta-cgec \
#     --source_dataset data/preprocessing/datasets/kovatchev_taule_2022.csv \
#     --test_dataset $TEST_DATASET \
#     --task component \
#     --output_dir results/xlm-roberta-cgec_kovatchev_taule_2022_component \
#     --n_trials $N_TRIALS \
#     --bf16

# ##### 5) Ruiz-Dolz 2021
# echo ""
# echo "=== Dataset 5/8: Ruiz-Dolz 2021 ==="
# echo "----------------------------------------"

# # BETO Base
# # Binary task
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path dccuchile/bert-base-spanish-wwm-uncased \
#     --source_dataset data/preprocessing/datasets/ruizdolz_2021.csv \
#     --test_dataset $TEST_DATASET \
#     --task binary \
#     --output_dir results/beto-base_ruizdolz_2021_binary \
#     --n_trials $N_TRIALS \
#     --bf16

# # Component task
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path dccuchile/bert-base-spanish-wwm-uncased \
#     --source_dataset data/preprocessing/datasets/ruizdolz_2021.csv \
#     --test_dataset $TEST_DATASET \
#     --task component \
#     --output_dir results/beto-base_ruizdolz_2021_component \
#     --n_trials $N_TRIALS \
#     --bf16

# # BETO-CGEC
# # Binary task
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path /scratch/nicolasal97/im_project/beto-cgec \
#     --source_dataset data/preprocessing/datasets/ruizdolz_2021.csv \
#     --test_dataset $TEST_DATASET \
#     --task binary \
#     --output_dir results/beto-cgec_ruizdolz_2021_binary \
#     --n_trials $N_TRIALS \
#     --bf16

# # Component task
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path /scratch/nicolasal97/im_project/beto-cgec \
#     --source_dataset data/preprocessing/datasets/ruizdolz_2021.csv \
#     --test_dataset $TEST_DATASET \
#     --task component \
#     --output_dir results/beto-cgec_ruizdolz_2021_component \
#     --n_trials $N_TRIALS \
#     --bf16

# # SpanBERTa Base
# # Binary task
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path skimai/spanberta-base-cased \
#     --source_dataset data/preprocessing/datasets/ruizdolz_2021.csv \
#     --test_dataset $TEST_DATASET \
#     --task binary \
#     --output_dir results/spanberta-base_ruizdolz_2021_binary \
#     --n_trials $N_TRIALS \
#     --bf16

# # Component task
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path skimai/spanberta-base-cased \
#     --source_dataset data/preprocessing/datasets/ruizdolz_2021.csv \
#     --test_dataset $TEST_DATASET \
#     --task component \
#     --output_dir results/spanberta-base_ruizdolz_2021_component \
#     --n_trials $N_TRIALS \
#     --bf16

# # SpanBERTa-CGEC
# # Binary task
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path /scratch/nicolasal97/im_project/spanberta-cgec \
#     --source_dataset data/preprocessing/datasets/ruizdolz_2021.csv \
#     --test_dataset $TEST_DATASET \
#     --task binary \
#     --output_dir results/spanberta-cgec_ruizdolz_2021_binary \
#     --n_trials $N_TRIALS \
#     --bf16

# # Component task
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path /scratch/nicolasal97/im_project/spanberta-cgec \
#     --source_dataset data/preprocessing/datasets/ruizdolz_2021.csv \
#     --test_dataset $TEST_DATASET \
#     --task component \
#     --output_dir results/spanberta-cgec_ruizdolz_2021_component \
#     --n_trials $N_TRIALS \
#     --bf16

# # XLM-RoBERTa Base
# # Binary task
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path FacebookAI/xlm-roberta-base \
#     --source_dataset data/preprocessing/datasets/ruizdolz_2021.csv \
#     --test_dataset $TEST_DATASET \
#     --task binary \
#     --output_dir results/xlm-roberta-base_ruizdolz_2021_binary \
#     --n_trials $N_TRIALS \
#     --bf16

# # Component task
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path FacebookAI/xlm-roberta-base \
#     --source_dataset data/preprocessing/datasets/ruizdolz_2021.csv \
#     --test_dataset $TEST_DATASET \
#     --task component \
#     --output_dir results/xlm-roberta-base_ruizdolz_2021_component \
#     --n_trials $N_TRIALS \
#     --bf16

# # XLM-RoBERTa-CGEC
# # Binary task
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path /scratch/nicolasal97/im_project/xlm-roberta-cgec \
#     --source_dataset data/preprocessing/datasets/ruizdolz_2021.csv \
#     --test_dataset $TEST_DATASET \
#     --task binary \
#     --output_dir results/xlm-roberta-cgec_ruizdolz_2021_binary \
#     --n_trials $N_TRIALS \
#     --bf16

# # Component task
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path /scratch/nicolasal97/im_project/xlm-roberta-cgec \
#     --source_dataset data/preprocessing/datasets/ruizdolz_2021.csv \
#     --test_dataset $TEST_DATASET \
#     --task component \
#     --output_dir results/xlm-roberta-cgec_ruizdolz_2021_component \
#     --n_trials $N_TRIALS \
#     --bf16

# ##### 6) Ruiz-Dolz 2024
# echo ""
# echo "=== Dataset 6/8: Ruiz-Dolz 2024 ==="
# echo "----------------------------------------"

# # BETO Base
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path dccuchile/bert-base-spanish-wwm-uncased \
#     --source_dataset data/preprocessing/datasets/ruizdolz_2024.csv \
#     --test_dataset $TEST_DATASET \
#     --task component \
#     --output_dir results/beto-base_ruizdolz_2024_component \
#     --n_trials $N_TRIALS \
#     --bf16

# # BETO-CGEC
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path /scratch/nicolasal97/im_project/beto-cgec \
#     --source_dataset data/preprocessing/datasets/ruizdolz_2024.csv \
#     --test_dataset $TEST_DATASET \
#     --task component \
#     --output_dir results/beto-cgec_ruizdolz_2024_component \
#     --n_trials $N_TRIALS \
#     --bf16

# # SpanBERTa Base
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path skimai/spanberta-base-cased \
#     --source_dataset data/preprocessing/datasets/ruizdolz_2024.csv \
#     --test_dataset $TEST_DATASET \
#     --task component \
#     --output_dir results/spanberta-base_ruizdolz_2024_component \
#     --n_trials $N_TRIALS \
#     --bf16

# # SpanBERTa-CGEC
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path /scratch/nicolasal97/im_project/spanberta-cgec \
#     --source_dataset data/preprocessing/datasets/ruizdolz_2024.csv \
#     --test_dataset $TEST_DATASET \
#     --task component \
#     --output_dir results/spanberta-cgec_ruizdolz_2024_component \
#     --n_trials $N_TRIALS \
#     --bf16

# # XLM-RoBERTa Base
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path FacebookAI/xlm-roberta-base \
#     --source_dataset data/preprocessing/datasets/ruizdolz_2024.csv \
#     --test_dataset $TEST_DATASET \
#     --task component \
#     --output_dir results/xlm-roberta-base_ruizdolz_2024_component \
#     --n_trials $N_TRIALS \
#     --bf16

# # XLM-RoBERTa-CGEC
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path /scratch/nicolasal97/im_project/xlm-roberta-cgec \
#     --source_dataset data/preprocessing/datasets/ruizdolz_2024.csv \
#     --test_dataset $TEST_DATASET \
#     --task component \
#     --output_dir results/xlm-roberta-cgec_ruizdolz_2024_component \
#     --n_trials $N_TRIALS \
#     --bf16

# ##### 7) Segura Tinoco 2022
# echo ""
# echo "=== Dataset 7/8: Segura Tinoco 2022 ==="
# echo "----------------------------------------"

# # BETO Base
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path dccuchile/bert-base-spanish-wwm-uncased \
#     --source_dataset data/preprocessing/datasets/segura_tinoco_2022.csv \
#     --test_dataset $TEST_DATASET \
#     --task component \
#     --output_dir results/beto-base_segura_tinoco_2022_component \
#     --n_trials $N_TRIALS \
#     --bf16

# # BETO-CGEC
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path /scratch/nicolasal97/im_project/beto-cgec \
#     --source_dataset data/preprocessing/datasets/segura_tinoco_2022.csv \
#     --test_dataset $TEST_DATASET \
#     --task component \
#     --output_dir results/beto-cgec_segura_tinoco_2022_component \
#     --n_trials $N_TRIALS \
#     --bf16

# # SpanBERTa Base
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path skimai/spanberta-base-cased \
#     --source_dataset data/preprocessing/datasets/segura_tinoco_2022.csv \
#     --test_dataset $TEST_DATASET \
#     --task component \
#     --output_dir results/spanberta-base_segura_tinoco_2022_component \
#     --n_trials $N_TRIALS \
#     --bf16

# # SpanBERTa-CGEC
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path /scratch/nicolasal97/im_project/spanberta-cgec \
#     --source_dataset data/preprocessing/datasets/segura_tinoco_2022.csv \
#     --test_dataset $TEST_DATASET \
#     --task component \
#     --output_dir results/spanberta-cgec_segura_tinoco_2022_component \
#     --n_trials $N_TRIALS \
#     --bf16

# # XLM-RoBERTa Base
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path FacebookAI/xlm-roberta-base \
#     --source_dataset data/preprocessing/datasets/segura_tinoco_2022.csv \
#     --test_dataset $TEST_DATASET \
#     --task component \
#     --output_dir results/xlm-roberta-base_segura_tinoco_2022_component \
#     --n_trials $N_TRIALS \
#     --bf16

# # XLM-RoBERTa-CGEC
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path /scratch/nicolasal97/im_project/xlm-roberta-cgec \
#     --source_dataset data/preprocessing/datasets/segura_tinoco_2022.csv \
#     --test_dataset $TEST_DATASET \
#     --task component \
#     --output_dir results/xlm-roberta-cgec_segura_tinoco_2022_component \
#     --n_trials $N_TRIALS \
#     --bf16

# ##### 8) Yeginbergen 2024
# echo ""
# echo "=== Dataset 8/8: Yeginbergen 2024 ==="
# echo "----------------------------------------"

# # BETO Base
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path dccuchile/bert-base-spanish-wwm-uncased \
#     --source_dataset data/preprocessing/datasets/yeginbergen_2024.csv \
#     --test_dataset $TEST_DATASET \
#     --task component \
#     --output_dir results/beto-base_yeginbergen_2024_component \
#     --n_trials $N_TRIALS \
#     --bf16

# # BETO-CGEC
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path /scratch/nicolasal97/im_project/beto-cgec \
#     --source_dataset data/preprocessing/datasets/yeginbergen_2024.csv \
#     --test_dataset $TEST_DATASET \
#     --task component \
#     --output_dir results/beto-cgec_yeginbergen_2024_component \
#     --n_trials $N_TRIALS \
#     --bf16

# # SpanBERTa Base
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path skimai/spanberta-base-cased \
#     --source_dataset data/preprocessing/datasets/yeginbergen_2024.csv \
#     --test_dataset $TEST_DATASET \
#     --task component \
#     --output_dir results/spanberta-base_yeginbergen_2024_component \
#     --n_trials $N_TRIALS \
#     --bf16

# # SpanBERTa-CGEC
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path /scratch/nicolasal97/im_project/spanberta-cgec \
#     --source_dataset data/preprocessing/datasets/yeginbergen_2024.csv \
#     --test_dataset $TEST_DATASET \
#     --task component \
#     --output_dir results/spanberta-cgec_yeginbergen_2024_component \
#     --n_trials $N_TRIALS \
#     --bf16

# # XLM-RoBERTa Base
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path FacebookAI/xlm-roberta-base \
#     --source_dataset data/preprocessing/datasets/yeginbergen_2024.csv \
#     --test_dataset $TEST_DATASET \
#     --task component \
#     --output_dir results/xlm-roberta-base_yeginbergen_2024_component \
#     --n_trials $N_TRIALS \
#     --bf16

# # XLM-RoBERTa-CGEC
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path /scratch/nicolasal97/im_project/xlm-roberta-cgec \
#     --source_dataset data/preprocessing/datasets/yeginbergen_2024.csv \
#     --test_dataset $TEST_DATASET \
#     --task component \
#     --output_dir results/xlm-roberta-cgec_yeginbergen_2024_component \
#     --n_trials $N_TRIALS \
#     --bf16

# echo ""
# echo "========================================="
# echo "ALL EXPERIMENTS COMPLETED!"
# echo "========================================="
# echo "Results saved in results/ directory"
# echo ""
# echo "Summary of experiments run:"
# echo "- 8 source datasets"
# echo "- 6 model variants (BETO, SpanBERTa, XLM-RoBERTa with base and domain-adapted versions)"
# echo "- Binary and component tasks where applicable"
# echo ""
# echo "Total experiments by dataset:"
# echo "  Furman 2023:             6 (component only)"
# echo "  Gorrostieta 2019:       12 (binary + component × 6 models)"
# echo "  Guzman 2023:             12 (binary + component × 6 models)"
# echo "  Kovatchev 2022:          12 (binary + component × 6 models)"
# echo "  Ruiz-Dolz 2021:          12 (binary + component × 6 models)"
# echo "  Ruiz-Dolz 2024:          6 (component only)"
# echo "  Segura Tinoco 2022:      6 (component only)"
# echo "  Yeginbergen 2024:        6 (component only)"
# echo ""
# echo "Total runs: 72 experiments"
# echo "========================================="

##### 1) Top-K
# echo ""
# echo "=== Dataset 1/4: Top-K ==="
# echo "----------------------------------------"

# # BETO Base
# # Binary task
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path dccuchile/bert-base-spanish-wwm-uncased \
#     --source_dataset data/preprocessing/datasets/topk.csv \
#     --test_dataset $TEST_DATASET \
#     --task binary \
#     --output_dir results/beto-base_topk_2026_binary \
#     --n_trials $N_TRIALS \
#     --bf16

# # Component task
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path dccuchile/bert-base-spanish-wwm-uncased \
#     --source_dataset data/preprocessing/datasets/topk.csv \
#     --test_dataset $TEST_DATASET \
#     --task component \
#     --output_dir results/beto-base_topk_2026_component \
#     --n_trials $N_TRIALS \
#     --bf16

# # BETO-CGEC
# # Binary task
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path /scratch/nicolasal97/im_project/beto-cgec \
#     --source_dataset data/preprocessing/datasets/topk.csv \
#     --test_dataset $TEST_DATASET \
#     --task binary \
#     --output_dir results/beto-cgec_topk_2026_binary \
#     --n_trials $N_TRIALS \
#     --bf16

# # Component task
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path /scratch/nicolasal97/im_project/beto-cgec \
#     --source_dataset data/preprocessing/datasets/topk.csv \
#     --test_dataset $TEST_DATASET \
#     --task component \
#     --output_dir results/beto-cgec_topk_2026_component \
#     --n_trials $N_TRIALS \
#     --bf16

# SpanBERTa Base
# Binary task
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path skimai/spanberta-base-cased \
#     --source_dataset data/preprocessing/datasets/topk.csv \
#     --test_dataset $TEST_DATASET \
#     --task binary \
#     --output_dir results/spanberta-base_topk_2026_binary \
#     --n_trials $N_TRIALS \
#     --bf16

# # Component task
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path skimai/spanberta-base-cased \
#     --source_dataset data/preprocessing/datasets/topk.csv \
#     --test_dataset $TEST_DATASET \
#     --task component \
#     --output_dir results/spanberta-base_topk_2026_component \
#     --n_trials $N_TRIALS \
#     --bf16

# # SpanBERTa-CGEC
# # Binary task
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path /scratch/nicolasal97/im_project/spanberta-cgec \
#     --source_dataset data/preprocessing/datasets/topk.csv \
#     --test_dataset $TEST_DATASET \
#     --task binary \
#     --output_dir results/spanberta-cgec_topk_2026_binary \
#     --n_trials $N_TRIALS \
#     --bf16

# # Component task
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path /scratch/nicolasal97/im_project/spanberta-cgec \
#     --source_dataset data/preprocessing/datasets/topk.csv \
#     --test_dataset $TEST_DATASET \
#     --task component \
#     --output_dir results/spanberta-cgec_topk_2026_component \
#     --n_trials $N_TRIALS \
#     --bf16

# # XLM-RoBERTa Base
# # Binary task
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path FacebookAI/xlm-roberta-base \
#     --source_dataset data/preprocessing/datasets/topk.csv \
#     --test_dataset $TEST_DATASET \
#     --task binary \
#     --output_dir results/xlm-roberta-base_topk_2026_binary \
#     --n_trials $N_TRIALS \
#     --bf16

# # Component task
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path FacebookAI/xlm-roberta-base \
#     --source_dataset data/preprocessing/datasets/topk.csv \
#     --test_dataset $TEST_DATASET \
#     --task component \
#     --output_dir results/xlm-roberta-base_topk_2026_component \
#     --n_trials $N_TRIALS \
#     --bf16

# # XLM-RoBERTa-CGEC
# # Binary task
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path /scratch/nicolasal97/im_project/xlm-roberta-cgec \
#     --source_dataset data/preprocessing/datasets/topk.csv \
#     --test_dataset $TEST_DATASET \
#     --task binary \
#     --output_dir results/xlm-roberta-cgec_topk_2026_binary \
#     --n_trials $N_TRIALS \
#     --bf16

# # Component task
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path /scratch/nicolasal97/im_project/xlm-roberta-cgec \
#     --source_dataset data/preprocessing/datasets/topk.csv \
#     --test_dataset $TEST_DATASET \
#     --task component \
#     --output_dir results/xlm-roberta-cgec_topk_2026_component \
#     --n_trials $N_TRIALS \
#     --bf16


# ##### 2) Similarity
# echo ""
# echo "=== Dataset 2/4: Similarity ==="
# echo "----------------------------------------"

# # BETO Base
# # Binary task
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path dccuchile/bert-base-spanish-wwm-uncased \
#     --source_dataset data/preprocessing/datasets/similarity.csv \
#     --test_dataset $TEST_DATASET \
#     --task binary \
#     --output_dir results/beto-base_similarity_2026_binary \
#     --n_trials $N_TRIALS \
#     --bf16

# # Component task
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path dccuchile/bert-base-spanish-wwm-uncased \
#     --source_dataset data/preprocessing/datasets/similarity.csv \
#     --test_dataset $TEST_DATASET \
#     --task component \
#     --output_dir results/beto-base_similarity_2026_component \
#     --n_trials $N_TRIALS \
#     --bf16

# # BETO-CGEC
# # Binary task
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path /scratch/nicolasal97/im_project/beto-cgec \
#     --source_dataset data/preprocessing/datasets/similarity.csv \
#     --test_dataset $TEST_DATASET \
#     --task binary \
#     --output_dir results/beto-cgec_similarity_2026_binary \
#     --n_trials $N_TRIALS \
#     --bf16

# # Component task
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path /scratch/nicolasal97/im_project/beto-cgec \
#     --source_dataset data/preprocessing/datasets/similarity.csv \
#     --test_dataset $TEST_DATASET \
#     --task component \
#     --output_dir results/beto-cgec_similarity_2026_component \
#     --n_trials $N_TRIALS \
#     --bf16

# # SpanBERTa Base
# # Binary task
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path skimai/spanberta-base-cased \
#     --source_dataset data/preprocessing/datasets/similarity.csv \
#     --test_dataset $TEST_DATASET \
#     --task binary \
#     --output_dir results/spanberta-base_similarity_2026_binary \
#     --n_trials $N_TRIALS \
#     --bf16

# # Component task
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path skimai/spanberta-base-cased \
#     --source_dataset data/preprocessing/datasets/similarity.csv \
#     --test_dataset $TEST_DATASET \
#     --task component \
#     --output_dir results/spanberta-base_similarity_2026_component \
#     --n_trials $N_TRIALS \
#     --bf16

# # SpanBERTa-CGEC
# # Binary task
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path /scratch/nicolasal97/im_project/spanberta-cgec \
#     --source_dataset data/preprocessing/datasets/similarity.csv \
#     --test_dataset $TEST_DATASET \
#     --task binary \
#     --output_dir results/spanberta-cgec_similarity_2026_binary \
#     --n_trials $N_TRIALS \
#     --bf16

# # Component task
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path /scratch/nicolasal97/im_project/spanberta-cgec \
#     --source_dataset data/preprocessing/datasets/similarity.csv \
#     --test_dataset $TEST_DATASET \
#     --task component \
#     --output_dir results/spanberta-cgec_similarity_2026_component \
#     --n_trials $N_TRIALS \
#     --bf16

# # XLM-RoBERTa Base
# # Binary task
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path FacebookAI/xlm-roberta-base \
#     --source_dataset data/preprocessing/datasets/similarity.csv \
#     --test_dataset $TEST_DATASET \
#     --task binary \
#     --output_dir results/xlm-roberta-base_similarity_2026_binary \
#     --n_trials $N_TRIALS \
#     --bf16

# # Component task
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path FacebookAI/xlm-roberta-base \
#     --source_dataset data/preprocessing/datasets/similarity.csv \
#     --test_dataset $TEST_DATASET \
#     --task component \
#     --output_dir results/xlm-roberta-base_similarity_2026_component \
#     --n_trials $N_TRIALS \
#     --bf16

# # XLM-RoBERTa-CGEC
# # Binary task
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path /scratch/nicolasal97/im_project/xlm-roberta-cgec \
#     --source_dataset data/preprocessing/datasets/similarity.csv \
#     --test_dataset $TEST_DATASET \
#     --task binary \
#     --output_dir results/xlm-roberta-cgec_similarity_2026_binary \
#     --n_trials $N_TRIALS \
#     --bf16

# # Component task
# python3 -u src/evaluating/finetune_and_eval_model.py \
#     --model_name_or_path /scratch/nicolasal97/im_project/xlm-roberta-cgec \
#     --source_dataset data/preprocessing/datasets/similarity.csv \
#     --test_dataset $TEST_DATASET \
#     --task component \
#     --output_dir results/xlm-roberta-cgec_similarity_2026_component \
#     --n_trials $N_TRIALS \
#     --bf16


##### 3) Diversity
echo ""
echo "=== Dataset 3/4: Diversity ==="
echo "----------------------------------------"

# BETO Base
# Binary task
python3 -u src/evaluating/finetune_and_eval_model.py \
    --model_name_or_path dccuchile/bert-base-spanish-wwm-uncased \
    --source_dataset data/preprocessing/datasets/diversity.csv \
    --test_dataset $TEST_DATASET \
    --task binary \
    --output_dir results/beto-base_diversity_2026_binary \
    --n_trials $N_TRIALS \
    --bf16

# Component task
python3 -u src/evaluating/finetune_and_eval_model.py \
    --model_name_or_path dccuchile/bert-base-spanish-wwm-uncased \
    --source_dataset data/preprocessing/datasets/diversity.csv \
    --test_dataset $TEST_DATASET \
    --task component \
    --output_dir results/beto-base_diversity_2026_component \
    --n_trials $N_TRIALS \
    --bf16

# BETO-CGEC
# Binary task
python3 -u src/evaluating/finetune_and_eval_model.py \
    --model_name_or_path /scratch/nicolasal97/im_project/beto-cgec \
    --source_dataset data/preprocessing/datasets/diversity.csv \
    --test_dataset $TEST_DATASET \
    --task binary \
    --output_dir results/beto-cgec_diversity_2026_binary \
    --n_trials $N_TRIALS \
    --bf16

# Component task
python3 -u src/evaluating/finetune_and_eval_model.py \
    --model_name_or_path /scratch/nicolasal97/im_project/beto-cgec \
    --source_dataset data/preprocessing/datasets/diversity.csv \
    --test_dataset $TEST_DATASET \
    --task component \
    --output_dir results/beto-cgec_diversity_2026_component \
    --n_trials $N_TRIALS \
    --bf16

# SpanBERTa Base
# Binary task
python3 -u src/evaluating/finetune_and_eval_model.py \
    --model_name_or_path skimai/spanberta-base-cased \
    --source_dataset data/preprocessing/datasets/diversity.csv \
    --test_dataset $TEST_DATASET \
    --task binary \
    --output_dir results/spanberta-base_diversity_2026_binary \
    --n_trials $N_TRIALS \
    --bf16

# Component task
python3 -u src/evaluating/finetune_and_eval_model.py \
    --model_name_or_path skimai/spanberta-base-cased \
    --source_dataset data/preprocessing/datasets/diversity.csv \
    --test_dataset $TEST_DATASET \
    --task component \
    --output_dir results/spanberta-base_diversity_2026_component \
    --n_trials $N_TRIALS \
    --bf16

# SpanBERTa-CGEC
# Binary task
python3 -u src/evaluating/finetune_and_eval_model.py \
    --model_name_or_path /scratch/nicolasal97/im_project/spanberta-cgec \
    --source_dataset data/preprocessing/datasets/diversity.csv \
    --test_dataset $TEST_DATASET \
    --task binary \
    --output_dir results/spanberta-cgec_diversity_2026_binary \
    --n_trials $N_TRIALS \
    --bf16

# Component task
python3 -u src/evaluating/finetune_and_eval_model.py \
    --model_name_or_path /scratch/nicolasal97/im_project/spanberta-cgec \
    --source_dataset data/preprocessing/datasets/diversity.csv \
    --test_dataset $TEST_DATASET \
    --task component \
    --output_dir results/spanberta-cgec_diversity_2026_component \
    --n_trials $N_TRIALS \
    --bf16

# XLM-RoBERTa Base
# Binary task
python3 -u src/evaluating/finetune_and_eval_model.py \
    --model_name_or_path FacebookAI/xlm-roberta-base \
    --source_dataset data/preprocessing/datasets/diversity.csv \
    --test_dataset $TEST_DATASET \
    --task binary \
    --output_dir results/xlm-roberta-base_diversity_2026_binary \
    --n_trials $N_TRIALS \
    --bf16

# Component task
python3 -u src/evaluating/finetune_and_eval_model.py \
    --model_name_or_path FacebookAI/xlm-roberta-base \
    --source_dataset data/preprocessing/datasets/diversity.csv \
    --test_dataset $TEST_DATASET \
    --task component \
    --output_dir results/xlm-roberta-base_diversity_2026_component \
    --n_trials $N_TRIALS \
    --bf16

# XLM-RoBERTa-CGEC
# Binary task
python3 -u src/evaluating/finetune_and_eval_model.py \
    --model_name_or_path /scratch/nicolasal97/im_project/xlm-roberta-cgec \
    --source_dataset data/preprocessing/datasets/diversity.csv \
    --test_dataset $TEST_DATASET \
    --task binary \
    --output_dir results/xlm-roberta-cgec_diversity_2026_binary \
    --n_trials $N_TRIALS \
    --bf16

# Component task
python3 -u src/evaluating/finetune_and_eval_model.py \
    --model_name_or_path /scratch/nicolasal97/im_project/xlm-roberta-cgec \
    --source_dataset data/preprocessing/datasets/diversity.csv \
    --test_dataset $TEST_DATASET \
    --task component \
    --output_dir results/xlm-roberta-cgec_diversity_2026_component \
    --n_trials $N_TRIALS \
    --bf16


##### 4) Full
echo ""
echo "=== Dataset 4/4: Full ==="
echo "----------------------------------------"

# BETO Base
# Binary task
python3 -u src/evaluating/finetune_and_eval_model.py \
    --model_name_or_path dccuchile/bert-base-spanish-wwm-uncased \
    --source_dataset data/preprocessing/datasets/full.csv \
    --test_dataset $TEST_DATASET \
    --task binary \
    --output_dir results/beto-base_full_2026_binary \
    --n_trials $N_TRIALS \
    --bf16

# Component task
python3 -u src/evaluating/finetune_and_eval_model.py \
    --model_name_or_path dccuchile/bert-base-spanish-wwm-uncased \
    --source_dataset data/preprocessing/datasets/full.csv \
    --test_dataset $TEST_DATASET \
    --task component \
    --output_dir results/beto-base_full_2026_component \
    --n_trials $N_TRIALS \
    --bf16

# BETO-CGEC
# Binary task
python3 -u src/evaluating/finetune_and_eval_model.py \
    --model_name_or_path /scratch/nicolasal97/im_project/beto-cgec \
    --source_dataset data/preprocessing/datasets/full.csv \
    --test_dataset $TEST_DATASET \
    --task binary \
    --output_dir results/beto-cgec_full_2026_binary \
    --n_trials $N_TRIALS \
    --bf16

# Component task
python3 -u src/evaluating/finetune_and_eval_model.py \
    --model_name_or_path /scratch/nicolasal97/im_project/beto-cgec \
    --source_dataset data/preprocessing/datasets/full.csv \
    --test_dataset $TEST_DATASET \
    --task component \
    --output_dir results/beto-cgec_full_2026_component \
    --n_trials $N_TRIALS \
    --bf16

# SpanBERTa Base
# Binary task
python3 -u src/evaluating/finetune_and_eval_model.py \
    --model_name_or_path skimai/spanberta-base-cased \
    --source_dataset data/preprocessing/datasets/full.csv \
    --test_dataset $TEST_DATASET \
    --task binary \
    --output_dir results/spanberta-base_full_2026_binary \
    --n_trials $N_TRIALS \
    --bf16

# Component task
python3 -u src/evaluating/finetune_and_eval_model.py \
    --model_name_or_path skimai/spanberta-base-cased \
    --source_dataset data/preprocessing/datasets/full.csv \
    --test_dataset $TEST_DATASET \
    --task component \
    --output_dir results/spanberta-base_full_2026_component \
    --n_trials $N_TRIALS \
    --bf16

# SpanBERTa-CGEC
# Binary task
python3 -u src/evaluating/finetune_and_eval_model.py \
    --model_name_or_path /scratch/nicolasal97/im_project/spanberta-cgec \
    --source_dataset data/preprocessing/datasets/full.csv \
    --test_dataset $TEST_DATASET \
    --task binary \
    --output_dir results/spanberta-cgec_full_2026_binary \
    --n_trials $N_TRIALS \
    --bf16

# Component task
python3 -u src/evaluating/finetune_and_eval_model.py \
    --model_name_or_path /scratch/nicolasal97/im_project/spanberta-cgec \
    --source_dataset data/preprocessing/datasets/full.csv \
    --test_dataset $TEST_DATASET \
    --task component \
    --output_dir results/spanberta-cgec_full_2026_component \
    --n_trials $N_TRIALS \
    --bf16

# XLM-RoBERTa Base
# Binary task
python3 -u src/evaluating/finetune_and_eval_model.py \
    --model_name_or_path FacebookAI/xlm-roberta-base \
    --source_dataset data/preprocessing/datasets/full.csv \
    --test_dataset $TEST_DATASET \
    --task binary \
    --output_dir results/xlm-roberta-base_full_2026_binary \
    --n_trials $N_TRIALS \
    --bf16

# Component task
python3 -u src/evaluating/finetune_and_eval_model.py \
    --model_name_or_path FacebookAI/xlm-roberta-base \
    --source_dataset data/preprocessing/datasets/full.csv \
    --test_dataset $TEST_DATASET \
    --task component \
    --output_dir results/xlm-roberta-base_full_2026_component \
    --n_trials $N_TRIALS \
    --bf16

# XLM-RoBERTa-CGEC
# Binary task
python3 -u src/evaluating/finetune_and_eval_model.py \
    --model_name_or_path /scratch/nicolasal97/im_project/xlm-roberta-cgec \
    --source_dataset data/preprocessing/datasets/full.csv \
    --test_dataset $TEST_DATASET \
    --task binary \
    --output_dir results/xlm-roberta-cgec_full_2026_binary \
    --n_trials $N_TRIALS \
    --bf16

# Component task
python3 -u src/evaluating/finetune_and_eval_model.py \
    --model_name_or_path /scratch/nicolasal97/im_project/xlm-roberta-cgec \
    --source_dataset data/preprocessing/datasets/full.csv \
    --test_dataset $TEST_DATASET \
    --task component \
    --output_dir results/xlm-roberta-cgec_full_2026_component \
    --n_trials $N_TRIALS \
    --bf16


deactivate
module purge
echo "Script finished"

# sbatch --dependency=afterok:JOBID preprocessing.bash