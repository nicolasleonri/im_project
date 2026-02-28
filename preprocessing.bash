#!/bin/bash
#SBATCH --job-name=preprocessing_annotator
#SBATCH --output=logs/preprocessing/slurm/preprocessing_annotator_%j.out
#SBATCH --partition=scavenger
#SBATCH --account=agfritz
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1

#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#alternative SBATCH --gres=gpu:h100:1
#SBATCH --time=00:00:30

# Load necessary modules
module purge
module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1
module load virtualenv/20.26.2-GCCcore-13.3.0

# Environment variables
export HF_HOME=/scratch/nicolasal97/.cache/huggingface
export NLTK_DATA=/home/nicolasal97/nltk_data
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TORCH_NCCL_ASYNC_ERROR_HANDLING=0
# export PYTORCH_ALLOC_CONF=expandable_segments:True
# export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1
# export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1
# export VLLM_WORKER_USE_GPU=1
# export TORCH_USE_CUDA_DSA=1
# export OMP_NUM_THREADS=4

# Activate virtual environment
source venv/preprocessing/bin/activate

echo "Setup done. Running python script..."
############# FIRST TASK: Paragrapher + Sentence Splitter #############
### Needs around 2h30min + 2x2GB + 1xA5000
# python3 -u src/preprocessing/paragrapher.py \
#     --input data/preprocessing/sentences_Meta-Llama-3.1-8B-Instruct-TurboMind-AWQ-4bit.csv \
#     --output data/preprocessing/test_set_llama.csv \
#     --model Aaron2599/Meta-Llama-3.1-8B-Instruct-TurboMind-AWQ-4bit \
#     --tokenizer meta-llama/Llama-3.1-8B-Instruct \
#     --tensor_parallel_size 1 \
#     --skip_vllm \
#     --skip_spacy 

### Needs around 3h + 2x2GB + 1xA5000
# python3 -u src/preprocessing/paragrapher.py \
#     --input data/preprocessing/sentences_Mistral-Small-3.2-24B-Instruct-2506-awq-sym.csv \
#     --output data/preprocessing/test_set_mistral.csv \
#     --model jeffcookio/Mistral-Small-3.2-24B-Instruct-2506-awq-sym \
#     --tokenizer mistralai/Mistral-Small-3.2-24B-Instruct-2506 \
#     --tokenizer_mode mistral \
#     --tensor_parallel_size 1 \
#     --skip_vllm \
#     --skip_spacy

### Needs around 4h20 + 2x2GB + 1xH100
# python3 -u src/preprocessing/paragrapher.py \
#     --input data/preprocessing/sentences_DeepSeek-R1-Distill-Qwen-32B-AWQ.csv \
#     --output data/preprocessing/test_set_deepseek.csv \
#     --model Valdemardi/DeepSeek-R1-Distill-Qwen-32B-AWQ \
#     --tokenizer deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
#     --tensor_parallel_size 1 \
#     --skip_vllm \
#     --skip_spacy

############# SECOND TASK: Annotator #############
## Needs around 1h + 2x2GB + 1xA5000
# python3 -u src/preprocessing/annotator.py \
#     --input data/preprocessing/test_set_mistral.csv \
#     --output data/preprocessing/test_set_llama_annotated.csv \
#     --guidelines data/preprocessing/annotation_guidelines_v1.md \
#     --model Aaron2599/Meta-Llama-3.1-8B-Instruct-TurboMind-AWQ-4bit \
#     --tokenizer meta-llama/Llama-3.1-8B-Instruct \
#     --tensor_parallel_size 1 \
#     --max_model_len 12288 \
#     --max_tokens 512

# ### Needs around 2h + 2x2GB + 1xA5000
# python3 -u src/preprocessing/annotator.py \
#     --input data/preprocessing/test_set_mistral.csv \
#     --output data/preprocessing/test_set_mistral_annotated.csv \
#     --guidelines data/preprocessing/annotation_guidelines_v1.md \
#     --model jeffcookio/Mistral-Small-3.2-24B-Instruct-2506-awq-sym \
#     --tokenizer mistralai/Mistral-Small-3.2-24B-Instruct-2506 \
#     --tokenizer_mode mistral \
#     --tensor_parallel_size 1 \
#     --max_model_len 12288 \
#     --max_tokens 512

## Needs around 1h + 2x2GB + 1xH100
# python3 -u src/preprocessing/annotator.py \
#     --input data/preprocessing/test_set_mistral.csv \
#     --output data/preprocessing/test_set_deepseek_annotated.csv \
#     --guidelines data/preprocessing/annotation_guidelines_v1.md \
#     --model Valdemardi/DeepSeek-R1-Distill-Qwen-32B-AWQ \
#     --tokenizer deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
#     --tensor_parallel_size 1 \
#     --max_model_len 12288 \
#     --max_tokens 512

## Needs around 2h + 2x2GB + 1xA5000
# python3 -u src/preprocessing/annotator.py \
#     --input data/preprocessing/test_set_mistral.csv \
#     --output data/preprocessing/test_set_magistral_annotated.csv \
#     --guidelines data/preprocessing/annotation_guidelines_v1.md \
#     --model mistralai/Magistral-Small-2506_gguf \
#     --tokenizer mistralai/Mistral-Small-3.2-24B-Instruct-2506 \
#     --tokenizer_mode mistral \
#     --tensor_parallel_size 1 \
#     --max_model_len 40960 \
#     --max_tokens 2048 \
#     --temperature 0.7 \
#     --top_p 0.95

## Needs around 2h + 2x2GB + 1xH100
# python3 -u src/preprocessing/annotator.py \
#     --input data/preprocessing/test_set_mistral.csv \
#     --output data/preprocessing/test_set_magistral_annotated.csv \
#     --guidelines data/preprocessing/annotation_guidelines_v1.md \
#     --model unsloth/Magistral-Small-2506-unsloth-bnb-4bit \
#     --tokenizer mistralai/Mistral-Small-3.2-24B-Instruct-2506 \
#     --tokenizer_mode mistral \
#     --tensor_parallel_size 1 \
#     --max_model_len 40960 \
#     --max_tokens 2048 \
#     --temperature 0.7 \
#     --top_p 0.95

## Needs around 15 secs + 1x1GB + 1xCPU
python3 -u scripts/preprocessing/calculate_iaa.py \
    --inputs data/preprocessing/test_set_mistral_annotated.csv data/preprocessing/test_set_llama_annotated.csv data/preprocessing/test_set_deepseek_annotated.csv data/preprocessing/test_set_magistral_annotated.csv \
    --output_dir data/preprocessing/iaa/

deactivate
module purge
echo "Script finished"

# sbatch --dependency=afterok:JOBID preprocessing.bash