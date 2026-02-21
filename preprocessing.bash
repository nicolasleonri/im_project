#!/bin/bash
#SBATCH --job-name=preprocessing_paragrapher
#SBATCH --output=logs/preprocessing/slurm/preprocessing_paragrapher_%j.out

#SBATCH --partition=scavenger
#SBATCH --account=agfritz
#SBATCH --qos=standard

#SBATCH --nodes=1
#SBATCH --ntasks=1

#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=3G
#SBATCH --gres=gpu:h100:1

#SBATCH --time=02:00:00

echo "Loading modules..."
module purge
module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1
module load virtualenv/20.26.2-GCCcore-13.3.0

echo "Setting folders..."
export HF_HOME=/scratch/nicolasal97/.cache/huggingface
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export NLTK_DATA=/home/nicolasal97/nltk_data
export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1
export OMP_NUM_THREADS=4

echo "Activating virtual environment..."
source venv/preprocessing/bin/activate

echo "Running python script..." 
python3 -u src/preprocessing/paragrapher.py \
    --input data/preprocessing/informal_economy.csv \
    --output data/preprocessing/test_set.csv \
    --model jeffcookio/Mistral-Small-3.2-24B-Instruct-2506-awq-sym \
    --tokenizer_mode mistral \
    --tensor_parallel_size 1
    --no_sample

python3 -u src/preprocessing/paragrapher.py \
    --input data/preprocessing/informal_economy.csv \
    --output data/preprocessing/test_set.csv \
    --model jeffcookio/Mistral-Small-3.2-24B-Instruct-2506-awq-sym \
    --tokenizer_mode mistral \
    --tensor_parallel_size 1

# python3 -u src/preprocessing/paragrapher.py \
#     --input data/preprocessing/informal_economy.csv \
#     --output data/preprocessing/test_set.csv \
#     --model BSC-LT/salamandra-7b-instruct \
#     --tokenizer BSC-LT/salamandra-7b-instruct \
#     --tensor_parallel_size 2 

deactivate
module purge
echo "Script finished"