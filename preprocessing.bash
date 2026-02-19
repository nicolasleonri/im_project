#!/bin/bash
#SBATCH --job-name=run_preprocessing
#SBATCH --output=logs/preprocessing/slurm/run_preprocessing_%j.out

#SBATCH --partition=scavenger
#SBATCH --account=agfritz
#SBATCH --qos=standard

#SBATCH --nodes=1
#SBATCH --ntasks=1

#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:h100:1
#SBATCH --mem-per-cpu=1G
#SBATCH --time=00:00:10

echo "Loading modules..."
module purge
module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1
module load virtualenv/20.32.0-GCCcore-14.3.0

echo "Setting folders..."
export HF_HOME=/scratch/nicolasal97/.cache/huggingface
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export NLTK_DATA=/home/nicolasal97/nltk_data
export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1

echo "Activating virtual environment..."
source venv/preprocessing/bin/activate

echo "Running python script..."
python3 -u src/preprocessing/paragrapher.py

deactivate
module purge

echo "Script finished"