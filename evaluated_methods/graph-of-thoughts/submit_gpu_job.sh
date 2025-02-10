#!/bin/bash

#SBATCH --job-name="huggingface_gpu_job"
#SBATCH --output=job_%x-%j.out
#SBATCH --error=job_%x-%j.err
#SBATCH --partition="gpu"
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:2
#SBATCH --mem=256G

python /p/llmreliability/test_repos/graph-of-thoughts/examples/GSM8k/gsm8k_process.py