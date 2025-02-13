#!/bin/bash
#SBATCH --job-name=deepseek_eval    # Job name
#SBATCH --output=deepseek_%j.out    # Output file name (%j expands to jobId)
#SBATCH --error=deepseek_%j.err     # Error file name (%j expands to jobId)
#SBATCH --time=24:00:00             # Time limit (24 hours)
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks-per-node=1         # Number of tasks per node
#SBATCH --cpus-per-task=4           # Number of CPU cores per task
#SBATCH --mem=16G                   # Memory per node
#SBATCH --partition=gpu             # GPU partition
#SBATCH --gres=gpu:1               # Request 1 GPU

# Run the evaluation script
python deepseek_unaided.py --model "deepseek-ai/DeepSeek-R1" 
