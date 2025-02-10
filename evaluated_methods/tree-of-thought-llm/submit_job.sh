#!/bin/bash

#SBATCH --job-name="huggingface_gpu_job"	
#SBATCH --output=job_%x-%j.out							# output file name of your choice
#SBATCH --error=job_%x-%j.err								# error file name of your choice
#SBATCH --partition="gpu"										# tells the machine you want to run with gpu	
#SBATCH --gres=gpu:1												

# Your bash command goes here
python run.py --backend gpt-4o --task mmlu --temperature 0.7 --prompt_sample standard --method_generate sample --method_evaluate value --method_select greedy --n_generate_sample 1 --n_evaluate_sample 1 --n_select_sample 1