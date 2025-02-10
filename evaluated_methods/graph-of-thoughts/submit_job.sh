#!/bin/bash

#SBATCH --job-name="huggingface_gpu_job"	
#SBATCH --output=job_%x-%j.out							# output file name of your choice
#SBATCH --error=job_%x-%j.err								# error file name of your choice
#SBATCH --partition="gpu"										# tells the machine you want to run with gpu	
#SBATCH --gres=gpu:1

# Your bash command goes here
python3 /p/llmreliability/test_repos/graph-of-thoughts/examples/doc_merge/doc_merge.py