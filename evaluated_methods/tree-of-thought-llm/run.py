import os
import json
import re
import argparse
from tot.tasks import get_task
from tot.methods.bfs import solve, naive_solve
from tot.models import gpt, gpt_usage, claude, anthropic_claude
from anthropic import Anthropic
import random

def extract_final_answer(response, task):
    if task == 'svamp':
        match = re.search(r'####\s*(\d+(?:\.\d+)?)', response)
        if match:
            return float(match.group(1))
        
        numbers = re.findall(r'\d+(?:\.\d+)?', response)
        if numbers:
            return float(numbers[-1])
        
        return "No numerical answer found"
    else:
        match = re.search(r'####\s*([A-E])', response)
        if match:
            return match.group(1)
        
        match = re.search(r'option\s*\(?([A-E])\)?\.?\s*$', response, re.IGNORECASE)
        if match:
            return match.group(1)
        
        options = re.findall(r'([A-E])\)', response)
        if options:
            return options[-1]
        
        match = re.search(r'([A-E])\s*$', response)
        if match:
            return match.group(1)
        
        return "No final answer found"
    
def get_model_function(backend):
    if backend.startswith('gpt'):
        return gpt
    elif backend.startswith('claude'):
        return anthropic_claude
    elif backend.startswith('mistralai'):
        return gpt
    else:
        raise ValueError(f"Unsupported backend: {backend}")

def get_usage_function(backend):
    if backend.startswith('gpt'):
        return gpt_usage
    elif backend.startswith('mistralai'):
        return gpt_usage
    elif backend.startswith('claude'):
        return lambda x: {"completion_tokens": 0, "prompt_tokens": 0, "cost": 0}  # Placeholder for Claude usage
    else:
        raise ValueError(f"Unsupported backend: {backend}")

def run(args):
    task = get_task(args.task)
    results = []
    
    total_items = len(task.data)
    
    num_samples = min(150, total_items)
    selected_indices = random.sample(range(total_items), num_samples)
    
    if args.naive_run:
        file = f'./logs/{args.task}/{args.backend}_{args.temperature}_naive_{args.prompt_sample}_sample_{args.n_generate_sample}_random{num_samples}.json'
    else:
        file = f'./logs/{args.task}/{args.backend}_{args.temperature}_{args.method_generate}{args.n_generate_sample}_{args.method_evaluate}{args.n_evaluate_sample}_{args.method_select}{args.n_select_sample}_random{num_samples}.json'
    
    os.makedirs(os.path.dirname(file), exist_ok=True)
    
    model_function = get_model_function(args.backend)
    usage_function = get_usage_function(args.backend)
    
    for count, i in enumerate(selected_indices, 1):
        try:
            print(f"Processing question {count}/{num_samples} (Dataset index: {i})")
            print(f"Input: {task.get_input(i)}")
            
            if args.naive_run:
                ys, info = naive_solve(args, task, i, to_print=True)
            else:
                ys, info = solve(args, task, i, to_print=True)
            
            print(f"Debug - ys: {ys}")
            print(f"Debug - info: {info}")
            
            model_response = ys[0] if ys else ""
            final_answer = task.extract_answer(model_response)
            
            print(f"Debug - model_response: {model_response}")
            print(f"Debug - final_answer: {final_answer}")
            
            result = {
                "dataset_index": i,
                "question": task.get_input(i),
                "model_response": model_response,
                "extracted_answer": final_answer,
                "correct_answer": task.data[i].get('answer', 'No correct answer found'),
                "solve_info": info
            }
            results.append(result)
            
            if not args.naive_run:
                infos = [task.test_output(i, y) for y in ys]
                info.update({'idx': i, 'ys': ys, 'infos': infos, 'usage_so_far': usage_function(args.backend)})
            
            print(f"Processed question {count}/{num_samples}")
            print("---")
        except Exception as e:
            print(f"Error processing question {i}: {str(e)}")
            results.append({
                "dataset_index": i,
                "question": task.get_input(i),
                "error": str(e)
            })
    
    with open(file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results written to {file}")
    print('usage_so_far', usage_function(args.backend))

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--backend', type=str, choices=['gpt-4', 'gpt-3.5-turbo', 'gpt-4o', 'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo', 'mistralai/Mixtral-8x22B-Instruct-v0.1', 'claude-3-5-sonnet-20240620'], default='gpt-4o')
    args.add_argument('--temperature', type=float, default=0.7)
    args.add_argument('--task', type=str, required=True, choices=['game24', 'text', 'crosswords', 'gsm8k', 'mmlu', 'aqua', 'svamp'])
    args.add_argument('--naive_run', action='store_true')
    args.add_argument('--prompt_sample', type=str, choices=['standard', 'cot'])
    args.add_argument('--method_generate', type=str, choices=['sample', 'propose'])
    args.add_argument('--method_evaluate', type=str, choices=['value', 'vote'])
    args.add_argument('--method_select', type=str, choices=['sample', 'greedy'], default='greedy')
    args.add_argument('--n_generate_sample', type=int, default=1)
    args.add_argument('--n_evaluate_sample', type=int, default=1)
    args.add_argument('--n_select_sample', type=int, default=1)
    return args.parse_args()

if __name__ == '__main__':
    args = parse_args()
    print(args)
    run(args)