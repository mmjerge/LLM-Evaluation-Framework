import datasets
import json
import random
from tqdm import tqdm
import os
import time
import argparse
from together import Together

def load_samples(dataset_name, num_samples=150):
    if dataset_name == "gsm8k":
        dataset = datasets.load_dataset('gsm8k', 'main', split='test')
    elif dataset_name == "gsm-symbolic":
        dataset = datasets.load_dataset('apple/GSM-Symbolic', 'main')
        dataset = dataset['test']
    elif dataset_name == "mmlu":
        dataset = datasets.load_dataset('cais/mmlu', 'all', split='test')
    elif dataset_name == "aqua":
        dataset = datasets.load_dataset('aqua_rat', 'raw', split='test')
    elif dataset_name == "svamp":
        dataset = datasets.load_dataset('ChilleD/SVAMP', split='test')
    
    total_samples = len(dataset)
    random_indices = random.sample(range(total_samples), num_samples)
    samples = [dataset[i] for i in random_indices]
    return samples

def format_question(sample, dataset_name):
    if dataset_name == "gsm8k":
        return f"Solve this math problem:\n{sample['question']}. Please provide your final answer as a single number marked by 'Final answer: #### [answer]'"
    
    elif dataset_name == "gsm-symbolic":
        return f"Solve this math problem:\n{sample['question']}"
    
    elif dataset_name == "mmlu":
        return f"""Question: {sample['question']}
A) {sample['choices'][0]}
B) {sample['choices'][1]}
C) {sample['choices'][2]}
D) {sample['choices'][3]}

Please provide your answer as a single letter (A, B, C, or D) followed by a brief explanation. Please mark it as 'Final answer: #### [answer].'"""
    
    elif dataset_name == "aqua":
        options = sample['options']
        formatted_options = "\n".join([f"{i+1}) {opt}" for i, opt in enumerate(options)])
        return f"""Question: {sample['question']}
{formatted_options}

Please solve this question and provide your final answer as a single letter (A - E) at the end of your response. Please mark it as 'Final answer: #### [answer].'"""
    
    elif dataset_name == "svamp":
        return f"""Solve this math word problem and put your final numerical answer at the end of your response. Please mark your final answer as 'Final answer: #### [answer]'.
Question: {sample['Body']} {sample['Question']}"""

def ask_model(formatted_question, model_name, client):
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": formatted_question}],
            max_tokens=2000,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

def collect_responses(model_name, dataset_name, client):
    samples = load_samples(dataset_name, 150)
    results = []
    
    for sample in tqdm(samples, desc=f"Processing {dataset_name}"):
        formatted_question = format_question(sample, dataset_name)
        
        if dataset_name == "mmlu":
            num_to_letter = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: '5'}
            ground_truth = num_to_letter.get(sample['answer'], str(sample['answer']))
        elif dataset_name == "aqua":
            ground_truth = str(sample['correct'])
        elif dataset_name == 'svamp':
            ground_truth = sample['Answer']
        else:
            ground_truth = sample['answer']
            
        model_response = ask_model(formatted_question, model_name, client)
        
        results.append({
            'question': formatted_question,
            'ground_truth': ground_truth,
            'model_response': model_response
        })
        time.sleep(0.1)  # Rate limiting
    
    model_short_name = model_name.split('/')[-1]
    filename = f'{dataset_name}_responses_{model_short_name}.json'
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate language models on multiple datasets using TogetherAI API')
    parser.add_argument('--model', type=str, default="deepseek-ai/DeepSeek-R1",
                        help='Name of the model to evaluate')
    parser.add_argument('--datasets', nargs='+', 
                        default=["gsm8k", "gsm-symbolic", "mmlu", "aqua", "svamp"],
                        help='List of datasets to evaluate on')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    api_key = "tgp_v1_2h6y0v7eHcP8-uRQrFXzBDVOfnMdTWq5FKwyMFU1Els"
    
    client = Together(api_key=api_key)
    
    print(f"Starting evaluation of {args.model}...")
    
    for dataset_name in args.datasets:
        print(f"\nEvaluating on {dataset_name}...")
        collect_responses(args.model, dataset_name, client)
        print(f"Results saved to {dataset_name}_responses_{args.model.split('/')[-1]}.json")
    
    print("\nEvaluation complete!")