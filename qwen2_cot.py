import datasets
import requests
import json
import random
from tqdm import tqdm
import time

def load_samples(dataset_name, num_samples=150, for_examples=False, exclude_indices=None):
    """
    Load samples from the dataset. If for_examples=True, load a separate set for CoT examples
    to avoid overlap with test samples. exclude_indices can be used to avoid overlap between
    examples and test samples when only one split is available.
    """
    if dataset_name == "gsm8k":
        dataset = datasets.load_dataset('gsm8k', 'main', split='train' if for_examples else 'test')
    elif dataset_name == "gsm-symbolic":
        dataset = datasets.load_dataset('apple/GSM-Symbolic', 'main')
        dataset = dataset['test']
    elif dataset_name == "mmlu":
        dataset = datasets.load_dataset('cais/mmlu', 'all', split='validation' if for_examples else 'test')
    elif dataset_name == "aqua":
        dataset = datasets.load_dataset('aqua_rat', 'raw', split='train' if for_examples else 'test')
    elif dataset_name == "svamp":
        dataset = datasets.load_dataset('ChilleD/SVAMP', split='train' if for_examples else 'test')
    
    total_samples = len(dataset)
    available_indices = set(range(total_samples))
    if exclude_indices:
        available_indices = available_indices - set(exclude_indices)
    
    num_to_sample = min(num_samples, len(available_indices))
    random_indices = random.sample(list(available_indices), num_to_sample)
    samples = [dataset[i] for i in random_indices]
    return samples, random_indices

def get_cot_examples(dataset_name, num_examples=3, exclude_indices=None):
    """Get chain of thought examples from the training/validation split of each dataset."""
    example_samples, used_indices = load_samples(dataset_name, num_examples, for_examples=True, exclude_indices=exclude_indices)
    examples = []
    
    for sample in example_samples:
        if dataset_name == "gsm8k":
            example = {
                "question": sample['question'],
                "reasoning": sample['answer'].split('####')[0].strip(),
                "answer": f"Final answer: #### {sample['answer'].split('####')[1].strip()}"
            }
        elif dataset_name == "svamp":
            example = {
                "question": f"{sample['Body']} {sample['Question']}",
                "reasoning": f"Let me solve this step by step:\n1. {sample['Body']}\n2. {sample['Question']}\n3. The answer is {sample['Answer']}",
                "answer": f"Final answer: #### {sample['Answer']}"
            }
        elif dataset_name == "mmlu":
            example = {
                "question": f"{sample['question']}\nA) {sample['choices'][0]}\nB) {sample['choices'][1]}\nC) {sample['choices'][2]}\nD) {sample['choices'][3]}",
                "reasoning": f"Let's analyze each option:\nA) {sample['choices'][0]}\nB) {sample['choices'][1]}\nC) {sample['choices'][2]}\nD) {sample['choices'][3]}",
                "answer": f"Final answer: #### {sample['answer']}"
            }
        elif dataset_name == "aqua":
            options = sample['options']
            formatted_options = "\n".join([f"{i+1}) {opt}" for i, opt in enumerate(options)])
            example = {
                "question": f"{sample['question']}\n{formatted_options}",
                "reasoning": "Let's solve this step by step:\n" + sample['rationale'] if 'rationale' in sample else "Let's analyze each option systematically.",
                "answer": f"Final answer: #### {sample['correct']}"
            }
        else: 
            example = {
                "question": sample['question'],
                "reasoning": "Let's solve this symbolically:\n" + sample['solution'] if 'solution' in sample else "Let's solve step by step.",
                "answer": f"Final answer: #### {sample['answer']}"
            }
        examples.append(example)
    
    return examples

def format_question(sample, dataset_name, use_cot=False):
    base_prompt = ""
    if use_cot:
        examples = get_cot_examples(dataset_name)
        base_prompt = "Here are some example solutions. Please follow a similar step-by-step reasoning approach:\n\n"
        for i, example in enumerate(examples, 1):
            base_prompt += f"Example {i}:\nQuestion: {example['question']}\n{example['reasoning']}\n{example['answer']}\n\n"
        base_prompt += "Now solve this problem:\n\n"

    if dataset_name == "gsm8k":
        return base_prompt + f"Solve this math problem:\n{sample['question']}. Please provide your final answer as a single number marked by 'Final answer: ####'"
    
    elif dataset_name == "gsm-symbolic":
        return base_prompt + f"Solve this math problem:\n{sample['question']}"
    
    elif dataset_name == "mmlu":
        return base_prompt + f"""Question: {sample['question']}
A) {sample['choices'][0]}
B) {sample['choices'][1]}
C) {sample['choices'][2]}
D) {sample['choices'][3]}

Please provide your answer as a single letter (A, B, C, or D) followed by a brief explanation. Please mark it as 'Final answer: #### [answer].'"""
    
    elif dataset_name == "aqua":
        options = sample['options']
        formatted_options = "\n".join([f"{i+1}) {opt}" for i, opt in enumerate(options)])
        return base_prompt + f"""Question: {sample['question']}
{formatted_options}

Please solve this question and provide your final answer as a single letter (A - E) at the end of your response. Please mark it as 'Final answer: #### [answer].'"""
    
    elif dataset_name == "svamp":
        return base_prompt + f"""Solve this math word problem and put your final numerical answer at the end of your response. Please mark your final answer as 'Final answer: #### [answer]'.
Question: {sample['Body']} {sample['Question']}"""

def ask_model(formatted_question, model_name):
    url = "http://localhost:8000/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": formatted_question
            }
        ]
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"Error: {response.status_code}"
    except Exception as e:
        return f"Error: {str(e)}"

def collect_responses(model_name="Qwen/Qwen2-7B-Instruct", dataset_name="gsm8k", use_cot=False):
    example_indices = None
    if use_cot:
        _, example_indices = load_samples(dataset_name, 3, for_examples=True)
    
    samples, _ = load_samples(dataset_name, 150, exclude_indices=example_indices)
    results = []
    
    for sample in tqdm(samples, desc=f"Processing {dataset_name}"):
        formatted_question = format_question(sample, dataset_name, use_cot)
        
        if dataset_name == "mmlu":
            num_to_letter = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: '5'}
            ground_truth = num_to_letter.get(sample['answer'], str(sample['answer']))
        elif dataset_name == "aqua":
            ground_truth = str(sample['correct'])
        elif dataset_name == 'svamp':
            ground_truth = sample['Answer']
        else:
            ground_truth = sample['answer']
            
        model_response = ask_model(formatted_question, model_name)
        results.append({
            'question': formatted_question,
            'ground_truth': ground_truth,
            'model_response': model_response
        })
        time.sleep(0.1)
    
    model_short_name = model_name.split('/')[-1]
    cot_suffix = "_cot" if use_cot else ""
    filename = f'{dataset_name}_responses_{model_short_name}{cot_suffix}.json'
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    model_name = "Qwen/Qwen2-7B-Instruct"
    datasets_to_evaluate = ["gsm8k", "gsm-symbolic", "mmlu", "aqua", "svamp"]
    use_cot = True  
    
    print(f"Starting evaluation of {model_name} on multiple datasets...")
    print(f"Chain of thought prompting: {'enabled' if use_cot else 'disabled'}")
    
    for dataset_name in datasets_to_evaluate:
        print(f"\nEvaluating on {dataset_name}...")
        collect_responses(model_name, dataset_name, use_cot)
        cot_suffix = "_cot" if use_cot else ""
        print(f"Results saved to {dataset_name}_responses_{model_name.split('/')[-1]}{cot_suffix}.json")
    
    print("\nEvaluation complete!")