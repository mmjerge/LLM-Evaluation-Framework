import datasets
import json
import random
from tqdm import tqdm
import time
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.llms import VLLM
from langchain.tools import Tool
from langchain.chains import LLMMathChain
import re

def load_samples(dataset_name, num_samples=150, for_examples=False, exclude_indices=None):
    """Load samples from the dataset."""
    if dataset_name == "gsm8k":
        dataset = datasets.load_dataset('gsm8k', 'main', split='test')
    elif dataset_name == "gsm-symbolic":
        dataset = datasets.load_dataset('apple/GSM-Symbolic', 'main')
        dataset = dataset['test']  # GSM-Symbolic only has test split
    elif dataset_name == "mmlu":
        dataset = datasets.load_dataset('cais/mmlu', 'all', split='test')
    elif dataset_name == "aqua":
        dataset = datasets.load_dataset('aqua_rat', 'raw', split='test')
    elif dataset_name == "svamp":
        dataset = datasets.load_dataset('ChilleD/SVAMP', split='test')
    
    total_samples = len(dataset)
    available_indices = set(range(total_samples))
    if exclude_indices:
        available_indices = available_indices - set(exclude_indices)
    
    num_to_sample = min(num_samples, len(available_indices))
    random_indices = random.sample(list(available_indices), num_to_sample)
    samples = [dataset[i] for i in random_indices]
    return samples, random_indices

def get_tools_for_dataset(dataset_name, llm):
    """Get appropriate tools for each dataset type."""
    if dataset_name in ["gsm8k", "gsm-symbolic", "svamp"]:
        llm_math = LLMMathChain.from_llm(llm=llm, verbose=False)
        calculator = Tool(
            name="Calculator",
            func=llm_math.run,
            description="Useful for performing mathematical calculations. Use this for solving math problems step by step."
        )
        return [calculator]
    else:
        return []  # MMLU and AQUA don't need special tools

def format_question(sample, dataset_name):
    """Format the question based on dataset type."""
    if dataset_name == "gsm8k":
        return f"Solve this math problem. Provide your final answer as a single number marked as 'Final answer: ####'.\nQuestion: {sample['question']}"
    
    elif dataset_name == "gsm-symbolic":
        return f"Solve this math problem. Provide your final answer as a single number marked as 'Final answer: ####'.\nQuestion: {sample['question']}"
    
    elif dataset_name == "mmlu":
        return f"""Answer this multiple choice question. Provide your final answer as a single letter (A, B, C, or D) marked as 'Final answer: ####'.
Question: {sample['question']}
A) {sample['choices'][0]}
B) {sample['choices'][1]}
C) {sample['choices'][2]}
D) {sample['choices'][3]}"""
    
    elif dataset_name == "aqua":
        options = sample['options']
        formatted_options = "\n".join([f"{i+1}) {opt}" for i, opt in enumerate(options)])
        return f"""Answer this multiple choice question. Provide your final answer as a number (1-5) marked as 'Final answer: ####'.
Question: {sample['question']}
{formatted_options}"""
    
    elif dataset_name == "svamp":
        return f"""Solve this math word problem. Provide your final answer as a single number marked as 'Final answer: ####'.
Question: {sample['Body']} {sample['Question']}"""

def extract_answer(response):
    """Extract the final answer from the model's response."""
    match = re.search(r'Final answer: #### (.+)', response)
    if match:
        return match.group(1).strip()
    return None

def evaluate_dataset(model_name, dataset_name, num_samples=150):
    """Evaluate a dataset using ReAct framework."""
    print(f"\nEvaluating {dataset_name}...")
    
    # Initialize vLLM
    llm = VLLM(
        model=model_name,
        trust_remote_code=True,
        max_new_tokens=512,
        top_k=10,
        top_p=0.95,
        temperature=0.8,
    )
    
    # Get tools for the dataset
    tools = get_tools_for_dataset(dataset_name, llm)
    
    # Get ReAct prompt
    prompt = hub.pull("hwchase17/react")
    
    # Create ReAct agent
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        max_iterations=5,
        early_stopping_method="generate"
    )
    
    # Load samples
    samples, _ = load_samples(dataset_name, num_samples)
    results = []
    
    for sample in tqdm(samples, desc=f"Processing {dataset_name}"):
        # Format question
        formatted_question = format_question(sample, dataset_name)
        
        try:
            # Get model response using ReAct
            response = agent_executor.invoke({"input": formatted_question})
            model_response = response['output']
            
            # Extract ground truth
            if dataset_name == "mmlu":
                # Convert numeric ground truth (0-3) to letter (A-D)
                num_to_letter = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
                ground_truth = num_to_letter.get(sample['answer'], str(sample['answer']))
            elif dataset_name == "aqua":
                ground_truth = str(sample['correct'])
            elif dataset_name == 'svamp':
                ground_truth = str(sample['Answer'])
            else:
                ground_truth = str(sample['answer'])
            
            # Extract model's final answer
            model_answer = extract_answer(model_response)
            
            results.append({
                'question': formatted_question,
                'ground_truth': ground_truth,
                'model_response': model_response,
                'extracted_answer': model_answer
            })
            
        except Exception as e:
            print(f"Error processing question: {str(e)}")
            results.append({
                'question': formatted_question,
                'ground_truth': ground_truth if 'ground_truth' in locals() else None,
                'model_response': f"Error: {str(e)}",
                'extracted_answer': None
            })
        
        time.sleep(0.1)  # Small delay between requests
    
    # Save results
    filename = f'{dataset_name}_responses_{model_name.split("/")[-1]}_react.json'
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    model_name = "Qwen/Qwen2-7B-Instruct"
    datasets_to_evaluate = ["gsm8k", "gsm-symbolic", "mmlu", "aqua", "svamp"]
    
    print(f"Starting evaluation of {model_name} using ReAct framework...")
    
    all_results = {}
    for dataset_name in datasets_to_evaluate:
        results = evaluate_dataset(model_name, dataset_name)
        all_results[dataset_name] = results
        print(f"Completed evaluation of {dataset_name}")
    
    print("\nEvaluation complete! Results saved to individual JSON files.")