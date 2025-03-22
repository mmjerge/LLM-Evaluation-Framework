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
import os
import logging
import gc
from typing import Dict, List, Any
from collections import Counter

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("vllm_tracking")

# Create a file handler if it doesn't exist
file_handler = logging.FileHandler("react_evaluation_tracking.log")
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Simple counters
tracked_problems = 0
tool_call_count = 0
total_tokens_generated = 0
total_prompt_tokens = 0

# Based on log analysis, each problem typically has around 7 vLLM API calls
VLLM_CALLS_PER_PROBLEM = 5

def load_samples(dataset_name, num_samples=150, for_examples=False, exclude_indices=None):
    """Load samples from the dataset."""
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
    available_indices = set(range(total_samples))
    if exclude_indices:
        available_indices = available_indices - set(exclude_indices)
    
    num_to_sample = min(num_samples, len(available_indices))
    random_indices = random.sample(list(available_indices), num_to_sample)
    samples = [dataset[i] for i in random_indices]
    return samples, random_indices

def calculator_with_tracking(llm_math, expression):
    """Wrapper function for the calculator that tracks usage"""
    global tool_call_count, total_tokens_generated
    
    # Track this tool call
    tool_call_count += 1
    logger.info(f"Calculator tool called, total calls: {tool_call_count}")
    
    # Call the LLMMathChain
    try:
        # Try to use invoke method first (newer LangChain versions)
        if hasattr(llm_math, "invoke") and callable(llm_math.invoke):
            result = llm_math.invoke(expression)
        # Fall back to __call__ if available
        elif hasattr(llm_math, "__call__") and callable(llm_math.__call__):
            result = llm_math(expression)
        else:
            # Last resort, try direct call
            result = llm_math(input=expression)
        
        # Estimate token count
        if isinstance(result, str):
            total_tokens_generated += len(result) // 4
        elif isinstance(result, dict) and "text" in result:
            total_tokens_generated += len(result["text"]) // 4
        
        # Return string result if possible
        if isinstance(result, str):
            return result
        elif isinstance(result, dict) and "text" in result:
            return result["text"]
        elif isinstance(result, dict) and "output" in result:
            return result["output"]
        else:
            return str(result)
    except Exception as e:
        logger.error(f"Error calling calculator: {str(e)}")
        return f"Error calculating {expression}: {str(e)}"

def get_tools_for_dataset(dataset_name, llm_math_chain):
    """Get appropriate tools for each dataset type."""
    if dataset_name in ["gsm8k", "gsm-symbolic", "svamp"]:
        # Create a tool that uses our tracking wrapper and the shared math chain
        calculator = Tool(
            name="Calculator",
            func=lambda expression: calculator_with_tracking(llm_math_chain, expression),
            description="Useful for performing mathematical calculations. Use this for solving math problems step by step."
        )
        
        return [calculator]
    else:
        return []

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
    
    match = re.findall(r'\d+(?:\.\d+)?', response)
    if match:
        return match[-1]
    
    return None

def run_agent_with_tracking(agent_executor, input_data):
    """Run the agent and track API calls without modifying any objects"""
    global tracked_problems, total_prompt_tokens, total_tokens_generated
    
    # Increment problem counter
    tracked_problems += 1
    logger.info(f"Processing problem {tracked_problems}")
    
    # Count the initial prompt tokens
    if isinstance(input_data, dict) and "input" in input_data and isinstance(input_data["input"], str):
        total_prompt_tokens += len(input_data["input"]) // 4
    
    # Call the agent executor
    try:
        # Run the agent
        result = agent_executor.invoke(input_data)
        
        # Count the generated tokens
        if isinstance(result, dict) and "output" in result and isinstance(result["output"], str):
            total_tokens_generated += len(result["output"]) // 4
        
        return result
    
    except Exception as e:
        logger.error(f"Error in agent execution: {str(e)}")
        raise e

def print_tracking_metrics():
    """Print the current tracking metrics"""
    # Calculate total API calls using the fixed multiplier approach
    vllm_api_calls = tracked_problems * VLLM_CALLS_PER_PROBLEM
    total_api_calls = vllm_api_calls + tool_call_count
    avg_calls_per_problem = total_api_calls / tracked_problems if tracked_problems > 0 else 0
    
    print("\n===== vLLM API Call Tracking Metrics =====")
    print(f"Total problems processed: {tracked_problems}")
    print(f"Estimated vLLM API calls ({VLLM_CALLS_PER_PROBLEM} per problem): {vllm_api_calls}")
    print(f"Total calculator tool calls: {tool_call_count}")
    print(f"Total combined calls (API + tools): {total_api_calls}")
    print(f"Average calls per problem: {avg_calls_per_problem:.2f}")
    print(f"Total tokens generated (estimate): {total_tokens_generated}")
    print(f"Total prompt tokens (estimate): {total_prompt_tokens}")
    print("==========================================")
    
    # Prepare metrics for JSON
    metrics = {
        "problems_processed": tracked_problems,
        "estimated_vllm_api_calls": vllm_api_calls,
        "tool_calls": tool_call_count,
        "total_calls": total_api_calls,
        "avg_calls_per_problem": round(avg_calls_per_problem, 2),
        "total_tokens_generated": total_tokens_generated,
        "total_prompt_tokens": total_prompt_tokens
    }
    
    # Save metrics to JSON file
    with open("react_evaluation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    return metrics

def evaluate_dataset(dataset_name, num_samples, llm, llm_math_chain):
    """Evaluate a dataset using ReAct framework with shared LLM and math chain."""
    print(f"\nEvaluating {dataset_name}...")

    try:
        # Get tools for the dataset using the shared math chain
        tools = get_tools_for_dataset(dataset_name, llm_math_chain)
        prompt = hub.pull("hwchase17/react")
        
        # Create ReAct agent
        agent = create_react_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=False,
            max_iterations=5,
            early_stopping_method="generate",
            handle_parsing_errors=True 
        )
        
        # Load samples
        samples, _ = load_samples(dataset_name, num_samples)
        results = []
        
        # Start tracking for this dataset
        dataset_start_time = time.time()
        
        for sample in tqdm(samples, desc=f"Processing {dataset_name}"):
            formatted_question = format_question(sample, dataset_name)
            
            try:
                # Use our tracked invoke function
                response = run_agent_with_tracking(agent_executor, {"input": formatted_question})
                model_response = response['output']
                
                if dataset_name == "mmlu":
                    num_to_letter = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}
                    ground_truth = num_to_letter.get(sample['answer'], str(sample['answer']))
                elif dataset_name == "aqua":
                    ground_truth = str(sample['correct'])
                elif dataset_name == 'svamp':
                    ground_truth = str(sample['Answer'])
                else:
                    ground_truth = str(sample['answer'])
                
                model_answer = extract_answer(model_response)
                
                results.append({
                    'question': formatted_question,
                    'ground_truth': ground_truth,
                    'model_response': model_response,
                    'extracted_answer': model_answer
                })
                
            except Exception as e:
                logger.error(f"Error processing question: {str(e)}")
                print(f"Error processing question: {str(e)}")
                results.append({
                    'question': formatted_question,
                    'ground_truth': ground_truth if 'ground_truth' in locals() else None,
                    'model_response': f"Error: {str(e)}",
                    'extracted_answer': None
                })
            
            # Add a small delay between samples to allow for resource cleanup
            time.sleep(0.1)
        
        # Calculate dataset metrics
        dataset_time = time.time() - dataset_start_time
        
        # Save results
        filename = f'{dataset_name}_responses_{model_name.split("/")[-1]}_react.json'
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print dataset statistics
        print(f"Completed {dataset_name} in {dataset_time:.2f} seconds")
        print(f"Processed {len(samples)} samples")
        print(f"Results saved to {filename}")
        
        return results
    
    except Exception as e:
        logger.error(f"Error evaluating dataset {dataset_name}: {str(e)}")
        print(f"Error evaluating dataset {dataset_name}: {str(e)}")
        return []

if __name__ == "__main__":
    # Change to Llama 3.1 model
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    datasets_to_evaluate = ["gsm8k", "gsm-symbolic", "mmlu", "aqua", "svamp"]
    
    # Number of samples to process per dataset
    num_samples = 150  # Adjust as needed (50 for faster testing, 150 for full evaluation)
    
    print(f"Starting evaluation of {model_name} using ReAct framework...")
    print(f"vLLM API call tracking: enabled (using fixed multiplier of {VLLM_CALLS_PER_PROBLEM} calls per problem)")
    print(f"Number of samples per dataset: {num_samples}")
    
    # Create a single VLLM instance to be used for all datasets
    llm = VLLM(
        model=model_name,
        trust_remote_code=True,
        max_new_tokens=512,
        top_k=10,
        top_p=0.95,
        temperature=0.8,
    )
    
    # Create a shared LLMMathChain to be reused across all dataset evaluations
    llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=False)
    
    all_results = {}
    start_time = time.time()
    
    for dataset_name in datasets_to_evaluate:
        try:
            results = evaluate_dataset(dataset_name, num_samples, llm, llm_math_chain)
            all_results[dataset_name] = results
            print(f"Completed evaluation of {dataset_name}")
            
            # Print intermediate tracking metrics
            print_tracking_metrics()
            
            # Force garbage collection between datasets to free memory
            gc.collect()
            
        except Exception as e:
            print(f"Error evaluating {dataset_name}: {str(e)}")
            print("Skipping to next dataset...")
            # Force garbage collection after error
            gc.collect()
    
    # Print final tracking metrics and statistics
    total_time = time.time() - start_time
    print(f"\nEvaluation complete! Total time: {total_time:.2f} seconds")
    final_metrics = print_tracking_metrics()
    
    # Calculate additional statistics
    total_api_calls = (tracked_problems * VLLM_CALLS_PER_PROBLEM) + tool_call_count
    
    if total_api_calls > 0:
        tokens_per_call = total_tokens_generated / total_api_calls
        print(f"Average tokens generated per call: {tokens_per_call:.2f}")
        
    if total_time > 0:
        calls_per_second = total_api_calls / total_time
        print(f"API calls per second: {calls_per_second:.2f}")



