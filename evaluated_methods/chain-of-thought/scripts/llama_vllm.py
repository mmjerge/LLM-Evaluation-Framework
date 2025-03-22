import datasets
import requests
import json
import random
from tqdm import tqdm
import time
import os
import logging
from functools import wraps
from typing import Dict, List, Optional, Union, Callable, Any
from collections import Counter, defaultdict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("vllm_evaluation")

# Embedded VLLMCallTracker class
class VLLMCallTracker:
    """
    A class to track vLLM API calls, providing metrics on method usage and performance.
    """
    def __init__(self, log_to_file: bool = False, log_file: str = "vllm_tracking.log"):
        self.call_counter = Counter()
        self.method_latency = defaultdict(list)
        self.prompt_counter = 0
        self.total_tokens_generated = 0
        self.total_prompt_tokens = 0
        self.log_to_file = log_to_file
        self._model_name = None  # Will store model name if available
        
        if log_to_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            logger.addHandler(file_handler)
    
    def track(self, method_name: Optional[str] = None):
        """
        Decorator to track calls to vLLM API methods.
        
        Args:
            method_name: Optional override for the method name to track
        
        Returns:
            Decorated function that tracks calls
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                tracked_name = method_name or func.__name__
                start_time = time.time()
                
                # Log the call
                self.call_counter[tracked_name] += 1
                logger.info(f"Calling vLLM method: {tracked_name}")
                
                # Track individual prompts if first arg is a list of strings (common vLLM pattern)
                prompt_count = 0
                if args and len(args) > 0 and isinstance(args[0], list):
                    prompts = args[0]
                    if all(isinstance(item, str) for item in prompts):
                        prompt_count = len(prompts)
                        self.prompt_counter += prompt_count
                        
                        # Count each prompt as an individual API call as well
                        self.call_counter[tracked_name] += (prompt_count - 1)  # -1 because we already incremented once
                        
                        logger.info(f"Processing {prompt_count} prompts in this batch (counting as {prompt_count} API calls)")
                else:
                    # If not a batch, still increment prompt counter for single prompt
                    self.prompt_counter += 1
                
                # Execute the function
                result = func(*args, **kwargs)
                
                # Calculate metrics
                latency = time.time() - start_time
                self.method_latency[tracked_name].append(latency)
                
                # Track token usage based on output type
                # For OpenAI-style responses
                if hasattr(result, 'usage'):
                    if hasattr(result.usage, 'completion_tokens'):
                        self.total_tokens_generated += result.usage.completion_tokens
                    if hasattr(result.usage, 'prompt_tokens'):
                        self.total_prompt_tokens += result.usage.prompt_tokens
                
                # For direct vLLM responses (list of RequestOutput objects)
                elif isinstance(result, list) and len(result) > 0:
                    try:
                        # Try to import tokenizers for counting if not already available
                        import transformers
                        from transformers import AutoTokenizer
                        
                        # For prompt tokens - get from args if possible
                        if args and isinstance(args[0], list) and all(isinstance(x, str) for x in args[0]):
                            # Get model name to load correct tokenizer
                            model_name = None
                            if hasattr(self, '_model_name'):
                                model_name = self._model_name
                            elif args and hasattr(args[0], '_model_name'):
                                model_name = args[0]._model_name
                            
                            # Default to a common tokenizer if we can't determine the model
                            if not model_name:
                                model_name = "gpt2"
                                
                            # Load tokenizer and count prompt tokens
                            tokenizer = AutoTokenizer.from_pretrained(model_name)
                            prompts = args[0]
                            for prompt in prompts:
                                prompt_tokens = len(tokenizer.encode(prompt))
                                self.total_prompt_tokens += prompt_tokens
                        
                        # Count completion tokens from outputs
                        for output in result:
                            if hasattr(output, 'outputs') and output.outputs:
                                for gen_output in output.outputs:
                                    if hasattr(gen_output, 'text') and gen_output.text:
                                        # Use the same tokenizer to count completion tokens
                                        if 'tokenizer' not in locals():
                                            tokenizer = AutoTokenizer.from_pretrained("gpt2")
                                        completion_tokens = len(tokenizer.encode(gen_output.text))
                                        self.total_tokens_generated += completion_tokens
                    except (ImportError, Exception) as e:
                        logger.warning(f"Failed to count tokens: {e}")
                
                # Try to estimate token count for request/response with simple heuristic
                # Only do this if we couldn't count tokens more precisely
                if self.total_prompt_tokens == 0 and isinstance(args[0], str):
                    # Very rough estimate: ~4 chars per token
                    self.total_prompt_tokens += len(args[0]) // 4
                
                if self.total_tokens_generated == 0 and isinstance(result, str):
                    # Very rough estimate: ~4 chars per token
                    self.total_tokens_generated += len(result) // 4
                
                logger.info(f"Completed vLLM method: {tracked_name} (latency: {latency:.4f}s)")
                return result
            
            return wrapper
        
        return decorator
    
    def get_call_count(self, method_name: Optional[str] = None) -> Union[int, Dict[str, int]]:
        """
        Get the number of calls for a specific method or all methods.
        
        Args:
            method_name: The method name to get call count for, or None for all methods
            
        Returns:
            Call count for the method or dictionary of all method counts
        """
        if method_name:
            return self.call_counter[method_name]
        return dict(self.call_counter)
    
    def get_avg_latency(self, method_name: Optional[str] = None) -> Union[float, Dict[str, float]]:
        """
        Get the average latency for a specific method or all methods.
        
        Args:
            method_name: The method name to get latency for, or None for all methods
            
        Returns:
            Average latency for the method or dictionary of all method latencies
        """
        if method_name and method_name in self.method_latency:
            latencies = self.method_latency[method_name]
            return sum(latencies) / len(latencies) if latencies else 0
        
        result = {}
        for method, latencies in self.method_latency.items():
            result[method] = sum(latencies) / len(latencies) if latencies else 0
        return result
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics about vLLM API usage.
        
        Returns:
            Dictionary containing all tracked metrics
        """
        return {
            "call_counts": dict(self.call_counter),
            "avg_latencies": self.get_avg_latency(),
            "total_calls": sum(self.call_counter.values()),
            "total_prompts": self.prompt_counter,
            "total_tokens_generated": self.total_tokens_generated,
            "total_prompt_tokens": self.total_prompt_tokens
        }
    
    def reset(self):
        """Reset all tracking metrics."""
        self.call_counter.clear()
        self.method_latency.clear()
        self.prompt_counter = 0
        self.total_tokens_generated = 0
        self.total_prompt_tokens = 0


# Create tracker instance
tracker = VLLMCallTracker(log_to_file=True, log_file="llm_evaluation_tracking.log")

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

@tracker.track(method_name="llm_inference")
def ask_model(formatted_question, model_name, host="localhost", port=8000):
    """Track each model inference call using the vLLM tracker"""
    url = f"http://{host}:{port}/v1/chat/completions"
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
            result = response.json()
            
            # Try to extract token counts from response if available
            if 'usage' in result:
                if 'completion_tokens' in result['usage']:
                    tracker.total_tokens_generated += result['usage']['completion_tokens']
                if 'prompt_tokens' in result['usage']:
                    tracker.total_prompt_tokens += result['usage']['prompt_tokens']
            
            return result["choices"][0]["message"]["content"]
        else:
            logger.error(f"API error: {response.status_code}")
            return f"Error: {response.status_code}"
    except Exception as e:
        logger.error(f"Exception during API call: {str(e)}")
        return f"Error: {str(e)}"

def collect_responses(model_name="meta-llama/Meta-Llama-3-8B-Instruct", dataset_name="gsm8k", use_cot=False, num_samples=150):
    example_indices = None
    if use_cot:
        _, example_indices = load_samples(dataset_name, 3, for_examples=True)
    
    samples, _ = load_samples(dataset_name, num_samples, exclude_indices=example_indices)
    results = []
    
    print(f"Processing {len(samples)} samples from {dataset_name} dataset...")
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
            
        model_response = ask_model(formatted_question, model_name, vllm_host, vllm_port)
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

def print_tracking_metrics():
    """Print the current tracking metrics"""
    metrics = tracker.get_metrics()
    print("\n===== vLLM API Call Tracking Metrics =====")
    print(f"Total API calls: {metrics['total_calls']}")
    print(f"Total prompts processed: {metrics['total_prompts']}")
    print(f"Call counts by method: {metrics['call_counts']}")
    print(f"Average latency: {metrics['avg_latencies'].get('llm_inference', 0):.4f} seconds")
    print(f"Total tokens generated: {metrics['total_tokens_generated']}")
    print(f"Total prompt tokens: {metrics['total_prompt_tokens']}")
    print("==========================================")
    
    # Save metrics to JSON file
    with open("evaluation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    return metrics

if __name__ == "__main__":
    # Change to the Llama-3 model
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    
    # Define the datasets to evaluate
    datasets_to_evaluate = ["gsm8k", "gsm-symbolic", "mmlu", "aqua", "svamp"]
    
    # Set to True to use chain-of-thought prompting
    use_cot = True
    
    num_samples = 150  # You can set this back to 150 for the full evaluation
    
    vllm_host = "localhost"  # Change this if your server is on a different host
    vllm_port = 8000         # Change this if your server is on a different port
    
    print(f"Starting evaluation of {model_name} on multiple datasets...")
    print(f"Chain of thought prompting: {'enabled' if use_cot else 'disabled'}")
    print(f"vLLM API call tracking: enabled")
    print(f"vLLM server: http://{vllm_host}:{vllm_port}")
    
    start_time = time.time()
    
    for dataset_name in datasets_to_evaluate:
        print(f"\nEvaluating on {dataset_name}...")
        try:
            collect_responses(model_name, dataset_name, use_cot, num_samples)
        except Exception as e:
            print(f"Error evaluating {dataset_name}: {str(e)}")
            print("Skipping to next dataset...")
        cot_suffix = "_cot" if use_cot else ""
        print(f"Results saved to {dataset_name}_responses_{model_name.split('/')[-1]}{cot_suffix}.json")
        
        # Print intermediate tracking metrics after each dataset
        print_tracking_metrics()
    
    # Print final tracking metrics
    total_time = time.time() - start_time
    print(f"\nEvaluation complete! Total time: {total_time:.2f} seconds")
    final_metrics = print_tracking_metrics()
    
    # Calculate some additional statistics
    if final_metrics['total_calls'] > 0:
        tokens_per_call = final_metrics['total_tokens_generated'] / final_metrics['total_calls']
        print(f"Average tokens generated per call: {tokens_per_call:.2f}")
        
    if total_time > 0:
        calls_per_second = final_metrics['total_calls'] / total_time
        print(f"API calls per second: {calls_per_second:.2f}")
        tokens_per_second = final_metrics['total_tokens_generated'] / total_time
        print(f"Tokens generated per second: {tokens_per_second:.2f}")