"""
USAGE EXAMPLES:

1. Quick test (Claude 3 Haiku on gsm8k with 10 samples):
   python script.py --quick-test

2. Test with multi-step reasoning (~3.5 API calls per question):
   python script.py --models claude-3-5-sonnet-20241022 --datasets gsm8k --use-multi-step --num-samples 20

3. Test specific models on specific datasets:
   python script.py --models claude-3-haiku-20240307 claude-3-5-sonnet-20241022 --datasets gsm8k mmlu --num-samples 25

4. Test all Claude models on all datasets:
   python script.py --all-models --all-datasets --num-samples 50

5. Test with chain-of-thought prompting:
   python script.py --models claude-3-5-sonnet-20241022 --datasets gsm8k --use-cot --num-samples 30

6. Test with both multi-step reasoning and chain-of-thought:
   python script.py --models claude-3-opus-20240229 --datasets gsm8k --use-multi-step --use-cot --num-samples 15

7. Custom delay between API calls:
   python script.py --models claude-3-5-sonnet-20241022 --datasets mmlu --delay 1.0 --num-samples 20

8. Test Claude 3.5 Sonnet on math datasets with multi-step reasoning:
   python script.py --models claude-3-5-sonnet-20241022 --datasets gsm8k gsm-symbolic svamp --use-multi-step --num-samples 25

Multi-step reasoning process:
- Step 1: Problem understanding and analysis
- Step 2: Solution planning and approach  
- Step 3: Step-by-step execution
- Step 4: Verification (conditional, for complex problems)

Available Claude models: 
- claude-3-haiku-20240307 (fastest, cheapest)
- claude-3-sonnet-20240229 (balanced)
- claude-3-opus-20240229 (most capable, slowest)
- claude-3-5-sonnet-20241022 (latest, high performance)
- claude-3-5-haiku-20241022 (latest fast model)

Available datasets: gsm8k, gsm-symbolic, mmlu, aqua, svamp

For help: python script.py --help
"""

import datasets
import json
import random
from tqdm import tqdm
import time
import os
import logging
import argparse
from functools import wraps
from typing import Dict, List, Optional, Union, Callable, Any
from collections import Counter, defaultdict
import anthropic

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("anthropic_evaluation")

# Note: This script uses Anthropic's Messages API for Claude models
# with multi-step reasoning that makes approximately 3.5 API calls per question

# Embedded AnthropicCallTracker class
class AnthropicCallTracker:
    """
    A class to track Anthropic API calls, providing metrics on method usage and performance.
    """
    def __init__(self, log_to_file: bool = False, log_file: str = "anthropic_tracking.log"):
        self.call_counter = Counter()
        self.method_latency = defaultdict(list)
        self.prompt_counter = 0
        self.total_tokens_generated = 0
        self.total_prompt_tokens = 0
        self.total_cost = 0.0  # Track estimated cost
        self.log_to_file = log_to_file
        
        if log_to_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            logger.addHandler(file_handler)
    
    def track(self, method_name: Optional[str] = None):
        """
        Decorator to track calls to Anthropic API methods.
        
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
                self.prompt_counter += 1
                logger.info(f"Calling Anthropic method: {tracked_name}")
                
                # Execute the function
                result = func(*args, **kwargs)
                
                # Calculate metrics
                latency = time.time() - start_time
                self.method_latency[tracked_name].append(latency)
                
                # Track token usage from Anthropic response
                if hasattr(result, 'usage'):
                    if hasattr(result.usage, 'output_tokens'):
                        self.total_tokens_generated += result.usage.output_tokens
                    if hasattr(result.usage, 'input_tokens'):
                        self.total_prompt_tokens += result.usage.input_tokens
                    
                    # Estimate cost for Claude models
                    model_name = kwargs.get('model', 'claude-3-haiku-20240307')
                    cost = self._estimate_cost(model_name, result.usage)
                    self.total_cost += cost
                
                logger.info(f"Completed Anthropic method: {tracked_name} (latency: {latency:.4f}s)")
                return result
            
            return wrapper
        
        return decorator
    
    def _estimate_cost(self, model_name: str, usage) -> float:
        """Estimate cost based on model and token usage (rough estimates)"""
        # Cost per 1M tokens for Claude models (as of 2024/2025 - these may change)
        pricing = {
            'claude-3-haiku-20240307': {'input': 0.25, 'output': 1.25},
            'claude-3-sonnet-20240229': {'input': 3.0, 'output': 15.0},
            'claude-3-opus-20240229': {'input': 15.0, 'output': 75.0},
            'claude-3-5-sonnet-20241022': {'input': 3.0, 'output': 15.0},
            'claude-3-5-haiku-20241022': {'input': 1.0, 'output': 5.0},
        }
        
        # Find the best matching model
        model_key = 'claude-3-haiku-20240307'  # default to cheapest
        for key in pricing.keys():
            if key in model_name:
                model_key = key
                break
        
        input_cost = (usage.input_tokens / 1_000_000) * pricing[model_key]['input']
        output_cost = (usage.output_tokens / 1_000_000) * pricing[model_key]['output']
        
        return input_cost + output_cost
    
    def get_call_count(self, method_name: Optional[str] = None) -> Union[int, Dict[str, int]]:
        """
        Get the number of calls for a specific method or all methods.
        """
        if method_name:
            return self.call_counter[method_name]
        return dict(self.call_counter)
    
    def get_avg_latency(self, method_name: Optional[str] = None) -> Union[float, Dict[str, float]]:
        """
        Get the average latency for a specific method or all methods.
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
        Get comprehensive metrics about Anthropic API usage.
        """
        return {
            "call_counts": dict(self.call_counter),
            "avg_latencies": self.get_avg_latency(),
            "total_calls": sum(self.call_counter.values()),
            "total_prompts": self.prompt_counter,
            "total_tokens_generated": self.total_tokens_generated,
            "total_prompt_tokens": self.total_prompt_tokens,
            "estimated_total_cost": self.total_cost
        }
    
    def reset(self):
        """Reset all tracking metrics."""
        self.call_counter.clear()
        self.method_latency.clear()
        self.prompt_counter = 0
        self.total_tokens_generated = 0
        self.total_prompt_tokens = 0
        self.total_cost = 0.0


# Create tracker instance
tracker = AnthropicCallTracker(log_to_file=True, log_file="anthropic_evaluation_tracking.log")

# Initialize Anthropic client
client = anthropic.Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)

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

@tracker.track(method_name="anthropic_inference")
def ask_model(formatted_question, model_name):
    """Track each model inference call using the Anthropic tracker"""
    try:
        response = client.messages.create(
            model=model_name,
            max_tokens=1000,
            temperature=0.1,
            messages=[
                {
                    "role": "user",
                    "content": formatted_question
                }
            ]
        )
        
        # Extract text from Anthropic response
        if hasattr(response, 'content') and len(response.content) > 0:
            # Handle case where content is a list of content blocks
            output_text = ""
            for content_block in response.content:
                if hasattr(content_block, 'text'):
                    output_text += content_block.text
                elif hasattr(content_block, 'type') and content_block.type == 'text':
                    output_text += content_block.text if hasattr(content_block, 'text') else str(content_block)
            return output_text
        else:
            return str(response)
        
    except Exception as e:
        logger.error(f"Exception during Anthropic API call: {str(e)}")
        return f"Error: {str(e)}"

def multi_step_reasoning(question, model_name, dataset_name, delay=0.5):
    """
    Implements a multi-step chain-of-thought reasoning process that makes approximately 3.5 API calls per question.
    
    Steps:
    1. Problem understanding and analysis
    2. Solution planning and approach
    3. Step-by-step execution
    4. Verification and final answer (conditional, based on complexity)
    """
    
    # Step 1: Problem Understanding and Analysis
    understanding_prompt = f"""
    Analyze and understand this problem thoroughly. Break down what is being asked and identify the key components:

    Problem: {question}

    Please provide:
    1. What type of problem this is
    2. What information is given
    3. What needs to be found or determined
    4. Any potential challenges or considerations

    Keep your analysis concise but thorough.
    """
    
    understanding = ask_model(understanding_prompt, model_name)
    time.sleep(delay)
    
    # Step 2: Solution Planning
    planning_prompt = f"""
    Based on this problem analysis:
    {understanding}

    Original problem: {question}

    Create a clear solution plan:
    1. What approach will you use to solve this?
    2. What are the specific steps needed?
    3. What calculations or reasoning will be required?

    Provide a structured plan without solving yet.
    """
    
    plan = ask_model(planning_prompt, model_name)
    time.sleep(delay)
    
    # Step 3: Step-by-step Execution
    execution_prompt = f"""
    Now execute the solution using this plan:
    {plan}

    Original problem: {question}

    Solve the problem step by step, showing your work clearly. For each step:
    1. State what you're doing
    2. Show the calculation or reasoning
    3. State the result

    Work through to get your answer.
    """
    
    execution = ask_model(execution_prompt, model_name)
    time.sleep(delay)
    
    # Step 4: Verification (conditional - approximately 50% of the time based on problem complexity)
    # We'll do verification for math problems and complex reasoning tasks
    needs_verification = any(keyword in question.lower() for keyword in [
        'calculate', 'solve', 'find', 'determine', 'what is', 'how many', 
        'mathematics', 'equation', 'problem', 'answer'
    ]) or dataset_name in ['gsm8k', 'gsm-symbolic', 'aqua', 'svamp']
    
    final_answer = execution
    
    if needs_verification:
        verification_prompt = f"""
        Review and verify this solution:

        Original problem: {question}
        Solution: {execution}

        Check:
        1. Is the reasoning correct?
        2. Are the calculations accurate?
        3. Does the answer make sense?
        4. Is this the final answer the problem is asking for?

        Provide the verified final answer clearly marked as 'Final answer: ####'
        """
        
        final_answer = ask_model(verification_prompt, model_name)
        time.sleep(delay)
    
    # Adjust prompt counter for multi-step reasoning
    # We made 3 or 4 API calls but it should count as 1 question
    # Since ask_model increments prompt_counter each time, we need to subtract the extra counts
    calls_made = 4 if needs_verification else 3
    tracker.prompt_counter = tracker.prompt_counter - calls_made + 1
    
    # Combine all reasoning steps for the complete response
    complete_reasoning = f"""
    PROBLEM ANALYSIS:
    {understanding}

    SOLUTION PLAN:
    {plan}

    STEP-BY-STEP EXECUTION:
    {execution}
    """
    
    if needs_verification:
        complete_reasoning += f"""

    VERIFICATION AND FINAL ANSWER:
    {final_answer}
    """
    else:
        complete_reasoning += f"""

    FINAL ANSWER:
    {final_answer}
    """
    
    return complete_reasoning

def collect_responses(model_name="claude-3-haiku-20240307", dataset_name="gsm8k", use_cot=False, use_multi_step=False, num_samples=150, delay=0.5):
    example_indices = None
    if use_cot:
        _, example_indices = load_samples(dataset_name, 3, for_examples=True)
    
    samples, _ = load_samples(dataset_name, num_samples, exclude_indices=example_indices)
    results = []
    
    method_description = "multi-step reasoning" if use_multi_step else ("chain-of-thought" if use_cot else "standard")
    print(f"Processing {len(samples)} samples from {dataset_name} dataset with {model_name} using {method_description}...")
    
    for sample in tqdm(samples, desc=f"Processing {dataset_name} with {model_name}"):
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
        
        # Use multi-step reasoning if enabled
        if use_multi_step:
            model_response = multi_step_reasoning(formatted_question, model_name, dataset_name, delay)
        else:
            model_response = ask_model(formatted_question, model_name)
            time.sleep(delay)
        
        results.append({
            'question': formatted_question,
            'ground_truth': ground_truth,
            'model_response': model_response
        })
    
    model_short_name = model_name.replace('/', '_').replace('-', '_')
    method_suffix = "_multi_step" if use_multi_step else ("_cot" if use_cot else "")
    filename = f'{dataset_name}_responses_{model_short_name}{method_suffix}.json'
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {filename}")
    return results

def parse_arguments():
    """Parse command line arguments for model and dataset selection"""
    parser = argparse.ArgumentParser(description="Evaluate Anthropic Claude models on multiple benchmarks")
    
    # Model selection
    parser.add_argument(
        "--models", 
        nargs="+",
        default=["claude-3-haiku-20240307"],
        choices=[
            "claude-3-haiku-20240307", 
            "claude-3-sonnet-20240229", 
            "claude-3-opus-20240229",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022"
        ],
        help="Claude models to evaluate (default: claude-3-haiku-20240307)"
    )
    
    # Dataset selection
    parser.add_argument(
        "--datasets",
        nargs="+", 
        default=["gsm8k"],
        choices=["gsm8k", "gsm-symbolic", "mmlu", "aqua", "svamp"],
        help="Datasets to evaluate on (default: gsm8k)"
    )
    
    # Evaluation parameters
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Number of samples per dataset (default: 50)"
    )
    
    parser.add_argument(
        "--use-cot",
        action="store_true",
        help="Enable chain-of-thought prompting"
    )
    
    parser.add_argument(
        "--use-multi-step",
        action="store_true",
        help="Enable multi-step reasoning (makes ~3.5 API calls per question)"
    )
    
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay between API calls in seconds (default: 0.5)"
    )
    
    # Convenience flags for common configurations
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Test all available Claude models"
    )
    
    parser.add_argument(
        "--all-datasets", 
        action="store_true",
        help="Test all available datasets"
    )
    
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Quick test: claude-3-haiku on gsm8k with 10 samples"
    )
    
    return parser.parse_args()

def get_models_and_datasets(args):
    """Get the final list of models and datasets based on arguments"""
    # All available options
    all_models = [
        "claude-3-haiku-20240307", 
        "claude-3-sonnet-20240229", 
        "claude-3-opus-20240229",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022"
    ]
    all_datasets = ["gsm8k", "gsm-symbolic", "mmlu", "aqua", "svamp"]
    
    # Handle convenience flags
    if args.quick_test:
        return ["claude-3-haiku-20240307"], ["gsm8k"], 10
    
    models = all_models if args.all_models else args.models
    datasets = all_datasets if args.all_datasets else args.datasets
    num_samples = args.num_samples
    
    return models, datasets, num_samples

def print_tracking_metrics():
    """Print the current tracking metrics"""
    metrics = tracker.get_metrics()
    print("\n===== Anthropic Claude API Call Tracking Metrics =====")
    print(f"Total API calls: {metrics['total_calls']}")
    print(f"Total prompts processed: {metrics['total_prompts']}")
    print(f"Call counts by method: {metrics['call_counts']}")
    print(f"Average latency: {metrics['avg_latencies'].get('anthropic_inference', 0):.4f} seconds")
    print(f"Total tokens generated: {metrics['total_tokens_generated']}")
    print(f"Total prompt tokens: {metrics['total_prompt_tokens']}")
    print(f"Estimated total cost: ${metrics['estimated_total_cost']:.4f}")
    print("=======================================================")
    
    # Save metrics to JSON file
    with open("evaluation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    return metrics

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Get models and datasets to evaluate
    models_to_evaluate, datasets_to_evaluate, num_samples = get_models_and_datasets(args)
    
    # Check if Anthropic API key is set
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: Please set the ANTHROPIC_API_KEY environment variable")
        print("You can set it by running: export ANTHROPIC_API_KEY='your-api-key-here'")
        exit(1)
    
    print("=" * 60)
    print("Anthropic Claude Model Evaluation on Multiple Benchmarks")
    print("=" * 60)
    print(f"Models to evaluate: {', '.join(models_to_evaluate)}")
    print(f"Datasets to evaluate: {', '.join(datasets_to_evaluate)}")
    print(f"Using Anthropic's Messages API for Claude models")
    
    if args.use_multi_step:
        print("Reasoning method: Multi-step reasoning (~3.5 API calls per question)")
    elif args.use_cot:
        print("Reasoning method: Chain-of-thought prompting")
    else:
        print("Reasoning method: Standard prompting")
    
    print(f"Number of samples per dataset: {num_samples}")
    print(f"Delay between API calls: {args.delay}s")
    print("Anthropic API call tracking: enabled")
    print("=" * 60)
    
    start_time = time.time()
    total_evaluations = len(models_to_evaluate) * len(datasets_to_evaluate)
    current_evaluation = 0
    
    # Track results for summary
    all_results = {}
    
    for model_name in models_to_evaluate:
        print(f"\nStarting evaluation with model: {model_name}")
        print("-" * 50)
        
        # Reset tracker for each model (optional - comment out to track across all models)
        # tracker.reset()
        
        model_results = {}
        
        for dataset_name in datasets_to_evaluate:
            current_evaluation += 1
            print(f"\nEvaluation {current_evaluation}/{total_evaluations}: {model_name} on {dataset_name}")
            
            try:
                results = collect_responses(
                    model_name=model_name,
                    dataset_name=dataset_name, 
                    use_cot=args.use_cot,
                    use_multi_step=args.use_multi_step,
                    num_samples=num_samples,
                    delay=args.delay
                )
                model_results[dataset_name] = len(results)
                
                # Print intermediate metrics for this model-dataset combination
                print(f"Completed {dataset_name} with {model_name}: {len(results)} samples processed")
                
            except Exception as e:
                print(f"Error evaluating {model_name} on {dataset_name}: {str(e)}")
                print("Continuing to next dataset...")
                model_results[dataset_name] = 0
        
        all_results[model_name] = model_results
        
        # Print summary for this model
        print(f"\nSummary for {model_name}:")
        for dataset, count in model_results.items():
            status = "COMPLETED" if count > 0 else "FAILED"
            print(f"  {status}: {dataset} - {count} samples")
    
    # Print final comprehensive metrics
    total_time = time.time() - start_time
    print(f"\nEvaluation complete! Total time: {total_time:.2f} seconds")
    
    # Print final tracking metrics
    final_metrics = print_tracking_metrics()
    
    # Print comprehensive summary
    print(f"\nCOMPREHENSIVE EVALUATION SUMMARY")
    print("=" * 60)
    for model_name, model_results in all_results.items():
        print(f"\nModel: {model_name}")
        total_samples = sum(model_results.values())
        print(f"   Total samples processed: {total_samples}")
        for dataset, count in model_results.items():
            status = "COMPLETED" if count > 0 else "FAILED"
            print(f"   {status}: {dataset} - {count} samples")
    
    # Calculate additional statistics
    if final_metrics['total_calls'] > 0:
        tokens_per_call = final_metrics['total_tokens_generated'] / final_metrics['total_calls']
        calls_per_question = final_metrics['total_calls'] / final_metrics['total_prompts'] if final_metrics['total_prompts'] > 0 else 0
        
        print(f"\nPerformance Metrics:")
        print(f"   Total questions processed: {final_metrics['total_prompts']}")
        print(f"   Total API calls made: {final_metrics['total_calls']}")
        print(f"   Average API calls per question: {calls_per_question:.1f}")
        print(f"   Average tokens generated per call: {tokens_per_call:.2f}")
        
        if total_time > 0:
            calls_per_second = final_metrics['total_calls'] / total_time
            tokens_per_second = final_metrics['total_tokens_generated'] / total_time
            questions_per_second = final_metrics['total_prompts'] / total_time
            print(f"   Questions processed per second: {questions_per_second:.2f}")
            print(f"   API calls per second: {calls_per_second:.2f}")
            print(f"   Tokens generated per second: {tokens_per_second:.2f}")
    
    print(f"\nEstimated total cost: ${final_metrics['estimated_total_cost']:.4f}")
    print(f"All results saved to individual JSON files")
    print(f"Metrics saved to evaluation_metrics.json")
    
    # Save comprehensive summary
    summary = {
        "evaluation_config": {
            "models": models_to_evaluate,
            "datasets": datasets_to_evaluate,
            "num_samples": num_samples,
            "use_cot": args.use_cot,
            "use_multi_step": args.use_multi_step,
            "delay": args.delay
        },
        "results": all_results,
        "metrics": final_metrics,
        "total_time": total_time
    }
    
    with open("evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Complete summary saved to evaluation_summary.json")