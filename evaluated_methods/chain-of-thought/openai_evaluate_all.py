"""
USAGE EXAMPLES:

1. Quick test (gpt-3.5-turbo on gsm8k with 10 samples):
   python script.py --quick-test

2. Test with multi-step reasoning (~3.5 API calls per question):
   python script.py --models gpt-4o --datasets gsm8k --use-multi-step --num-samples 20

3. Test specific models on specific datasets:
   python script.py --models gpt-3.5-turbo gpt-4o --datasets gsm8k mmlu --num-samples 25

4. Test all models on all datasets:
   python script.py --all-models --all-datasets --num-samples 50

5. Test with chain-of-thought prompting:
   python script.py --models gpt-4o --datasets gsm8k --use-cot --num-samples 30

6. Test with both multi-step reasoning and chain-of-thought:
   python script.py --models gpt-4o --datasets gsm8k --use-multi-step --use-cot --num-samples 15

7. Custom delay between API calls:
   python script.py --models gpt-4o --datasets mmlu --delay 1.0 --num-samples 20

8. Test gpt-4o on math datasets with multi-step reasoning:
   python script.py --models gpt-4o --datasets gsm8k gsm-symbolic svamp --use-multi-step --num-samples 25

Multi-step reasoning process:
- Step 1: Problem understanding and analysis
- Step 2: Solution planning and approach  
- Step 3: Step-by-step execution
- Step 4: Verification (conditional, for complex problems)

Available models: gpt-3.5-turbo, gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-4
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
import re
from functools import wraps
from typing import Dict, List, Optional, Union, Callable, Any
from collections import Counter, defaultdict
import openai

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("openai_evaluation")

# Note: This script uses OpenAI's Responses API (introduced March 2025)
# which combines Chat Completions simplicity with Assistants API capabilities
# and includes built-in tools like web search, file search, and computer use

# Embedded OpenAICallTracker class (renamed from VLLMCallTracker)
class OpenAICallTracker:
    """
    A class to track OpenAI API calls, providing metrics on method usage and performance.
    """
    def __init__(self, log_to_file: bool = False, log_file: str = "openai_tracking.log"):
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
        Decorator to track calls to OpenAI API methods.
        
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
                logger.info(f"Calling OpenAI method: {tracked_name}")
                
                # Execute the function
                result = func(*args, **kwargs)
                
                # Calculate metrics
                latency = time.time() - start_time
                self.method_latency[tracked_name].append(latency)
                
                # Track token usage from OpenAI response
                if hasattr(result, 'usage'):
                    if hasattr(result.usage, 'completion_tokens'):
                        self.total_tokens_generated += result.usage.completion_tokens
                    if hasattr(result.usage, 'prompt_tokens'):
                        self.total_prompt_tokens += result.usage.prompt_tokens
                    
                    # Estimate cost (rough estimates for GPT models)
                    model_name = kwargs.get('model', 'gpt-3.5-turbo')
                    cost = self._estimate_cost(model_name, result.usage)
                    self.total_cost += cost
                
                logger.info(f"Completed OpenAI method: {tracked_name} (latency: {latency:.4f}s)")
                return result
            
            return wrapper
        
        return decorator
    
    def _estimate_cost(self, model_name: str, usage) -> float:
        """Estimate cost based on model and token usage (rough estimates)"""
        # Cost per 1K tokens (as of 2024 - these may change)
        pricing = {
            'gpt-4': {'input': 0.03, 'output': 0.06},
            'gpt-4-turbo': {'input': 0.01, 'output': 0.03},
            'gpt-3.5-turbo': {'input': 0.0015, 'output': 0.002},
            'gpt-4o': {'input': 0.005, 'output': 0.015},
            'gpt-4o-mini': {'input': 0.00015, 'output': 0.0006},
        }
        
        # Find the best matching model
        model_key = 'gpt-3.5-turbo'  # default
        for key in pricing.keys():
            if key in model_name:
                model_key = key
                break
        
        input_cost = (usage.prompt_tokens / 1000) * pricing[model_key]['input']
        output_cost = (usage.completion_tokens / 1000) * pricing[model_key]['output']
        
        return input_cost + output_cost
    
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
        Get comprehensive metrics about OpenAI API usage.
        
        Returns:
            Dictionary containing all tracked metrics
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
tracker = OpenAICallTracker(log_to_file=True, log_file="openai_evaluation_tracking.log")

# Initialize OpenAI client for Responses API
client = openai.OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")  # Make sure to set this environment variable
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

def process_ground_truth(sample, dataset_name):
    """Process ground truth answers for different datasets - taken from the unaided evaluation script"""
    if dataset_name == "mmlu":
        num_to_letter = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}
        return num_to_letter.get(sample['answer'], str(sample['answer']))
    elif dataset_name == "aqua":
        return str(sample['correct'])
    elif dataset_name == 'svamp':
        return str(sample['Answer'])
    elif dataset_name == "gsm8k":
        # Extract numerical answer from gsm8k format
        answer_text = sample['answer']
        match = re.search(r'####\s*(\d+(?:\.\d+)?)', answer_text)
        if match:
            return match.group(1)
        else:
            # Fallback: try to find the last number in the answer
            numbers = re.findall(r'\d+(?:\.\d+)?', answer_text)
            return numbers[-1] if numbers else str(sample['answer'])
    elif dataset_name == "gsm-symbolic":
        # Extract numerical answer from gsm-symbolic format (same as gsm8k)
        answer_text = sample['answer']
        match = re.search(r'####\s*(\d+(?:\.\d+)?)', answer_text)
        if match:
            return match.group(1)
        else:
            # Fallback: try to find the last number in the answer
            numbers = re.findall(r'\d+(?:\.\d+)?', answer_text)
            return numbers[-1] if numbers else str(sample['answer'])
    else:
        return str(sample['answer'])

def evaluate_accuracy(results, dataset_name):
    """Evaluate model performance based on dataset-specific criteria - taken from the unaided evaluation script"""
    accuracies = {}
    
    for model, responses in results.items():
        if not responses:
            print(f"No responses for {model} on {dataset_name}")
            accuracies[model] = 0
            continue
            
        correct = 0
        total = len(responses)
        
        for item in responses:
            ground_truth = item['ground_truth']
            model_response = item['model_response']
            
            if isinstance(model_response, str) and isinstance(ground_truth, str):
                if dataset_name == "gsm8k":
                    # STRICT evaluation for GSM8K only
                    is_correct = False
                    
                    # Pattern 1: Final answer: #### [number] (most preferred format)
                    pattern1 = rf'FINAL ANSWER:\s*####\s*{re.escape(ground_truth)}\b'
                    if re.search(pattern1, model_response.upper()):
                        is_correct = True
                    
                    # Pattern 2: #### [number] (standard format)
                    elif re.search(rf'####\s*{re.escape(ground_truth)}\s*$', model_response.strip()):
                        is_correct = True
                    
                    # Pattern 3: #### [number] followed by whitespace/punctuation (but not other numbers)
                    elif re.search(rf'####\s*{re.escape(ground_truth)}(?:\s*$|\s*\.|$)', model_response):
                        is_correct = True
                    
                    if is_correct:
                        correct += 1
                        
                elif dataset_name == "gsm-symbolic":
                    # More lenient evaluation for gsm-symbolic
                    is_correct = False
                    
                    # Pattern 1: Final answer: #### [number]
                    pattern1 = rf'FINAL ANSWER:\s*####\s*{re.escape(ground_truth)}\b'
                    if re.search(pattern1, model_response.upper()):
                        is_correct = True
                    
                    # Pattern 2: #### [number] at end or followed by whitespace/punctuation
                    pattern2 = rf'####\s*{re.escape(ground_truth)}(?:\s|$|\.|,)'
                    if re.search(pattern2, model_response):
                        is_correct = True
                    
                    # Pattern 3: Final answer: [number] (without ####)
                    pattern3 = rf'FINAL ANSWER:\s*{re.escape(ground_truth)}(?:\s|$|\.|,)'
                    if re.search(pattern3, model_response.upper()):
                        is_correct = True
                    
                    # Pattern 4: The answer is [number] (common model response pattern)
                    pattern4 = rf'(?:THE ANSWER IS|ANSWER:|ANSWER IS)\s*{re.escape(ground_truth)}(?:\s|$|\.|,)'
                    if re.search(pattern4, model_response.upper()):
                        is_correct = True
                    
                    # Pattern 5: Last line contains just the number (fallback)
                    lines = model_response.strip().split('\n')
                    if lines and lines[-1].strip() == ground_truth:
                        is_correct = True
                    
                    # Pattern 6: Very end of response has the number (more restrictive fallback)
                    if model_response.strip().endswith(ground_truth):
                        is_correct = True
                    elif re.search(rf'{re.escape(ground_truth)}[\.\s]*$', model_response.strip()):
                        is_correct = True
                    
                    if is_correct:
                        correct += 1
                        
                elif dataset_name == "svamp":
                    # Strict evaluation for SVAMP - look for the answer in final answer format or at the end
                    is_correct = False
                    
                    # Pattern 1: Final answer: #### [number] (most preferred format)
                    pattern1 = rf'FINAL ANSWER:\s*####\s*{re.escape(ground_truth)}\b'
                    if re.search(pattern1, model_response.upper()):
                        is_correct = True
                    
                    # Pattern 2: #### [number] (standard format)
                    elif re.search(rf'####\s*{re.escape(ground_truth)}(?:\s|$|\.|,)', model_response):
                        is_correct = True
                    
                    # Pattern 3: Final answer: [number] (without ####)
                    elif re.search(rf'FINAL ANSWER:?\s*{re.escape(ground_truth)}(?:\s|$|\.|,)', model_response.upper()):
                        is_correct = True
                    
                    # Pattern 4: The answer is [number] 
                    elif re.search(rf'(?:THE ANSWER IS|ANSWER:|ANSWER IS)\s*{re.escape(ground_truth)}(?:\s|$|\.|,)', model_response.upper()):
                        is_correct = True
                    
                    # Pattern 5: Number appears at the very end of response (more restrictive)
                    elif re.search(rf'{re.escape(ground_truth)}[\.\s]*$', model_response.strip()):
                        is_correct = True
                    
                    # Pattern 6: Last line contains just the number
                    else:
                        lines = model_response.strip().split('\n')
                        if lines and lines[-1].strip() == ground_truth:
                            is_correct = True
                    
                    if is_correct:
                        correct += 1
                        
                elif dataset_name in ["mmlu"]:
                    ground_truth = ground_truth.upper()
                    model_response_upper = model_response.upper()
                    
                    is_correct = False
                    extracted_answer = None
                    
                    if f"FINAL ANSWER: #### {ground_truth}" in model_response_upper:
                        is_correct = True
                        extracted_answer = ground_truth
                    elif f"#### {ground_truth}" in model_response_upper:
                        is_correct = True
                        extracted_answer = ground_truth
                    else:
                        patterns = [
                            r'FINAL ANSWER:?\s*####?\s*([ABCD])',
                            r'####\s*([ABCD])',
                            r'FINAL ANSWER:?\s*([ABCD])',
                            r'ANSWER:?\s*([ABCD])',
                            r'(?:^|\n)\s*([ABCD])\s*(?:\.|$)', 
                            r'\b([ABCD])\)\s*(?:[A-Z]|$)',  
                        ]
                        
                        for pattern in patterns:
                            match = re.search(pattern, model_response_upper)
                            if match:
                                extracted_answer = match.group(1)
                                if extracted_answer == ground_truth:
                                    is_correct = True
                                break
                    
                    if is_correct:
                        correct += 1
                        
                elif dataset_name == "aqua":
                    if f"Final answer: {ground_truth}" in model_response or f"#### {ground_truth}" in model_response:
                        correct += 1
                    letter_to_num = {'A': '1', 'B': '2', 'C': '3', 'D': '4', 'E': '5'}
                    for letter, num in letter_to_num.items():
                        if ground_truth == num and (f"Final answer: {letter}" in model_response or f"#### {letter}" in model_response):
                            correct += 1
                            break
            else:
                print(f"Warning: Invalid ground_truth or model_response type for item: {item}")
        
        accuracies[model] = correct / total if total > 0 else 0
        print(f"{model} on {dataset_name}: {correct}/{total} correct ({accuracies[model]:.3f})")
        
    return accuracies

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

@tracker.track(method_name="openai_inference")
def ask_model(formatted_question, model_name):
    """Track each model inference call using the OpenAI tracker"""
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": formatted_question}
            ],
            temperature=0.1,  # Low temperature for more consistent results
            max_tokens=1000   # Adjust as needed
        )
        
        # The tracker will automatically capture token usage from response.usage
        # Extract the response text from the chat completion
        if response.choices and len(response.choices) > 0:
            return response.choices[0].message.content
        else:
            return "No response generated"
            
    except Exception as e:
        logger.error(f"Exception during OpenAI API call: {str(e)}")
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

def collect_responses(model_name="gpt-3.5-turbo", dataset_name="gsm8k", use_cot=False, use_multi_step=False, num_samples=150, delay=0.5):
    example_indices = None
    if use_cot:
        _, example_indices = load_samples(dataset_name, 3, for_examples=True)
    
    samples, _ = load_samples(dataset_name, num_samples, exclude_indices=example_indices)
    results = []
    
    method_description = "multi-step reasoning" if use_multi_step else ("chain-of-thought" if use_cot else "standard")
    print(f"Processing {len(samples)} samples from {dataset_name} dataset with {model_name} using {method_description}...")
    
    for sample in tqdm(samples, desc=f"Processing {dataset_name} with {model_name}"):
        formatted_question = format_question(sample, dataset_name, use_cot)
        ground_truth = process_ground_truth(sample, dataset_name)
        
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
    parser = argparse.ArgumentParser(description="Evaluate OpenAI models on multiple benchmarks")
    
    # Model selection
    parser.add_argument(
        "--models", 
        nargs="+",
        default=["gpt-3.5-turbo"],
        choices=["gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4"],
        help="Models to evaluate (default: gpt-3.5-turbo)"
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
        help="Test all available models"
    )
    
    parser.add_argument(
        "--all-datasets", 
        action="store_true",
        help="Test all available datasets"
    )
    
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Quick test: gpt-3.5-turbo on gsm8k with 10 samples"
    )
    
    return parser.parse_args()

def get_models_and_datasets(args):
    """Get the final list of models and datasets based on arguments"""
    # All available options
    all_models = ["gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4"]
    all_datasets = ["gsm8k", "gsm-symbolic", "mmlu", "aqua", "svamp"]
    
    # Handle convenience flags
    if args.quick_test:
        return ["gpt-3.5-turbo"], ["gsm8k"], 10
    
    models = all_models if args.all_models else args.models
    datasets = all_datasets if args.all_datasets else args.datasets
    num_samples = args.num_samples
    
    return models, datasets, num_samples

def print_tracking_metrics():
    """Print the current tracking metrics"""
    metrics = tracker.get_metrics()
    print("\n===== OpenAI Responses API Call Tracking Metrics =====")
    print(f"Total API calls: {metrics['total_calls']}")
    print(f"Total prompts processed: {metrics['total_prompts']}")
    print(f"Call counts by method: {metrics['call_counts']}")
    print(f"Average latency: {metrics['avg_latencies'].get('openai_inference', 0):.4f} seconds")
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
    
    # Check if OpenAI API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: Please set the OPENAI_API_KEY environment variable")
        print("You can set it by running: export OPENAI_API_KEY='your-api-key-here'")
        exit(1)
    
    print("=" * 60)
    print("OpenAI Model Evaluation on Multiple Benchmarks")
    print("=" * 60)
    print(f"Models to evaluate: {', '.join(models_to_evaluate)}")
    print(f"Datasets to evaluate: {', '.join(datasets_to_evaluate)}")
    print(f"Using OpenAI's Responses API (introduced March 2025)")
    
    if args.use_multi_step:
        print("Reasoning method: Multi-step reasoning (~3.5 API calls per question)")
    elif args.use_cot:
        print("Reasoning method: Chain-of-thought prompting")
    else:
        print("Reasoning method: Standard prompting")
    
    print(f"Number of samples per dataset: {num_samples}")
    print(f"Delay between API calls: {args.delay}s")
    print("OpenAI API call tracking: enabled")
    print("=" * 60)
    
    start_time = time.time()
    total_evaluations = len(models_to_evaluate) * len(datasets_to_evaluate)
    current_evaluation = 0
    
    # Track results for summary and evaluation
    all_results = {}
    all_evaluation_results = {}
    
    for model_name in models_to_evaluate:
        print(f"\nStarting evaluation with model: {model_name}")
        print("-" * 50)
        
        model_results = {}
        model_dataset_results = {}
        
        for dataset_name in datasets_to_evaluate:
            current_evaluation += 1
            print(f"\nEvaluation {current_evaluation}/{total_evaluations}: {model_name} on {dataset_name}")
            
            try:
                # Collect responses
                results = collect_responses(
                    model_name=model_name,
                    dataset_name=dataset_name, 
                    use_cot=args.use_cot,
                    use_multi_step=args.use_multi_step,
                    num_samples=num_samples,
                    delay=args.delay
                )
                model_results[dataset_name] = len(results)
                
                # Store results for evaluation
                model_dataset_results[dataset_name] = results
                
                print(f"Completed {dataset_name} with {model_name}: {len(results)} samples processed")
                
            except Exception as e:
                print(f"Error evaluating {model_name} on {dataset_name}: {str(e)}")
                print("Continuing to next dataset...")
                model_results[dataset_name] = 0
                model_dataset_results[dataset_name] = []
        
        all_results[model_name] = model_results
        
        # Evaluate accuracy for this model across all datasets
        print(f"\n=== Evaluating Accuracy for {model_name} ===")
        model_accuracies = {}
        for dataset_name in datasets_to_evaluate:
            if dataset_name in model_dataset_results and model_dataset_results[dataset_name]:
                # Format results for evaluation function (expects dict with model as key)
                eval_input = {model_name: model_dataset_results[dataset_name]}
                dataset_accuracies = evaluate_accuracy(eval_input, dataset_name)
                model_accuracies[dataset_name] = dataset_accuracies[model_name]
            else:
                model_accuracies[dataset_name] = 0.0
        
        all_evaluation_results[model_name] = model_accuracies
        
        # Print summary for this model
        print(f"\nSummary for {model_name}:")
        for dataset, count in model_results.items():
            accuracy = model_accuracies.get(dataset, 0.0)
            status = "COMPLETED" if count > 0 else "FAILED"
            print(f"  {status}: {dataset} - {count} samples - Accuracy: {accuracy:.3f} ({accuracy:.1%})")
    
    # Print final comprehensive metrics
    total_time = time.time() - start_time
    print(f"\nEvaluation complete! Total time: {total_time:.2f} seconds")
    
    # Print final tracking metrics
    final_metrics = print_tracking_metrics()
    
    # Print comprehensive summary including accuracies
    print(f"\nCOMPREHENSIVE EVALUATION SUMMARY WITH ACCURACIES")
    print("=" * 70)
    for model_name, model_results in all_results.items():
        print(f"\nModel: {model_name}")
        total_samples = sum(model_results.values())
        print(f"   Total samples processed: {total_samples}")
        
        # Calculate average accuracy across datasets
        accuracies = [acc for acc in all_evaluation_results[model_name].values() if acc > 0]
        avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
        print(f"   Average accuracy across datasets: {avg_accuracy:.3f} ({avg_accuracy:.1%})")
        
        for dataset, count in model_results.items():
            accuracy = all_evaluation_results[model_name].get(dataset, 0.0)
            status = "COMPLETED" if count > 0 else "FAILED"
            print(f"   {status}: {dataset} - {count} samples - Accuracy: {accuracy:.3f} ({accuracy:.1%})")
    
    # Print accuracy comparison table
    print(f"\nACCURACY COMPARISON TABLE")
    print("=" * 70)
    print(f"{'Model':<20} {'Dataset':<15} {'Accuracy':<10} {'Percentage':<10}")
    print("-" * 70)
    for model_name in models_to_evaluate:
        for dataset_name in datasets_to_evaluate:
            accuracy = all_evaluation_results.get(model_name, {}).get(dataset_name, 0.0)
            print(f"{model_name:<20} {dataset_name:<15} {accuracy:<10.3f} {accuracy*100:<10.1f}%")
    
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
    
    # Save comprehensive summary including accuracies
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
        "accuracies": all_evaluation_results,
        "metrics": final_metrics,
        "total_time": total_time
    }
    
    with open("evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Complete summary with accuracies saved to evaluation_summary.json")

# """
# USAGE EXAMPLES:

# 1. Quick test (gpt-3.5-turbo on gsm8k with 10 samples):
#    python script.py --quick-test

# 2. Test with multi-step reasoning (~3.5 API calls per question):
#    python script.py --models gpt-4o --datasets gsm8k --use-multi-step --num-samples 20

# 3. Test specific models on specific datasets:
#    python script.py --models gpt-3.5-turbo gpt-4o --datasets gsm8k mmlu --num-samples 25

# 4. Test all models on all datasets:
#    python script.py --all-models --all-datasets --num-samples 50

# 5. Test with chain-of-thought prompting:
#    python script.py --models gpt-4o --datasets gsm8k --use-cot --num-samples 30

# 6. Test with both multi-step reasoning and chain-of-thought:
#    python script.py --models gpt-4o --datasets gsm8k --use-multi-step --use-cot --num-samples 15

# 7. Custom delay between API calls:
#    python script.py --models gpt-4o --datasets mmlu --delay 1.0 --num-samples 20

# 8. Test gpt-4o on math datasets with multi-step reasoning:
#    python script.py --models gpt-4o --datasets gsm8k gsm-symbolic svamp --use-multi-step --num-samples 25

# Multi-step reasoning process:
# - Step 1: Problem understanding and analysis
# - Step 2: Solution planning and approach  
# - Step 3: Step-by-step execution
# - Step 4: Verification (conditional, for complex problems)

# Available models: gpt-3.5-turbo, gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-4
# Available datasets: gsm8k, gsm-symbolic, mmlu, aqua, svamp

# For help: python script.py --help

# """

# import datasets
# import json
# import random
# from tqdm import tqdm
# import time
# import os
# import logging
# import argparse
# from functools import wraps
# from typing import Dict, List, Optional, Union, Callable, Any
# from collections import Counter, defaultdict
# import openai

# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger("openai_evaluation")

# # Note: This script uses OpenAI's Responses API (introduced March 2025)
# # which combines Chat Completions simplicity with Assistants API capabilities
# # and includes built-in tools like web search, file search, and computer use

# # Embedded OpenAICallTracker class (renamed from VLLMCallTracker)
# class OpenAICallTracker:
#     """
#     A class to track OpenAI API calls, providing metrics on method usage and performance.
#     """
#     def __init__(self, log_to_file: bool = False, log_file: str = "openai_tracking.log"):
#         self.call_counter = Counter()
#         self.method_latency = defaultdict(list)
#         self.prompt_counter = 0
#         self.total_tokens_generated = 0
#         self.total_prompt_tokens = 0
#         self.total_cost = 0.0  # Track estimated cost
#         self.log_to_file = log_to_file
        
#         if log_to_file:
#             file_handler = logging.FileHandler(log_file)
#             file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
#             logger.addHandler(file_handler)
    
#     def track(self, method_name: Optional[str] = None):
#         """
#         Decorator to track calls to OpenAI API methods.
        
#         Args:
#             method_name: Optional override for the method name to track
        
#         Returns:
#             Decorated function that tracks calls
#         """
#         def decorator(func):
#             @wraps(func)
#             def wrapper(*args, **kwargs):
#                 tracked_name = method_name or func.__name__
#                 start_time = time.time()
                
#                 # Log the call
#                 self.call_counter[tracked_name] += 1
#                 self.prompt_counter += 1
#                 logger.info(f"Calling OpenAI method: {tracked_name}")
                
#                 # Execute the function
#                 result = func(*args, **kwargs)
                
#                 # Calculate metrics
#                 latency = time.time() - start_time
#                 self.method_latency[tracked_name].append(latency)
                
#                 # Track token usage from OpenAI response
#                 if hasattr(result, 'usage'):
#                     if hasattr(result.usage, 'completion_tokens'):
#                         self.total_tokens_generated += result.usage.completion_tokens
#                     if hasattr(result.usage, 'prompt_tokens'):
#                         self.total_prompt_tokens += result.usage.prompt_tokens
                    
#                     # Estimate cost (rough estimates for GPT models)
#                     model_name = kwargs.get('model', 'gpt-3.5-turbo')
#                     cost = self._estimate_cost(model_name, result.usage)
#                     self.total_cost += cost
                
#                 logger.info(f"Completed OpenAI method: {tracked_name} (latency: {latency:.4f}s)")
#                 return result
            
#             return wrapper
        
#         return decorator
    
#     def _estimate_cost(self, model_name: str, usage) -> float:
#         """Estimate cost based on model and token usage (rough estimates)"""
#         # Cost per 1K tokens (as of 2024 - these may change)
#         pricing = {
#             'gpt-4': {'input': 0.03, 'output': 0.06},
#             'gpt-4-turbo': {'input': 0.01, 'output': 0.03},
#             'gpt-3.5-turbo': {'input': 0.0015, 'output': 0.002},
#             'gpt-4o': {'input': 0.005, 'output': 0.015},
#             'gpt-4o-mini': {'input': 0.00015, 'output': 0.0006},
#         }
        
#         # Find the best matching model
#         model_key = 'gpt-3.5-turbo'  # default
#         for key in pricing.keys():
#             if key in model_name:
#                 model_key = key
#                 break
        
#         input_cost = (usage.prompt_tokens / 1000) * pricing[model_key]['input']
#         output_cost = (usage.completion_tokens / 1000) * pricing[model_key]['output']
        
#         return input_cost + output_cost
    
#     def get_call_count(self, method_name: Optional[str] = None) -> Union[int, Dict[str, int]]:
#         """
#         Get the number of calls for a specific method or all methods.
        
#         Args:
#             method_name: The method name to get call count for, or None for all methods
            
#         Returns:
#             Call count for the method or dictionary of all method counts
#         """
#         if method_name:
#             return self.call_counter[method_name]
#         return dict(self.call_counter)
    
#     def get_avg_latency(self, method_name: Optional[str] = None) -> Union[float, Dict[str, float]]:
#         """
#         Get the average latency for a specific method or all methods.
        
#         Args:
#             method_name: The method name to get latency for, or None for all methods
            
#         Returns:
#             Average latency for the method or dictionary of all method latencies
#         """
#         if method_name and method_name in self.method_latency:
#             latencies = self.method_latency[method_name]
#             return sum(latencies) / len(latencies) if latencies else 0
        
#         result = {}
#         for method, latencies in self.method_latency.items():
#             result[method] = sum(latencies) / len(latencies) if latencies else 0
#         return result
    
#     def get_metrics(self) -> Dict[str, Any]:
#         """
#         Get comprehensive metrics about OpenAI API usage.
        
#         Returns:
#             Dictionary containing all tracked metrics
#         """
#         return {
#             "call_counts": dict(self.call_counter),
#             "avg_latencies": self.get_avg_latency(),
#             "total_calls": sum(self.call_counter.values()),
#             "total_prompts": self.prompt_counter,
#             "total_tokens_generated": self.total_tokens_generated,
#             "total_prompt_tokens": self.total_prompt_tokens,
#             "estimated_total_cost": self.total_cost
#         }
    
#     def reset(self):
#         """Reset all tracking metrics."""
#         self.call_counter.clear()
#         self.method_latency.clear()
#         self.prompt_counter = 0
#         self.total_tokens_generated = 0
#         self.total_prompt_tokens = 0
#         self.total_cost = 0.0


# # Create tracker instance
# tracker = OpenAICallTracker(log_to_file=True, log_file="openai_evaluation_tracking.log")

# # Initialize OpenAI client for Responses API
# client = openai.OpenAI(
#     api_key=os.environ.get("OPENAI_API_KEY")  # Make sure to set this environment variable
# )

# def load_samples(dataset_name, num_samples=150, for_examples=False, exclude_indices=None):
#     """
#     Load samples from the dataset. If for_examples=True, load a separate set for CoT examples
#     to avoid overlap with test samples. exclude_indices can be used to avoid overlap between
#     examples and test samples when only one split is available.
#     """
#     if dataset_name == "gsm8k":
#         dataset = datasets.load_dataset('gsm8k', 'main', split='train' if for_examples else 'test')
#     elif dataset_name == "gsm-symbolic":
#         dataset = datasets.load_dataset('apple/GSM-Symbolic', 'main')
#         dataset = dataset['test']
#     elif dataset_name == "mmlu":
#         dataset = datasets.load_dataset('cais/mmlu', 'all', split='validation' if for_examples else 'test')
#     elif dataset_name == "aqua":
#         dataset = datasets.load_dataset('aqua_rat', 'raw', split='train' if for_examples else 'test')
#     elif dataset_name == "svamp":
#         dataset = datasets.load_dataset('ChilleD/SVAMP', split='train' if for_examples else 'test')
    
#     total_samples = len(dataset)
#     available_indices = set(range(total_samples))
#     if exclude_indices:
#         available_indices = available_indices - set(exclude_indices)
    
#     num_to_sample = min(num_samples, len(available_indices))
#     random_indices = random.sample(list(available_indices), num_to_sample)
#     samples = [dataset[i] for i in random_indices]
#     return samples, random_indices

# def get_cot_examples(dataset_name, num_examples=3, exclude_indices=None):
#     """Get chain of thought examples from the training/validation split of each dataset."""
#     example_samples, used_indices = load_samples(dataset_name, num_examples, for_examples=True, exclude_indices=exclude_indices)
#     examples = []
    
#     for sample in example_samples:
#         if dataset_name == "gsm8k":
#             example = {
#                 "question": sample['question'],
#                 "reasoning": sample['answer'].split('####')[0].strip(),
#                 "answer": f"Final answer: #### {sample['answer'].split('####')[1].strip()}"
#             }
#         elif dataset_name == "svamp":
#             example = {
#                 "question": f"{sample['Body']} {sample['Question']}",
#                 "reasoning": f"Let me solve this step by step:\n1. {sample['Body']}\n2. {sample['Question']}\n3. The answer is {sample['Answer']}",
#                 "answer": f"Final answer: #### {sample['Answer']}"
#             }
#         elif dataset_name == "mmlu":
#             example = {
#                 "question": f"{sample['question']}\nA) {sample['choices'][0]}\nB) {sample['choices'][1]}\nC) {sample['choices'][2]}\nD) {sample['choices'][3]}",
#                 "reasoning": f"Let's analyze each option:\nA) {sample['choices'][0]}\nB) {sample['choices'][1]}\nC) {sample['choices'][2]}\nD) {sample['choices'][3]}",
#                 "answer": f"Final answer: #### {sample['answer']}"
#             }
#         elif dataset_name == "aqua":
#             options = sample['options']
#             formatted_options = "\n".join([f"{i+1}) {opt}" for i, opt in enumerate(options)])
#             example = {
#                 "question": f"{sample['question']}\n{formatted_options}",
#                 "reasoning": "Let's solve this step by step:\n" + sample['rationale'] if 'rationale' in sample else "Let's analyze each option systematically.",
#                 "answer": f"Final answer: #### {sample['correct']}"
#             }
#         else: 
#             example = {
#                 "question": sample['question'],
#                 "reasoning": "Let's solve this symbolically:\n" + sample['solution'] if 'solution' in sample else "Let's solve step by step.",
#                 "answer": f"Final answer: #### {sample['answer']}"
#             }
#         examples.append(example)
    
#     return examples

# def format_question(sample, dataset_name, use_cot=False):
#     base_prompt = ""
#     if use_cot:
#         examples = get_cot_examples(dataset_name)
#         base_prompt = "Here are some example solutions. Please follow a similar step-by-step reasoning approach:\n\n"
#         for i, example in enumerate(examples, 1):
#             base_prompt += f"Example {i}:\nQuestion: {example['question']}\n{example['reasoning']}\n{example['answer']}\n\n"
#         base_prompt += "Now solve this problem:\n\n"

#     if dataset_name == "gsm8k":
#         return base_prompt + f"Solve this math problem:\n{sample['question']}. Please provide your final answer as a single number marked by 'Final answer: ####'"
    
#     elif dataset_name == "gsm-symbolic":
#         return base_prompt + f"Solve this math problem:\n{sample['question']}"
    
#     elif dataset_name == "mmlu":
#         return base_prompt + f"""Question: {sample['question']}
# A) {sample['choices'][0]}
# B) {sample['choices'][1]}
# C) {sample['choices'][2]}
# D) {sample['choices'][3]}

# Please provide your answer as a single letter (A, B, C, or D) followed by a brief explanation. Please mark it as 'Final answer: #### [answer].'"""
    
#     elif dataset_name == "aqua":
#         options = sample['options']
#         formatted_options = "\n".join([f"{i+1}) {opt}" for i, opt in enumerate(options)])
#         return base_prompt + f"""Question: {sample['question']}
# {formatted_options}

# Please solve this question and provide your final answer as a single letter (A - E) at the end of your response. Please mark it as 'Final answer: #### [answer].'"""
    
#     elif dataset_name == "svamp":
#         return base_prompt + f"""Solve this math word problem and put your final numerical answer at the end of your response. Please mark your final answer as 'Final answer: #### [answer]'.
# Question: {sample['Body']} {sample['Question']}"""

# @tracker.track(method_name="openai_inference")
# def ask_model(formatted_question, model_name):
#     """Track each model inference call using the OpenAI tracker"""
#     try:
#         response = client.responses.create(
#             model=model_name,
#             input=formatted_question,
#             temperature=0.1,  # Low temperature for more consistent results
#             max_tokens=1000   # Adjust as needed
#         )
        
#         # The tracker will automatically capture token usage from response.usage
#         # For Responses API, we need to extract text from the output
#         if hasattr(response, 'output_text'):
#             return response.output_text
#         elif hasattr(response, 'output') and len(response.output) > 0:
#             # Handle case where output is a list of items
#             output_text = ""
#             for item in response.output:
#                 if hasattr(item, 'content') and len(item.content) > 0:
#                     for content_item in item.content:
#                         if hasattr(content_item, 'text'):
#                             output_text += content_item.text
#                         elif content_item.get('type') == 'output_text':
#                             output_text += content_item.get('text', '')
#             return output_text
#         else:
#             return str(response)
        
#     except Exception as e:
#         logger.error(f"Exception during OpenAI API call: {str(e)}")
#         return f"Error: {str(e)}"

# def multi_step_reasoning(question, model_name, dataset_name, delay=0.5):
#     """
#     Implements a multi-step chain-of-thought reasoning process that makes approximately 3.5 API calls per question.
    
#     Steps:
#     1. Problem understanding and analysis
#     2. Solution planning and approach
#     3. Step-by-step execution
#     4. Verification and final answer (conditional, based on complexity)
#     """
    
#     # Step 1: Problem Understanding and Analysis
#     understanding_prompt = f"""
#     Analyze and understand this problem thoroughly. Break down what is being asked and identify the key components:

#     Problem: {question}

#     Please provide:
#     1. What type of problem this is
#     2. What information is given
#     3. What needs to be found or determined
#     4. Any potential challenges or considerations

#     Keep your analysis concise but thorough.
#     """
    
#     understanding = ask_model(understanding_prompt, model_name)
#     time.sleep(delay)
    
#     # Step 2: Solution Planning
#     planning_prompt = f"""
#     Based on this problem analysis:
#     {understanding}

#     Original problem: {question}

#     Create a clear solution plan:
#     1. What approach will you use to solve this?
#     2. What are the specific steps needed?
#     3. What calculations or reasoning will be required?

#     Provide a structured plan without solving yet.
#     """
    
#     plan = ask_model(planning_prompt, model_name)
#     time.sleep(delay)
    
#     # Step 3: Step-by-step Execution
#     execution_prompt = f"""
#     Now execute the solution using this plan:
#     {plan}

#     Original problem: {question}

#     Solve the problem step by step, showing your work clearly. For each step:
#     1. State what you're doing
#     2. Show the calculation or reasoning
#     3. State the result

#     Work through to get your answer.
#     """
    
#     execution = ask_model(execution_prompt, model_name)
#     time.sleep(delay)
    
#     # Step 4: Verification (conditional - approximately 50% of the time based on problem complexity)
#     # We'll do verification for math problems and complex reasoning tasks
#     needs_verification = any(keyword in question.lower() for keyword in [
#         'calculate', 'solve', 'find', 'determine', 'what is', 'how many', 
#         'mathematics', 'equation', 'problem', 'answer'
#     ]) or dataset_name in ['gsm8k', 'gsm-symbolic', 'aqua', 'svamp']
    
#     final_answer = execution
    
#     if needs_verification:
#         verification_prompt = f"""
#         Review and verify this solution:

#         Original problem: {question}
#         Solution: {execution}

#         Check:
#         1. Is the reasoning correct?
#         2. Are the calculations accurate?
#         3. Does the answer make sense?
#         4. Is this the final answer the problem is asking for?

#         Provide the verified final answer clearly marked as 'Final answer: ####'
#         """
        
#         final_answer = ask_model(verification_prompt, model_name)
#         time.sleep(delay)
    
#     # Adjust prompt counter for multi-step reasoning
#     # We made 3 or 4 API calls but it should count as 1 question
#     # Since ask_model increments prompt_counter each time, we need to subtract the extra counts
#     calls_made = 4 if needs_verification else 3
#     tracker.prompt_counter = tracker.prompt_counter - calls_made + 1
    
#     # Combine all reasoning steps for the complete response
#     complete_reasoning = f"""
#     PROBLEM ANALYSIS:
#     {understanding}

#     SOLUTION PLAN:
#     {plan}

#     STEP-BY-STEP EXECUTION:
#     {execution}
#     """
    
#     if needs_verification:
#         complete_reasoning += f"""

#     VERIFICATION AND FINAL ANSWER:
#     {final_answer}
#     """
#     else:
#         complete_reasoning += f"""

#     FINAL ANSWER:
#     {final_answer}
#     """
    
#     return complete_reasoning

# def collect_responses(model_name="gpt-3.5-turbo", dataset_name="gsm8k", use_cot=False, use_multi_step=False, num_samples=150, delay=0.5):
#     example_indices = None
#     if use_cot:
#         _, example_indices = load_samples(dataset_name, 3, for_examples=True)
    
#     samples, _ = load_samples(dataset_name, num_samples, exclude_indices=example_indices)
#     results = []
    
#     method_description = "multi-step reasoning" if use_multi_step else ("chain-of-thought" if use_cot else "standard")
#     print(f"Processing {len(samples)} samples from {dataset_name} dataset with {model_name} using {method_description}...")
    
#     for sample in tqdm(samples, desc=f"Processing {dataset_name} with {model_name}"):
#         formatted_question = format_question(sample, dataset_name, use_cot)
        
#         if dataset_name == "mmlu":
#             num_to_letter = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: '5'}
#             ground_truth = num_to_letter.get(sample['answer'], str(sample['answer']))
#         elif dataset_name == "aqua":
#             ground_truth = str(sample['correct'])
#         elif dataset_name == 'svamp':
#             ground_truth = sample['Answer']
#         else:
#             ground_truth = sample['answer']
        
#         # Use multi-step reasoning if enabled
#         if use_multi_step:
#             model_response = multi_step_reasoning(formatted_question, model_name, dataset_name, delay)
#         else:
#             model_response = ask_model(formatted_question, model_name)
#             time.sleep(delay)
        
#         results.append({
#             'question': formatted_question,
#             'ground_truth': ground_truth,
#             'model_response': model_response
#         })
    
#     model_short_name = model_name.replace('/', '_').replace('-', '_')
#     method_suffix = "_multi_step" if use_multi_step else ("_cot" if use_cot else "")
#     filename = f'{dataset_name}_responses_{model_short_name}{method_suffix}.json'
    
#     with open(filename, 'w') as f:
#         json.dump(results, f, indent=2)
    
#     print(f"Results saved to {filename}")
#     return results

# def parse_arguments():
#     """Parse command line arguments for model and dataset selection"""
#     parser = argparse.ArgumentParser(description="Evaluate OpenAI models on multiple benchmarks")
    
#     # Model selection
#     parser.add_argument(
#         "--models", 
#         nargs="+",
#         default=["gpt-3.5-turbo"],
#         choices=["gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4"],
#         help="Models to evaluate (default: gpt-3.5-turbo)"
#     )
    
#     # Dataset selection
#     parser.add_argument(
#         "--datasets",
#         nargs="+", 
#         default=["gsm8k"],
#         choices=["gsm8k", "gsm-symbolic", "mmlu", "aqua", "svamp"],
#         help="Datasets to evaluate on (default: gsm8k)"
#     )
    
#     # Evaluation parameters
#     parser.add_argument(
#         "--num-samples",
#         type=int,
#         default=50,
#         help="Number of samples per dataset (default: 50)"
#     )
    
#     parser.add_argument(
#         "--use-cot",
#         action="store_true",
#         help="Enable chain-of-thought prompting"
#     )
    
#     parser.add_argument(
#         "--use-multi-step",
#         action="store_true",
#         help="Enable multi-step reasoning (makes ~3.5 API calls per question)"
#     )
    
#     parser.add_argument(
#         "--delay",
#         type=float,
#         default=0.5,
#         help="Delay between API calls in seconds (default: 0.5)"
#     )
    
#     # Convenience flags for common configurations
#     parser.add_argument(
#         "--all-models",
#         action="store_true",
#         help="Test all available models"
#     )
    
#     parser.add_argument(
#         "--all-datasets", 
#         action="store_true",
#         help="Test all available datasets"
#     )
    
#     parser.add_argument(
#         "--quick-test",
#         action="store_true",
#         help="Quick test: gpt-3.5-turbo on gsm8k with 10 samples"
#     )
    
#     return parser.parse_args()

# def get_models_and_datasets(args):
#     """Get the final list of models and datasets based on arguments"""
#     # All available options
#     all_models = ["gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4"]
#     all_datasets = ["gsm8k", "gsm-symbolic", "mmlu", "aqua", "svamp"]
    
#     # Handle convenience flags
#     if args.quick_test:
#         return ["gpt-3.5-turbo"], ["gsm8k"], 10
    
#     models = all_models if args.all_models else args.models
#     datasets = all_datasets if args.all_datasets else args.datasets
#     num_samples = args.num_samples
    
#     return models, datasets, num_samples

# def print_tracking_metrics():
#     """Print the current tracking metrics"""
#     metrics = tracker.get_metrics()
#     print("\n===== OpenAI Responses API Call Tracking Metrics =====")
#     print(f"Total API calls: {metrics['total_calls']}")
#     print(f"Total prompts processed: {metrics['total_prompts']}")
#     print(f"Call counts by method: {metrics['call_counts']}")
#     print(f"Average latency: {metrics['avg_latencies'].get('openai_inference', 0):.4f} seconds")
#     print(f"Total tokens generated: {metrics['total_tokens_generated']}")
#     print(f"Total prompt tokens: {metrics['total_prompt_tokens']}")
#     print(f"Estimated total cost: ${metrics['estimated_total_cost']:.4f}")
#     print("=======================================================")
    
#     # Save metrics to JSON file
#     with open("evaluation_metrics.json", "w") as f:
#         json.dump(metrics, f, indent=2)
    
#     return metrics
#     """Print the current tracking metrics"""
#     metrics = tracker.get_metrics()
#     print("\n===== OpenAI Responses API Call Tracking Metrics =====")
#     print(f"Total API calls: {metrics['total_calls']}")
#     print(f"Total prompts processed: {metrics['total_prompts']}")
#     print(f"Call counts by method: {metrics['call_counts']}")
#     print(f"Average latency: {metrics['avg_latencies'].get('openai_inference', 0):.4f} seconds")
#     print(f"Total tokens generated: {metrics['total_tokens_generated']}")
#     print(f"Total prompt tokens: {metrics['total_prompt_tokens']}")
#     print(f"Estimated total cost: ${metrics['estimated_total_cost']:.4f}")
#     print("=======================================================")
    
#     # Save metrics to JSON file
#     with open("evaluation_metrics.json", "w") as f:
#         json.dump(metrics, f, indent=2)
    
#     return metrics

# if __name__ == "__main__":
#     # Parse command line arguments
#     args = parse_arguments()
    
#     # Get models and datasets to evaluate
#     models_to_evaluate, datasets_to_evaluate, num_samples = get_models_and_datasets(args)
    
#     # Check if OpenAI API key is set
#     if not os.environ.get("OPENAI_API_KEY"):
#         print("Error: Please set the OPENAI_API_KEY environment variable")
#         print("You can set it by running: export OPENAI_API_KEY='your-api-key-here'")
#         exit(1)
    
#     print("=" * 60)
#     print("OpenAI Model Evaluation on Multiple Benchmarks")
#     print("=" * 60)
#     print(f"Models to evaluate: {', '.join(models_to_evaluate)}")
#     print(f"Datasets to evaluate: {', '.join(datasets_to_evaluate)}")
#     print(f"Using OpenAI's Responses API (introduced March 2025)")
    
#     if args.use_multi_step:
#         print("Reasoning method: Multi-step reasoning (~3.5 API calls per question)")
#     elif args.use_cot:
#         print("Reasoning method: Chain-of-thought prompting")
#     else:
#         print("Reasoning method: Standard prompting")
    
#     print(f"Number of samples per dataset: {num_samples}")
#     print(f"Delay between API calls: {args.delay}s")
#     print("OpenAI API call tracking: enabled")
#     print("=" * 60)
    
#     start_time = time.time()
#     total_evaluations = len(models_to_evaluate) * len(datasets_to_evaluate)
#     current_evaluation = 0
    
#     # Track results for summary
#     all_results = {}
    
#     for model_name in models_to_evaluate:
#         print(f"\nStarting evaluation with model: {model_name}")
#         print("-" * 50)
        
#         # Reset tracker for each model (optional - comment out to track across all models)
#         # tracker.reset()
        
#         model_results = {}
        
#         for dataset_name in datasets_to_evaluate:
#             current_evaluation += 1
#             print(f"\nEvaluation {current_evaluation}/{total_evaluations}: {model_name} on {dataset_name}")
            
#             try:
#                 results = collect_responses(
#                     model_name=model_name,
#                     dataset_name=dataset_name, 
#                     use_cot=args.use_cot,
#                     use_multi_step=args.use_multi_step,
#                     num_samples=num_samples,
#                     delay=args.delay
#                 )
#                 model_results[dataset_name] = len(results)
                
#                 # Print intermediate metrics for this model-dataset combination
#                 print(f"Completed {dataset_name} with {model_name}: {len(results)} samples processed")
                
#             except Exception as e:
#                 print(f"Error evaluating {model_name} on {dataset_name}: {str(e)}")
#                 print("Continuing to next dataset...")
#                 model_results[dataset_name] = 0
        
#         all_results[model_name] = model_results
        
#         # Print summary for this model
#         print(f"\nSummary for {model_name}:")
#         for dataset, count in model_results.items():
#             status = "COMPLETED" if count > 0 else "FAILED"
#             print(f"  {status}: {dataset} - {count} samples")
    
#     # Print final comprehensive metrics
#     total_time = time.time() - start_time
#     print(f"\nEvaluation complete! Total time: {total_time:.2f} seconds")
    
#     # Print final tracking metrics
#     final_metrics = print_tracking_metrics()
    
#     # Print comprehensive summary
#     print(f"\nCOMPREHENSIVE EVALUATION SUMMARY")
#     print("=" * 60)
#     for model_name, model_results in all_results.items():
#         print(f"\nModel: {model_name}")
#         total_samples = sum(model_results.values())
#         print(f"   Total samples processed: {total_samples}")
#         for dataset, count in model_results.items():
#             status = "COMPLETED" if count > 0 else "FAILED"
#             print(f"   {status}: {dataset} - {count} samples")
    
#     # Calculate additional statistics
#     if final_metrics['total_calls'] > 0:
#         tokens_per_call = final_metrics['total_tokens_generated'] / final_metrics['total_calls']
#         calls_per_question = final_metrics['total_calls'] / final_metrics['total_prompts'] if final_metrics['total_prompts'] > 0 else 0
        
#         print(f"\nPerformance Metrics:")
#         print(f"   Total questions processed: {final_metrics['total_prompts']}")
#         print(f"   Total API calls made: {final_metrics['total_calls']}")
#         print(f"   Average API calls per question: {calls_per_question:.1f}")
#         print(f"   Average tokens generated per call: {tokens_per_call:.2f}")
        
#         if total_time > 0:
#             calls_per_second = final_metrics['total_calls'] / total_time
#             tokens_per_second = final_metrics['total_tokens_generated'] / total_time
#             questions_per_second = final_metrics['total_prompts'] / total_time
#             print(f"   Questions processed per second: {questions_per_second:.2f}")
#             print(f"   API calls per second: {calls_per_second:.2f}")
#             print(f"   Tokens generated per second: {tokens_per_second:.2f}")
    
#     print(f"\nEstimated total cost: ${final_metrics['estimated_total_cost']:.4f}")
#     print(f"All results saved to individual JSON files")
#     print(f"Metrics saved to evaluation_metrics.json")
    
#     # Save comprehensive summary
#     summary = {
#         "evaluation_config": {
#             "models": models_to_evaluate,
#             "datasets": datasets_to_evaluate,
#             "num_samples": num_samples,
#             "use_cot": args.use_cot,
#             "use_multi_step": args.use_multi_step,
#             "delay": args.delay
#         },
#         "results": all_results,
#         "metrics": final_metrics,
#         "total_time": total_time
#     }
    
#     with open("evaluation_summary.json", "w") as f:
#         json.dump(summary, f, indent=2)
    
#     print(f"Complete summary saved to evaluation_summary.json")