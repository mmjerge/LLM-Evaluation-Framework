import os
import re
import json
from multiprocessing.pool import ThreadPool
import pandas as pd
from tqdm import tqdm
import anthropic
from datasets import load_dataset
import random
import time

# Set random seed for reproducibility
random.seed(42)

# Initialize Anthropic client
client = anthropic.Anthropic()  # Make sure to set ANTHROPIC_API_KEY environment variable

# Load the privacy_policy_qa dataset from LegalBench
print("Loading privacy_policy_qa dataset from LegalBench...")
dataset = load_dataset("nguha/legalbench", "privacy_policy_qa")
test_data = dataset["test"]

# Convert the dataset to a pandas DataFrame for easier handling
test_df = pd.DataFrame(test_data)
print(f"Loaded {len(test_df)} test samples in total")

# Select random subset of 150 samples
SAMPLE_SIZE = 150
test_df = test_df.sample(n=SAMPLE_SIZE, random_state=42)
print(f"Selected random subset of {len(test_df)} samples for evaluation")

# Define Anthropic models to evaluate
MODELS = {
    "claude-3-5-sonnet-20240620": "claude-3-5-sonnet-20240620"
}

def get_response(system_prompt, user_prompt, model="claude-3-5-sonnet-20240620", temp=0.5):
    """Get response from a Claude model through the Anthropic API."""
    try:
        time.sleep(0.5)
        
        response = client.messages.create(
            model=model,
            system=system_prompt,
            max_tokens=1000,
            temperature=temp,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.content[0].text
    except Exception as e:
        print(f"Error getting response from model {model}: {e}")
        if "rate" in str(e).lower():
            print("Rate limit hit, waiting 60 seconds and retrying...")
            time.sleep(60)
            try:
                response = client.messages.create(
                    model=model,
                    system=system_prompt,
                    max_tokens=1000,
                    temperature=temp,
                    messages=[
                        {"role": "user", "content": user_prompt}
                    ]
                )
                return response.content[0].text
            except Exception as retry_error:
                print(f"Retry failed: {retry_error}")
                return "NA"
        return "NA"

def self_consistency_solver(question, text, n_experts, model):
    """
    Uses the self-consistency approach with multiple experts to determine relevance.
    """
    system_prompt = f'''
    You are helping to evaluate if a privacy policy text contains an answer to a user's question.
    Imagine {n_experts} completely independent experts who reason differently are analyzing this task.
    
    Follow these steps in your response:
    1. For each expert, provide their step-by-step reasoning and their determination of whether the privacy policy text addresses the question.
    2. Determine the final answer by majority vote among the experts.
    3. Return the final answer, prefixed by 'Final answer:' followed by either 'Relevant' or 'Irrelevant'.
    
    'Relevant' means the privacy policy text contains information that directly addresses the user's question.
    'Irrelevant' means the privacy policy text does not contain information that addresses the user's question.
    '''
    
    user_prompt = f'''
    Question: {question}
    
    Privacy Policy Text: {text}
    
    Task: Determine if the privacy policy text contains an answer to the question. 
    Classify as "Relevant" or "Irrelevant".
    '''
    
    return get_response(system_prompt=system_prompt, user_prompt=user_prompt, model=model, temp=0.5)

def identify_final_answer(solution):
    """
    Extracts the final answer from the expert solution.
    """
    try:
        if 'Final answer:' in solution:
            answer = solution.split('Final answer:')[1].strip()
            if 'Relevant' in answer:
                return 'Relevant'
            elif 'Irrelevant' in answer:
                return 'Irrelevant'
            else:
                return 'NA'
        else:
            if 'Relevant' in solution and not 'Irrelevant' in solution:
                return 'Relevant'
            elif 'Irrelevant' in solution and not 'Relevant' in solution:
                return 'Irrelevant'
            last_sentences = solution.split('.')[-3:]
            for sentence in reversed(last_sentences):
                if 'Relevant' in sentence and not 'Irrelevant' in sentence:
                    return 'Relevant'
                elif 'Irrelevant' in sentence and not 'Relevant' in sentence:
                    return 'Irrelevant'
            return 'NA'
    except Exception as e:
        print(f"Error in identify_final_answer: {e}")
        return 'NA'

def get_best_answer(options):
    """Get the most frequent answer from a list of options."""
    answer_count = [[x, options.count(x)] for x in set(options) if x not in ['', 'NA']]
    answer_count_sorted = sorted(answer_count, key=lambda x: x[1], reverse=True)
    if len(answer_count_sorted) > 0:
        return answer_count_sorted[0][0]
    else:
        return 'NA'

def multi_step_solver(question, text, true_answer, n_experts, n_attempts, model):
    """
    Run multiple attempts of self-consistency with multiple experts.
    """
    pool = ThreadPool(min(n_attempts, 4))  
    attempts = pool.starmap(
        self_consistency_solver, 
        [(question, text, n_experts, model) for _ in range(n_attempts)]
    )
    pool.close()
    pool.join()
    
    answers_parsed = [identify_final_answer(attempt) for attempt in attempts]
    
    best_answer = get_best_answer(answers_parsed)
    
    is_correct = best_answer == true_answer
    
    return {
        "best_answer": best_answer,
        "is_correct": is_correct,
        "attempts": attempts,
        "answers_parsed": answers_parsed
    }

def evaluate_privacy_policy_qa():
    """Main evaluation function."""
    results = {}
    for model_key in MODELS:
        results[model_key] = {}
    
    output_file = 'anthropic_privacy_policy_qa_results.json'
    
    try:
        with open(output_file, 'r') as f:
            results = json.load(f)
            print(f"Loaded existing results")
    except:
        print("No existing results found, starting fresh")
    
    models = list(MODELS.keys())
    expert_attempt_configs = [
        (1, 1),  # (n_experts, n_attempts)
        (3, 1),
        (5, 1),
        (1, 3),
        (3, 3)
    ]
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Processing examples"):
        question = row["question"]
        text = row["text"]
        true_answer = row["answer"]
        index = str(row["index"])
        
        for model_key in models:
            model_name = MODELS[model_key]
            
            if model_key not in results:
                results[model_key] = {}

            for n_experts, n_attempts in expert_attempt_configs:
                config_key = f"experts_{n_experts}_attempts_{n_attempts}"
                
                if config_key not in results[model_key]:
                    results[model_key][config_key] = {
                        "examples": {},
                        "metrics": {
                            "total": 0,
                            "correct": 0,
                            "accuracy": 0
                        }
                    }
                
                if index in results[model_key][config_key]["examples"]:
                    print(f"Skipping already processed example {index} with {model_key}, {n_experts} experts, {n_attempts} attempts")
                    continue
                
                print(f"Evaluating example {index} with {model_key}, {n_experts} experts, {n_attempts} attempts")
                
                eval_results = multi_step_solver(
                    question=question,
                    text=text,
                    true_answer=true_answer,
                    n_experts=n_experts,
                    n_attempts=n_attempts,
                    model=model_name
                )
                
                results[model_key][config_key]["examples"][index] = {
                    "question": question,
                    "text": text,
                    "true_answer": true_answer,
                    "predicted_answer": eval_results["best_answer"],
                    "is_correct": eval_results["is_correct"],
                    "answers_parsed": eval_results["answers_parsed"]
                }
                
                results[model_key][config_key]["metrics"]["total"] += 1
                if eval_results["is_correct"]:
                    results[model_key][config_key]["metrics"]["correct"] += 1
                results[model_key][config_key]["metrics"]["accuracy"] = (
                    results[model_key][config_key]["metrics"]["correct"] / 
                    results[model_key][config_key]["metrics"]["total"]
                )
                
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)
    
    print("\n=== FINAL RESULTS ===")
    for model_key in models:
        print(f"\nResults for {model_key}:")
        for config_key in results[model_key]:
            metrics = results[model_key][config_key]["metrics"]
            print(f"  {config_key}: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})")
    
    return results

if __name__ == "__main__":
    evaluate_privacy_policy_qa()