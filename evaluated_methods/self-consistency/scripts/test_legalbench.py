import os
import re
import json
from multiprocessing.pool import ThreadPool
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from datasets import load_dataset
import random

# Set random seed for reproducibility
random.seed(42)

# Initialize OpenAI client
client = OpenAI()  # Make sure to set OPENAI_API_KEY environment variable

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

def get_response(msg, model="gpt-3.5-turbo", temp=0):
    """Get response from a model through the OpenAI API."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=msg,
            temperature=temp
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error getting response from model {model}: {e}")
        return "NA"

def self_consistency_solver(question, text, n_experts, model):
    """
    Uses the self-consistency approach with multiple experts to determine relevance.
    """
    instructions = f'''
    Imagine {n_experts} completely independent experts who reason differently
    are determining if a privacy policy text contains an answer to a user's question.
    
    Step 1. For each of the experts, give their step-by-step reasoning and determine if the 
    privacy policy text contains an answer to the question.
    
    Step 2. Determine the final answer by majority vote.
    
    Step 3. Return the final answer, obtained by majority vote, prefixed by 'Final answer:' 
    followed by either 'Relevant' or 'Irrelevant'.
    
    'Relevant' means the privacy policy text contains information that directly addresses the user's question.
    'Irrelevant' means the privacy policy text does not contain information that addresses the user's question.
    '''
    
    user_content = f'''
    Question: {question}
    
    Privacy Policy Text: {text}
    
    Task: Determine if the privacy policy text contains an answer to the question. 
    Classify as "Relevant" or "Irrelevant".
    '''
    
    msg = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": user_content}
    ]
    
    return get_response(msg=msg, model=model, temp=0.5)

def identify_final_answer(solution):
    """
    Extracts the final answer from the expert solution.
    """
    try:
        # Attempt to split the solution by "Final answer:"
        if 'Final answer:' in solution:
            answer = solution.split('Final answer:')[1].strip()
            # Check if the answer contains "Relevant" or "Irrelevant"
            if 'Relevant' in answer:
                return 'Relevant'
            elif 'Irrelevant' in answer:
                return 'Irrelevant'
            else:
                return 'NA'
        else:
            # If 'Final answer:' is not found, try to detect "Relevant" or "Irrelevant"
            if 'Relevant' in solution:
                return 'Relevant'
            elif 'Irrelevant' in solution:
                return 'Irrelevant'
            else:
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
    pool = ThreadPool(n_attempts)
    
    # Run multiple attempts of self-consistency
    attempts = pool.starmap(
        self_consistency_solver, 
        [(question, text, n_experts, model) for _ in range(n_attempts)]
    )
    
    # Parse the final answers from each attempt
    answers_parsed = [identify_final_answer(attempt) for attempt in attempts]
    
    # Get the best answer by majority voting across attempts
    best_answer = get_best_answer(answers_parsed)
    
    # Calculate accuracy
    is_correct = best_answer == true_answer
    
    return {
        "best_answer": best_answer,
        "is_correct": is_correct,
        "attempts": attempts,
        "answers_parsed": answers_parsed
    }

def evaluate_privacy_policy_qa():
    """Main evaluation function."""
    # Results dictionary
    results = {
        "gpt-3.5-turbo": {},
        "gpt-4o": {}
    }
    
    # Results file
    output_file = 'privacy_policy_qa_results.json'
    
    # Check if we already have some results
    try:
        with open(output_file, 'r') as f:
            results = json.load(f)
            print(f"Loaded existing results")
    except:
        print("No existing results found, starting fresh")
    
    # Define expert/attempt configurations
    models = ["gpt-3.5-turbo", "gpt-4o"]
    expert_attempt_configs = [
        (1, 1),  # (n_experts, n_attempts)
        (3, 1),
        (5, 1),
        (1, 3),
        (3, 3)
    ]
    
    # Process each example in the dataset
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Processing examples"):
        question = row["question"]
        text = row["text"]
        true_answer = row["answer"]
        index = str(row["index"])
        
        # Evaluate with each model
        for model in models:
            if model not in results:
                results[model] = {}

            # Evaluate with each configuration
            for n_experts, n_attempts in expert_attempt_configs:
                config_key = f"experts_{n_experts}_attempts_{n_attempts}"
                
                # Initialize configuration if not exists
                if config_key not in results[model]:
                    results[model][config_key] = {
                        "examples": {},
                        "metrics": {
                            "total": 0,
                            "correct": 0,
                            "accuracy": 0
                        }
                    }
                
                # Skip if already processed
                if index in results[model][config_key]["examples"]:
                    print(f"Skipping already processed example {index} with {model}, {n_experts} experts, {n_attempts} attempts")
                    continue
                
                print(f"Evaluating example {index} with {model}, {n_experts} experts, {n_attempts} attempts")
                
                # Run the solver
                eval_results = multi_step_solver(
                    question=question,
                    text=text,
                    true_answer=true_answer,
                    n_experts=n_experts,
                    n_attempts=n_attempts,
                    model=model
                )
                
                # Update the results
                results[model][config_key]["examples"][index] = {
                    "question": question,
                    "text": text,
                    "true_answer": true_answer,
                    "predicted_answer": eval_results["best_answer"],
                    "is_correct": eval_results["is_correct"],
                    "answers_parsed": eval_results["answers_parsed"]
                }
                
                # Update metrics
                results[model][config_key]["metrics"]["total"] += 1
                if eval_results["is_correct"]:
                    results[model][config_key]["metrics"]["correct"] += 1
                results[model][config_key]["metrics"]["accuracy"] = (
                    results[model][config_key]["metrics"]["correct"] / 
                    results[model][config_key]["metrics"]["total"]
                )
                
                # Save results after each example
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)
    
    # Print final metrics summary
    print("\n=== FINAL RESULTS ===")
    for model in models:
        print(f"\nResults for {model}:")
        for config_key in results[model]:
            metrics = results[model][config_key]["metrics"]
            print(f"  {config_key}: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})")
    
    return results

if __name__ == "__main__":
    evaluate_privacy_policy_qa()