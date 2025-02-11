import os
import json
import time
from tqdm import tqdm
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from datasets import load_dataset
import random

# Set your Mistral API key
api_key = os.environ["MISTRAL_API_KEY"]
client = MistralClient(api_key=api_key)

# Define the 5-shot prompt
FIVE_SHOT_PROMPT = """Let me show you some example math problems and their answers:

Problem 1: Janet has 3 brothers and 2 sisters. Each of her siblings has 2 pets. How many pets do Janet's siblings have in total?
Answer: 10

Problem 2: A restaurant sold 145 pizzas last week. This week they sold 15% more pizzas than last week. How many pizzas did they sell this week?
Answer: 167

Problem 3: Tom buys a notebook for $4.50 and three pens for $1.25 each. If he pays with a $10 bill, how much change will he receive?
Answer: 1.75

Problem 4: A train travels 120 miles in 2 hours. If it maintains the same speed, how many miles will it travel in 5 hours?
Answer: 300

Problem 5: Sarah has twice as many marbles as John. John has 5 fewer marbles than Amy. If Amy has 15 marbles, how many marbles does Sarah have?
Answer: 20

Now, please solve this new problem:
"""

def load_gsm8k_dataset(file_path, sample_size=150):
    """
    Load the GSM8K dataset from a JSONL file and randomly sample questions.
    Each line is a JSON object with 'question' and 'answer' fields.
    """
    dataset = []

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cannot find file at path: {file_path}")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # Split by newline and filter out empty lines
            lines = [line.strip() for line in content.split('\n') if line.strip()]

            for line in lines:
                try:
                    # Parse each line as a JSON object
                    item = json.loads(line)
                    if 'question' in item and 'answer' in item:
                        dataset.append(item)
                except json.JSONDecodeError as e:
                    continue  # Skip invalid lines

        print(f"Successfully loaded {len(dataset)} questions from GSM8K")

        # Randomly sample if we have more than we need
        if len(dataset) > sample_size:
            dataset = random.sample(dataset, sample_size)

        return dataset

    except Exception as e:
        print(f"Error reading GSM8K file: {str(e)}")
        return None

def load_gsm_symbolic_dataset(sample_size=150):
    """
    Load the GSM-Symbolic dataset and randomly sample questions.
    """
    try:
        print("Loading GSM-Symbolic dataset...")
        dataset = load_dataset("apple/GSM-Symbolic", "main")

        print(f"Available splits in GSM-Symbolic: {list(dataset.keys())}")

        split_data = dataset["main" if "main" in dataset else list(dataset.keys())[0]]

        shuffled_dataset = split_data.shuffle(seed=42)
        sampled_dataset = shuffled_dataset.select(range(min(sample_size, len(shuffled_dataset))))

        print(f"Successfully loaded {len(sampled_dataset)} questions from GSM-Symbolic")
        return sampled_dataset

    except Exception as e:
        print(f"Error loading GSM-Symbolic dataset: {str(e)}")
        return None

def evaluate_model(model_name, dataset, dataset_type="gsm8k", use_five_shot=False):
    """
    Evaluate the model on the specified dataset and save results to a JSON file.
    Added parameter use_five_shot to toggle between prompt styles.
    """
    if dataset is None or len(dataset) == 0:
        print(f"No data available for {dataset_type}")
        return 0

    correct = 0
    total = len(dataset)
    model_results = []

    dataset_iter = dataset if dataset_type == "gsm8k" else dataset

    for sample in tqdm(dataset_iter, desc=f"Evaluating {model_name} on {dataset_type}"):
        if dataset_type == "gsm8k":
            question = sample['question']
            correct_answer = sample['answer']
        else:  # gsm_symbolic
            question = sample['question']
            correct_answer = sample['answer']
            original_data = {
                'original_id': sample['original_id'],
                'original_question': sample['original_question'],
                'original_answer': sample['original_answer']
            }

        try:
            if use_five_shot:
                prompt = f"{FIVE_SHOT_PROMPT}\n{question}"
            else:
                prompt = f"""Please solve this math problem.
        
                <HOW_TO_RESPOND> Respond and provide your answer in the format "Final answer: #### [your final answer]", where your final answer is a singular numerical response. 
                
                For example, respond with only: "Final answer: #### 53", "Final Answer: #### 0", "Final answer: #### -1203923" </HOW_TO_RESPOND>
                
                Here's the question:

                {question}"""

            message = ChatMessage(role="user", content=prompt)
            
            response = client.chat(
                model=model_name,
                messages=[message]
            )

            model_response = response.choices[0].message.content

            result = {
                "question": question,
                "model_response": model_response,
                "correct_answer": correct_answer
            }

            if dataset_type == "gsm_symbolic":
                result.update(original_data)

            model_results.append(result)

            if str(correct_answer) in model_response:
                correct += 1

            time.sleep(1)

        except Exception as e:
            print(f"Error occurred during evaluation: {str(e)}")
            continue

    accuracy = correct / total if total > 0 else 0

    prompt_type = "five_shot" if use_five_shot else "standard"
    output_filename = f"{model_name}_{dataset_type}_{prompt_type}_results.json"
    with open(output_filename, 'w') as outfile:
        json.dump(model_results, outfile, indent=4)

    return accuracy

import argparse

def main():
    parser = argparse.ArgumentParser(description='Evaluate language models on math problems')
    parser.add_argument('--five-shot', action='store_true', help='Use 5-shot prompt')
    parser.add_argument('--dataset', type=str, choices=['gsm8k', 'gsm_symbolic', 'both'], default='both',
                      help='Which dataset to evaluate on')
    args = parser.parse_args()
    
    gsm8k_path = "/scratch/mj6ux/Projects/llm_reliability_framework/paper_evaluated_methods/unaided/datasets/gsm8k/test.jsonl"

    print("\nLoading datasets...")
    gsm8k_dataset = load_gsm8k_dataset(gsm8k_path, sample_size=150)
    gsm_symbolic_dataset = load_gsm_symbolic_dataset(sample_size=150)

    if gsm8k_dataset is None and gsm_symbolic_dataset is None:
        print("Failed to load both datasets. Exiting.")
        return

    if gsm8k_dataset:
        print(f"Sampled {len(gsm8k_dataset)} questions from GSM8K")
    if gsm_symbolic_dataset:
        print(f"Sampled {len(gsm_symbolic_dataset)} questions from GSM-Symbolic")

    models_to_evaluate = [
        "open-mixtral-8x22b",
    ]

    results = {}

    # Evaluate on selected datasets
    for model in models_to_evaluate:
        if (args.dataset in ['gsm8k', 'both']) and gsm8k_dataset:
            gsm8k_accuracy = evaluate_model(model, gsm8k_dataset, dataset_type="gsm8k", use_five_shot=args.five_shot)
            results[f"{model}_gsm8k"] = gsm8k_accuracy

        if (args.dataset in ['gsm_symbolic', 'both']) and gsm_symbolic_dataset:
            gsm_symbolic_accuracy = evaluate_model(model, gsm_symbolic_dataset, dataset_type="gsm_symbolic", use_five_shot=args.five_shot)
            results[f"{model}_gsm_symbolic"] = gsm_symbolic_accuracy

    print("\nFinal Results:")
    prompt_type = "5-shot prompt" if use_five_shot else "standard prompt"
    print(f"\nResults using {prompt_type}:")
    for model_dataset, accuracy in results.items():
        print(f"{model_dataset}: {accuracy:.2%}")

if __name__ == "__main__":
    main()
