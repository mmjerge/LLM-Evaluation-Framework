import os
import json
import time
from tqdm import tqdm
from anthropic import Anthropic
from datasets import load_dataset
import random

client = Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
)

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
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            
            for line in lines:
                try:
                    item = json.loads(line)
                    if 'question' in item and 'answer' in item:
                        dataset.append(item)
                except json.JSONDecodeError as e:
                    continue  
                    
        print(f"Successfully loaded {len(dataset)} questions from GSM8K")
        
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

def evaluate_model(model_name, dataset, dataset_type="gsm8k"):
    """
    Evaluate the model on the specified dataset and save results to a JSON file.
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
        else:  
            question = sample['question']
            correct_answer = sample['answer']
            original_data = {
                'original_id': sample['original_id'],
                'original_question': sample['original_question'],
                'original_answer': sample['original_answer']
            }

        try:
            prompt = f"""Please solve this math problem and provide your answer in the format "Final answer: #### [your final answer]". You must add the hashtags in the final answer block.

Here's the question:
{question}"""

            message = client.messages.create(
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=model_name,
            )
            response = message.content[0].text

            import re
            final_answer_match = re.search(r"Final answer: #### \[(.*?)\]", response)
            extracted_answer = final_answer_match.group(1) if final_answer_match else None

            result = {
                "question": question,
                "model_response": response,
                "extracted_answer": extracted_answer,
                "correct_answer": correct_answer
            }

            if dataset_type == "gsm_symbolic":
                result.update(original_data)

            model_results.append(result)

            if extracted_answer and str(correct_answer) in extracted_answer:
                correct += 1

            time.sleep(1)  

        except Exception as e:
            print(f"Error occurred during evaluation: {e}")
            continue

    accuracy = correct / total if total > 0 else 0

    output_filename = f"{model_name}_{dataset_type}_results.json"
    with open(output_filename, 'w') as outfile:
        json.dump(model_results, outfile, indent=4)

    return accuracy

def main(dataset_choice="both"):
    """
    Run evaluation with specified dataset choice.
    dataset_choice can be "both", "gsm8k", or "gsm_symbolic"
    """
    print(f"\nRunning evaluation on {dataset_choice} dataset(s)...")
    
    gsm8k_dataset = None
    gsm_symbolic_dataset = None
    
    if dataset_choice in ["both", "gsm8k"]:
        gsm8k_path = "/scratch/mj6ux/Projects/llm_reliability_framework/paper_evaluated_methods/no_ensemble/datasets/gsm8k/test.jsonl"
        gsm8k_dataset = load_gsm8k_dataset(gsm8k_path, sample_size=150)
        if gsm8k_dataset:
            print(f"Sampled {len(gsm8k_dataset)} questions from GSM8K")
    
    if dataset_choice in ["both", "gsm_symbolic"]:
        gsm_symbolic_dataset = load_gsm_symbolic_dataset(sample_size=150)
        if gsm_symbolic_dataset:
            print(f"Sampled {len(gsm_symbolic_dataset)} questions from GSM-Symbolic")

    if dataset_choice == "both" and gsm8k_dataset is None and gsm_symbolic_dataset is None:
        print("Failed to load both datasets. Exiting.")
        return
    elif dataset_choice == "gsm8k" and gsm8k_dataset is None:
        print("Failed to load GSM8K dataset. Exiting.")
        return
    elif dataset_choice == "gsm_symbolic" and gsm_symbolic_dataset is None:
        print("Failed to load GSM-Symbolic dataset. Exiting.")
        return

    models_to_evaluate = [
        "claude-3-5-sonnet-20240620"
    ]

    results = {}
    
    for model in models_to_evaluate:
        if gsm8k_dataset and dataset_choice in ["both", "gsm8k"]:
            gsm8k_accuracy = evaluate_model(model, gsm8k_dataset, dataset_type="gsm8k")
            results[f"{model}_gsm8k"] = gsm8k_accuracy
        
        if gsm_symbolic_dataset and dataset_choice in ["both", "gsm_symbolic"]:
            gsm_symbolic_accuracy = evaluate_model(model, gsm_symbolic_dataset, dataset_type="gsm_symbolic")
            results[f"{model}_gsm_symbolic"] = gsm_symbolic_accuracy

    print("\nFinal Results:")
    for model_dataset, accuracy in results.items():
        print(f"{model_dataset}: {accuracy:.2%}")

if __name__ == "__main__":
    main(dataset_choice="gsm_symbolic")