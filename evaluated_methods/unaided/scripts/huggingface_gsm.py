import re
import json
import os
import time
from tqdm import tqdm
import random
from datasets import load_dataset
import argparse

from mistral_inference.transformer import Transformer
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_inference.generate import generate

def check_model_dir(model_dir):
    """
    Check if model directory exists and has required files.
    Returns True if successful.
    """
    if not os.path.exists(model_dir):
        print(f"\nModel directory {model_dir} doesn't exist.")
        return False
        
    required_files = {
        "params.json": "params.json",
        "tokenizer": "tokenizer.model.v3",
        "model": "consolidated.safetensors"
    }
    
    missing_files = []
    for key, filename in required_files.items():
        if not os.path.exists(os.path.join(model_dir, filename)):
            missing_files.append(f"{key} ({filename})")
    
    if missing_files:
        print(f"\nMissing required files in {model_dir}:")
        for file in missing_files:
            print(f"- {file}")
        print("\nPlease ensure all required files are present.")
        return False
        
    return True

def initialize_model(model_dir):
    """
    Initialize the Mistral model and tokenizer.
    """
    try:
        tokenizer = MistralTokenizer.from_file(os.path.join(model_dir, "tokenizer.model.v3"))
        model = Transformer.from_folder(model_dir)
        return model, tokenizer
    except Exception as e:
        print(f"Error initializing model: {e}")
        return None, None

def load_gsm8k_dataset(file_path, sample_size=150):
    """
    Load the GSM8K dataset from a JSONL file and randomly sample questions.
    """
    dataset = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            
            for line in lines:
                try:
                    item = json.loads(line)
                    if 'question' in item and 'answer' in item:
                        dataset.append(item)
                except json.JSONDecodeError:
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
        
        split_data = dataset["main" if "main" in dataset else list(dataset.keys())[0]]
        shuffled_dataset = split_data.shuffle(seed=42)
        sampled_dataset = shuffled_dataset.select(range(min(sample_size, len(shuffled_dataset))))
        
        print(f"Successfully loaded {len(sampled_dataset)} questions from GSM-Symbolic")
        return sampled_dataset
        
    except Exception as e:
        print(f"Error loading GSM-Symbolic dataset: {str(e)}")
        return None

def generate_response(model, tokenizer, prompt, max_tokens=512):
    """
    Generate a response using the Mistral model.
    """
    try:
        completion_request = ChatCompletionRequest(messages=[UserMessage(content=prompt)])
        tokens = tokenizer.encode_chat_completion(completion_request).tokens
        
        out_tokens, _ = generate(
            [tokens], 
            model, 
            max_tokens=max_tokens,
            temperature=0.7,
            eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id
        )
        
        result = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])
        return result.strip()
            
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        print(f"Prompt that caused the error: {prompt[:100]}...")
        return None

def evaluate_model(model, tokenizer, dataset, dataset_type="gsm8k"):
    """
    Evaluate the model on the specified dataset and save results to a JSON file.
    """
    if dataset is None or len(dataset) == 0:
        print(f"No data available for {dataset_type}")
        return 0
        
    correct = 0
    total = len(dataset)
    model_results = []
    
    successful_generations = 0
    failed_generations = 0

    dataset_iter = dataset if dataset_type == "gsm8k" else dataset

    for sample in tqdm(dataset_iter, desc=f"Evaluating model on {dataset_type}"):
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

        prompt = f"""Solve this math word problem step by step. Show your work clearly. Your solution must end with "Final answer: #### [number]" where [number] is your final numerical answer

Question: {question}

Solution:"""

        try:
            response = generate_response(model, tokenizer, prompt)
            if response is None or not response.strip():
                print(f"Empty response generated for question: {question[:50]}...")
                failed_generations += 1
                continue
            successful_generations += 1

            result = {
                "question": question,
                "model_response": response,
                "correct_answer": correct_answer
            }

            if dataset_type == "gsm_symbolic":
                result.update(original_data)

            model_results.append(result)

            final_answer_match = re.search(r"Final answer: #### (.+)", response)
            if final_answer_match:
                extracted_answer = final_answer_match.group(1).strip()
                if str(correct_answer) in extracted_answer:
                    correct += 1

        except Exception as e:
            print(f"Error occurred during evaluation: {e}")
            continue

    accuracy = correct / total if total > 0 else 0

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_filename = f"mixtral_8x22b_{dataset_type}_results_{timestamp}.json"
    with open(output_filename, 'w') as outfile:
        json.dump(model_results, outfile, indent=4)

    print(f"\nGeneration Statistics:")
    print(f"Successful generations: {successful_generations}/{total} ({successful_generations/total*100:.2f}%)")
    print(f"Failed generations: {failed_generations}/{total} ({failed_generations/total*100:.2f}%)")
    
    return accuracy

def main(dataset_type="both", username=None, download_only=False):
    """
    Run the evaluation on specified dataset(s).
    """
    if username is None:
        raise ValueError("Please provide your username for the scratch directory path")

    model_dir = f"/scratch/{username}/hf_models/mistralai"
    
    os.makedirs(model_dir, exist_ok=True)
    
    if not check_model_dir(model_dir):
        return

    if download_only:
        print("\nDownload-only mode specified. Exiting without evaluation.")
        required_files = ["config.json", "tokenizer.model", "consolidated.00.pth"]
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_dir, f))]
        if missing_files:
            print(f"Warning: Some required files are missing: {missing_files}")
        else:
            print("All required model files are present.")
        return

    print(f"\nInitializing Mistral model...")
    model, tokenizer = initialize_model(model_dir)
    
    if model is None or tokenizer is None:
        print("Failed to initialize model. Exiting.")
        return
        
    print("\nLoading datasets...")
    gsm8k_dataset = None
    gsm_symbolic_dataset = None
    
    if dataset_type in ["gsm8k", "both"]:
        gsm8k_path = "/scratch/mj6ux/Projects/llm_reliability_framework/paper_evaluated_methods/no_ensemble/datasets/gsm8k/test.jsonl"
        gsm8k_dataset = load_gsm8k_dataset(gsm8k_path, sample_size=150)
        
    if dataset_type in ["gsm_symbolic", "both"]:
        gsm_symbolic_dataset = load_gsm_symbolic_dataset(sample_size=150)

    if gsm8k_dataset is None and gsm_symbolic_dataset is None:
        print("Failed to load datasets. Exiting.")
        return

    results = {}
    
    if gsm8k_dataset:
        print(f"Starting GSM8K evaluation...")
        gsm8k_accuracy = evaluate_model(model, tokenizer, gsm8k_dataset, dataset_type="gsm8k")
        results["mixtral_8x22b_gsm8k"] = gsm8k_accuracy
    
    if gsm_symbolic_dataset:
        print(f"Starting GSM-Symbolic evaluation...")
        gsm_symbolic_accuracy = evaluate_model(model, tokenizer, gsm_symbolic_dataset, dataset_type="gsm_symbolic")
        results["mixtral_8x22b_gsm_symbolic"] = gsm_symbolic_accuracy

    print("\nFinal Results:")
    for model_dataset, accuracy in results.items():
        print(f"{model_dataset}: {accuracy:.2%}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Mixtral-8x22B on math datasets')
    parser.add_argument('--dataset', type=str, choices=['gsm8k', 'gsm_symbolic', 'both'], 
                      default='gsm_symbolic', help='Which dataset to evaluate on')
    parser.add_argument('--username', type=str, required=True,
                      help='Your username for the scratch directory path')
    parser.add_argument('--download_only', action='store_true',
                      help='Only verify model files without running evaluation')
    
    args = parser.parse_args()
    main(dataset_type=args.dataset, username=args.username, download_only=args.download_only)