import random
from datasets import load_dataset
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI
from anthropic import Anthropic
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import torch
import pandas as pd
import time
from typing import List, Dict, Literal
from tqdm import tqdm
import re
import json
from datetime import datetime
import os
import pathlib
import argparse

ProviderType = Literal["openai", "anthropic", "mistral", "huggingface"]

def ensure_directory_exists(directory: str) -> str:
    """
    Ensures the specified directory exists, creates it if it doesn't.
    Returns the absolute path to the directory.
    """
    dir_path = os.path.abspath(directory)
    pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)
    return dir_path

def get_few_shot_examples() -> str:
    """
    Returns few-shot examples for chain-of-thought prompting.
    """
    return """Example 1:
Question: Angelo and Melanie want to plan how many hours over the next week they should study together for their test next week. They have 2 chapters of their textbook to study and 4 worksheets to memorize. They figure out that they should dedicate 3 hours to each chapter of their textbook and 1.5 hours for each worksheet. If they plan to study no more than 4 hours each day, how many days should they plan to study total over the next week if they take a 10-minute break every hour, include 3 10-minute snack breaks each day, and 30 minutes for lunch each day?

Let's think step by step:
1. Calculate total study hours for chapters: 3 hours × 2 chapters = 6 hours
2. Calculate total study hours for worksheets: 1.5 hours × 4 worksheets = 6 hours
3. Base study time needed: 6 + 6 = 12 hours
4. Calculate break time:
   - Hourly breaks: 12 hours × 10 minutes = 120 minutes
   - Daily snack breaks: 3 × 10 minutes = 30 minutes per day
   - Daily lunch: 30 minutes per day
5. Total break time per day: 30 + 30 = 60 minutes = 1 hour
6. Total study + break time: 12 hours + 3 hours = 15 hours
7. Days needed at 4 hours per day: 15 ÷ 4 = 3.75 days

Final answer: 4

Example 2:
Question: Mark's basketball team scores 25 2 pointers, 8 3 pointers and 10 free throws. Their opponents score double the 2 pointers but half the 3 pointers and free throws. What's the total number of points scored by both teams added together?

Let's think step by step:
1. Calculate Mark's team points:
   - 2 pointers: 25 × 2 = 50 points
   - 3 pointers: 8 × 3 = 24 points
   - Free throws: 10 × 1 = 10 points
   - Total: 50 + 24 + 10 = 84 points
2. Calculate opponent's points:
   - 2 pointers: 50 × 2 = 100 points
   - 3 pointers: 24 ÷ 2 = 12 points
   - Free throws: 10 ÷ 2 = 5 points
   - Total: 100 + 12 + 5 = 117 points
3. Total points in game: 84 + 117 = 201

Final answer: 201

Now solve this new problem:
"""

def create_cot_prompt(question: str) -> str:
    """
    Creates a chain-of-thought prompt with few-shot examples for a given math question.
    """
    return f"{get_few_shot_examples()}\nQuestion: {question}\n\nLet's think step by step:"

def extract_final_answer(response: str) -> str:
    """
    Extracts the final numerical answer from the model's response.
    """
    if "Final answer:" in response:
        final_answer_line = response.split("Final answer:")[-1].strip()
        numbers = re.findall(r'-?\d*\.?\d+', final_answer_line)
        if numbers:
            return numbers[0]
    
    if "The answer is" in response:
        final_answer_line = response.split("The answer is")[-1].strip()
        numbers = re.findall(r'-?\d*\.?\d+', final_answer_line)
        if numbers:
            return numbers[0]
    
    lines = [line.strip() for line in response.split('\n') if line.strip()]
    for line in reversed(lines):
        numbers = re.findall(r'-?\d*\.?\d+', line)
        if numbers:
            return numbers[-1]
    
    return ""

def evaluate_response(predicted: str, actual: str) -> bool:
    """
    Evaluates if the predicted answer matches the actual answer within a tolerance.
    """
    try:
        # Clean up the predicted and actual answers to get just the numbers
        pred_clean = re.search(r'-?\d*\.?\d+', str(predicted)).group()
        actual_clean = re.search(r'-?\d*\.?\d+', str(actual)).group()
        
        pred_num = float(pred_clean)
        actual_num = float(actual_clean)
        
        # Allow for small floating point differences
        return abs(pred_num - actual_num) < 0.01
    except (ValueError, AttributeError) as e:
        print(f"Evaluation error: {e}")
        print(f"Predicted: {predicted}")
        print(f"Actual: {actual}")
        return False

def get_model_response(
    prompt: str,
    provider: ProviderType,
    model_name: str,
    temperature: float = 0.0,
    huggingface_pipeline = None
) -> str:
    """
    Gets response from specified model provider.
    """
    system_prompt = """You are a mathematical reasoning assistant. For each question:
    1. Always show your step-by-step thinking
    2. Show all calculations clearly
    3. End with 'Final answer: [number]'
    4. Do not engage in conversation or add extra commentary"""    

    if provider == "huggingface":
        if huggingface_pipeline is None:
            raise ValueError("HuggingFace pipeline must be provided for HuggingFace models")
        
        # Format prompt with system prompt for consistency
        formatted_prompt = f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:"
        
        # Generate response
        generation_config = {
            "max_new_tokens": 1000,
            "do_sample": True,
            "temperature": max(0.1, temperature),
            "top_p": 0.9,
            "num_return_sequences": 1,
            "repetition_penalty": 1.1  # Add this to prevent repetitive outputs
        }
        
        response = huggingface_pipeline(
            formatted_prompt,
            **generation_config
        )
        
        generated_text = response[0]['generated_text']
        assistant_response = generated_text.split("Assistant:")[-1].strip()
        return assistant_response
    
    if provider == "openai":
        client = OpenAI()
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature
        )
        return response.choices[0].message.content
    
    elif provider == "anthropic":
        client = Anthropic()
        response = client.messages.create(
            model=model_name,
            max_tokens=1000,
            temperature=temperature,
            system=system_prompt,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text
    
    elif provider == "mistral":
        client = MistralClient()
        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=prompt)
        ]
        response = client.chat(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=1000
        )
        return response.choices[0].message.content
    
    else:
        raise ValueError(f"Unsupported provider: {provider}")

def save_results_to_json(results: Dict, output_dir: str = "results") -> str:
    """
    Saves the test results to a JSON file with detailed comparisons.
    Returns the path to the saved file.
    """
    results_dir = ensure_directory_exists(output_dir)
    
    clean_model_name = re.sub(r'[^\w\-]', '_', results['model_name'])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{results['provider']}_{clean_model_name}_{timestamp}.json"
    output_file = os.path.join(results_dir, filename)
    
    json_results = {
        'metadata': {
            'model_name': results['model_name'],
            'provider': results['provider'],
            'accuracy': results['accuracy'],
            'total_samples': results['total_samples'],
            'correct_count': results['correct_count'],
            'timestamp': datetime.now().isoformat()
        },
        'questions': []
    }
    
    for _, row in results['results_df'].iterrows():
        question_result = {
            'question_id': int(row['question_id']),
            'question': row['question'],
            'ground_truth': row['actual_answer'],
            'model_answer': row['predicted_answer'],
            'full_response': row['full_response'],
            'is_correct': bool(row['correct'])
        }
        json_results['questions'].append(question_result)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed results saved to: {output_file}")
    return output_file

def test_gsm_symbolic(
    num_samples: int = 150,
    provider: ProviderType = "openai",
    model_name: str = "gpt-4o",
    temperature: float = 0.0,
    output_dir: str = "results",
    huggingface_pipeline = None
) -> Dict:
    """
    Tests the specified model on random samples from GSM-Symbolic dataset.
    """
    ds = load_dataset("apple/GSM-Symbolic", "main")
    
    if provider == "huggingface" and huggingface_pipeline is None:
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
        
        huggingface_pipeline = pipeline(
            "text-generation",
            model=model_name,
            tokenizer=tokenizer,
            torch_dtype=torch.float16,
            device_map="auto",
            max_new_tokens=1000,
            do_sample=False,
            temperature=0.1,
            repetition_penalty=1.2, 
            early_stopping=True
        )

    test_indices = random.sample(range(len(ds['test'])), num_samples)
    test_samples = [ds['test'][i] for i in test_indices]
    
    results = []
    correct = 0
    
    for i, sample in tqdm(enumerate(test_samples), total=num_samples, desc=f"Testing {model_name}"):
        question = sample['question']
        actual_answer = sample['answer']
        
        prompt = create_cot_prompt(question)
        
        try:
            response = get_model_response(
                prompt,
                provider,
                model_name,
                temperature,
                huggingface_pipeline
            )
            
            predicted_answer = extract_final_answer(response)
            
            is_correct = evaluate_response(predicted_answer, actual_answer)
            if is_correct:
                correct += 1
                
            results.append({
                'question_id': i,
                'question': question,
                'actual_answer': actual_answer,
                'predicted_answer': predicted_answer,
                'full_response': response,
                'correct': is_correct
            })
            
            time.sleep(1)
            
        except Exception as e:
            print(f"Error processing question {i}: {str(e)}")
            continue
    
    accuracy = correct / num_samples
    
    results_df = pd.DataFrame(results)
    
    results_dict = {
        'accuracy': accuracy,
        'total_samples': num_samples,
        'correct_count': correct,
        'results_df': results_df,
        'model_name': model_name,
        'provider': provider
    }
    
    save_results_to_json(results_dict, output_dir)
    
    return results_dict

def analyze_results(results: Dict) -> None:
    """
    Analyzes and prints the test results.
    """
    print(f"\nTest Results for {results['provider']} - {results['model_name']}:")
    print(f"Accuracy: {results['accuracy']*100:.2f}%")
    print(f"Correct: {results['correct_count']}/{results['total_samples']}")
    
    incorrect_samples = results['results_df'][~results['results_df']['correct']].head(3)
    if len(incorrect_samples) > 0:
        print("\nSample of incorrect answers:")
        for _, row in incorrect_samples.iterrows():
            print(f"\nQuestion: {row['question']}")
            print(f"Expected: {row['actual_answer']}")
            print(f"Predicted: {row['predicted_answer']}")
            print("Full response:")
            print(row['full_response'])

def save_comparative_results(all_results: Dict, output_dir: str = "results") -> str:
    """
    Saves comparative results to a JSON file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"comparative_results_{timestamp}.json"
    output_file = os.path.join(ensure_directory_exists(output_dir), filename)
    
    comparative_data = {
        'timestamp': datetime.now().isoformat(),
        'results': {}
    }
    
    for model_key, results in all_results.items():
        comparative_data['results'][model_key] = {
            'accuracy': results['accuracy'],
            'correct_count': results['correct_count'],
            'total_samples': results['total_samples'],
            'model_name': results['model_name'],
            'provider': results['provider']
        }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(comparative_data, f, indent=2)
    
    print(f"\nComparative results saved to: {output_file}")
    return output_file

if __name__ == "__main__":
    print(f"\nPyTorch version: {torch.__version__}")
    
    parser = argparse.ArgumentParser(description='Test language models on GSM-Symbolic dataset')
    parser.add_argument('--output-dir', type=str, default="gsm_results",
                      help='Directory to save results (default: gsm_results)')
    parser.add_argument('--providers', nargs='+', type=str, default=['openai', 'anthropic', 'mistral', 'huggingface'],
                      help='List of providers to test (default: all providers)')
    parser.add_argument('--gpu-ids', nargs='+', type=int, default=[0],
                      help='List of GPU IDs to use for HuggingFace models (default: [0])')
    parser.add_argument('--num-samples', type=int, default=150,
                      help='Number of samples to test (default: 150)')
    parser.add_argument('--huggingface-models', nargs='+', type=str, 
                      default=['meta-llama/Llama-2-70b-chat-hf', 'mistralai/Mixtral-8x7B-Instruct-v0.1'],
                      help='List of HuggingFace model IDs to test')
    
    args = parser.parse_args()
    
    if 'huggingface' in args.providers:
        print("\nCUDA Diagnostic Information:")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device() if torch.cuda.is_available() else 'None'}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
        print(f"NVIDIA_VISIBLE_DEVICES: {os.environ.get('NVIDIA_VISIBLE_DEVICES', 'Not set')}")
        print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")

        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpu_ids))
        
        if torch.cuda.is_available():
            print(f"\nUsing GPU: {torch.cuda.get_device_name(0)}")
            device = "cuda"
        else:
            print("\nNo GPU available, using CPU")
            device = "cpu"
    
    output_dir = args.output_dir
    
    provider_models = {
        "openai": ["gpt-3.5-turbo", "gpt-4o"],
        "anthropic": ["claude-3-5-sonnet-20241022"],
        "mistral": ["open-mixtral-8x22b"],
        "huggingface": args.huggingface_models
    }
    
    models_to_test = []
    for provider in args.providers:
        if provider in provider_models:
            models_to_test.extend([
                {"provider": provider, "model_name": model_name}
                for model_name in provider_models[provider]
            ])
        else:
            print(f"Warning: Unknown provider '{provider}' specified")
    
    all_results = {}
    
    for model in models_to_test:
        try:
            print(f"\nStarting evaluation of {model['provider']} - {model['model_name']}")
            
            huggingface_pipeline = None
            if model['provider'] == "huggingface":
                huggingface_pipeline = pipeline(
                    "text-generation",
                    model=model['model_name'],
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
            
            results = test_gsm_symbolic(
                num_samples=args.num_samples,
                provider=model["provider"],
                model_name=model["model_name"],
                temperature=0.0,
                output_dir=output_dir,
                huggingface_pipeline=huggingface_pipeline
            )
            
            all_results[f"{model['provider']}_{model['model_name']}"] = results
            analyze_results(results)
            
            if huggingface_pipeline is not None:
                del huggingface_pipeline
                torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error with {model['provider']} - {model['model_name']}: {str(e)}")
            continue
    
    save_comparative_results(all_results, output_dir)
    
    print("\n=== Final Comparative Results ===")
    print("\nAccuracy by Model:")
    for model_key, results in all_results.items():
        print(f"{model_key}: {results['accuracy']*100:.2f}%")