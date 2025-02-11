import random
from datasets import load_dataset
from openai import OpenAI
from anthropic import Anthropic
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import pandas as pd
import time
from typing import List, Dict, Literal
from tqdm import tqdm
import re
import json
from datetime import datetime

ProviderType = Literal["openai", "anthropic", "mistral"]

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
        pred_clean = re.search(r'-?\d*\.?\d+', str(predicted)).group()
        actual_clean = re.search(r'-?\d*\.?\d+', str(actual)).group()
        
        pred_num = float(pred_clean)
        actual_num = float(actual_clean)
        
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
    temperature: float = 0.0
) -> str:
    """
    Gets response from specified model provider.
    """
    system_prompt = "You are a mathematical reasoning assistant. Always show your step-by-step thinking and conclude with 'Final answer: #### [number]'."
    
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

def save_results_to_json(results: Dict, output_file: str = None) -> None:
    """
    Saves the test results to a JSON file with detailed comparisons.
    """
    if output_file is None:
        clean_model_name = re.sub(r'[^\w\-]', '_', results['model_name'])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"{results['provider']}_{clean_model_name}_{timestamp}.json"
    
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

def test_gsm_symbolic(
    num_samples: int = 150,
    provider: ProviderType = "openai",
    model_name: str = "gpt-3.5-turbo",
    temperature: float = 0.0
) -> Dict:
    """
    Tests the specified model on random samples from GSM-Symbolic dataset.
    """
    ds = load_dataset("apple/GSM-Symbolic", "main")
    
    test_indices = random.sample(range(len(ds['test'])), num_samples)
    test_samples = [ds['test'][i] for i in test_indices]
    
    results = []
    correct = 0
    
    for i, sample in tqdm(enumerate(test_samples), total=num_samples, desc=f"Testing {model_name}"):
        question = sample['question']
        actual_answer = sample['answer']
        
        prompt = create_cot_prompt(question)
        
        try:
            response = get_model_response(prompt, provider, model_name, temperature)
            
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
    
    save_results_to_json(results_dict)
    
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

def run_model_evaluations(
    num_samples: int = 150,
    models_to_test: List[Dict[str, str]] = None
) -> Dict:
    """
    Runs evaluations for multiple models and returns their results.
    """
    if models_to_test is None:
        models_to_test = [
            {"provider": "openai", "model_name": "gpt-3.5-turbo"},
            {"provider": "openai", "model_name": "gpt-4-turbo-preview"},
            {"provider": "anthropic", "model_name": "claude-3-opus-20240229"},
            {"provider": "mistral", "model_name": "mistral-large-latest"}
        ]
    
    all_results = {}
    
    for model in models_to_test:
        provider = model["provider"]
        model_name = model["model_name"]
        
        print(f"\nEvaluating {provider} - {model_name}")
        try:
            results = test_gsm_symbolic(
                num_samples=num_samples,
                provider=provider,
                model_name=model_name
            )
            all_results[f"{provider}_{model_name}"] = results
            analyze_results(results)
            
        except Exception as e:
            print(f"Error evaluating {provider} - {model_name}: {str(e)}")
            continue
    
    print("\n=== Comparative Results ===")
    print("\nAccuracy by Model:")
    for model_key, results in all_results.items():
        print(f"{model_key}: {results['accuracy']*100:.2f}%")
    
    return all_results

if __name__ == "__main__":
    # Define your models
    models_to_test = [
        {"provider": "openai", "model_name": "gpt-3.5-turbo"},
        {"provider": "openai", "model_name": "gpt-4-turbo-preview"},
        {"provider": "anthropic", "model_name": "claude-3-opus-20240229"},
        {"provider": "mistral", "model_name": "mistral-large-latest"}
    ]
    
    # Initialize results dictionary
    all_results = {}
    
    # Explicitly iterate through each model
    for model in models_to_test:
        try:
            print(f"\nStarting evaluation of {model['provider']} - {model['model_name']}")
            results = test_gsm_symbolic(
                num_samples=150,
                provider=model["provider"],
                model_name=model["model_name"],
                temperature=0.0
            )
            
            all_results[f"{model['provider']}_{model['model_name']}"] = results
            analyze_results(results)
            
        except Exception as e:
            print(f"Error with {model['provider']} - {model['model_name']}: {str(e)}")
            continue
    
    # Print final comparative results
    print("\n=== Final Comparative Results ===")
    for model_key, model_results in all_results.items():
        print(f"{model_key}: {model_results['accuracy']*100:.2f}%")