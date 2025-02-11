import argparse
import json
import requests
import random
import time
import tiktoken
from tqdm import tqdm
import re

def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name on Hugging Face")
    parser.add_argument("--API_URL", type=str, required=True, help="Hugging Face Inference API URL")
    parser.add_argument("--API_TOKEN", type=str, required=True, help="Hugging Face API Token")
    parser.add_argument("--n_samples", type=int, default=5, help="Number of samples for self-consistency")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum number of new tokens to generate")
    parser.add_argument("--dataset_path", type=str, default="data/SVAMP/SVAMP.json", help="Path to the SVAMP dataset file")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save the result file")
    parser.add_argument("--num_questions", type=int, default=10, help="Number of questions to process")
    parser.add_argument("--cot", action="store_true", help="Use Chain-of-Thought prompting")
    return parser.parse_args()

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def generate_answer(API_URL, headers, formatted_prompt, max_new_tokens):
    input_tokens = num_tokens_from_string(formatted_prompt)
    max_input_tokens = 1512 - max_new_tokens
    if input_tokens > max_input_tokens:
        print(f"Warning: Input exceeds token limit. Truncating input.")
        encoding = tiktoken.get_encoding("cl100k_base")
        truncated_tokens = encoding.encode(formatted_prompt)[:max_input_tokens]
        formatted_prompt = encoding.decode(truncated_tokens)
        input_tokens = num_tokens_from_string(formatted_prompt)

    payload = {
        "inputs": formatted_prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": 0.7,
            "top_p": 0.95,
            "do_sample": True,
            "no_repeat_ngram_size": 2
        }
    }

    try:
        resp = requests.post(API_URL, json=payload, headers=headers)
        resp.raise_for_status()
        response = resp.json()
        
        if not response or not isinstance(response, list) or len(response) == 0:
            raise ValueError("Unexpected response format")
        
        return response[0].get("generated_text", "")
    except requests.RequestException as e:
        print(f"Request error: {e}")
        print(f"Response content: {resp.text}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    print("Retrying due to an error......")
    time.sleep(5)
    return generate_answer(API_URL, headers, formatted_prompt, max_new_tokens)

def parse_answer(response):
    match = re.search(r'\\boxed{(.*?)}', response)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    # If no boxed answer or not a number, try to find the last number in the response
    numbers = re.findall(r'-?\d+(?:\.\d+)?', response)
    return float(numbers[-1]) if numbers else None

def self_consistency(question, API_URL, headers, max_new_tokens, n_samples, use_cot):
    prompt = f"Can you solve the following math problem? {question} Explain your reasoning. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response."
    if use_cot:
        prompt += " Let's think step by step."
    
    reasoning_paths = []
    for _ in range(n_samples):
        response = generate_answer(API_URL, headers, prompt, max_new_tokens)
        reasoning_paths.append(response)
    
    final_answers = [parse_answer(path) for path in reasoning_paths]
    final_answers = [answer for answer in final_answers if answer is not None]  # Remove None values
    
    if not final_answers:
        return None, reasoning_paths
    
    # Use the most common answer (rounded to 2 decimal places)
    from collections import Counter
    rounded_answers = [round(answer, 2) for answer in final_answers]
    most_common_answer = Counter(rounded_answers).most_common(1)[0][0]
    
    return most_common_answer, reasoning_paths

def compute_accuracy(correct_answer, pred_answer, tolerance=1e-6):
    if pred_answer is None:
        return 0.0
    return 1.0 if abs(float(correct_answer) - float(pred_answer)) < tolerance else 0.0

def main():
    args = args_parse()
    
    headers = {
        "Authorization": f"Bearer {args.API_TOKEN}",
        "Content-Type": "application/json"
    }
    
    with open(args.dataset_path, 'r') as f:
        svamp_questions = json.load(f)
    
    random.seed(0)
    random.shuffle(svamp_questions)
    
    results = []
    accuracies = []
    
    for idx in tqdm(range(min(args.num_questions, len(svamp_questions))), desc="Processing questions"):
        question_data = svamp_questions[idx]
        question = f"{question_data['Body']} {question_data['Question']}"
        correct_answer = question_data['Answer']
        
        predicted_answer, reasoning_paths = self_consistency(
            question, args.API_URL, headers, args.max_new_tokens, args.n_samples, args.cot
        )
        
        accuracy = compute_accuracy(correct_answer, predicted_answer)
        accuracies.append(accuracy)
        
        results.append({
            "question_id": idx,
            "question": question,
            "correct_answer": correct_answer,
            "predicted_answer": predicted_answer,
            "reasoning_paths": reasoning_paths,
            "accuracy": accuracy
        })
    
    output_file = f"{args.output_dir}/svamp_self_consistency_results_{args.model.replace('/', '_')}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    
    overall_accuracy = sum(accuracies) / len(accuracies)
    print(f"Overall accuracy: {overall_accuracy:.2%}")
    
    performance = [{"performance": overall_accuracy}]
    performance_file = f"{args.output_dir}/svamp_performance_{'cot' if args.cot else 'no_cot'}.json"
    with open(performance_file, "w") as f:
        json.dump(performance, f, indent=4)
    
    print(f"Results saved to {output_file}")
    print(f"Performance saved to {performance_file}")
    print("All done!")

if __name__ == "__main__":
    main()