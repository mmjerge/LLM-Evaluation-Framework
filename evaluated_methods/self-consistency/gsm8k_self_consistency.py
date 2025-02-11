import argparse
import os
import json
import requests
import random
from tqdm import tqdm
from collections import Counter

def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name on Hugging Face")
    parser.add_argument("--API_URL", type=str, required=True, help="Hugging Face Inference API URL")
    parser.add_argument("--API_TOKEN", type=str, required=True, help="Hugging Face API Token")
    parser.add_argument("--n_samples", type=int, default=5, help="Number of samples for self-consistency")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum number of new tokens to generate")
    parser.add_argument("--dataset_path", type=str, default="data/GSM8K/gsm8k_test.jsonl", help="Path to the dataset file")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save the result file")
    parser.add_argument("--num_questions", type=int, default=50, help="Number of questions to process")
    return parser.parse_args()

def generate_answer(API_URL, headers, formatted_prompt, max_new_tokens):
    payload = {
        "inputs": formatted_prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens
        }
    }
    try:
        resp = requests.post(API_URL, json=payload, headers=headers)
        response = resp.json()
        return response[0]["generated_text"]
    except:
        print("Retrying due to an error...")
        return generate_answer(API_URL, headers, formatted_prompt, max_new_tokens)

def extract_final_answer(response):
    # Assuming the final answer is in the form \boxed{answer}
    import re
    match = re.search(r'\\boxed{(.*?)}', response)
    if match:
        return match.group(1)
    # If no boxed answer, return the last line
    return response.strip().split('\n')[-1]

def self_consistency(question, API_URL, headers, max_new_tokens, n_samples):
    prompt = f"Solve the following math problem. Explain your reasoning step by step. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response.\n\nQuestion: {question}"
    
    reasoning_paths = []
    for _ in range(n_samples):
        response = generate_answer(API_URL, headers, prompt, max_new_tokens)
        reasoning_paths.append(response)
    
    final_answers = [extract_final_answer(path) for path in reasoning_paths]
    answer_counts = Counter(final_answers)
    most_consistent_answer = answer_counts.most_common(1)[0][0]
    
    return most_consistent_answer, reasoning_paths

def read_jsonl(path: str):
    with open(path, "r") as fh:
        return [json.loads(line) for line in fh.readlines() if line]

def main():
    args = args_parse()
    
    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    headers = {
        "Authorization": f"Bearer {args.API_TOKEN}",
        "Content-Type": "application/json"
    }
    
    questions = read_jsonl(args.dataset_path)
    random.seed(0)
    random.shuffle(questions)
    
    results = []
    
    for idx in tqdm(range(min(args.num_questions, len(questions)))):
        question = questions[idx]["question"]
        answer = questions[idx]["answer"]
        
        predicted_answer, reasoning_paths = self_consistency(
            question, args.API_URL, headers, args.max_new_tokens, args.n_samples
        )
        
        results.append({
            "question_id": idx,
            "question": question,
            "true_answer": answer,
            "predicted_answer": predicted_answer,
            "reasoning_paths": reasoning_paths
        })
    
    output_file = f"{args.output_dir}/self_consistency_results_{args.model.replace('/', '_')}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    
    correct_predictions = sum(1 for result in results if result['true_answer'] == result['predicted_answer'])
    accuracy = correct_predictions / len(results)
    print(f"Overall accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    main()