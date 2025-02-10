import argparse
import json
import requests
import random
from tqdm import tqdm
from collections import Counter
import numpy as np

def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name on Hugging Face")
    parser.add_argument("--API_URL", type=str, required=True, help="Hugging Face Inference API URL")
    parser.add_argument("--API_TOKEN", type=str, required=True, help="Hugging Face API Token")
    parser.add_argument("--n_samples", type=int, default=5, help="Number of samples for self-consistency")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum number of new tokens to generate")
    parser.add_argument("--dataset_path", type=str, default="/p/llmreliability/test_repos/LLM-Agora/data/AQUA/test.jsonl", help="Path to the AQUA dataset file")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save the result file")
    parser.add_argument("--num_questions", type=int, default=50, help="Number of questions to process")
    parser.add_argument("--cot", action="store_true", help="Use Chain-of-Thought prompting")
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

def parse_answer(input_str):
    uppercase_letters = [char for char in input_str if char.isupper()]
    return uppercase_letters[-1] if uppercase_letters else None

def self_consistency(question, options, API_URL, headers, max_new_tokens, n_samples, use_cot):
    if use_cot:
        prompt = f"Answer the following question step by step, then select the correct answer from the given options. Your final answer should be a single uppercase letter.\n\nQuestion: {question}\nOptions: {options}"
    else:
        prompt = f"Answer the following question and select the correct answer from the given options. Your final answer should be a single uppercase letter.\n\nQuestion: {question}\nOptions: {options}"
    
    reasoning_paths = []
    for _ in range(n_samples):
        response = generate_answer(API_URL, headers, prompt, max_new_tokens)
        reasoning_paths.append(response)
    
    final_answers = [parse_answer(path) for path in reasoning_paths]
    final_answers = [answer for answer in final_answers if answer]  # Remove None values
    
    if not final_answers:
        return None, reasoning_paths
    
    answer_counts = Counter(final_answers)
    most_consistent_answer = answer_counts.most_common(1)[0][0]
    
    return most_consistent_answer, reasoning_paths

def compute_accuracy(correct_answer, pred_answer):
    return 1.0 if pred_answer == correct_answer else 0.0

def main():
    args = args_parse()
    
    headers = {
        "Authorization": f"Bearer {args.API_TOKEN}",
        "Content-Type": "application/json"
    }
    
    with open(args.dataset_path, 'r') as f:
        dataset = [json.loads(line) for line in f]
    
    random.seed(0)
    random.shuffle(dataset)
    
    results = []
    accuracies = []
    
    for idx in tqdm(range(min(args.num_questions, len(dataset)))):
        entry = dataset[idx]
        question = entry["question"]
        options = entry["options"]
        correct_answer = entry["correct"]
        
        predicted_answer, reasoning_paths = self_consistency(
            question, options, args.API_URL, headers, args.max_new_tokens, args.n_samples, args.cot
        )
        
        accuracy = compute_accuracy(correct_answer, predicted_answer)
        accuracies.append(accuracy)
        
        results.append({
            "question_id": idx,
            "question": question,
            "options": options,
            "correct_answer": correct_answer,
            "predicted_answer": predicted_answer,
            "reasoning_paths": reasoning_paths,
            "accuracy": accuracy
        })
    
    output_file = f"{args.output_dir}/aqua_self_consistency_results_{args.model.replace('/', '_')}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    
    overall_accuracy = np.mean(accuracies)
    print(f"Overall accuracy: {overall_accuracy:.2%}")
    
    performance = [{"performance": overall_accuracy}]
    performance_file = f"{args.output_dir}/aqua_performance_{'cot' if args.cot else 'no_cot'}.json"
    with open(performance_file, "w") as f:
        json.dump(performance, f, indent=4)
    
    print(f"Results saved to {output_file}")
    print(f"Performance saved to {performance_file}")
    print("All done!")

if __name__ == "__main__":
    main()