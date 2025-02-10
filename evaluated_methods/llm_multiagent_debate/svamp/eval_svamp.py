import json
import numpy as np
import re

def parse_answer(input_str):
    # More flexible regex to capture the answer
    pattern = r"\\boxed\{(.*?)\}"
    matches = re.findall(pattern, input_str)
    if matches:
        return matches[-1].strip()  # Return the last match as the likely answer
    return None

def compute_accuracy(gt, pred_solutions):
    correct_answer = str(gt).strip()
    pred_answers = []
    for pred in pred_solutions:
        pred_answer = parse_answer(pred)
        if pred_answer:
            pred_answers.append(pred_answer)
    
    if not pred_answers:
        return 0  # No valid answers found
    
    # Select the most frequent predicted answer
    pred_answer = most_frequent(pred_answers)
    
    # Compare the predicted answer with the ground truth
    if is_close_enough(pred_answer, correct_answer):
        return 1
    else:
        return 0

def most_frequent(lst):
    return max(set(lst), key=lst.count)

def is_close_enough(pred, gt):
    # Try to convert both to float for numerical comparison
    try:
        pred_float = float(pred)
        gt_float = float(gt)
        return abs(pred_float - gt_float) < 1e-6
    except ValueError:
        # If conversion fails, fall back to string comparison
        return pred.lower() == gt.lower()

if __name__ == "__main__":
    response_dict = json.load(open("/p/llmreliability/test_repos/llm_multiagent_debate/svamp/svamp_3_2_openai_with_counterfactuals.json", "r"))
    questions = list(response_dict.keys())  # Evaluate all questions
    accuracies = []
    
    for question in questions:
        responses = response_dict[question]['contexts']
        gt = response_dict[question]['correct_answer']
        pred_solutions = []
        for response in responses:
            # Use the last message in the context as the predicted solution
            pred_solution = response[-1]['content']
            pred_solutions.append(pred_solution)
        
        accurate = compute_accuracy(gt, pred_solutions)
        accuracies.append(float(accurate))
        
        if accurate == 0:
            print(f"Incorrect prediction for question: {question}")
            print(f"Ground truth: {gt}")
            print(f"Predicted solutions: {pred_solutions}")
            print("---")

    if accuracies:
        mean_accuracy = np.mean(accuracies)
        std_error = np.std(accuracies) / np.sqrt(len(accuracies))
        print(f"Accuracy: {mean_accuracy:.4f} ± {std_error:.4f}")
        print(f"Number of evaluated questions: {len(accuracies)}")
    else:
        print("No valid accuracies were calculated.")