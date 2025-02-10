import json
import numpy as np
import re

def parse_bullets(sentence):
    bullets_preprocess = sentence.split("\n")
    bullets = []
    for bullet in bullets_preprocess:
        try:
            idx = bullet.find(next(filter(str.isalpha, bullet)))
        except:
            continue
        bullet = bullet[idx:]
        if len(bullet) != 0:
            bullets.append(bullet)
    return bullets

def parse_yes_no(string):
    if "yes" in string.lower():
        return True
    elif "no" in string.lower():
        return False
    else:
        return None

def parse_answer(input_str):
    # Extracts answer from the string, assuming answer is in a format like "Answer: A"
    pattern = r"[A-E]"
    matches = re.findall(pattern, input_str)
    if matches:
        return matches[-1]  # Return the last match as the likely answer
    return None

def compute_accuracy(gt, pred_solutions):
    correct_option = gt.strip()  # Ground truth answer is a single option like 'A', 'B', etc.
    
    pred_answers = []
    for pred in pred_solutions:
        pred_answer = parse_answer(pred)
        if pred_answer:
            pred_answers.append(pred_answer)
    
    if not pred_answers:
        return 0
    
    pred_answer = most_frequent(pred_answers)
    
    if pred_answer == correct_option:
        return 1
    else:
        return 0

def most_frequent(List):
    return max(set(List), key=List.count)

if __name__ == "__main__":
    response_dict = json.load(open("/p/llmreliability/test_repos/llm_multiagent_debate/aqua/gpt4o_with_counterfactuals.json", "r"))
    questions = list(response_dict.keys())[:50]
    accuracies = []

    for question in questions:
        responses, gt = response_dict[question]['contexts'], response_dict[question]['correct_answer']
        pred_solutions = []
        for response in responses:
            pred_solution = response[-1]['content']
            pred_solutions.append(pred_solution)
        
        accurate = compute_accuracy(gt, pred_solutions)
        if accurate is not None:
            accuracies.append(float(accurate))
        else:
            print(f"Skipping question: {question}")
            print(f"Ground truth: {gt}")
            print(f"Predicted solutions: {pred_solutions}")

    if accuracies:
        mean_accuracy = np.mean(accuracies)
        std_error = np.std(accuracies) / np.sqrt(len(accuracies))
        print(f"Accuracy: {mean_accuracy:.4f} ± {std_error:.4f}")
        print(f"Number of evaluated questions: {len(accuracies)}")
    else:
        print("No valid accuracies were calculated.")
