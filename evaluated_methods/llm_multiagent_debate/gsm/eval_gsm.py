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

def solve_math_problems(input_str):
    pattern = r"\d+\.?\d*"
    matches = re.findall(pattern, input_str)
    if matches:
        return matches[-1]
    return None

def parse_answer(input_str):
    pattern = r"\{([0-9.,$]*)\}"
    matches = re.findall(pattern, input_str)
    solution = None
    for match_str in matches[::-1]:
        solution = re.sub(r"[^0-9.]", "", match_str)
        if solution:
            break
    return solution

def compute_accuracy(gt, pred_solution):
    answers = solve_math_problems(gt)
    if answers is None:
        return None
    
    if isinstance(pred_solution, list):
        pred_answers = []
        for pred in pred_solution:
            pred_answer = parse_answer(pred)
            if pred_answer is None:
                pred_answer = solve_math_problems(pred)
            pred_answers.append(pred_answer)
        pred_answer = most_frequent(pred_answers)
    else:
        pred_answer = parse_answer(pred_solution)
        if pred_answer is None:
            pred_answer = solve_math_problems(pred_solution)
    
    if pred_answer is None:
        return 1
    
    try:
        if float(answers) == float(pred_answer):
            return 1
        else:
            return 0
    except ValueError:
        print(f"Error: Unable to compare {answers} and {pred_answer}")
        return None

def most_frequent(List):
    return max(set(List), key=List.count)

if __name__ == "__main__":
    response_dict = json.load(open("/p/llmreliability/test_repos/llm_multiagent_debate/gsm/gpt4o_counterfactuals.json", "r"))
    questions = list(response_dict.keys())[:50]
    accuracies = []

    for question in questions:
        responses, gt = response_dict[question]
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

