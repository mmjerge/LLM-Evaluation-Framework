import json
import openai
import numpy as np
import time
import re
import argparse

def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_1",
        type=str,
        help="It should be the same model used in svamp_inference.py"
    )
    parser.add_argument(
        "--model_2",
        type=str,
        help="It should be the same model used in svamp_inference.py"
    )
    parser.add_argument(
        "--model_3",
        type=str,
        help="It should be the same model used in svamp_inference.py"
    )
    parser.add_argument(
        "--cot",
        action="store_true"
    )
    parser.add_argument(
        "--output_dir",
        default="Math",
        type=str
    )
    return parser.parse_args()

def solve_math_problems(input_str):
    pattern = r"\d+\.?\d*"
    matches = re.findall(pattern, input_str)
    if matches:
        return float(matches[-1])
    return None

def parse_answer(input_str):
    pattern = r"([0-9]*\.?[0-9]+)"
    matches = re.findall(pattern, input_str)
    if matches:
        return float(matches[-1])
    return None

def answer_check(pred, gt):
    if abs(pred - gt) < 1e-6:
        return 1.0
    else:
        return 0.0

def compute_accuracy(gt, pred_solutions):
    gt_answer = float(gt)
    if type(pred_solutions) == list:
        pred_answers = []
        for pred_solution in pred_solutions:
            pred_answer = parse_answer(pred_solution)
            if not pred_answer:
                pred_answer = solve_math_problems(pred_solution)
            if pred_answer is not None:
                pred_answers.append(pred_answer)
        
        for pred_answer in pred_answers:
            if answer_check(pred_answer, gt_answer):
                return 1.0
        return 0.0
    else:
        pred_answer = parse_answer(pred_solutions)
        if not pred_answer:
            pred_answer = solve_math_problems(pred_solutions)
        if pred_answer is not None:
            return answer_check(pred_answer, gt_answer)
        return 0.0

if __name__ == "__main__":
    args = args_parse()
    model_list = [args.model_1, args.model_2, args.model_3]

    if args.cot:
        file_name = "_cot.json"
    else:
        file_name = ".json"

    with open(f"{args.output_dir}/svamp_result{file_name}", "r") as f:
        response_dict = json.load(f)

    questions = [response_dict[i]["question"] for i in range(len(response_dict))]
    performance = []

    for turn in range(3):
        accuracies = []
        for idx in range(len(questions)):
            responses = [response_dict[idx]["agent_response"][model][turn] for model in model_list]
            gt = response_dict[idx]["answer"]
            accurate = compute_accuracy(gt, responses)
            accuracies.append(float(accurate))

        performance.append({f"{turn+1}_performance": np.mean(accuracies)})
        print({f"{turn+1}_performance": np.mean(accuracies)})

    print(f"The performance file 'svamp_performance{file_name}' is saving...")
    with open(args.output_dir + f"/svamp_performance{file_name}", "w") as f:
        json.dump(performance, f, indent=4)

    print("All done!!")