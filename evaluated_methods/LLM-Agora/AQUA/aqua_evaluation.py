import json
import numpy as np
import argparse

def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_1",
        type=str,
        help="It should be the same model used in aqua_inference.py"
    )
    parser.add_argument(
        "--model_2",
        type=str,
        help="It should be the same model used in aqua_inference.py"
    )
    parser.add_argument(
        "--model_3",
        type=str,
        help="It should be the same model used in aqua_inference.py"
    )
    parser.add_argument(
        "--cot",
        action="store_true"
    )
    parser.add_argument(
        "--output_dir",
        default="AQUA",
        type=str
    )
    return parser.parse_args()

def parse_answer(input_str):
    # Extract the last uppercase letter from the input string
    uppercase_letters = [char for char in input_str if char.isupper()]
    return uppercase_letters[-1] if uppercase_letters else None

def answer_check(pred_answer, correct_answer):
    return 1.0 if pred_answer == correct_answer else 0.0

def compute_accuracy(correct_answer, pred_solutions):
    pred_answers = [parse_answer(pred_solution) for pred_solution in pred_solutions]
    return max(answer_check(pred_answer, correct_answer) for pred_answer in pred_answers if pred_answer)

if __name__ == "__main__":
    args = args_parse()
    model_list = [args.model_1, args.model_2, args.model_3]

    if args.cot:
        file_name = "_cot.json"
    else:
        file_name = ".json"

    with open(f"{args.output_dir}/aqua_result{file_name}", "r") as f:
        response_dict = json.load(f)

    questions = [response_dict[i]["question"] for i in range(len(response_dict))]
    performance = []

    for turn in range(3):
        accuracies = []
        for idx in range(len(questions)):
            responses = [response_dict[idx]["agent_response"][model][turn] for model in model_list]
            correct_answer = response_dict[idx]["correct"]
            accurate = compute_accuracy(correct_answer, responses)
            accuracies.append(float(accurate))

        performance.append({f"{turn+1}_performance": np.mean(accuracies)})
        print({f"{turn+1}_performance": np.mean(accuracies)})

    print(f"The performance file 'aqua_performance{file_name}' is saving...")
    with open(args.output_dir + f"/aqua_performance{file_name}", "w") as f:
        json.dump(performance, f, indent=4)

    print("All done!!")