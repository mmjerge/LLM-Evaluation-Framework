import os
import json

def calculate_accuracy(directory):
    total_problems = 0
    problems_solved = 0

    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)
                
                # Find the "ground_truth_evaluator" operation
                for operation in data:
                    if operation.get('operation') == 'ground_truth_evaluator':
                        problem_solved = operation.get('problem_solved', [])
                        total_problems += len(problem_solved)
                        problems_solved += sum(problem_solved)
                        break

    accuracy = (problems_solved / total_problems) * 100 if total_problems > 0 else 0
    return accuracy, problems_solved, total_problems

# Replace 'path_to_your_directory' with the actual path to your JSON files
directory = '/p/llmreliability/test_repos/graph-of-thoughts/examples/GSM8k/results/claude-3.5_cot-got_2024-08-19_14-05-36/cot'
accuracy, solved, total = calculate_accuracy(directory)

print(f"Total problems: {total}")
print(f"Problems solved: {solved}")
print(f"Accuracy: {accuracy:.2f}%")