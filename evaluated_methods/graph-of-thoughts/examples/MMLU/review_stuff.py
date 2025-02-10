import os
import json

def count_problems_solved(directory):
    total_solved = 0
    
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            
            with open(file_path, 'r') as file:
                data = json.load(file)
                
                for item in data:
                    if 'operation' in item and item['operation'] == 'ground_truth_evaluator':
                        if 'problem_solved' in item:
                            total_solved += sum(1 for solved in item['problem_solved'] if solved)
    
    return total_solved

if __name__ == "__main__":
    # Usage
    directory_path = '/p/llmreliability/test_repos/graph-of-thoughts/examples/MMLU/results/claude-3.5_cot-got_2024-08-19_14-56-28/cot'
    problems_solved = count_problems_solved(directory_path)
    print(f"Total problems solved: {problems_solved}")