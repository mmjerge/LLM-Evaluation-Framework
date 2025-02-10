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
                    if item.get('operation') == 'ground_truth_evaluator':
                        if 'problem_solved' in item:
                            # Treat all problem_solved values as one
                            if isinstance(item['problem_solved'], list) and all(item['problem_solved']):
                                total_solved += 1
    
    return total_solved

if __name__ == "__main__":
    # Usage
    directory_path = '/p/llmreliability/test_repos/graph-of-thoughts/examples/SVAMP/results/svamp_claude-3.5_cot-got_2024-08-19_17-05-38/got'
    problems_solved = count_problems_solved(directory_path)
    print(f"Total problems solved: {problems_solved}")