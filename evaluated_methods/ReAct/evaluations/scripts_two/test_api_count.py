import json

def analyze_api_calls(filename):
    total_api_calls = 0
    question_count = 0
    
    with open(filename, 'r') as file:
        for line in file:
            try:
                # Parse each line as JSON
                data = json.loads(line)
                
                # Extract API calls if the field exists
                if 'api_calls' in data:
                    total_api_calls += data['api_calls']
                    question_count += 1
            except json.JSONDecodeError:
                # Skip lines that aren't valid JSON
                continue
    
    # Calculate average if we have questions
    if question_count > 0:
        average_calls = total_api_calls / question_count
    else:
        average_calls = 0
    
    print(f"File: {filename}")
    print(f"Total questions: {question_count}")
    print(f"Total API calls: {total_api_calls}")
    print(f"Average API calls per question: {average_calls:.2f}")
    print("-" * 50)

analyze_api_calls("/scratch/mj6ux/Projects/LLM-Evaluation-Framework/evaluated_methods/ReAct/evaluations/results/GSM8K/anthropic_api_count_gsm8k_react_results_multiple_choice.jsonl")
analyze_api_calls("/scratch/mj6ux/Projects/LLM-Evaluation-Framework/evaluated_methods/ReAct/evaluations/results/GSM8K/_gsm8k_react_results_gpt4o_multiple_choice.jsonl")
analyze_api_calls("/scratch/mj6ux/Projects/LLM-Evaluation-Framework/evaluated_methods/ReAct/evaluations/results/MMLU/mmlu_api_claude_claude_3_5_sonnet_20240620.jsonl")
analyze_api_calls("/scratch/mj6ux/Projects/LLM-Evaluation-Framework/evaluated_methods/ReAct/evaluations/results/MMLU/mmlu_api_openai_gpt_4o.jsonl")
analyze_api_calls("/scratch/mj6ux/Projects/LLM-Evaluation-Framework/evaluated_methods/ReAct/evaluations/results/AQUA/aqua_api_claude_3_5_sonnet_20240620.jsonl")
analyze_api_calls("/scratch/mj6ux/Projects/LLM-Evaluation-Framework/evaluated_methods/ReAct/evaluations/results/AQUA/aqua_api_react_results_openai_gpt_4o.jsonl")
