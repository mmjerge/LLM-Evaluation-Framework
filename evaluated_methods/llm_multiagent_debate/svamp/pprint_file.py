import json
import re

def find_last_boxed_number(text):
    boxed_numbers = re.findall(r'\\boxed{(\d+(?:\.\d+)?)}', text)
    return float(boxed_numbers[-1]) if boxed_numbers else None

def add_final_answer_and_prettify(input_file, output_file):
    # Read the input JSON file
    with open(input_file, 'r') as file:
        data = json.load(file)
    
    # Find the last assistant message in the contexts
    last_assistant_message = None
    for context in data.get('contexts', []):
        for message in reversed(context):
            if message.get('role') == 'assistant':
                last_assistant_message = message.get('content')
                break
        if last_assistant_message:
            break
    
    # Find the last boxed number in the last assistant message
    final_answer = find_last_boxed_number(last_assistant_message) if last_assistant_message else None
    
    # Add the final_answer key
    if final_answer is not None:
        data['final_answer'] = final_answer
    
    # Write the prettified JSON to the output file
    with open(output_file, 'w') as file:
        json.dump(data, file, indent=2, sort_keys=True)

    print(f"Added final_answer and prettified JSON has been written to {output_file}")
    if final_answer is not None:
        print(f"Final answer parsed: {final_answer}")
    else:
        print("No final answer found in the last assistant message.")

# Usage
input_file = '/p/llmreliability/test_repos/llm_multiagent_debate/svamp/svamp_3_2_openai_with_counterfactuals.json'  # Replace with your input file path
output_file = 'prettified_output.json'  # Replace with your desired output file path

add_final_answer_and_prettify(input_file, output_file)