import json
import re

# Helper function to normalize text by removing unnecessary whitespace and converting to lowercase
def normalize(text):
    return re.sub(r'\s+', '', text).strip().lower()

# Function to extract the numerical value (or key phrases) from the text
def extract_key_info(text):
    # This regex looks for numbers, which are often the key info in math problems
    numbers = re.findall(r'\d+', text)
    return numbers

# Load the JSON file
with open('/p/llmreliability/test_repos/graph-of-thoughts/examples/GSM8k/results/chatgpt_got_2024-08-19_13-43-19/cot/0.json', 'r') as f:
    data = json.load(f)

# Iterate through the items in the JSON data
for item in data:
    if item.get('operation') == 'ground_truth_evaluator':
        # Extract the generated answer and the ground truth answer
        generated_answer = item['thoughts'][0]['current']
        ground_truth_answer = item['thoughts'][0]['problem']['answer']
        
        # Normalize and extract key numerical information
        norm_generated_answer = extract_key_info(normalize(generated_answer))
        norm_ground_truth_answer = extract_key_info(normalize(ground_truth_answer))
        
        # Compare the normalized key info (numbers in this case)
        if norm_generated_answer == norm_ground_truth_answer:
            item['problem_solved'][0] = True
        else:
            item['problem_solved'][0] = False

# Save the updated JSON file
with open('updated_0.json', 'w') as f:
    json.dump(data, f, indent=4)
