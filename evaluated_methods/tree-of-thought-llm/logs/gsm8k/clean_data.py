import json
import re

# Load the JSON data
file_path = '/p/llmreliability/test_repos/tree-of-thought-llm/logs/gsm8k/claude-3-5-sonnet-20240620_0.7_sample1_value1_greedy1_random150.json'
with open(file_path, 'r') as f:
    data = json.load(f)

# Function to extract the correct answer after "####"
def extract_correct_answer(entry):
    correct_answer_full = entry['correct_answer']
    correct_answer = correct_answer_full.split("####")[-1].strip()
    entry['correct_answer'] = correct_answer
    return entry

# Function to clean and extract the numeric value, handling commas
def clean_and_extract_number(text):
    # Remove any commas from the text
    text = text.replace(',', '')
    # Use regex to find the first numeric pattern (including those with decimal points)
    match = re.search(r'[-+]?\d*\.\d+|\d+', text)
    return match.group(0) if match else text


# Function to update both model_response and correct_answer fields
def clean_entry(entry):
    entry['model_response'] = clean_and_extract_number(entry['model_response'])
    entry = extract_correct_answer(entry)
    return entry

# Clean the dataset
cleaned_data = [clean_entry(entry) for entry in data]

# Save the updated data back to a JSON file
output_file_path = 'cleaned_claude_responses.json'
with open(output_file_path, 'w') as f:
    json.dump(cleaned_data, f, indent=4)

print(f"Correct answers updated and saved to {output_file_path}")