import json

# Load the dataset from a JSON file
with open('/p/llmreliability/test_repos/tree-of-thought-llm/logs/svamp/claude-3-5-sonnet-20240620_0.7_sample1_value1_greedy1_random150.json', 'r') as file:
    data = json.load(file)

def calculate_accuracy(data):
    correct_count = 0
    total_count = 0

    for entry in data:
        if 'extracted_answer' in entry and 'correct_answer' in entry:
            total_count += 1
            if entry['extracted_answer'] == entry['correct_answer']:
                correct_count += 1

    if total_count == 0:
        return 0.0  # Avoid division by zero

    accuracy = correct_count / total_count
    return accuracy

accuracy = calculate_accuracy(data)
print(f'Accuracy: {accuracy * 100:.2f}%')