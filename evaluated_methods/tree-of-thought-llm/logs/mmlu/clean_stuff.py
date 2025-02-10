import json
import re
from collections import Counter, defaultdict

def clean_and_calculate_accuracy(data):
    correct_count = 0
    total_count = 0
    confusion_matrix = defaultdict(lambda: defaultdict(int))
    incorrect_examples = []
    no_extraction_count = 0

    for item in data:
        # Extract the letter from the model response
        model_response = item['model_response']
        extracted_letter = extract_letter(model_response)

        # Add the extracted letter to a new key
        item['extracted_letter_value'] = extracted_letter

        # Compare with the correct answer
        correct_answer = item['correct_answer'].lower()
        
        print(f"Question {item['dataset_index']}:")
        print(f"Correct answer: {correct_answer}")
        print(f"Extracted answer: {extracted_letter}")
        
        if extracted_letter:
            extracted_letter = extracted_letter.lower()
            confusion_matrix[correct_answer][extracted_letter] += 1
            if extracted_letter == correct_answer:
                correct_count += 1
                print("Correct!")
            else:
                incorrect_examples.append({
                    'dataset_index': item['dataset_index'],
                    'question': item['question'],
                    'correct_answer': correct_answer,
                    'model_answer': extracted_letter,
                    'model_response': model_response
                })
                print("Incorrect.")
        else:
            no_extraction_count += 1
            confusion_matrix[correct_answer]['no_answer'] += 1
            incorrect_examples.append({
                'dataset_index': item['dataset_index'],
                'question': item['question'],
                'correct_answer': correct_answer,
                'model_answer': 'No answer extracted',
                'model_response': model_response
            })
            print("No answer extracted.")
        
        total_count += 1
        print("\n")

    # Calculate accuracy
    accuracy = correct_count / total_count if total_count > 0 else 0
    print(f"Total questions: {total_count}")
    print(f"Correct answers: {correct_count}")
    print(f"No extraction: {no_extraction_count}")
    print(f"Accuracy: {accuracy:.2%}")

    return data, accuracy, confusion_matrix, incorrect_examples

def extract_letter(response):
    # Define all patterns to search for A, B, C, D, or E
    patterns = [
        r'####\s*([A-Ea-e])\s*$',  # #### X at the end of the response
        r'The correct answer is\s*\(?([A-Ea-e])\)?',  # The correct answer is (X) or The correct answer is X
        r'####\s*\[([A-Ea-e])\]',  # #### [X]
        r'^\s*\(?([A-Ea-e])\)',  # (X) at the beginning of the response
        r'\(([A-Ea-e])\)',  # (X) anywhere in the response
        r'\b([A-Ea-e])\b',  # A, B, C, D, or E as a word boundary
        r'\[([A-Ea-e])\]'  # [X] anywhere in the response
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            return match.group(1).upper()  # Ensure the returned letter is uppercase
    
    return None  # If no valid letter is found

def print_accuracy_report(accuracy, confusion_matrix, incorrect_examples):
    print(f"Accuracy: {accuracy:.2%}")
    print("\nConfusion Matrix:")
    all_answers = sorted(set(key for subdict in confusion_matrix.values() for key in subdict.keys()))
    
    # Print header
    print("   " + " ".join(f"{ans:>5}" for ans in all_answers))
    
    # Print rows
    for true_answer in sorted(confusion_matrix.keys()):
        row = [true_answer] + [str(confusion_matrix[true_answer][ans]).rjust(5) for ans in all_answers]
        print(" ".join(row))
    
    print("\nMost Common Incorrect Answers:")
    incorrect_counter = Counter((item['correct_answer'], item['model_answer']) for item in incorrect_examples)
    for (correct, incorrect), count in incorrect_counter.most_common(5):
        print(f"  Correct: {correct}, Model: {incorrect}, Count: {count}")
    
    print("\nSample Incorrect Examples:")
    for example in incorrect_examples[:5]:  # Print first 5 examples
        print(f"  Q: {example['question'][:100]}...")  # Truncate long questions
        print(f"     Correct: {example['correct_answer']}, Model: {example['model_answer']}")
        print()

# Load the data
with open('/p/llmreliability/test_repos/tree-of-thought-llm/logs/mmlu/gpt-4o_0.7_sample1_value1_greedy1_random150.json', 'r') as f:
    data = json.load(f)

# Clean data and calculate accuracy
cleaned_data, accuracy, confusion_matrix, incorrect_examples = clean_and_calculate_accuracy(data)

# Save the cleaned data
with open('cleaned_data.json', 'w') as f:
    json.dump(cleaned_data, f, indent=2)

# Print the detailed accuracy report
print_accuracy_report(accuracy, confusion_matrix, incorrect_examples)