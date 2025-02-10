import json
import re

def parse_answer(question, model_answer):
    # Extract multiple choice options from the question
    options = re.findall(r'\(([A-D])\)\s*(\d+)', question)
    options_dict = {value: key for key, value in options}

    # List of patterns to match, in order of preference
    patterns = [
        r'\*\*\(?([A-D])\)?\s*[^*]+\*\*\s*$',  # Matches **(A) Something** or **A) Something** at the end of the string
        r'Therefore,\s*the\s*(only\s*)?correct\s*answer\s*is:?\s*\n?\s*\*?\(?([A-D])\)?',  # Matches various "Therefore, the correct answer is: (A)" patterns
        r'So,?\s*the\s*(correct\s*)?answer\s*is:?\s*\n?\s*\*?\(?([A-D])\)?',  # Matches "So, the answer is: (A)" patterns
        r'The\s*(correct\s*)?answer\s*is:?\s*\n?\s*\*?\(?([A-D])\)?',  # Matches "The correct answer is: (A)" patterns
        r'\(([A-D])\)\s*[^(]*$',  # Matches (A) Something at the end of the string
        r'\$\\boxed\{([A-D])\}\$',  # Matches $\boxed{A}$
        r'\$\\boxed\{(\d+)\}\$',  # Matches $\boxed{2}$
        r'The\s*final\s*answer\s*is:?\s*\n?\s*\*?\(?([A-D])\)?',  # Matches "The final answer is: (A)"
        r'The\s*final\s*answer\s*is:?\s*\n?\s*\*?\(?(\d+)\)?',  # Matches "The final answer is: 2"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, model_answer, re.IGNORECASE | re.DOTALL)
        if match:
            answer = match.group(1) if len(match.groups()) == 1 else match.group(2)
            if answer in 'ABCD':
                return answer
            elif answer in options_dict:
                return options_dict[answer]
    
    # If no match found, try to find any mention of A, B, C, D, or the corresponding numbers in the last sentence
    sentences = model_answer.split('.')
    if sentences:
        last_sentence = sentences[-1]
        match = re.search(r'\b([A-D])\b|\b(\d+)\b', last_sentence)
        if match:
            answer = match.group(1) or match.group(2)
            if answer in 'ABCD':
                return answer
            elif answer in options_dict:
                return options_dict[answer]
    
    return None

def process_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    processed_data = {}
    
    for subject, subject_data in data.items():
        if isinstance(subject_data, dict) and 'questions' in subject_data:
            processed_questions = []
            for question in subject_data['questions']:
                parsed_answer = parse_answer(question['question'], question['model_answer'])
                question['parsed_answer'] = parsed_answer
                question['is_correct'] = (parsed_answer == question['correct_answer']) if parsed_answer else False
                processed_questions.append(question)
            
            correct_count = sum(q['is_correct'] for q in processed_questions)
            total_count = len(processed_questions)
            accuracy = correct_count / total_count if total_count > 0 else 0
            
            processed_data[subject] = {
                'accuracy': accuracy,
                'questions': processed_questions
            }
        else:
            processed_data[subject] = subject_data
    
    return processed_data

def print_results(processed_data):
    for subject, subject_data in processed_data.items():
        print(f"\nSubject: {subject}")
        if isinstance(subject_data, dict) and 'questions' in subject_data:
            print(f"Accuracy: {subject_data['accuracy']:.2f}")
            for i, question in enumerate(subject_data['questions'], 1):
                print(f"\nQuestion {i}:")
                print(f"Parsed Answer: {question.get('parsed_answer', 'N/A')}")
                print(f"Correct Answer: {question['correct_answer']}")
                print(f"Is Correct: {question.get('is_correct', 'N/A')}")
        else:
            print(f"Data: {subject_data}")

# Usage
file_path = '/p/llmreliability/test_repos/chain-of-thought-hub/MMLU/llama31_mmlu_results.json'
processed_data = process_json(file_path)

# Print results
print_results(processed_data)

# Save the processed data back to a JSON file
with open('processed_data.json', 'w') as file:
    json.dump(processed_data, file, indent=2)