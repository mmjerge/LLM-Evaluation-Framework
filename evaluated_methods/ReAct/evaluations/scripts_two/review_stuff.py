import json
import re

def extract_numeric_value(text):
    match = re.search(r"[-+]?\d*\.\d+|\d+", text)
    return float(match.group()) if match else None

def evaluate_accuracy_with_options(json_file):
    with open(json_file, 'r') as infile:
        data = json.load(infile)

    total_questions = len(data)
    total_correct = 0

    for entry in data:
        model_response = entry.get("model_response", "")
        correct_answer = entry.get("correct_answer", "")
        options = entry.get("options", [])

        # Extract numeric value from model response
        extracted_model_response = extract_numeric_value(model_response)

        if extracted_model_response is not None:
            # Find the correct option's numeric value
            correct_option_text = next((opt for opt in options if opt.startswith(correct_answer)), None)

            if correct_option_text:
                # Extract the numeric value from the correct option
                correct_numeric_value = extract_numeric_value(correct_option_text)

                if correct_numeric_value is not None and extracted_model_response == correct_numeric_value:
                    total_correct += 1

    total_incorrect = total_questions - total_correct

    return {
        "total_questions": total_questions,
        "total_correct": total_correct,
        "total_incorrect": total_incorrect
    }

print(result)



