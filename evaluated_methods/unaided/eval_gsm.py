import json
import re
import pandas as pd
import numpy as np

def extract_answer(text):
    """Extract the numerical answer from ground truth."""
    match = re.search(r'####\s*([\d,]+)', text)
    if match:
        return int(match.group(1).replace(',', ''))
    return None

def extract_prediction(text):
    """Extract the numerical answer from model prediction."""
    patterns = [
        r'Final answer:?\s*([\d,]+)',
        r'The answer is:?\s*([\d,]+)',
        r'Therefore,? the answer is:?\s*([\d,]+)',
        r'So,? the answer is:?\s*([\d,]+)',
        r'Thus,? the answer is:?\s*([\d,]+)',
        r'[\.\n](\d+)$'  # Last number in text
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return int(match.group(1).replace(',', ''))
    return None

with open('cot_gsm8k_eval_results.json', 'r') as f:
    data = json.load(f)

rows = []
for idx, item in enumerate(data):
    ground_truth = extract_answer(item['ground_truth'])
    prediction = extract_prediction(item['model_prediction'])
    
    rows.append({
        'index': idx,
        'question': item['question'],
        'ground_truth': ground_truth,
        'prediction': prediction,
        'is_correct': ground_truth == prediction if (ground_truth is not None and prediction is not None) else None,
        'is_evaluated': (ground_truth is not None and prediction is not None)
    })

df = pd.DataFrame(rows)

total_problems = len(df)
evaluated_problems = df['is_evaluated'].sum()
correct_predictions = df['is_correct'].sum()

accuracy_all = (correct_predictions / total_problems) * 100
accuracy_evaluated = (correct_predictions / evaluated_problems) * 100 if evaluated_problems > 0 else 0

print("\nAccuracy Metrics:")
print(f"Total problems: {total_problems}")
print(f"Successfully evaluated problems: {evaluated_problems}")
print(f"Failed to evaluate: {total_problems - evaluated_problems}")
print(f"Correct predictions: {correct_predictions}")
print(f"Accuracy (all problems): {accuracy_all:.2f}%")
print(f"Accuracy (evaluated problems only): {accuracy_evaluated:.2f}%")

print("\nFirst few rows of the DataFrame:")
display_df = df.copy()
display_df['question'] = display_df['question'].str[:50] + '...'
print(display_df.head())

print("\nSample of unevaluated problems:")
unevaluated = df[~df['is_evaluated']].head()
unevaluated['question'] = unevaluated['question'].str[:50] + '...'
print(unevaluated)

print("\nSample of incorrect predictions:")
incorrect = df[df['is_correct'] == False].head()
incorrect['question'] = incorrect['question'].str[:50] + '...'
print(incorrect)
