def extract_answer(text):
  # Extract number after ####
    if '####' not in text:
        return None
    try:
        return int(text.split('####')[1].strip())
    except:
        return None

def extract_model_answer(text):
    # Extract number after "Final answer:" or similar
    if 'final answer:' not in text.lower():
        return None
    try:
        ans = text.lower().split('final answer:')[1].strip().split()[0]
        # Remove any $ signs and convert to int
        ans = ans.replace('$','').replace(',','')
        return int(float(ans))
    except:
        return None

# Load the JSON file
with open('cot_gsm-symbolic_eval_results.json', 'r') as f:
    data = json.load(f)

correct = 0
total = 0
evaluated = []

for item in data:
    ground_truth = extract_answer(item['ground_truth'])
    prediction = extract_model_answer(item['model_prediction'])
    
    if ground_truth is not None and prediction is not None:
        total += 1
        if ground_truth == prediction:
            correct += 1
        evaluated.append({
            'id': item['id'],
            'question': item['question'][:50] + '...',
            'ground_truth': ground_truth,
            'prediction': prediction,
            'correct': ground_truth == prediction
        })

accuracy = (correct / total) * 100 if total > 0 else 0

print(f"\nResults:")
print(f"Total evaluated: {total}")
print(f"Correct: {correct}")
print(f"Accuracy: {accuracy:.1f}%")

print("\nFirst few evaluations:")
for item in evaluated[:5]:
    print(f"\nQ{item['id']}: {item['question']}")
    print(f"Ground truth: {item['ground_truth']}")
    print(f"Prediction: {item['prediction']}")
    print(f"Correct: {item['correct']}")
