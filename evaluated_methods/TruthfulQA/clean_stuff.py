import json
import csv

def extract_final_answer(responses):
    last_response = responses[-1]
    assistant_content = last_response[-1]['content']
    final_answer_start = assistant_content.rfind("Final Answer:")
    if final_answer_start != -1:
        return assistant_content[final_answer_start + 14:].strip()
    else:
        return assistant_content.strip()

with open('/p/llmreliability/test_repos/TruthfulQA/mixtral_truthfulqa_3_2_openai.json', 'r') as file:
    data = json.load(file)

csv_data = []
for question, responses in data.items():
    if isinstance(responses, list) and len(responses) > 0 and isinstance(responses[0], list):
        final_answer = extract_final_answer(responses[0])
    else:
        final_answer = str(responses)
    csv_data.append([question, final_answer])

# Write to CSV
with open('output.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["Question", "Mixtral-8x22b_Answer"])
    writer.writerows(csv_data)

print("CSV file has been created successfully.")