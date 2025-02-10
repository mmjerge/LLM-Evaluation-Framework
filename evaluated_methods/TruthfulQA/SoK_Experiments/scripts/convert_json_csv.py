import json
import csv
import sys

def convert_json_to_csv(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as jsonfile:
        data = json.load(jsonfile)


    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['Question', 'Llama-3.1-8b_Answer'])
        writer.writeheader()
        
        # Write the data
        for item in data:
            writer.writerow({
                'Question': item['question'],
                'Llama-3.1-8b_Answer': item['model_response']['output']
            })

    print(f"CSV file '{output_file}' has been created successfully.")

if __name__ == "__main__":
    input_file = "/p/llmreliability/test_repos/TruthfulQA/SoK_Experiments/scripts/llama-3.1-8b_truthfulqa_react_results.json"
    output_file = "llama-3.1-8b_responses.csv"
    
    convert_json_to_csv(input_file, output_file)