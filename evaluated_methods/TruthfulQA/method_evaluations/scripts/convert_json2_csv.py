import json
import pandas as pd

# Load the JSON file
with open('data/generative_agent_responses.json', 'r') as file:
    data = json.load(file)

# Define the models you want to evaluate
models_to_evaluate = ['gpt-3.5-turbo', 'gpt-4', 'mistral-base', 'llama-base']

# Iterate over each model and save the responses to a separate CSV file
for model in models_to_evaluate:
    rows = []
    for question, responses in data.items():
        if model in responses:
            rows.append({
                "Question": question,
                f"{model}_Answer": responses[model]
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(rows)
    
    # Save to CSV
    output_csv = f'{model}_responses.csv'
    df.to_csv(output_csv, index=False)
    print(f"Responses for {model} saved to {output_csv}")