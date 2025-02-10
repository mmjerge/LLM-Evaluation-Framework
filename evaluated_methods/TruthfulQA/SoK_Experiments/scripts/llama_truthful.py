import os
import json
from tqdm import tqdm
from together import Together
from datasets import load_dataset

# Initialize the Together client
client = Together(api_key=os.environ.get('TOGETHER_API_KEY'))

# Load the TruthfulQA dataset with the "generation" config
dataset = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")

# Define a function to generate answers using the Together API
def generate_answer(prompt, max_length=256):
    response = client.chat.completions.create(
        model="mistralai/Mixtral-8x22B-Instruct-v0.1",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_length,
    )
    return response.choices[0].message.content

# Create a list to store the results
results = []

for example in tqdm(dataset):
    question = example["question"]
    answer = generate_answer(question)
    results.append({"question": question, "model_answer": answer})

# Save the results to a JSON file
output_file = "truthfulqa_llama31_8b_results.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=4)

print(f"Results saved to {output_file}")

