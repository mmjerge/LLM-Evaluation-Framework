import torch
from datasets import load_dataset
from mistralai import Mistral
import json
import os

# Load the Mistral 8x22B model and tokenizer from Hugging Face
model_name = "open-mixtral-8x22b"

# Load the TruthfulQA dataset with the "generation" config
dataset = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")

# Define a function to generate answers using the model
def generate_answer(prompt, model_name, max_length=50):
    api_key = os.environ["MISTRAL_API_KEY"]
    model = model_name

    client = Mistral(api_key=api_key)

    chat_response = client.chat.complete(
        model = model_name,
        messages = [
            {
                "role": "user",
                "content": prompt,
            },
        ]
    )

    return chat_response.choices[0].message.content

results = []

# Iterate over the dataset and generate answers
for example in dataset:
    question = example["question"]
    answer = generate_answer(question, model_name)
    results.append({"question": question, "model_answer": answer})

# Save the results to a JSON file
output_file = "truthfulqa_mixtral_8x22b_results.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=4)

print(f"Results saved to {output_file}")
