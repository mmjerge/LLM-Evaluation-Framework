import os
import json
from tqdm import tqdm
from datasets import load_dataset
from openai import OpenAI

dataset = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")

def get_response(question, temp=0):
    client = OpenAI()
    response = client.chat.completions.create(
      model="gpt-4o",
      messages=[
          {
              "role": "user",
              "content": question
          }
      ],
      temperature=temp
    )
    return response.choices[0].message.content

results = []

for example in tqdm(dataset):
    question = example["question"]
    answer = get_response(question)
    results.append({"question": question, "model_answer": answer})

# Save the results to a JSON file
output_file = "truthfulqa_gpt4o_results.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=4)

print(f"Results saved to {output_file}")
