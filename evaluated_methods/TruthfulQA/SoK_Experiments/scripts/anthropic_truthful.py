import anthropic
from datasets import load_dataset
import json
from tqdm import tqdm

# Initialize the Anthropic client
client = anthropic.Anthropic(api_key="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")

# Load the TruthfulQA dataset with the "generation" config
dataset = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")

# Define a function to generate answers using the Anthropic API
def generate_answer(prompt, max_tokens=50):
    client = anthropic.Anthropic(api_key="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")
    message = client.messages.create(
    max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="claude-3-5-sonnet-20240620",
    )
    return message.content[0].text

# Create a list to store the results
results = []

for example in tqdm(dataset):
    question = example["question"]
    answer = generate_answer(question)
    
    # Store the additional fields from the dataset along with the model-generated answer
    result = {
        "type": example["type"],
        "category": example["category"],
        "question": question,
        "best_answer": example["best_answer"],
        "correct_answers": example["correct_answers"],
        "model_answer": answer
    }
    results.append(result)

# Save the results to a JSON file
output_file = "truthfulqa_claude35_results.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=4)

print(f"Results saved to {output_file}")