import json
import random

def split_dataset(file_path, split_ratio=0.5, random_seed=42):
    # Read the JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Shuffle the data
    random.shuffle(data)
    
    # Calculate the split point
    split_point = int(len(data) * split_ratio)
    
    # Split the data
    part1 = data[:split_point]
    part2 = data[split_point:]
    
    # Save the split datasets as JSON
    with open('dataset_part1.json', 'w') as f:
        json.dump(part1, f, indent=2)
    
    with open('dataset_part2.json', 'w') as f:
        json.dump(part2, f, indent=2)
    
    print(f"Dataset split complete.")
    print(f"Part 1 entries: {len(part1)}")
    print(f"Part 2 entries: {len(part2)}")

# Usage example
file_path = '/p/llmreliability/test_repos/truthfulqa_claude35_results.json'  # Replace with your actual file path
split_dataset(file_path)