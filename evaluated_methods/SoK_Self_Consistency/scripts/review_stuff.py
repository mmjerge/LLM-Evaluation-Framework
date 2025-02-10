import pandas as pd
import json

# Read the NDJSON file
with open('/p/llmreliability/test_repos/llmpromptboosting/SoK_Self_Consistency/scripts/claude_combined.jsonl', 'r') as file:
    df = pd.DataFrame([json.loads(line) for line in file])

# Perform the initial transformations
df['is_correct'] = df['best_answer'] == df['true_answer']
df['n_experts'] = df['n_experts'].astype('category')
df['n_attempts'] = df['n_attempts'].astype('category')

# Print the total number of rows
print(f"Total number of rows: {len(df)}")

# Analysis for n_experts = 1 and n_attempts = 1
filtered_df = df[(df['n_experts'] == 1) & (df['n_attempts'] == 1)]
if len(filtered_df) > 0:
    pct_correct = filtered_df['is_correct'].mean() * 100
    print(f"\nPercentage correct for n_experts=1 and n_attempts=1: {pct_correct:.2f}%")
else:
    print("\nNo rows match the condition n_experts=1 and n_attempts=1")

print(f"Total rows in filtered DataFrame: {len(filtered_df)}")
print("Value counts of 'is_correct' in filtered DataFrame:")
print(filtered_df['is_correct'].value_counts(normalize=True) * 100)

# Group by n_attempts and calculate percentage correct
summary = df.groupby('n_attempts')['is_correct'].mean().reset_index()
summary['pct_correct'] = summary['is_correct'] * 100

# Format the percentage to two decimal places
summary['pct_correct'] = summary['pct_correct'].round(2)

# Rename columns for clarity
summary = summary.rename(columns={'n_attempts': '# of Attempts', 'pct_correct': '% Correct Answers'})

# Drop the 'is_correct' column as it's no longer needed
summary = summary.drop('is_correct', axis=1)

# Display the summary
print("\nSummary of correct answers by number of attempts:")
print(summary.to_string(index=False))