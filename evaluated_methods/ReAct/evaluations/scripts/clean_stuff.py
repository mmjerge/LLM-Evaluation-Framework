import json

def remove_entries_with_stopped_agent(json_file, output_file):
    with open(json_file, 'r') as infile:
        data = json.load(infile)

    # Filter out the entries where the model_response contains "Agent stopped due to"
    filtered_data = [entry for entry in data if "Agent stopped due to" not in entry.get("model_response", "")]

    # Write the filtered data back to a new file
    with open(output_file, 'w') as outfile:
        json.dump(filtered_data, outfile, indent=4)

# Example usage:
remove_entries_with_stopped_agent('/p/llmreliability/test_repos/ReAct/SoK_Experiments/scripts/gpt35_aqua_react_results_random_150_1.json', 'output.json')