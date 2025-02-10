import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def get_complete_results(base_directory):
    results_complete = {}
    for folder_name in os.listdir(base_directory):
        folder_path = os.path.join(base_directory, folder_name)
        if os.path.isdir(folder_path):
            results_complete[folder_name] = []
            for file_name in os.listdir(folder_path):
                if file_name.endswith(".json"):
                    file_path = os.path.join(folder_path, file_name)
                    with open(file_path, "r") as f:
                        data = json.load(f)
                        results_complete[folder_name].append(
                            {"key": int(file_name.split(".")[0]), "data": data}
                        )
        for key in results_complete.keys():
            results_complete[key] = sorted(
                results_complete[key], key=lambda x: x["key"]
            )
    return results_complete

def get_final_scores(results_complete):
    scores = {}
    for method in results_complete.keys():
        scores[method] = []
        for result in results_complete[method]:
            score = 100
            solved = False
            cost = 1
            prompt_tokens = 0
            completion_tokens = 0
            for op in result["data"]:
                if "operation" in op and op["operation"] == "ground_truth_evaluator":
                    try:
                        score = min(op["scores"])
                        solved = any(op["problem_solved"])
                    except:
                        continue
                if "cost" in op:
                    cost = op["cost"]
                    prompt_tokens = op["prompt_tokens"]
                    completion_tokens = op["completion_tokens"]
            scores[method].append(
                [result["key"], score, solved, prompt_tokens, completion_tokens, cost]
            )
        scores[method] = sorted(scores[method], key=lambda x: x[0])
    return scores

def get_plotting_data(base_directory):
    results_complete = get_complete_results(base_directory)
    scores = get_final_scores(results_complete)
    results_plotting = {
        method: {
            "scores": [x[1] for x in scores[method]],
            "solved": sum([1 for x in scores[method] if x[2]]),
            "costs": [x[5] for x in scores[method]],
        }
        for method in scores.keys()
    }
    return results_plotting

def prepare_data_for_seaborn(results, methods_order):
    data = []
    for method in methods_order:
        if method in results:
            scores = results[method]["scores"]
            data.extend([(method, score) for score in scores])
    
    df = pd.DataFrame(data, columns=['Method', 'Incorrectly Sorted Items'])
    
    cost_data = {method: sum(results[method]['costs']) for method in methods_order if method in results}
    solved_data = {method: results[method]['solved'] for method in methods_order if method in results}
    
    return df, cost_data, solved_data

def plot_multiple_models(
    base_directory,
    models,
    methods_order=["io", "cot", "tot", "tot2", "got"],
    num_ndas=32,
    y_lower=0,
    y_upper=32,
    display_left_ylabel=True,
):
    sns.set_style("whitegrid")
    plt.figure(figsize=(16, 10), dpi=150)

    all_data = []

    for model in models:
        model_dir = os.path.join(base_directory, f"{model}_io-cot-tot-tot2-got_{num_ndas:03d}")
        results = get_plotting_data(model_dir)
        
        df, _, _ = prepare_data_for_seaborn(results, methods_order)
        df['Model'] = model
        all_data.append(df)

    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Create a custom palette with distinct colors
    palette = sns.color_palette("husl", n_colors=len(models))
    
    # Box plot with swarm plot overlay
    ax = sns.boxplot(x='Method', y='Incorrectly Sorted Items', hue='Model', data=combined_df,
                     order=methods_order, palette=palette, whis=[5, 95], width=0.6)
    
    sns.swarmplot(x='Method', y='Incorrectly Sorted Items', hue='Model', data=combined_df,
                  order=methods_order, palette=palette, dodge=True, size=4, alpha=0.6)

    # Customize the plot
    plt.ylim(y_lower, y_upper)
    if display_left_ylabel:
        plt.ylabel("Incorrectly Sorted Items", fontsize=14, fontweight='bold')
    plt.xlabel("Approach", fontsize=14, fontweight='bold')
    
    method_labels = ["IO", "CoT", "ToT", "ToT2", "GoT"]
    plt.xticks(range(len(method_labels)), method_labels, fontsize=12)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Adjust legend
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles[:len(models)], labels[:len(models)], title='Model', 
               bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, title_fontsize=12)

    # Add a title
    plt.title("Comparison of Sorting Methods Across Models", fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(f"improved_sorting_comparison_{num_ndas}.pdf", bbox_inches="tight")

# Example usage
base_directory = "/p/llmreliability/test_repos/graph-of-thoughts/examples/sorting/results"
models = ["Claude-3.5", "GPT-3.5-turbo", "GPT-4o", "Llama-3.1-8B", "Mixtral-8x22b"]

plot_multiple_models(
    base_directory=base_directory,
    models=models,
    num_ndas=32,
    y_upper=32,
    display_left_ylabel=True,
)