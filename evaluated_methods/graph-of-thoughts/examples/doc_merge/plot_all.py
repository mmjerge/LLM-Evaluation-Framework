import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
import csv

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
            score = 0
            solved = False
            cost = 1
            prompt_tokens = 0
            completion_tokens = 0
            for op in reversed(result["data"]):
                if "cost" in op:
                    cost = op["cost"]
                    prompt_tokens = op["prompt_tokens"]
                    completion_tokens = op["completion_tokens"]
                if "operation" in op and op["operation"] == "score":
                    try:
                        score = max(op["scores"])
                        break
                    except:
                        continue
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
            costs = results[method]["costs"]
            data.extend([(method, score, cost) for score, cost in zip(scores, costs)])
    
    df = pd.DataFrame(data, columns=['Method', 'Score', 'Cost'])
    
    cost_data = {method: sum(results[method]['costs']) for method in methods_order if method in results}
    solved_data = {method: results[method]['solved'] for method in methods_order if method in results}
    
    return df, cost_data, solved_data

def plot_multiple_models(
    base_directory,
    models,
    methods_order=["io", "cot", "tot", "got"],
    num_ndas=4,
    y_lower=5,
    y_upper=10,
    cost_upper=15,
    display_solved=False,
    display_left_ylabel=True,
    display_right_ylabel=False,
):
    sns.set(style="whitegrid", palette="pastel", rc={"axes.labelsize": 16, "axes.titlesize": 20, "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 12})

    fig, ax = plt.subplots(figsize=(20, 12), dpi=150)

    all_data = []
    all_costs = {}
    all_solved = {}

    for model in models:
        model_dir = os.path.join(base_directory, f"{model}_io-cot-tot-got-got2_doc_merge")
        results = get_plotting_data(model_dir)
        
        df, cost_data, solved_data = prepare_data_for_seaborn(results, methods_order)
        df['Model'] = model
        all_data.append(df)
        all_costs[model] = cost_data
        all_solved[model] = solved_data

    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Export data to CSV
    combined_df.to_csv(f"doc_merge_comparison_data_{num_ndas}.csv", index=False)
    
    # Export costs and solved data to CSV
    with open(f"doc_merge_comparison_costs_solved_{num_ndas}.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Model', 'Method', 'Total Cost', 'Solved'])
        for model in models:
            for method in methods_order:
                writer.writerow([model, method, all_costs[model].get(method, 0), all_solved[model].get(method, 0)])

    # Box plot with swarm plot
    sns.boxplot(x='Method', y='Score', hue='Model', data=combined_df, order=methods_order, ax=ax, palette="Set2")
    sns.swarmplot(x='Method', y='Score', hue='Model', data=combined_df, order=methods_order, dodge=True, ax=ax, palette="Set2", linewidth=1, edgecolor='gray')
    
    # # Add statistical significance
    # for i, method in enumerate(methods_order):
    #     method_data = combined_df[combined_df['Method'] == method]
    #     models_data = [method_data[method_data['Model'] == model]['Score'] for model in models]
    #     f_statistic, p_value = stats.f_oneway(*models_data)
    #     ax.text(i, y_upper + 1.5, f'p={p_value:.3f}', ha='center', va='bottom', fontsize=14, fontweight='bold')

    # Remove the duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    ax.legend().remove()  # Remove the default legend

    # Add cost information
    ax2 = ax.twinx()
    for i, model in enumerate(models):
        costs = [all_costs[model].get(method, 0) for method in methods_order]
        ax2.plot(range(len(methods_order)), costs, marker='o', linestyle='--', label=f'{model} Cost')
    ax2.set_ylabel('Total Cost', fontsize=30, labelpad=15)
    ax2.set_ylim(0, cost_upper)

    # Create a single legend for both plots
    handles_ax2, labels_ax2 = ax2.get_legend_handles_labels()
    all_handles = handles[:len(models)] + handles_ax2
    all_labels = labels[:len(models)] + labels_ax2

    # Position the legend outside the plot
    fig.legend(all_handles, all_labels, loc='upper left', bbox_to_anchor=(1, 0.5), borderaxespad=0., ncol=1, fontsize=16, markerscale=1.5)

    # Adjust the layout to make room for the legend
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)  # Adjust this value to fine-tune legend placement

    # Customize the plot
    ax.set_ylim(y_lower, y_upper + 2 if display_solved else y_upper)
    if display_left_ylabel:
        ax.set_ylabel("Accuracy", fontsize=30, labelpad=15)
    ax.set_xlabel("Approach", fontsize=30, labelpad=15)
    
    method_labels = ["Input-Output", "Chain-of-Thought", "Tree-of-Thoughts", "Graph-of-Thoughts", "Graph-of-Thoughts2"]
    ax.set_xticklabels(method_labels, fontsize=25, fontweight='bold', rotation=30, ha='right')
    ax.set_title(f"Document Merging Task Comparison", fontsize=30, fontweight='bold', pad=20)

    # Add performance improvement percentage
    baseline_method = 'io'
    for model in models:
        baseline_score = combined_df[(combined_df['Method'] == baseline_method) & (combined_df['Model'] == model)]['Score'].mean()
        for method in methods_order[1:]:
            method_score = combined_df[(combined_df['Method'] == method) & (combined_df['Model'] == model)]['Score'].mean()
            improvement = (method_score - baseline_score) / baseline_score * 100
            # ax.text(methods_order.index(method), y_lower - 0.5, f'{improvement:.1f}%', ha='center', va='top', fontsize=12, rotation=90)

    # Improve general aesthetics
    sns.despine(left=True, bottom=True)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    fig.savefig(f"enhanced_doc_merge_comparison_{num_ndas}.pdf", bbox_inches="tight")

# Example usage
base_directory = "/p/llmreliability/test_repos/graph-of-thoughts/examples/doc_merge/results"
models = ["GPT-3.5-turbo", "GPT-4o", "Claude-3.5", "Llama-3.1-8B", "Mixtral-8x22B"]  

plot_multiple_models(
    base_directory,
    models,
    num_ndas=4,
    display_solved=False,
    y_upper=10,
    display_left_ylabel=True,
    cost_upper=20,
)
