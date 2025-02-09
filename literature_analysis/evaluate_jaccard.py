import pandas as pd
import json
import os
import numpy as np

def load_and_parse_data(file_path):
    """Load and parse the JSON data into a pandas DataFrame."""
    with open(file_path, 'r') as file:
        data = json.load(file)
    return pd.DataFrame(data['paper_comparisons'])

def analyze_similarities(df):
    """Analyze the benchmark and model similarities."""
    similarity_stats = {
        'Benchmark': {
            'mean': df['benchmark_similarity'].mean(),
            'median': df['benchmark_similarity'].median(),
            'std': df['benchmark_similarity'].std()
        },
        'Model': {
            'mean': df['model_similarity'].mean(),
            'median': df['model_similarity'].median(),
            'std': df['model_similarity'].std()
        }
    }
    
    ranges = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    
    def get_distribution(series):
        dist = {}
        for start, end in ranges:
            label = f"{start:.1f}-{end:.1f}"
            count = len(series[(series > start) & (series <= end)])
            dist[label] = count
        return dist
    
    distributions = {
        'Benchmark': get_distribution(df['benchmark_similarity']),
        'Model': get_distribution(df['model_similarity'])
    }
    
    perfect_matches = df[
        (df['benchmark_similarity'] == 1.0) & 
        (df['model_similarity'] == 1.0)
    ]['title'].tolist()
    
    low_matches = df[
        (df['benchmark_similarity'] < 0.6) | 
        (df['model_similarity'] < 0.6)
    ][['title', 'benchmark_similarity', 'model_similarity']]
    
    return {
        'statistics': similarity_stats,
        'distributions': distributions,
        'perfect_matches': perfect_matches,
        'low_matches': low_matches
    }

def print_analysis(analysis):
    """Print the analysis results in a formatted way."""
    print("=== Similarity Analysis ===\n")
    
    print("Statistics:")
    for metric in ['Benchmark', 'Model']:
        print(f"\n{metric} Similarity:")
        stats = analysis['statistics'][metric]
        print(f"  Mean: {stats['mean']:.4f}")
        print(f"  Median: {stats['median']:.4f}")
        print(f"  Std Dev: {stats['std']:.4f}")
    
    print("\nDistributions:")
    for metric in ['Benchmark', 'Model']:
        print(f"\n{metric} Similarity Distribution:")
        for range_label, count in analysis['distributions'][metric].items():
            print(f"  {range_label}: {count}")
    
    print("\nPapers with Perfect Similarity:")
    for title in analysis['perfect_matches']:
        print(f"  - {title}")
    
    print("\nPapers with Low Similarity (< 0.6):")
    low_matches = analysis['low_matches']
    for _, row in low_matches.iterrows():
        print(f"  - {row['title']}")
        print(f"    Benchmark: {row['benchmark_similarity']:.4f}")
        print(f"    Model: {row['model_similarity']:.4f}")

df = load_and_parse_data('annotation_comparison_results.json')
analysis_results = analyze_similarities(df)
print_analysis(analysis_results)

correlation = df['benchmark_similarity'].corr(df['model_similarity'])
print(f"\nCorrelation between benchmark and model similarities: {correlation:.4f}")