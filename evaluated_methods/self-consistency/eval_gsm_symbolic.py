import json
from collections import Counter

def evaluate_dataset(data):
    """
    Evaluates the accuracy of self-consistency results in the dataset.
    
    Args:
        data (dict): Dictionary containing the problem data
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    metrics = {
        'total_problems': 0,
        'correct_best_answers': 0,
        'path_consistency': [],
        'path_accuracy': [],
        'path_distribution': []
    }
    
    # Analyze each problem
    for problem in data:
        metrics['total_problems'] += 1
        true_answer = problem['true_answer']
        best_answer = problem['best_answer']
        path_answers = problem['answers']
        
        # Check if best answer matches true answer
        if best_answer == true_answer:
            metrics['correct_best_answers'] += 1
        
        # Analyze path consistency and accuracy
        path_counter = Counter(path_answers)
        total_paths = len(path_answers)
        majority_answer = path_counter.most_common(1)[0]
        
        # Calculate consistency (percentage of paths giving the majority answer)
        consistency = majority_answer[1] / total_paths
        metrics['path_consistency'].append(consistency)
        
        # Calculate accuracy (percentage of paths giving the correct answer)
        correct_paths = path_counter[true_answer]
        accuracy = correct_paths / total_paths
        metrics['path_accuracy'].append(accuracy)
        
        # Store answer distribution for this problem
        distribution = [(answer, count/total_paths) for answer, count in path_counter.items()]
        metrics['path_distribution'].append(distribution)
    
    # Calculate aggregate metrics
    metrics['avg_consistency'] = sum(metrics['path_consistency']) / len(metrics['path_consistency'])
    metrics['avg_path_accuracy'] = sum(metrics['path_accuracy']) / len(metrics['path_accuracy'])
    metrics['overall_accuracy'] = metrics['correct_best_answers'] / metrics['total_problems']
    
    return metrics

def format_results(metrics):
    """
    Formats the evaluation metrics into a readable string.
    
    Args:
        metrics (dict): Dictionary containing evaluation metrics
        
    Returns:
        str: Formatted results string
    """
    results = []
    results.append(f"Total problems evaluated: {metrics['total_problems']}")
    results.append(f"Overall accuracy (best answers): {metrics['overall_accuracy']:.2%}")
    results.append(f"Average path consistency: {metrics['avg_consistency']:.2%}")
    results.append(f"Average path accuracy: {metrics['avg_path_accuracy']:.2%}")
    
    return "\n".join(results)

# Example usage:
if __name__ == "__main__":
    # Load your JSONL data
    data = []
    with open('~/results_gsm_symbolic_mistral_open_mixtral_8x22b.jsonl', 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    
    # Run evaluation
    metrics = evaluate_dataset(data)
    
    # Print results
    print(format_results(metrics))