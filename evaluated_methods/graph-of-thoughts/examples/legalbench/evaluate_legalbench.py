#!/usr/bin/env python3
"""
Evaluation script for relevance classification GOT results.
Calculates accuracy across all JSON files by comparing answer vs current predictions.
"""

import json
import os
import glob
from typing import List, Dict, Any, Tuple
from collections import Counter


def load_json_file(filepath: str) -> List[Dict[str, Any]]:
    """Load and parse a JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return []


def extract_questions_from_operations(data: List[Dict[str, Any]], operation_filter: str = "keep_best_n") -> List[Dict[str, Any]]:
    """
    Extract questions from specific operations in the data.
    
    Args:
        data: The loaded JSON data
        operation_filter: Which operation to extract from ("keep_best_n", "all", "generate", "score")
    """
    all_questions = []
    
    for operation in data:
        operation_type = operation.get('operation', '')
        
        # Skip if we're filtering and this isn't the right operation
        if operation_filter != "all" and operation_type != operation_filter:
            continue
            
        if 'thoughts' in operation:
            for thought in operation['thoughts']:
                # Check for the required keys for relevance classification
                if all(key in thought for key in ['answer', 'current']):
                    all_questions.append(thought)
    
    return all_questions


def evaluate_questions(questions: List[Dict[str, Any]]) -> Tuple[int, int, float, Dict[str, int]]:
    """
    Evaluate questions by comparing correct answers with predicted answers.
    
    Returns:
        Tuple of (correct_count, total_count, accuracy, label_distribution)
    """
    correct_count = 0
    total_count = len(questions)
    
    # Track label distributions
    true_labels = []
    predicted_labels = []
    
    for question in questions:
        correct_answer = question.get('answer', '').strip()
        predicted_answer = question.get('current', '').strip()
        
        true_labels.append(correct_answer)
        predicted_labels.append(predicted_answer)
        
        if correct_answer == predicted_answer:
            correct_count += 1
    
    accuracy = (correct_count / total_count * 100) if total_count > 0 else 0.0
    
    # Create label distribution summary
    label_stats = {
        'true_labels': Counter(true_labels),
        'predicted_labels': Counter(predicted_labels)
    }
    
    return correct_count, total_count, accuracy, label_stats


def print_detailed_results(questions: List[Dict[str, Any]], show_incorrect: bool = False):
    """Print detailed results including incorrect answers if requested."""
    print(f"\nDetailed Results:")
    print(f"Total questions: {len(questions)}")
    
    correct_count = 0
    incorrect_questions = []
    
    for i, question in enumerate(questions, 1):
        correct_answer = question.get('answer', '').strip()
        predicted_answer = question.get('current', '').strip()
        is_correct = correct_answer == predicted_answer
        
        if is_correct:
            correct_count += 1
        else:
            incorrect_questions.append((i, question, correct_answer, predicted_answer))
    
    print(f"Correct answers: {correct_count}")
    print(f"Incorrect answers: {len(incorrect_questions)}")
    
    if show_incorrect and incorrect_questions:
        print(f"\nIncorrect Classifications:")
        print("-" * 80)
        for idx, question, correct, predicted in incorrect_questions:
            question_text = question.get('question', 'N/A')
            text_context = question.get('text', 'N/A')
            
            question_preview = question_text[:100] + "..." if len(question_text) > 100 else question_text
            text_preview = text_context[:100] + "..." if len(text_context) > 100 else text_context
            
            print(f"Q{idx}: {question_preview}")
            print(f"  Context: {text_preview}")
            print(f"  True Label: {correct}, Predicted: {predicted}")
            print()


def print_label_statistics(label_stats: Dict[str, Counter]):
    """Print statistics about label distributions."""
    print(f"\nLabel Distribution:")
    print("-" * 40)
    
    print("True Labels:")
    for label, count in label_stats['true_labels'].most_common():
        print(f"  {label}: {count}")
    
    print("\nPredicted Labels:")
    for label, count in label_stats['predicted_labels'].most_common():
        print(f"  {label}: {count}")


def evaluate_got_directory(directory_path: str = "got", 
                          show_details: bool = False, 
                          show_incorrect: bool = False,
                          show_label_stats: bool = False,
                          operation_filter: str = "keep_best_n"):
    """
    Evaluate all JSON files in the GOT directory.
    
    Args:
        directory_path: Path to the directory containing JSON files
        show_details: Whether to show detailed per-file results
        show_incorrect: Whether to show incorrect answers in detail
        show_label_stats: Whether to show label distribution statistics
        operation_filter: Which operation to evaluate ("keep_best_n", "all", "generate", "score")
    """
    
    # Find all JSON files in the directory
    json_pattern = os.path.join(directory_path, "*.json")
    json_files = glob.glob(json_pattern)
    
    if not json_files:
        print(f"No JSON files found in directory: {directory_path}")
        return
    
    print(f"Found {len(json_files)} JSON files in '{directory_path}' directory")
    print(f"Evaluating operation: {operation_filter}")
    print("=" * 60)
    
    all_questions = []
    file_results = []
    
    # Process each file
    for filepath in sorted(json_files):
        filename = os.path.basename(filepath)
        data = load_json_file(filepath)
        
        if not data:
            continue
            
        questions = extract_questions_from_operations(data, operation_filter)
        correct, total, accuracy, _ = evaluate_questions(questions)
        
        file_results.append({
            'filename': filename,
            'correct': correct,
            'total': total,
            'accuracy': accuracy
        })
        
        all_questions.extend(questions)
        
        if show_details:
            print(f"File: {filename}")
            print(f"  Questions: {total}")
            print(f"  Correct: {correct}")
            print(f"  Accuracy: {accuracy:.2f}%")
            print()
    
    # Calculate overall results
    if all_questions:
        total_correct, total_questions, overall_accuracy, label_stats = evaluate_questions(all_questions)
        
        print("=" * 60)
        print("OVERALL RESULTS")
        print("=" * 60)
        print(f"Total files processed: {len(file_results)}")
        print(f"Total questions: {total_questions}")
        print(f"Total correct: {total_correct}")
        print(f"Total incorrect: {total_questions - total_correct}")
        print(f"Overall accuracy: {overall_accuracy:.2f}%")
        
        if show_label_stats:
            print_label_statistics(label_stats)
        
        if show_details:
            print(f"\nPer-file breakdown:")
            print("-" * 40)
            for result in file_results:
                print(f"{result['filename']}: {result['accuracy']:.2f}% ({result['correct']}/{result['total']})")
        
        if show_incorrect:
            print_detailed_results(all_questions, show_incorrect=True)
            
    else:
        print("No valid questions found in any files.")


def main():
    """Main function with command line argument handling."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate relevance classification GOT results and calculate accuracy")
    parser.add_argument("--directory", "-d", default="got", 
                       help="Directory containing JSON files (default: got)")
    parser.add_argument("--details", action="store_true",
                       help="Show detailed per-file results")
    parser.add_argument("--show-incorrect", action="store_true",
                       help="Show details of incorrect classifications")
    parser.add_argument("--show-labels", action="store_true",
                       help="Show label distribution statistics")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Only show overall accuracy")
    parser.add_argument("--operation", default="keep_best_n",
                       choices=["keep_best_n", "all", "generate", "score"],
                       help="Which operation to evaluate (default: keep_best_n)")
    
    args = parser.parse_args()
    
    if args.quiet:
        # Quick mode - just print overall accuracy
        json_pattern = os.path.join(args.directory, "*.json")
        json_files = glob.glob(json_pattern)
        
        all_questions = []
        for filepath in json_files:
            data = load_json_file(filepath)
            questions = extract_questions_from_operations(data, args.operation)
            all_questions.extend(questions)
        
        if all_questions:
            _, _, accuracy, _ = evaluate_questions(all_questions)
            print(f"{accuracy:.2f}%")
        else:
            print("0.00%")
    else:
        evaluate_got_directory(
            directory_path=args.directory,
            show_details=args.details,
            show_incorrect=args.show_incorrect,
            show_label_stats=args.show_labels,
            operation_filter=args.operation
        )


if __name__ == "__main__":
    main()