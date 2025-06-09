#!/usr/bin/env python3
"""
Evaluation script for GOT (Generate, Optimize, Test) results.
Calculates accuracy across all JSON files in the got directory.
"""

import json
import os
import glob
from typing import List, Dict, Any, Tuple


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
                if all(key in thought for key in ['answer_idx', 'current']):
                    all_questions.append(thought)
    
    return all_questions


def evaluate_questions(questions: List[Dict[str, Any]]) -> Tuple[int, int, float]:
    """
    Evaluate questions by comparing correct answers with selected answers.
    
    Returns:
        Tuple of (correct_count, total_count, accuracy)
    """
    correct_count = 0
    total_count = len(questions)
    
    for question in questions:
        correct_answer = question.get('answer_idx', '').strip().upper()
        selected_answer = question.get('current', '').strip().upper()
        
        if correct_answer == selected_answer:
            correct_count += 1
    
    accuracy = (correct_count / total_count * 100) if total_count > 0 else 0.0
    return correct_count, total_count, accuracy


def print_detailed_results(questions: List[Dict[str, Any]], show_incorrect: bool = False):
    """Print detailed results including incorrect answers if requested."""
    print(f"\nDetailed Results:")
    print(f"Total questions: {len(questions)}")
    
    correct_count = 0
    incorrect_questions = []
    
    for i, question in enumerate(questions, 1):
        correct_answer = question.get('answer_idx', '').strip().upper()
        selected_answer = question.get('current', '').strip().upper()
        is_correct = correct_answer == selected_answer
        
        if is_correct:
            correct_count += 1
        else:
            incorrect_questions.append((i, question, correct_answer, selected_answer))
    
    print(f"Correct answers: {correct_count}")
    print(f"Incorrect answers: {len(incorrect_questions)}")
    
    if show_incorrect and incorrect_questions:
        print(f"\nIncorrect Questions:")
        print("-" * 80)
        for idx, question, correct, selected in incorrect_questions:
            question_text = question.get('question', 'N/A')
            question_preview = question_text[:100] + "..." if len(question_text) > 100 else question_text
            print(f"Q{idx}: {question_preview}")
            print(f"  Correct: {correct}, Selected: {selected}")
            print()


def evaluate_got_directory(directory_path: str = "got", 
                          show_details: bool = False, 
                          show_incorrect: bool = False,
                          operation_filter: str = "keep_best_n"):
    """
    Evaluate all JSON files in the GOT directory.
    
    Args:
        directory_path: Path to the directory containing JSON files
        show_details: Whether to show detailed per-file results
        show_incorrect: Whether to show incorrect answers in detail
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
        correct, total, accuracy = evaluate_questions(questions)
        
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
        total_correct, total_questions, overall_accuracy = evaluate_questions(all_questions)
        
        print("=" * 60)
        print("OVERALL RESULTS")
        print("=" * 60)
        print(f"Total files processed: {len(file_results)}")
        print(f"Total questions: {total_questions}")
        print(f"Total correct: {total_correct}")
        print(f"Total incorrect: {total_questions - total_correct}")
        print(f"Overall accuracy: {overall_accuracy:.2f}%")
        
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
    
    parser = argparse.ArgumentParser(description="Evaluate GOT results and calculate accuracy")
    parser.add_argument("--directory", "-d", default="got", 
                       help="Directory containing JSON files (default: got)")
    parser.add_argument("--details", action="store_true",
                       help="Show detailed per-file results")
    parser.add_argument("--show-incorrect", action="store_true",
                       help="Show details of incorrect answers")
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
            _, _, accuracy = evaluate_questions(all_questions)
            print(f"{accuracy:.2f}%")
        else:
            print("0.00%")
    else:
        evaluate_got_directory(
            directory_path=args.directory,
            show_details=args.details,
            show_incorrect=args.show_incorrect,
            operation_filter=args.operation
        )


if __name__ == "__main__":
    main()