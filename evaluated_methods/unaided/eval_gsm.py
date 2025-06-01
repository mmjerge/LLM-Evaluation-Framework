#!/usr/bin/env python3
"""
GSM8K Accuracy Calculator

This script calculates the accuracy of math word problem datasets by comparing
model responses with correct answers. It extracts numerical answers from text
and handles various formats (integers, decimals, dollar amounts, etc.).

Usage:
    python accuracy_calculator.py <json_file_path>
    
"""

import json
import re
import sys
import argparse
from pathlib import Path
from typing import Optional, Dict, List, Tuple


def extract_final_answer(text: str) -> Optional[float]:
    """
    Extract the final numerical answer from a text response.
    
    Handles multiple formats:
    - #### 123 (GSM8K format)
    - Final answer: 123
    - Dollar amounts ($123)
    - Plain numbers at the end
    
    Args:
        text: The text to extract answer from
        
    Returns:
        The extracted number as float, or None if no valid number found
    """
    if not text or not isinstance(text, str):
        return None
    
    text = text.replace(',', '')
    
    hash_match = re.search(r'####\s*(-?\d+(?:\.\d+)?)', text)
    if hash_match:
        try:
            return float(hash_match.group(1))
        except ValueError:
            pass
    
    final_answer_patterns = [
        r'(?:Final answer|Answer):\s*\$?(-?\d+(?:\.\d+)?)',
        r'(?:Final answer|Answer):\s*(-?\d+(?:\.\d+)?)',
        r'(?:The answer is|So the answer is)\s*\$?(-?\d+(?:\.\d+)?)',
    ]
    
    for pattern in final_answer_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except (ValueError, IndexError):
                continue
    
    boxed_match = re.search(r'\\boxed\{(-?\d+(?:\.\d+)?)\}', text)
    if boxed_match:
        try:
            return float(boxed_match.group(1))
        except ValueError:
            pass
    
    dollar_pattern = r'(?:costs?|pays?|price|total|amount|worth|value)\s+(?:is\s+)?\$(-?\d+(?:\.\d+)?)|(?:^|\s)\$(-?\d+(?:\.\d+)?)(?:\s|$|\.)'
    dollar_matches = re.findall(dollar_pattern, text, re.IGNORECASE)
    if dollar_matches:
        flat_matches = [match for group in dollar_matches for match in group if match]
        if flat_matches:
            try:
                return float(flat_matches[-1])
            except ValueError:
                pass
    
    standalone_dollar = re.findall(r'\$(-?\d+(?:\.\d+)?)', text)
    if standalone_dollar:
        try:
            return float(standalone_dollar[-1])
        except ValueError:
            pass
    
    end_number_patterns = [
        r'(?:is|equals?|totals?)\s+(-?\d+(?:\.\d+)?)(?:\s*\.|\s*$)',
        r'(-?\d+(?:\.\d+)?)(?:\s*\.|\s*$)',
    ]
    
    for pattern in end_number_patterns:
        matches = re.findall(pattern, text)
        if matches:
            try:
                return float(matches[-1])
            except ValueError:
                continue
    
    all_numbers = re.findall(r'(-?\d+(?:\.\d+)?)', text)
    if all_numbers:
        try:
            candidates = []
            for num_str in all_numbers:
                num = float(num_str)
                if 1900 <= num <= 2100:
                    continue
                if num == 100 or num == 50 or num == 25:  
                    continue
                candidates.append(num)
            
            if candidates:
                return candidates[-1]
            else:
                return float(all_numbers[-1])
        except ValueError:
            pass
    
    return None


def compare_answers(model_answer: Optional[float], correct_answer: Optional[float], 
                   tolerance: float = 1e-9) -> bool:
    """
    Compare two numerical answers with a small tolerance for floating point differences.
    
    Args:
        model_answer: The model's predicted answer
        correct_answer: The correct answer
        tolerance: Tolerance for floating point comparison
        
    Returns:
        True if answers match within tolerance, False otherwise
    """
    if model_answer is None or correct_answer is None:
        return False
    
    if correct_answer == int(correct_answer) and model_answer == int(model_answer):
        return int(model_answer) == int(correct_answer)
    
    if correct_answer == 0:
        return abs(model_answer) < tolerance
    
    relative_error = abs(model_answer - correct_answer) / abs(correct_answer)
    return relative_error < tolerance or abs(model_answer - correct_answer) < tolerance


def analyze_errors(errors: List[Dict]) -> Dict:
    """
    Analyze the types of errors made by the model.
    
    Args:
        errors: List of error dictionaries
        
    Returns:
        Dictionary with error analysis
    """
    if not errors:
        return {
            'total_errors': 0,
            'null_errors': 0,
            'order_of_magnitude_errors': 0,
            'major_errors': 0,
            'moderate_errors': 0,
            'minor_errors': 0
        }
    
    null_errors = 0
    order_of_magnitude_errors = 0
    major_errors = 0
    moderate_errors = 0
    minor_errors = 0
    
    for error in errors:
        model_ans = error['model_answer']
        correct_ans = error['correct_answer']
        
        if model_ans is None or correct_ans is None:
            null_errors += 1
        elif correct_ans == 0 and model_ans != 0:
            major_errors += 1
        elif correct_ans != 0:
            diff = abs(model_ans - correct_ans)
            relative_error = diff / abs(correct_ans)
            
            if model_ans >= 10 * correct_ans or correct_ans >= 10 * model_ans:
                order_of_magnitude_errors += 1
            elif relative_error > 0.5:
                major_errors += 1
            elif relative_error > 0.1:
                moderate_errors += 1
            else:  
                minor_errors += 1
    
    return {
        'total_errors': len(errors),
        'null_errors': null_errors,
        'order_of_magnitude_errors': order_of_magnitude_errors,
        'major_errors': major_errors,
        'moderate_errors': moderate_errors,
        'minor_errors': minor_errors
    }


def calculate_accuracy(json_file_path: str, verbose: bool = True) -> Dict:
    """
    Calculate accuracy for a math dataset JSON file.
    
    Args:
        json_file_path: Path to the JSON file containing the dataset
        verbose: Whether to print detailed results
        
    Returns:
        Dictionary containing accuracy results and error analysis
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {json_file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON file: {e}")
    
    if not isinstance(data, list):
        raise ValueError("JSON file should contain a list of questions")
    
    correct_count = 0
    total_count = len(data)
    errors = []
    
    for i, question in enumerate(data):
        if not isinstance(question, dict):
            print(f"Warning: Question {i+1} is not a dictionary, skipping")
            continue
            
        if 'model_response' not in question or 'correct_answer' not in question:
            print(f"Warning: Question {i+1} missing required fields, skipping")
            continue
        
        model_answer = extract_final_answer(question['model_response'])
        correct_answer = extract_final_answer(question['correct_answer'])
        
        is_correct = compare_answers(model_answer, correct_answer)
        
        if is_correct:
            correct_count += 1
        else:
            errors.append({
                'question_index': i + 1,
                'model_answer': model_answer,
                'correct_answer': correct_answer,
                'model_response': question['model_response'][:200] + '...' if len(question['model_response']) > 200 else question['model_response'],
                'correct_response': question['correct_answer'][:200] + '...' if len(question['correct_answer']) > 200 else question['correct_answer'],
                'question': question.get('question', 'N/A')[:100] + '...' if len(question.get('question', '')) > 100 else question.get('question', 'N/A')
            })
    
    accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
    
    error_analysis = analyze_errors(errors)
    
    results = {
        'total_questions': total_count,
        'correct_answers': correct_count,
        'incorrect_answers': total_count - correct_count,
        'accuracy_percentage': accuracy,
        'error_analysis': error_analysis,
        'errors': errors[:10] if verbose else [] 
    }
    
    if verbose:
        print(f"\n{'='*50}")
        print(f"ACCURACY ANALYSIS: {Path(json_file_path).name}")
        print(f"{'='*50}")
        print(f"Total questions: {total_count}")
        print(f"Correct answers: {correct_count}")
        print(f"Incorrect answers: {total_count - correct_count}")
        print(f"Accuracy: {accuracy:.1f}%")
        
        if error_analysis['total_errors'] > 0:
            print(f"\nError Breakdown:")
            print(f"  Null/extraction errors: {error_analysis['null_errors']}")
            print(f"  Order of magnitude errors (10x+ off): {error_analysis['order_of_magnitude_errors']}")
            print(f"  Major errors (>50% off): {error_analysis['major_errors']}")
            print(f"  Moderate errors (10-50% off): {error_analysis['moderate_errors']}")
            print(f"  Minor errors (<10% off): {error_analysis['minor_errors']}")
            
            print(f"\nFirst few incorrect answers:")
            for error in errors[:5]:
                print(f"  Q{error['question_index']}: Model={error['model_answer']}, Correct={error['correct_answer']}")
                print(f"    Model response: {error['model_response'][:100]}...")
                print(f"    Correct response: {error['correct_response'][:100]}...")
                print()
    
    return results


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description='Calculate accuracy for math problem datasets')
    parser.add_argument('json_file', help='Path to the JSON file containing the dataset')
    parser.add_argument('--quiet', '-q', action='store_true', help='Only print the accuracy percentage')
    parser.add_argument('--tolerance', '-t', type=float, default=1e-9, help='Tolerance for answer comparison (default: 1e-9)')
    
    args = parser.parse_args()
    
    try:
        results = calculate_accuracy(args.json_file, verbose=not args.quiet)
        
        if args.quiet:
            print(f"{results['accuracy_percentage']:.1f}%")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
