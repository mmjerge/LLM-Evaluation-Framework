#!/usr/bin/env python3
"""
Evaluation Script for GSM8K Math Word Problems
"""

import json
import re
import sys
from typing import Dict, List, Tuple, Optional

def extract_ground_truth_answer(ground_truth_text: str) -> Optional[str]:
    if not ground_truth_text:
        return None
    pattern = r'####\s*([\d,]+(?:\.\d+)?)'
    match = re.search(pattern, ground_truth_text)
    if match:
        return match.group(1).replace(',', '')
    return None

def extract_model_answer(model_answer_text: str) -> Optional[str]:
    if not model_answer_text:
        return None
    
    final_answer_patterns = [
        r'final\s+answer\s*:\s*\$?\s*([\d,]+(?:\.\d+)?)',  
        r'\\text\s*\{\s*final\s*answer\s*:?\s*\}\s*\$?\s*([\d,]+(?:\.\d+)?)',
        r'\\boxed\s*\{\s*([\d,]+(?:\.\d+)?)\s*\}',
        r'therefore,\s*the\s*answer\s*is\s*\$?\s*([\d,]+(?:\.\d+)?)',  
        r'the\s*answer\s*is\s*\$?\s*([\d,]+(?:\.\d+)?)',
    ]
    
    for pattern in final_answer_patterns:
        match = re.search(pattern, model_answer_text, re.IGNORECASE)
        if match:
            return match.group(1).replace(',', '')
    
    return None

def normalize_answer(answer: str) -> Optional[float]:
    if not answer:
        return None
    try:
        return float(answer)
    except (ValueError, TypeError):
        return None

def evaluate_single_question(ground_truth: str, model_answer: str) -> Tuple[bool, Dict]:
    gt_answer = extract_ground_truth_answer(ground_truth)
    model_ans = extract_model_answer(model_answer)
    
    gt_normalized = normalize_answer(gt_answer) if gt_answer else None
    model_normalized = normalize_answer(model_ans) if model_ans else None
    
    if gt_normalized is None or model_normalized is None:
        is_correct = False
    else:
        numerically_correct = abs(gt_normalized - model_normalized) < 1e-9
        
        if numerically_correct and gt_answer and model_ans:
            is_correct = (gt_answer == model_ans)
        else:
            is_correct = False
    
    return is_correct, {
        'ground_truth_raw': gt_answer,
        'model_answer_raw': model_ans,
        'ground_truth_normalized': gt_normalized,
        'model_answer_normalized': model_normalized,
        'is_correct': is_correct
    }

def evaluate_dataset(questions: List[Dict], strict_format: bool = False) -> Dict:
    total_questions = len(questions)
    correct_count = 0
    detailed_results = []
    errors = []
    format_issues = []
    
    for i, question in enumerate(questions):
        question_id = i + 1
        ground_truth = question.get('ground_truth', '')
        model_answer = question.get('model_response', '')
        
        if strict_format:
            gt_answer = extract_ground_truth_answer(ground_truth)
            gt_normalized = normalize_answer(gt_answer) if gt_answer else None
            
            model_ans = None
            model_normalized = None
            is_correct = False
            
            pattern1 = r'Final answer: (\d+(?:\.\d+)?)$'
            pattern2 = r'Final answer: (\d+(?:\.\d+)?)\.$'
            
            match1 = re.search(pattern1, model_answer.strip())
            match2 = re.search(pattern2, model_answer.strip())
            
            if match1:
                model_ans = match1.group(1)
                model_normalized = normalize_answer(model_ans)
                is_correct = (gt_normalized is not None and 
                             model_normalized is not None and 
                             abs(gt_normalized - model_normalized) < 1e-9)
            elif match2:
                model_ans = match2.group(1)
                model_normalized = normalize_answer(model_ans)
                is_correct = (gt_normalized is not None and 
                             model_normalized is not None and 
                             abs(gt_normalized - model_normalized) < 1e-9)
            
            if not match1 and not match2 and "Final answer:" in model_answer:
                format_issues.append(question_id)
        else:
            is_correct, details = evaluate_single_question(ground_truth, model_answer)
            gt_answer = details['ground_truth_raw']
            model_ans = details['model_answer_raw']
            gt_normalized = details['ground_truth_normalized']
            model_normalized = details['model_answer_normalized']
        
        if is_correct:
            correct_count += 1
        else:
            errors.append({
                'question_id': question_id,
                'ground_truth_extracted': gt_answer,
                'model_answer_extracted': model_ans,
                'question_snippet': question.get('question', '')[:100] + "..." if len(question.get('question', '')) > 100 else question.get('question', ''),
                'original_model_response': model_answer[:150] + "..." if len(model_answer) > 150 else model_answer
            })
        
        detailed_results.append({
            'question_id': question_id,
            'ground_truth_raw': gt_answer,
            'model_answer_raw': model_ans,
            'ground_truth_normalized': gt_normalized,
            'model_answer_normalized': model_normalized,
            'is_correct': is_correct
        })
    
    accuracy = correct_count / total_questions if total_questions > 0 else 0
    
    result = {
        'total_questions': total_questions,
        'correct_count': correct_count,
        'accuracy': accuracy,
        'accuracy_percentage': accuracy * 100,
        'detailed_results': detailed_results,
        'errors': errors,
        'error_count': len(errors),
        'strict_format': strict_format
    }
    
    if strict_format:
        result['format_issues'] = len(format_issues)
        result['format_issue_questions'] = format_issues
    
    return result

def print_results(results: Dict, show_details: bool = False):
    evaluation_type = "STRICT" if results.get('strict_format', False) else "LENIENT"
    
    print("=" * 60)
    print(f"GSM8K EVALUATION RESULTS ({evaluation_type})")
    print("=" * 60)
    print(f"Total Questions: {results['total_questions']}")
    print(f"Correct Answers: {results['correct_count']}")
    print(f"Incorrect Answers: {results['error_count']}")
    print(f"Accuracy: {results['accuracy_percentage']:.1f}%")
    
    if results.get('format_issues'):
        print(f"Format Issues: {results['format_issues']} questions")
    
    print()
    
    if results['error_count'] > 0:
        print("INCORRECT ANSWERS:")
        print("-" * 60)
        for i, error in enumerate(results['errors'][:15]):
            print(f"{i+1}. Question {error['question_id']}:")
            print(f"   Expected: {error['ground_truth_extracted']}")
            print(f"   Got: {error['model_answer_extracted']}")
            if show_details:
                print(f"   Question: {error['question_snippet']}")
            print()
    
    if show_details and len(results['errors']) > 15:
        print(f"... and {len(results['errors']) - 15} more errors")
        print()
    
    if show_details:
        print("SAMPLE CORRECT ANSWERS (first 5):")
        print("-" * 40)
        correct_samples = [r for r in results['detailed_results'] if r['is_correct']][:5]
        for result in correct_samples:
            print(f"✓ Q{result['question_id']}: Expected={result['ground_truth_raw']} | Got={result['model_answer_raw']}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <json_file_path> [--debug]")
        sys.exit(1)
    
    json_file_path = sys.argv[1]
    debug_mode = '--debug' in sys.argv
    
    try:
        print(f"Loading evaluation data from: {json_file_path}")
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            print("Error: Expected a JSON array of question objects.")
            sys.exit(1)
        
        if not data:
            print("Error: No questions found in the JSON file.")
            sys.exit(1)
        
        print(f"Found {len(data)} questions.")
        if debug_mode:
            print("Debug mode enabled")
        print("Running evaluation...")
        
        print("\n" + "="*60)
        print("LENIENT EVALUATION")
        print("="*60)
        results_lenient = evaluate_dataset(data, strict_format=False)
        print_results(results_lenient, show_details=debug_mode)
        
        if debug_mode:
            print("\nDEBUG: Sample extraction failures:")
            print("-" * 40)
            failures = [r for r in results_lenient['detailed_results'] if not r['is_correct']]
            for i, failure in enumerate(failures[:5]):
                print(f"Q{failure['question_id']}: Expected={failure['ground_truth_raw']}, Got={failure['model_answer_raw']}")
        
        print("\n" + "="*60)
        print("STRICT EVALUATION")
        print("="*60)
        results_strict = evaluate_dataset(data, strict_format=True)
        print_results(results_strict, show_details=True)
        
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        print(f"Lenient accuracy:  {results_lenient['accuracy_percentage']:.1f}% ({results_lenient['correct_count']}/{results_lenient['total_questions']})")
        print(f"Strict accuracy:   {results_strict['accuracy_percentage']:.1f}% ({results_strict['correct_count']}/{results_strict['total_questions']})")
        print(f"Difference:        {results_lenient['accuracy_percentage'] - results_strict['accuracy_percentage']:.1f} percentage points")
        
        if 'format_issues' in results_strict:
            print(f"Format issues:     {results_strict['format_issues']} questions had LaTeX/formatting issues")
        
        output_file_lenient = json_file_path.replace('.json', '_evaluation_lenient.json')
        output_file_strict = json_file_path.replace('.json', '_evaluation_strict.json')
        
        with open(output_file_lenient, 'w', encoding='utf-8') as f:
            json.dump(results_lenient, f, indent=2, ensure_ascii=False)
        
        with open(output_file_strict, 'w', encoding='utf-8') as f:
            json.dump(results_strict, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to:")
        print(f"  Lenient: {output_file_lenient}")
        print(f"  Strict:  {output_file_strict}")
        
    except FileNotFoundError:
        print(f"Error: File '{json_file_path}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON file. {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
