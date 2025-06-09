#!/usr/bin/env python
import json
import re
import os
import sys
import argparse
from collections import Counter


def extract_answer_letter(text):
    """
    Extract the answer letter from various boxed formats in the response.
    Handles multiple formats including:
    - \boxed{A}
    - \(boxed{A}\)
    - \(\\boxed{A}\)
    - \(\\boxed{A. Option Text}\)
    - etc.
    """
    # Try different regex patterns to extract the letter
    patterns = [
        r'\\boxed\{([A-E])\}',                      # \boxed{A}
        r'\\boxed\{([A-E])\.',                      # \boxed{A. Text}
        r'\\boxed\{([A-E])[\s\}]',                  # \boxed{A } or \boxed{A}
        r'\\\(\\boxed\{([A-E])[\.|\}]',             # \(\boxed{A} or \(\boxed{A.
        r'\\\(\\\\boxed\{([A-E])[\.|\}]',           # \(\\boxed{A} or \(\\boxed{A.
        r'boxed\{([A-E])[\.|\}]',                   # boxed{A} or boxed{A.
        r'boxed\s*\{([A-E])[\.|\}]',                # boxed {A} or boxed {A.
        r'\\boxed\s*\{([A-E])[\.|\}]',              # \boxed {A} or \boxed {A.
        r'Box\s*\{([A-E])[\.|\}]',                  # Box {A} or Box {A.
        r'Box\s*\[([A-E])[\.|\]]',                  # Box [A] or Box [A.
        r'\\box\{([A-E])[\.|\}]',                   # \box{A} or \box{A.
        r'\\Box\{([A-E])[\.|\}]',                   # \Box{A} or \Box{A.
        r'Answer:\s*([A-E])[\.|\s]',                # Answer: A or Answer: A.
        r'Final\s*Answer:\s*([A-E])[\.|\s]',        # Final Answer: A or Final Answer: A.
        r'The\s*answer\s*is\s*([A-E])[\.|\s]',      # The answer is A or The answer is A.
        r'Therefore,\s*([A-E])\s',                  # Therefore, A is
        r'My\s*answer\s*is\s*([A-E])[\.|\s]',       # My answer is A or My answer is A.
        r'Choose\s*([A-E])[\.|\s]',                 # Choose A or Choose A.
        r'Option\s*([A-E])[\.|\s]',                 # Option A or Option A.
        r'[^A-Z]([A-E])[\.\)]$',                    # A. or A) at the end
        r'[^A-Z]([A-E])[\.\)]?\s*$'                 # A or A. or A) at the end
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1]  # Return the last match (most likely the final answer)
    
    # If no pattern matches, try to find any standalone A-E in the last few characters
    last_100_chars = text[-100:] if len(text) > 100 else text
    standalone_letters = re.findall(r'[^A-Za-z]([A-E])[^A-Za-z]', last_100_chars)
    if standalone_letters:
        return standalone_letters[-1]
    
    # If still no match, return None
    return None


def evaluate_results(results_file):
    """
    Evaluate the accuracy of model responses against correct answers.
    Returns a dictionary with accuracy metrics.
    """
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    results = data.get("results", {})
    
    correct_count = 0
    incorrect_count = 0
    no_answer_count = 0
    answer_distribution = Counter()
    correct_questions = []
    incorrect_questions = []
    no_answer_questions = []
    
    for question, (agent_contexts, correct_answer) in results.items():
        # Assuming we're evaluating the final answer of the final agent
        if not agent_contexts:
            continue
            
        # Get last agent context (all rounds completed)
        last_agent_context = agent_contexts[-1]
        
        # Get the final message from the last agent
        if not last_agent_context:
            continue
            
        last_message = last_agent_context[-1]["content"] if last_agent_context[-1]["role"] == "assistant" else ""
        
        # Extract the answer letter
        extracted_answer = extract_answer_letter(last_message)
        
        if extracted_answer:
            answer_distribution[extracted_answer] += 1
            if extracted_answer == correct_answer:
                correct_count += 1
                correct_questions.append((question, correct_answer, extracted_answer))
            else:
                incorrect_count += 1
                incorrect_questions.append((question, correct_answer, extracted_answer))
        else:
            no_answer_count += 1
            no_answer_questions.append((question, correct_answer))
    
    total_evaluated = correct_count + incorrect_count + no_answer_count
    accuracy = correct_count / total_evaluated if total_evaluated > 0 else 0
    
    # Return the evaluation results
    return {
        "accuracy": accuracy,
        "correct_count": correct_count,
        "incorrect_count": incorrect_count,
        "no_answer_count": no_answer_count,
        "total_evaluated": total_evaluated,
        "answer_distribution": dict(answer_distribution),
        "correct_questions": correct_questions,
        "incorrect_questions": incorrect_questions,
        "no_answer_questions": no_answer_questions
    }


def print_evaluation_report(eval_results):
    """Print a detailed evaluation report."""
    print("\n===== MEDQA EVALUATION REPORT =====")
    print(f"Accuracy: {eval_results['accuracy']:.2%} ({eval_results['correct_count']}/{eval_results['total_evaluated']})")
    print(f"Correct: {eval_results['correct_count']}")
    print(f"Incorrect: {eval_results['incorrect_count']}")
    print(f"No Answer: {eval_results['no_answer_count']}")
    
    print("\nAnswer Distribution:")
    for answer, count in eval_results['answer_distribution'].items():
        print(f"  {answer}: {count}")
    
    print("\nIncorrect Questions:")
    for i, (question, correct, extracted) in enumerate(eval_results['incorrect_questions'][:5], 1):
        print(f"{i}. Expected: {correct}, Got: {extracted}")
        print(f"   Question: {question[:100]}..." if len(question) > 100 else f"   Question: {question}")
    
    if eval_results['no_answer_count'] > 0:
        print("\nNo Answer Questions:")
        for i, (question, correct) in enumerate(eval_results['no_answer_questions'][:5], 1):
            print(f"{i}. Expected: {correct}")
            print(f"   Question: {question[:100]}..." if len(question) > 100 else f"   Question: {question}")
    
    print("\n=====================================")


def main():
    parser = argparse.ArgumentParser(description="Evaluate MedQA results")
    parser.add_argument("--results_file", type=str, required=True, help="Path to the JSON results file")
    parser.add_argument("--output_file", type=str, help="Optional path to save evaluation results as JSON")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_file):
        print(f"Error: Results file '{args.results_file}' not found.")
        sys.exit(1)
    
    # Evaluate the results
    eval_results = evaluate_results(args.results_file)
    
    # Print the report
    print_evaluation_report(eval_results)
    
    # Save the results if an output file is specified
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(eval_results, f, indent=2)
        print(f"Evaluation results saved to {args.output_file}")


if __name__ == "__main__":
    main()