#!/usr/bin/env python
import json
import re
import os
import sys
import argparse
from collections import Counter


def extract_answer(text):
    """
    Extract Relevant/Irrelevant answer from various formats in the response.
    Handles multiple formats including:
    - \boxed{Relevant}
    - \(boxed{Irrelevant}\)
    - answer is Relevant
    - final answer: Irrelevant
    - etc.
    """
    # Check for boxed answers
    boxed_patterns = [
        r'\\boxed\{(Relevant|Irrelevant)\}',
        r'\\\(\\boxed\{(Relevant|Irrelevant)\}\\\)',
        r'\\\(\\\\boxed\{(Relevant|Irrelevant)\}\\\)',
        r'boxed\{(Relevant|Irrelevant)\}',
    ]
    
    for pattern in boxed_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].lower()
    
    # Check for answer statements
    statement_patterns = [
        r'answer is[:\s]*(Relevant|Irrelevant)',
        r'final answer[:\s]*(Relevant|Irrelevant)',
        r'conclusion[:\s]*(Relevant|Irrelevant)',
        r'the answer is[:\s]*(Relevant|Irrelevant)',
        r'my answer is[:\s]*(Relevant|Irrelevant)',
        r'answer:\s*(Relevant|Irrelevant)',
        r'decision:\s*(Relevant|Irrelevant)',
        r'classification:\s*(Relevant|Irrelevant)',
        r'clause is\s*(Relevant|Irrelevant)',
        r'I classify this as\s*(Relevant|Irrelevant)',
    ]
    
    for pattern in statement_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].lower()
    
    # Look for Relevant/Irrelevant in the last 50 words of the text
    words = text.split()
    last_words = ' '.join(words[-50:])
    
    # Check if "relevant" appears without "irrelevant" nearby
    relevant_match = re.search(r'\brelevant\b', last_words, re.IGNORECASE)
    irrelevant_match = re.search(r'\birrelevant\b', last_words, re.IGNORECASE)
    
    if relevant_match and not irrelevant_match:
        return 'relevant'
    elif irrelevant_match and not relevant_match:
        return 'irrelevant'
    
    # If we found both relevant and irrelevant, see which one is more likely the answer
    if relevant_match and irrelevant_match:
        # Check if one is in a stronger position (near the end)
        last_20_words = ' '.join(words[-20:])
        if 'relevant' in last_20_words.lower() and 'irrelevant' not in last_20_words.lower():
            return 'relevant'
        if 'irrelevant' in last_20_words.lower() and 'relevant' not in last_20_words.lower():
            return 'irrelevant'
    
    # If still can't determine, return None
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
        # For each agent, check their final answer
        for agent_idx, agent_context in enumerate(agent_contexts):
            # Get the final message from the agent
            if not agent_context or len(agent_context) < 2:
                no_answer_count += 1
                no_answer_questions.append((question, correct_answer))
                continue
                
            last_message = agent_context[-1]["content"] if agent_context[-1]["role"] == "assistant" else ""
            
            # Extract the answer (relevant/irrelevant)
            extracted_answer = extract_answer(last_message)
            
            if extracted_answer:
                answer_distribution[extracted_answer] += 1
                if extracted_answer.lower() == correct_answer.lower():
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
    print("\n===== LEGALBENCH PRIVACY POLICY QA EVALUATION REPORT =====")
    print(f"Accuracy: {eval_results['accuracy']:.2%} ({eval_results['correct_count']}/{eval_results['total_evaluated']})")
    print(f"Correct: {eval_results['correct_count']}")
    print(f"Incorrect: {eval_results['incorrect_count']}")
    print(f"No Answer: {eval_results['no_answer_count']}")
    
    print("\nAnswer Distribution:")
    for answer, count in eval_results['answer_distribution'].items():
        print(f"  {answer}: {count}")
    
    print("\nIncorrect Questions (up to 5):")
    for i, (question, correct, extracted) in enumerate(eval_results['incorrect_questions'][:5], 1):
        print(f"{i}. Expected: {correct}, Got: {extracted}")
        print(f"   Question: {question[:100]}..." if len(question) > 100 else f"   Question: {question}")
    
    if eval_results['no_answer_count'] > 0:
        print("\nNo Answer Questions (up to 5):")
        for i, (question, correct) in enumerate(eval_results['no_answer_questions'][:5], 1):
            print(f"{i}. Expected: {correct}")
            print(f"   Question: {question[:100]}..." if len(question) > 100 else f"   Question: {question}")
    
    print("\n=======================================================")


def main():
    parser = argparse.ArgumentParser(description="Evaluate LegalBench Privacy Policy QA results")
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