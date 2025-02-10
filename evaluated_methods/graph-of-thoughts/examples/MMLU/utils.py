from typing import Dict, Union
import re

def extract_answer(solution: Union[str, Dict]) -> str:
    """
    Extracts the answer letter from a solution string or dictionary.
    
    :param solution: The solution string or dictionary.
    :return: The extracted answer letter (A, B, C, or D).
    """
    if isinstance(solution, dict):
        solution = solution.get('current', '')
    
    if not isinstance(solution, str):
        return ''  # Return empty string if the input is neither a string nor a dict
    
    # Extract the first letter of the solution, which should be the answer
    match = re.search(r'^([A-Da-d])', solution.strip())
    if match:
        return match.group(1).upper()
    return ''

def test_mmlu(problem: Dict, solution: Union[str, Dict]) -> bool:
    """
    Tests if the solution to an MMLU problem is correct.
    
    :param problem: The problem dictionary containing the question and answer.
    :param solution: The solution string or dictionary.
    :return: True if the solution is correct, False otherwise.
    """
    extracted_answer = extract_answer(solution)
    correct_answer = problem['answer'].strip().upper()
    return extracted_answer == correct_answer

def calculate_error(problem: Dict, solution: Union[str, Dict]) -> float:
    """
    Calculates the error for an MMLU problem solution.
    For multiple-choice questions, the error is binary: 0 for correct, 1 for incorrect.
    
    :param problem: The problem dictionary containing the question and answer.
    :param solution: The solution string or dictionary.
    :return: 0 if the solution is correct, 1 otherwise.
    """
    return 0 if test_mmlu(problem, solution) else 1

def num_errors(state: Dict) -> float:
    """
    Function to calculate the error that serves as a score.
    
    :param state: Thought state to be scored.
    :return: The error value (0 for correct, 1 for incorrect).
    """
    try:
        problem = state["original"]
        return calculate_error(problem, state["current"])
    except:
        return 1  # Return maximum error if there's an exception

def score_solution(problem: Dict, solution: Union[str, Dict]) -> float:
    """
    Scores a solution based on its correctness.
    
    :param problem: The problem dictionary containing the question and answer.
    :param solution: The solution string or dictionary.
    :return: 1.0 if the solution is correct, 0.0 otherwise.
    """
    return 1.0 if test_mmlu(problem, solution) else 0.0