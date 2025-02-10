from typing import Dict, Union
import re

def extract_answer(solution: Union[str, Dict]) -> str:
    """Extracts the answer from a solution string or dictionary."""
    if isinstance(solution, dict):
        solution = solution.get('current', '')  # Get the 'current' field if it's a dict
    if not isinstance(solution, str):
        return ''  # Return empty string if the input is neither a string nor a dict
    match = re.search(r'Answer:\s*([A-E])', solution, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return ''

def test_aqua(problem: Dict, solution: Union[str, Dict]) -> bool:
    """Tests if the solution to an AQUA problem is correct."""
    extracted_answer = extract_answer(solution)
    correct_answer = problem['correct']
    return extracted_answer == correct_answer

def calculate_error(problem: Dict, solution: str) -> float:
    """
    Calculates the error for AQUA problems (0 for correct, 1 for incorrect).
    :param problem: The problem dictionary containing the question and correct answer.
    :type problem: Dict
    :param solution: The solution string.
    :type solution: str
    :return: 0 if the answer is correct, 1 if it's incorrect.
    :rtype: float
    """
    extracted_answer = extract_answer(solution)
    correct_answer = problem['correct']
    return 0 if extracted_answer == correct_answer else 1

def num_errors(state: Dict) -> float:
    """
    Function to calculate the error that serves as a score.
    :param state: Thought state to be scored.
    :type state: Dict
    :return: The error value.
    :rtype: float
    """
    try:
        problem = state["original"]
        return calculate_error(problem, state["current"])
    except:
        return float('inf')

def score_solution(problem: dict, solution: str) -> float:
    """Scores a solution based on its correctness."""
    if test_aqua(problem, solution):
        return 1.0
    return 0.0