from typing import Dict, Union
import re

def extract_answer(solution: Union[str, Dict]) -> float:
    """Extracts the numerical answer from a solution string or dictionary."""
    if isinstance(solution, dict):
        solution = solution.get('current', '')  # Get the 'current' field if it's a dict
    
    if not isinstance(solution, str):
        return float('nan')  # Return NaN if the input is neither a string nor a dict
    
    match = re.search(r'Answer:\s*\$?(\d+(?:\.\d+)?)', solution)
    if match:
        return float(match.group(1))
    return float('nan')

def test_gsm8k(problem: Dict, solution: Union[str, Dict]) -> bool:
    """Tests if the solution to a GSM8K problem is correct."""
    extracted_answer = extract_answer(solution)
    correct_answer = float(problem['answer'].split('####')[1].strip())
    return abs(extracted_answer - correct_answer) < 1e-6

def calculate_error(problem: Dict, solution: str) -> float:
    """
    Calculates the error between the extracted answer and the correct answer.

    :param problem: The problem dictionary containing the question and answer.
    :type problem: Dict
    :param solution: The solution string.
    :type solution: str
    :return: The absolute error between the extracted and correct answers.
    :rtype: float
    """
    extracted_answer = extract_answer(solution)
    correct_answer = float(problem['answer'].split('####')[1].strip())
    return abs(extracted_answer - correct_answer)

def num_errors(state: Dict) -> float:
    """
    Function to calculate the error that serves as a score.

    :param state: Thought state to be scored.
    :type state: Dict
    :return: The error value.
    :rtype: float
    """
    try:
        problem = {"question": state["original"], "answer": state["answer"]}
        return calculate_error(problem, state["current"])
    except:
        return float('inf')

def score_solution(problem: dict, solution: str) -> float:
    """Scores a solution based on its correctness."""
    if test_gsm8k(problem, solution):
        return 1.0
    return 0.0