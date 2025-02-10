import numpy as np
from datasets import load_dataset

def load_truthfulqa_dataset():
    return load_dataset("truthfulqa", "generation")

def score_truthfulness(answer, reference):
    # This is a simple implementation and should be replaced with a more sophisticated method
    overlap = len(set(answer.lower().split()) & set(reference.lower().split()))
    return overlap / len(set(reference.lower().split()))

def evaluate_truthfulqa(generated_answer, problem):
    dataset = load_truthfulqa_dataset()
    reference_answers = dataset['validation'][problem['id']]['correct_answers']
    
    scores = [score_truthfulness(generated_answer, ref) for ref in reference_answers]
    return max(scores)

def test_truthfulqa(state):
    generated_answer = state['current']
    problem = state['problem']
    return evaluate_truthfulqa(generated_answer, problem)