import os
import json
from tot.tasks.base import Task, DATA_PATH
import re

class MMLUTask(Task):
    """
    Input (x)   : a multiple choice question
    Output (y)  : a letter corresponding to the correct answer
    Reward (r)  : 1 if the answer is correct, 0 otherwise
    """

    def __init__(self, file='mmlu_dataset_pretty.json'):
        super().__init__()
        path = os.path.join(DATA_PATH, 'MMLU', file)
        with open(path, 'r') as f:
            self.data = json.load(f)
        self.value_cache = {}
        self.steps = 3  # Set a default number of steps for solving a multiple-choice question
        self.stops = ['\n'] * self.steps

    def __len__(self) -> int:
        return len(self.data)

    def get_input(self, idx: int) -> str:
        return self.data[idx]['question']

    def test_output(self, idx: int, output: str):
        return {'r': int(output.strip().lower() == self.data[idx]['answer'].lower())}
    
    @staticmethod
    def extract_answer(output: str) -> float:
        # First, try to find the '####' format
        match = re.search(r'####\s*(\d+(?:\.\d+)?)', output)
        if match:
            return float(match.group(1))
        
        # If not found, look for "The final answer is: [number]"
        match = re.search(r'The final answer is:\s*(\d+(?:\.\d+)?)', output)
        if match:
            return float(match.group(1))
        
        # If still not found, look for the last number in the string
        numbers = re.findall(r'\d+(?:\.\d+)?', output)
        if numbers:
            return float(numbers[-1])
        
        return None

    @staticmethod
    def standard_prompt_wrap(x: str, y: str='') -> str:
        return f"Question: {x}\n\nAnswer: {y}"

    @staticmethod
    def cot_prompt_wrap(x: str, y: str='') -> str:
        return f"Question: {x}\n\nLet's approach this step-by-step:\n{y}"

    @staticmethod
    def propose_prompt_wrap(x: str, y: str='') -> str:
        return f"Question: {x}\n\nLet's think about this step-by-step. What's the next step in solving this problem?\n\nCurrent solution:\n{y}"

    @staticmethod
    def value_prompt_wrap(x: str, y: str) -> str:
        return f"Question: {x}\n\nProposed solution:\n{y}\n\nHow likely is this solution to be correct? (impossible/unlikely/likely/very likely/certain)"

    @staticmethod
    def value_outputs_unwrap(x: str, y: str, value_outputs: list) -> float:
        value_map = {
            'impossible': 0,
            'unlikely': 0.25,
            'likely': 0.5,
            'very likely': 0.75,
            'certain': 1
        }
        values = [value_map.get(output.strip().lower(), 0) for output in value_outputs]
        return sum(values) / len(values) if values else 0