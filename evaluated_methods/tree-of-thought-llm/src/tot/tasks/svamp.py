import re
import os
import json
from tot.tasks.base import Task, DATA_PATH
from tot.prompts.svamp import * 

class SVAMPTask(Task):
    def __init__(self, file='SVAMP.json'):
        super().__init__()
        path = os.path.join(DATA_PATH, 'SVAMP', file)
        with open(path, 'r') as f:
            self.data = json.load(f)
        self.value_cache = {}
        self.steps = 5
        self.stops = ['\n'] * 5

    def __len__(self) -> int:
        return len(self.data)

    def get_input(self, idx: int) -> str:
        return f"{self.data[idx]['Body']} {self.data[idx]['Question']}"

    def test_output(self, idx: int, output: str):
        correct_answer = float(self.data[idx]['Answer'])
        try:
            # Extract the numerical answer from the output
            match = re.search(r'####\s*(\d+(?:\.\d+)?)', output)
            if match:
                predicted_answer = float(match.group(1))
            else:
                # If no #### format, try to find the last number in the string
                numbers = re.findall(r'\d+(?:\.\d+)?', output)
                predicted_answer = float(numbers[-1]) if numbers else None

            if predicted_answer is not None:
                return {'r': int(abs(predicted_answer - correct_answer) < 1e-6)}
            else:
                return {'r': 0}
        except (ValueError, IndexError):
            return {'r': 0}
    
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
        return f"""
Solve the following math word problem step by step:
{x}
Current solution:
{y}
Provide the next step in the solution. If you have the final answer, format it as: #### [your answer]
"""

    @staticmethod
    def cot_prompt_wrap(x: str, y: str='') -> str:
        return f"""
Solve the following math word problem step by step:
{x}
Current solution:
{y}
Provide the next step in the solution, showing your work clearly. If you have the final answer, format it as: #### [your answer]
"""

    @staticmethod
    def propose_prompt_wrap(x: str, y: str='') -> str:
        return f"""
Here's a math word problem:
{x}
Current solution steps:
{y}
What is the next step in solving this problem? Be specific and use the numbers given in the problem. If you have enough information to provide the final answer, do so in the format: #### [your answer]
"""

    @staticmethod
    def value_prompt_wrap(x: str, y: str) -> str:
        return f"""
Here's a math word problem:
{x}
Proposed solution:
{y}
How likely is this partial solution to lead to the correct answer? (impossible/unlikely/likely/very likely/certain)
Explain your reasoning briefly.
"""

    @staticmethod
    def value_outputs_unwrap(x: str, y: str, value_outputs: list) -> float:
        value_map = {'impossible': 0.0, 'unlikely': 0.25, 'likely': 0.5, 'very likely': 0.75, 'certain': 1.0}
        values = []
        for vo in value_outputs:
            for key in value_map:
                if key in vo.lower():
                    values.append(value_map[key])
                    break
        return sum(values) / len(values) if values else 0.0

    @staticmethod
    def answer_clean(s):
        s = s.strip()
        s = s.split('####')[-1]  # Get the last #### segment
        s = re.sub(r'[^\d.]+', '', s)  # Remove all non-numeric characters except '.'
        return s