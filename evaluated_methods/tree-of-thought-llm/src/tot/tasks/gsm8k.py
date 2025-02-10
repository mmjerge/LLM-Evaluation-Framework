import re
import os
import json
from tot.tasks.base import Task, DATA_PATH
from tot.prompts.gsm8k import *

def get_current_step(y: str) -> str:
    steps = y.strip().split('\n')
    return steps[-1] if steps else ""

class GSM8KTask(Task):
    """
    Input (x)   : a math word problem
    Output (y)  : a step-by-step solution with final answer
    Reward (r)  : 1 if the final answer is correct, 0 otherwise
    Input Example:
        Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
    Output Example:
        Let's solve this step by step:
        1. Calculate how many eggs Janet has left to sell:
           16 eggs laid - 3 eggs eaten - 4 eggs for muffins = 9 eggs to sell
        2. Calculate the money Janet makes from selling the eggs:
           9 eggs * $2 per egg = $18
        Therefore, Janet makes $18 every day at the farmers' market.
        #### 18
    """

    def __init__(self, file='test.jsonl'):
        super().__init__()
        path = os.path.join(DATA_PATH, 'GSM8k', file)
        with open(path, 'r') as f:
            self.data = [json.loads(line) for line in f]
        self.value_cache = {}
        self.steps = 5
        self.stops = ['\n'] * 5

    def __len__(self) -> int:
        return len(self.data)

    def get_input(self, idx: int) -> str:
        return self.data[idx]['question']

    def test_output(self, idx: int, output: str):
        match = re.search(r'####\s*(\d+)', output)
        if match:
            predicted_answer = int(match.group(1))
            correct_answer = int(self.data[idx]['answer'].split('####')[-1].strip())
            return {'r': int(predicted_answer == correct_answer)}
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
        return standard_prompt.format(input=x) + y

    @staticmethod
    def cot_prompt_wrap(x: str, y: str='') -> str:
        return cot_prompt.format(input=x) + y

    @staticmethod
    def propose_prompt_wrap(x: str, y: str='') -> str:
        return f"""
    Question: {x}

    Current solution steps:
    {y}

    Provide the next step in solving this problem. If you have enough information to calculate the final answer, do so and format it as '#### [number]'.
    Always end your response with the final answer in the format '#### [number]', even if you're not sure it's correct.
    """

    @staticmethod
    def value_prompt_wrap(x: str, y: str) -> str:
        return f"""
        Question: {x}

        Proposed solution:
        {y}

        On a scale of 1 to 5, how complete and correct is this solution? (1: Not started, 2: Poor progress, 3: Halfway there, 4: Nearly complete, 5: Complete and correct)
        """

    @staticmethod
    def value_outputs_unwrap(x: str, y: str, value_outputs: list) -> float:
        # Check if the final answer format is present
        final_answer_match = re.search(r'####\s*(\d+)', y)
        if final_answer_match:
            return 1.0  # If a final answer is provided, give it full value
        
        # Assign partial values for progress
        value = 0.0
        
        # Check for numerical calculations
        calculations = re.findall(r'\d+\s*[\+\-\*/]\s*\d+\s*=\s*\d+', y)
        value += min(len(calculations) * 0.2, 0.6)  # Up to 0.6 for calculations
        
        # Check for presence of dollar amounts
        dollar_amounts = re.findall(r'\$\d+', y)
        value += min(len(dollar_amounts) * 0.1, 0.2)  # Up to 0.2 for dollar amounts
        
        # Check for relevant keywords
        keywords = ['total', 'subtract', 'multiply', 'divide', 'add', 'sum', 'difference', 'product']
        for keyword in keywords:
            if keyword in y.lower():
                value += 0.05
        value = min(value, 0.2)  # Up to 0.2 for keywords
        
        return min(value, 0.9)  # Cap at 0.9 for solutions without the exact formatted answer