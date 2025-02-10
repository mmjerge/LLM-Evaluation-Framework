import re
import os
import json
from tot.tasks.base import Task, DATA_PATH
from tot.prompts.aqua import *

def get_current_step(y: str) -> str:
    steps = y.strip().split('\n')
    return steps[-1] if steps else ""

class AQUATask(Task):
    def __init__(self, file='test.jsonl'):
        super().__init__()
        path = os.path.join(DATA_PATH, 'AQUA', file)
        with open(path, 'r') as f:
            self.data = [json.loads(line) for line in f]
        self.value_cache = {}
        self.steps = 5
        self.stops = ['\n'] * 5

    def __len__(self) -> int:
        return len(self.data)

    def get_input(self, idx: int) -> str:
        question = self.data[idx].get('question', 'No question found')
        options = '\n'.join(self.data[idx].get('options', ['No options found']))
        return f"Question: {question}\n\nOptions:\n{options}"

    def test_output(self, idx: int, output: str):
        correct_answer = self.data[idx].get('correct', 'No correct answer found')
        if correct_answer == 'No correct answer found':
            print(f"Warning: No correct answer found for index {idx}")
            return {'r': 0}

        predicted_answer = self.extract_answer(output)
        if predicted_answer:
            return {'r': int(predicted_answer == correct_answer)}
        else:
            print(f"Warning: No predicted answer found in output for index {idx}")
            return {'r': 0}

    @staticmethod
    def extract_answer(output: str) -> str:
        # First, try to find the '####' format
        match = re.search(r'####\s*([A-E])', output)
        if match:
            return match.group(1)

        # If not found, look for "The final answer is: [A-E]"
        match = re.search(r'The final answer is:\s*([A-E])', output, re.IGNORECASE)
        if match:
            return match.group(1)

        # If still not found, look for the last occurrence of A), B), C), D), or E)
        matches = re.findall(r'([A-E])\)', output)
        if matches:
            return matches[-1]

        # If still not found, look for the last occurrence of A, B, C, D, or E
        matches = re.findall(r'\b([A-E])\b', output)
        if matches:
            return matches[-1]

        print(f"Warning: No answer could be extracted from output: {output}")
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
    {x}

    Current solution steps:
    {y}

    Provide the next step in solving this problem. If you have enough information to select the correct answer, do so and format it as '#### [A/B/C/D/E]'.
    Always end your response with the final answer in the format '#### [A/B/C/D/E]', even if you're not sure it's correct.
    """

    @staticmethod
    def value_prompt_wrap(x: str, y: str) -> str:
        return f"""
        {x}

        Proposed solution:
        {y}

        On a scale of 1 to 5, how complete and correct is this solution? (1: Not started, 2: Poor progress, 3: Halfway there, 4: Nearly complete, 5: Complete and correct)
        """

    @staticmethod
    def value_outputs_unwrap(x: str, y: str, value_outputs: list) -> float:
        final_answer_match = re.search(r'####\s*([A-E])', y)
        if final_answer_match:
            return 1.0  # If a final answer is provided, give it full value
        
        value = 0.0
        
        # Check for presence of calculations or logical reasoning
        reasoning_steps = re.findall(r'\d+\.|\([a-z]\)', y)
        value += min(len(reasoning_steps) * 0.2, 0.6)  # Up to 0.6 for reasoning steps
        
        # Check for mentions of the options
        options_mentioned = len(re.findall(r'\b[A-E]\)', y))
        value += min(options_mentioned * 0.1, 0.2)  # Up to 0.2 for mentioning options
        
        # Check for relevant keywords
        keywords = ['therefore', 'because', 'result', 'conclusion', 'answer']
        for keyword in keywords:
            if keyword in y.lower():
                value += 0.05
        value = min(value, 0.2)  # Up to 0.2 for keywords
        
        return min(value, 0.9)  # Cap at 0.9 for solutions without the exact formatted answer