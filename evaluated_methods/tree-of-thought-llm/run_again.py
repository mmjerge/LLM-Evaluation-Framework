import json
import random
from typing import List, Tuple
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
import re
from langchain_openai import ChatOpenAI

# Initialize the language model with the correct model name
llm = OpenAI(temperature=0.7, max_tokens=256, model="gpt-3.5-turbo-instruct")

class ThoughtGenerator:
    def __init__(self, llm: OpenAI):
        self.llm = llm
        self.prompt = PromptTemplate(
            template="""You are solving a math problem step by step. Here's the problem:

{problem_description}

Previous thoughts:
{thoughts}

Generate the next thought to solve this problem. Focus on one step at a time.

Next thought:""",
            input_variables=["problem_description", "thoughts"]
        )

    def generate_thoughts(self, problem_description: str, thoughts: List[str], num_thoughts: int) -> List[str]:
        formatted_thoughts = "\n".join(thoughts)
        prompt_value = self.prompt.format(problem_description=problem_description, thoughts=formatted_thoughts)
        
        new_thoughts = []
        for _ in range(num_thoughts):
            response = self.llm.invoke(prompt_value)
            new_thoughts.append(response.strip())
        
        return new_thoughts

class GSM8KChecker:
    def evaluate(self, thought: str) -> Tuple[bool, bool, str]:
        if "final answer" in thought.lower():
            return True, True, thought
        return True, False, None

def tree_of_thoughts(problem_description: str, max_depth: int, num_thoughts: int):
    thought_generator = ThoughtGenerator(llm)
    checker = GSM8KChecker()
    
    def solve(thoughts: List[str], depth: int):
        if depth >= max_depth:
            return None, None
        
        new_thoughts = thought_generator.generate_thoughts(problem_description, thoughts, num_thoughts)
        
        for thought in new_thoughts:
            is_valid, is_final, answer = checker.evaluate(thought)
            
            if is_valid:
                if is_final:
                    return thoughts + [thought], answer
                if depth + 1 < max_depth:
                    result, answer = solve(thoughts + [thought], depth + 1)
                    if result:
                        return result, answer
        
        return None, None

    result, answer = solve([], 0)
    return result, answer

def process_gsm8k_problems(file_path: str, num_problems: int, max_depth: int, num_thoughts: int):
    with open(file_path, 'r') as f:
        dataset = [json.loads(line) for line in f]
    
    random.shuffle(dataset)
    selected_problems = dataset[:num_problems]
    
    results = []
    for i, problem in enumerate(selected_problems, 1):
        print(f"Processing problem {i}/{num_problems}")
        question = problem['question']
        correct_answer = problem['answer'] 
        
        _, model_answer = tree_of_thoughts(question, max_depth, num_thoughts)
        
        result = {
            "question": question,
            "model_answer": model_answer if model_answer is not None else "No answer found",
            "correct_answer": correct_answer
        }
        results.append(result)
    
    return results

# Main execution
if __name__ == "__main__":
    input_file = "/p/llmreliability/test_repos/tree-of-thought-llm/src/tot/data/GSM8k/test.jsonl"
    output_file = "gsm8k_results.json"
    num_problems = 25
    max_depth = 5
    num_thoughts = 3

    results = process_gsm8k_problems(input_file, num_problems, max_depth, num_thoughts)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results written to {output_file}")
    print(f"Processed {len(results)} problems")