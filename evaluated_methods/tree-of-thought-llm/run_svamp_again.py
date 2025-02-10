import json
import random
from typing import List, Tuple
from langchain_openai import OpenAI, ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate
import wandb
import time
import os

llm = ChatAnthropic(
    model="claude-3-5-sonnet-20240620",
    temperature=0,
    max_tokens=1024,
    timeout=None,
    max_retries=2,
)


class ThoughtGenerator:
    def __init__(self, llm: OpenAI):
        self.llm = llm
        self.prompt = PromptTemplate(
            template="""You are solving a math word problem step by step. Here's the problem:

{body}
{question}

Previous thoughts:
{thoughts}

Generate the next thought to solve this problem. Focus on reasoning through the solution.

Next thought:""",
            input_variables=["body", "question", "thoughts"]
        )

    def generate_thoughts(self, body: str, question: str, thoughts: List[str], num_thoughts: int) -> List[str]:
        formatted_thoughts = "\n".join(thoughts)
        prompt_value = self.prompt.format(body=body, question=question, thoughts=formatted_thoughts)

        new_thoughts = []
        for _ in range(num_thoughts):
            response = self.llm.invoke(prompt_value)
            new_thoughts.append(response.content.strip())

        return new_thoughts

class SWAMPChecker:
    def evaluate(self, thought: str) -> Tuple[bool, bool, float]:
        if "final answer" in thought.lower():
            try:
                answer = float(thought.split("final answer")[-1].strip().split()[0])
                return True, True, answer
            except ValueError:
                return False, True, None  # Invalid but final
        return True, False, None  # Valid but not final

def tree_of_thoughts(body: str, question: str, max_depth: int, num_thoughts: int):
    thought_generator = ThoughtGenerator(llm)
    checker = SWAMPChecker()

    def solve(thoughts: List[str], depth: int):
        if depth >= max_depth:
            return None, None

        new_thoughts = thought_generator.generate_thoughts(body, question, thoughts, num_thoughts)

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

def process_swamp_problems(dataset: List[dict], num_problems: int, max_depth: int, num_thoughts: int, log_file: str):
    random.shuffle(dataset)
    selected_problems = dataset[:num_problems]

    results = []
    for i, problem in enumerate(selected_problems, 1):
        print(f"Processing problem {i}/{num_problems}")
        body = problem['Body']
        question = problem['Question']
        correct_answer = problem['Answer']

        start_time = time.time()
        _, model_answer = tree_of_thoughts(body, question, max_depth, num_thoughts)
        end_time = time.time()

        result = {
            "id": problem['ID'],
            "body": body,
            "question": question,
            "model_answer": model_answer if model_answer is not None else "No answer found",
            "correct_answer": correct_answer,
            "processing_time": end_time - start_time,
            "type": problem['Type']
        }
        results.append(result)

        with open(log_file, 'a') as log_f:
            log_f.write(json.dumps(result) + '\n')

    return results

# Main execution
if __name__ == "__main__":
    input_file = "/p/llmreliability/test_repos/tree-of-thought-llm/src/tot/data/SVAMP/SVAMP.json"
    output_file = "swamp_results.json"
    log_file = "swamp_log.jsonl"
    num_problems = 1
    max_depth = 3
    num_thoughts = 3

    # Load the dataset
    with open(input_file, 'r') as f:
        dataset = json.load(f)

    results = process_swamp_problems(dataset, num_problems, max_depth, num_thoughts, log_file)

    # Write results to JSON file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results written to {output_file}")