import json
import random
from typing import List, Tuple
from langchain_openai import OpenAI, ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate
import wandb
import time
import os
import re

llm = ChatAnthropic(
    model="claude-3-5-sonnet-20240620",
    temperature=0,
    max_tokens=1024,
    timeout=None,
    max_retries=2
)

class ThoughtGenerator:
    def __init__(self, llm: OpenAI):
        self.llm = llm
        self.prompt = PromptTemplate(
            template="""You are solving a multiple-choice question step by step. Here's the question:

{question}

Options:
{options}

Previous thoughts:
{thoughts}

Generate the next thought to solve this problem. Focus on reasoning through the options.

Next thought:""",
            input_variables=["question", "options", "thoughts"]
        )

    def generate_thoughts(self, question: str, options: List[str], thoughts: List[str], num_thoughts: int) -> List[str]:
        formatted_thoughts = "\n".join(thoughts)
        formatted_options = "\n".join(f"{option}" for option in options)
        prompt_value = self.prompt.format(question=question, options=formatted_options, thoughts=formatted_thoughts)

        new_thoughts = []
        for _ in range(num_thoughts):
            start_time = time.time()
            response = self.llm.invoke(prompt_value)
            end_time = time.time()

            latency = end_time - start_time
            # tokens = len(response.split())
            tokens = len(response.content.split())  # Approximate token count

            # new_thoughts.append(response.strip())
            new_thoughts.append(response.content.strip())

        return new_thoughts

class AQuAChecker:
    def evaluate(self, thought: str, options: List[str]) -> Tuple[bool, bool, str]:
        thought_lower = thought.lower()

        # Check for phrases indicating a final answer
        final_answer_phrases = ["final answer", "therefore", "thus", "in conclusion", "the answer is", "the correct answer is"]
        is_final = any(phrase in thought_lower for phrase in final_answer_phrases)

        if is_final:
            # Try to extract the chosen option
            for idx, option in enumerate(options):
                option_letter = chr(ord('A') + idx)
                option_content = option[2:].strip().lower()  # Remove the letter and parenthesis

                # Check for various ways the answer might be expressed
                patterns = [
                    rf"\b{option_letter}\)",
                    rf"\boption\s+{option_letter}\b",
                    rf"\b{option_letter}\s+is correct\b",
                    rf"\b{option_content}\b",
                    rf"\b{idx + 1}\)",
                    rf"\boption\s+{idx + 1}\b"
                ]

                for pattern in patterns:
                    if re.search(pattern, thought_lower):
                        return True, True, option_letter

            # If no specific option is found, look for the last mention of a letter
            letter_matches = re.findall(r'\b([A-E])\)', thought_lower)
            if letter_matches:
                return True, True, letter_matches[-1]

        return True, False, None  # Valid but not final

def tree_of_thoughts(question: str, options: List[str], max_depth: int, num_thoughts: int):
    thought_generator = ThoughtGenerator(llm)
    checker = AQuAChecker()

    def solve(thoughts: List[str], depth: int):
        print(f"\nExploring at depth {depth}")  # Debug print
        if depth >= max_depth:
            print(f"Max depth {max_depth} reached, backtracking")  # Debug print
            return None, None, None

        new_thoughts = thought_generator.generate_thoughts(question, options, thoughts, num_thoughts)

        for i, thought in enumerate(new_thoughts):
            print(f"\nEvaluating thought {i+1} at depth {depth}: {thought}")  # Debug print
            is_valid, is_final, answer = checker.evaluate(thought, options)

            if is_valid:
                if is_final:
                    print(f"Final answer found: {answer} - Thought: {thought}")  # Debug print
                    return thoughts + [thought], answer, thought
                if depth + 1 < max_depth:
                    print(f"Exploring deeper for thought {i+1}")  # Debug print
                    result, answer, final_thought = solve(thoughts + [thought], depth + 1)
                    if result:
                        return result, answer, final_thought

        print(f"No valid solution found at depth {depth}, backtracking")  # Debug print
        return None, None, None

    result, answer, final_thought = solve([], 0)
    return result, answer, final_thought

def process_aqua_problems(dataset: List[dict], num_problems: int, max_depth: int, num_thoughts: int, log_file: str):
    results = []
    for i, problem in enumerate(dataset, 1):
        print(f"\nProcessing problem {i}/{num_problems}")
        question = problem['question']
        options = problem['options']
        correct_answer = problem['correct']

        print(f"Question: {question}")
        print(f"Options: {options}")

        start_time = time.time()
        thoughts, model_answer, final_thought = tree_of_thoughts(question, options, max_depth, num_thoughts)
        end_time = time.time()

        print(f"Model's answer: {model_answer}")
        print(f"Correct answer: {correct_answer}")

        result = {
            "question": question,
            "options": options,
            "model_answer": model_answer if model_answer is not None else "No answer found",
            "final_thought": final_thought if final_thought is not None else "No final thought",
            "correct_answer": correct_answer,
            "processing_time": end_time - start_time
        }
        results.append(result)

        with open(log_file, 'a') as log_f:
            log_f.write(json.dumps(result) + '\n')

    return results

# Main execution
if __name__ == "__main__":
    input_file = "/p/llmreliability/test_repos/tree-of-thought-llm/src/tot/data/AQUA/test.jsonl"
    output_file = "aqua_results.json"
    log_file = "aqua_log.jsonl"
    num_problems = 150
    max_depth = 3
    num_thoughts = 3

    dataset = []
    with open(input_file, 'r') as f:
        for line in f:
            dataset.append(json.loads(line))

    results = process_aqua_problems(dataset, num_problems, max_depth, num_thoughts, log_file)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results written to {output_file}")