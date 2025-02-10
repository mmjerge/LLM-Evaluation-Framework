import json
import random
from typing import List, Tuple
from langchain_openai import OpenAI, ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate
import wandb
import time
import os
import logging
from tqdm import tqdm
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
            template="""You are solving a multiple-choice question step by step. Here's the question:

{question}

Previous thoughts:
{thoughts}

Generate the next thought to solve this problem. Focus on reasoning through the options.
If you're ready to give a final answer, state it as "Final Answer: (X)" where X is a single letter A, B, C, or D corresponding to your chosen option.

Next thought:""",
            input_variables=["question", "thoughts"]
        )

    def generate_thoughts(self, question: str, thoughts: List[str], num_thoughts: int) -> List[str]:
        formatted_thoughts = "\n".join(thoughts)
        prompt_value = self.prompt.format(question=question, thoughts=formatted_thoughts)
        
        new_thoughts = []
        for _ in range(num_thoughts):
            try:
                response = self.llm.invoke(prompt_value)
                new_thoughts.append(response.content.strip())
            except Exception as e:
                logging.error(f"Error generating thought: {e}")
        
        return new_thoughts

class MMlUChecker:
    def evaluate(self, thought: str) -> Tuple[bool, bool, str]:
        # Check for final answer in the format "Final Answer: (X)" or "Final Answer: (x)"
        final_answer_match = re.search(r'Final Answer:\s*\(([A-Da-d])\)', thought)
        
        # Check for final answer in the format "Final Answer: X" or "Final Answer: x"
        if not final_answer_match:
            final_answer_match = re.search(r'Final Answer:\s*([A-Da-d])', thought)
        
        if final_answer_match:
            return True, True, final_answer_match.group(1).upper()
        
        # If no final answer format is found, check if any option is mentioned
        option_match = re.search(r'\(([A-Da-d])\)', thought)
        if option_match:
            return True, True, option_match.group(1).upper()
        
        return True, False, None  # Valid but not final

def tree_of_thoughts(question: str, max_depth: int, num_thoughts: int, timeout: int = 60):
    thought_generator = ThoughtGenerator(llm)
    checker = MMlUChecker()
    
    def solve(thoughts: List[str], depth: int, start_time: float):
        if depth >= max_depth or time.time() - start_time > timeout:
            return None, None, None
        
        try:
            new_thoughts = thought_generator.generate_thoughts(question, thoughts, num_thoughts)
        except Exception as e:
            logging.error(f"Error generating thoughts: {e}")
            return None, None, None
        
        for thought in new_thoughts:
            is_valid, is_final, answer = checker.evaluate(thought)
            
            if is_valid:
                if is_final:
                    return thoughts + [thought], answer, thought
                if depth + 1 < max_depth:
                    result, answer, final_thought = solve(thoughts + [thought], depth + 1, start_time)
                    if result:
                        return result, answer, final_thought
        
        # If we've reached this point, return the last thought as the final response
        return thoughts, None, thoughts[-1] if thoughts else None

    return solve([], 0, time.time())

def process_mmlu_problems(dataset: List[dict], num_problems: int, max_depth: int, num_thoughts: int, log_file: str):
    random.shuffle(dataset)
    selected_problems = dataset[:num_problems]
    
    results = []
    for problem in tqdm(selected_problems, desc="Processing problems"):
        question = problem['question']
        correct_answer = problem['answer']
        
        start_time = time.time()
        try:
            _, model_answer, final_response = tree_of_thoughts(question, max_depth, num_thoughts)
        except Exception as e:
            logging.error(f"Error processing question: {e}")
            model_answer = "Error"
            final_response = "Error occurred during processing"
        end_time = time.time()
        
        result = {
            "question": question,
            "model_answer": model_answer if model_answer is not None else "No answer found",
            "final_response": final_response if final_response is not None else "No final response",
            "correct_answer": correct_answer,
            "processing_time": end_time - start_time
        }
        results.append(result)
        
        with open(log_file, 'a') as log_f:
            log_f.write(json.dumps(result) + '\n')
    
    return results

# Main execution
if __name__ == "__main__":
    input_file = "/p/llmreliability/test_repos/tree-of-thought-llm/src/tot/data/MMLU/mmlu_dataset_pretty.json"
    output_file = "mmlu_results.json"
    log_file = "mmlu_log.jsonl"
    num_problems = 150
    max_depth = 3
    num_thoughts = 3

    # Load the dataset
    try:
        with open(input_file, 'r') as f:
            dataset = json.load(f)
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        exit(1)

    results = process_mmlu_problems(dataset, num_problems, max_depth, num_thoughts, log_file)

    # Write results to JSON file
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logging.info(f"Results written to {output_file}")
    except Exception as e:
        logging.error(f"Error writing results to file: {e}")

    logging.info("Processing completed.")
