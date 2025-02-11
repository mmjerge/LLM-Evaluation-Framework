import os
import sys
import logging
import datetime
import time
import json
import random
from typing import Dict, List, Callable, Union
from datasets import load_dataset
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(project_root))

from graph_of_thoughts import (
    controller,
    language_models,
    operations,
    prompter,
    parser
)
from graph_of_thoughts.language_models.a_claude import AClaude
from graph_of_thoughts.language_models.mistral_models import Mistral
from graph_of_thoughts.language_models.together_ai import ATogetherAI
from graph_of_thoughts.language_models.llamachat_hf import Llama2HF

try:
    from . import utils
except ImportError:
    import utils

class GSM8KPrompter(prompter.Prompter):
    """
    GSM8KPrompter provides the generation of prompts specific to the GSM8K
    example for the language models.

    Inherits from the Prompter class and implements its abstract methods.
    """

    solve_prompt = """<Instruction> Solve the following math problem step by step. Provide your final answer at the end, prefixed with "Answer: ". </Instruction>

<Examples>
Input: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and sells the rest at the farmers market daily for $2 per egg. How much in dollars does she make every day at the farmers market?

Step 1: Calculate the total number of eggs Janet's ducks lay per day
Total eggs = 16

Step 2: Calculate the number of eggs Janet eats for breakfast
Eggs eaten = 3

Step 3: Calculate the number of eggs left to sell
Eggs to sell = Total eggs - Eggs eaten
Eggs to sell = 16 - 3 = 13

Step 4: Calculate the money Janet makes from selling the eggs
Money made = Eggs to sell × Price per egg
Money made = 13 × $2 = $26

Answer: $26

Input: {input}
</Examples>
"""

    cot_prompt = """<Instruction> Solve the following math problem step by step. Show your reasoning for each step. Provide your final answer at the end, prefixed with "Answer: ". </Instruction>

<Examples>
Input: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and sells the rest at the farmers market daily for $2 per egg. How much in dollars does she make every day at the farmers market?

Step 1: Calculate the total number of eggs Janet's ducks lay per day
Total eggs = 16
Reasoning: The problem states that Janet's ducks lay 16 eggs per day.

Step 2: Calculate the number of eggs Janet eats for breakfast
Eggs eaten = 3
Reasoning: The problem states that Janet eats three eggs for breakfast every morning.

Step 3: Calculate the number of eggs left to sell
Eggs to sell = Total eggs - Eggs eaten
Eggs to sell = 16 - 3 = 13
Reasoning: To find out how many eggs Janet can sell, we subtract the number of eggs she eats from the total number of eggs laid.

Step 4: Calculate the money Janet makes from selling the eggs
Money made = Eggs to sell × Price per egg
Money made = 13 × $2 = $26
Reasoning: Janet sells each egg for $2, so we multiply the number of eggs she sells by the price per egg.

Answer: $26

Input: {input}
</Examples>
"""

    tot_improve_prompt = """<Instruction> Review the following solution to a math problem. Identify any errors or areas for improvement in the reasoning or calculations. Then, provide an improved solution with corrected steps and calculations. </Instruction>

<Examples>
Problem: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and sells the rest at the farmers market daily for $2 per egg. How much in dollars does she make every day at the farmers market?

Original Solution:
Step 1: Calculate the total number of eggs Janet's ducks lay per day
Total eggs = 16

Step 2: Calculate the number of eggs Janet eats for breakfast
Eggs eaten = 3

Step 3: Calculate the number of eggs left to sell
Eggs to sell = Total eggs - Eggs eaten
Eggs to sell = 16 - 3 = 12

Step 4: Calculate the money Janet makes from selling the eggs
Money made = Eggs to sell × Price per egg
Money made = 12 × $2 = $24

Answer: $24

Improved Solution:
Step 1: Calculate the total number of eggs Janet's ducks lay per day
Total eggs = 16
(This step is correct)

Step 2: Calculate the number of eggs Janet eats for breakfast
Eggs eaten = 3
(This step is correct)

Step 3: Calculate the number of eggs left to sell
Eggs to sell = Total eggs - Eggs eaten
Eggs to sell = 16 - 3 = 13
(The calculation in this step was incorrect. It should be 13, not 12)

Step 4: Calculate the money Janet makes from selling the eggs
Money made = Eggs to sell × Price per egg
Money made = 13 × $2 = $26
(This step needed to be updated with the correct number of eggs to sell)

Answer: $26

Problem: {problem}

Original Solution:
{solution}

Improved Solution:
</Examples>
"""

    def aggregation_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        """
        Generate an aggregation prompt for the language model.

        :param state_dicts: The thought states that should be aggregated.
        :type state_dicts: List[Dict]
        :param kwargs: Additional keyword arguments.
        :return: The aggregation prompt.
        :rtype: str
        """
        problem = state_dicts[0]['original']
        solutions = [state['current'] for state in state_dicts]
        
        prompt = f"""<Instruction> Review the following solutions to the given math problem. Analyze each solution, identifying strengths and weaknesses. Then, synthesize the best elements from each solution to create a comprehensive, step-by-step final solution. Provide your final answer at the end, prefixed with "Answer: ". </Instruction>

Problem: {problem}

Solution 1:
{solutions[0]}

Solution 2:
{solutions[1]}

Synthesized Solution:
"""
        return prompt

    def generate_prompt(
        self, num_branches: int, original: str, current: str, method: str, **kwargs
    ) -> str:
        """
        Generate a generate prompt for the language model.

        :param num_branches: The number of responses the prompt should ask the LM to generate.
        :type num_branches: int
        :param original: Original problem statement.
        :type original: str
        :param current: Intermediate solution.
        :type current: str
        :param method: Method for which the generate prompt is generated.
        :type method: str
        :param kwargs: Additional keyword arguments.
        :return: The generate prompt.
        :rtype: str
        """
        if method.startswith("io"):
            return self.solve_prompt.format(input=original)
        elif method.startswith("cot"):
            return self.cot_prompt.format(input=original)
        elif method.startswith("tot"):
            if current is None or current == "":
                return self.cot_prompt.format(input=original)
            return self.tot_improve_prompt.format(problem=original, solution=current)
        elif method.startswith("got"):
            return self.cot_prompt.format(input=original)

    def improve_prompt(self, **kwargs) -> str:
        pass

    def validation_prompt(self, **kwargs) -> str:
        pass

    def score_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        pass

class GSM8KParser(parser.Parser):
    """
    GSM8KParser provides the parsing of language model responses specific to
    the GSM8K example.

    Inherits from the Parser class and implements its abstract methods.
    """

    def __init__(self) -> None:
        self.cache = {}

    def parse_aggregation_answer(
        self, states: List[Dict], texts: List[str]
    ) -> Union[Dict, List[Dict]]:
        new_states = []
        for text in texts:
            new_state = states[0].copy()
            new_state['current'] = text
            new_states.append(new_state)
        return new_states

    def parse_generate_answer(self, state: Dict, texts: List[str]) -> List[Dict]:
        new_states = []
        for text in texts:
            new_state = state.copy()
            new_state['current'] = text
            new_states.append(new_state)
        return new_states

    def parse_improve_answer(self, state: Dict, texts: List[str]) -> Dict:
        pass

    def parse_validation_answer(self, state: Dict, texts: List[str]) -> bool:
        pass

    def parse_score_answer(self, states: List[Dict], texts: List[str]) -> List[float]:
        pass

def io() -> operations.GraphOfOperations:
    operations_graph = operations.GraphOfOperations()
    operations_graph.append_operation(operations.Generate(1, 1))
    operations_graph.append_operation(operations.Score(1, False, utils.score_solution))
    operations_graph.append_operation(operations.GroundTruth(utils.test_gsm8k))
    return operations_graph

def cot() -> operations.GraphOfOperations:
    operations_graph = operations.GraphOfOperations()
    operations_graph.append_operation(operations.Generate(1, 1))
    operations_graph.append_operation(operations.Score(1, False, utils.extract_answer))
    operations_graph.append_operation(operations.GroundTruth(utils.test_gsm8k))
    return operations_graph

def tot() -> operations.GraphOfOperations:
    """
    Generates an optimized Graph of Operations for the ToT method with reduced thoughts.

    :return: Graph of Operations
    :rtype: GraphOfOperations
    """
    operations_graph = operations.GraphOfOperations()

    logging.info("Creating optimized ToT graph")
    
    # Generate initial solutions
    generate_op = operations.Generate(1, 2)
    operations_graph.append_operation(generate_op)
    logging.info("Added initial Generate operation")

    # Score initial solutions
    score_op = operations.Score(1, False, utils.extract_answer)
    score_op.add_predecessor(generate_op)
    operations_graph.append_operation(score_op)
    logging.info("Added Score operation")

    # Keep the best initial solution
    keep_best = operations.KeepBestN(1, False)
    keep_best.add_predecessor(score_op)
    operations_graph.append_operation(keep_best)
    logging.info("Added KeepBestN operation")

    # Generate improvements
    improve_generate = operations.Generate(1, 2)
    improve_generate.add_predecessor(keep_best)
    operations_graph.append_operation(improve_generate)
    logging.info("Added Improve Generate operation")

    # Score improvements
    improve_score = operations.Score(1, False, utils.extract_answer)
    improve_score.add_predecessor(improve_generate)
    operations_graph.append_operation(improve_score)
    logging.info("Added Improve Score operation")

    # Keep the best final solution
    final_keep_best = operations.KeepBestN(1, False)
    final_keep_best.add_predecessor(improve_score)
    operations_graph.append_operation(final_keep_best)
    logging.info("Added final KeepBestN operation")

    # Evaluate against ground truth
    operations_graph.append_operation(operations.GroundTruth(utils.test_gsm8k))
    logging.info("Added GroundTruth operation")

    return operations_graph

def got() -> operations.GraphOfOperations:
    """
    Generates a minimal Graph of Operations for the GoT method with a single thought.

    :return: Graph of Operations
    :rtype: GraphOfOperations
    """
    operations_graph = operations.GraphOfOperations()

    logging.info("Creating minimal GoT graph")

    # Generate a single solution
    generate_op = operations.Generate(1, 1)
    operations_graph.append_operation(generate_op)
    logging.info("Added Generate operation")

    # Evaluate against ground truth
    ground_truth_op = operations.GroundTruth(utils.test_gsm8k)
    ground_truth_op.add_predecessor(generate_op)
    operations_graph.append_operation(ground_truth_op)
    logging.info("Added GroundTruth operation")

    return operations_graph

def load_gsm8k_data(num_samples: int = 150, seed: int = 42) -> List[Dict]:
    """
    Load and sample questions from the GSM8K dataset.
    """
    dataset = load_dataset("gsm8k", "main", split="test")
    random.seed(seed)
    
    total_samples = len(dataset)
    sample_indices = random.sample(range(total_samples), min(num_samples, total_samples))
    
    sampled_data = []
    for idx in sample_indices:
        item = dataset[idx]
        formatted_item = {
            'question': item['question'],
            'answer': item['answer'],
        }
        sampled_data.append(formatted_item)
    
    return sampled_data

def load_gsm_symbolic_data(num_samples: int = 150, seed: int = 42) -> List[Dict]:
    """
    Load and sample questions from the GSM-Symbolic dataset.
    
    :param num_samples: Number of questions to sample
    :param seed: Random seed for reproducibility
    :return: List of sampled questions
    """
    # Load the dataset with specific configuration
    dataset = load_dataset("apple/GSM-Symbolic", "main", split="test")
    
    # The dataset is now directly a Dataset object, not a DatasetDict
    train_data = dataset
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Randomly sample indices
    total_samples = len(train_data)
    sample_indices = random.sample(range(total_samples), min(num_samples, total_samples))
    
    # Create the formatted data
    sampled_data = []
    for idx in sample_indices:
        item = train_data[idx]
        # Format the data to match the expected structure
        formatted_item = {
            'question': item['question'],
            'answer': item['answer'],
            # Add any additional fields that might be needed
        }
        sampled_data.append(formatted_item)
    
    return sampled_data

def run(
    num_samples: int,
    methods: List[Callable[[], operations.GraphOfOperations]],
    budget: float,
    lm_name: str,
    dataset_name: str = "gsm8k",
    seed: int = 42,
    timeout: int = 60,
) -> float:
    orig_budget = budget
    
    if dataset_name.lower() == "gsm8k":
        data = load_gsm8k_data(num_samples=num_samples, seed=seed)
    else:
        data = load_gsm_symbolic_data(num_samples=num_samples, seed=seed)
    
    selected_data = data

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    extra_info = f"{dataset_name}_{lm_name}_{'-'.join([method.__name__ for method in methods])}"
    folder_name = f"{extra_info}_{timestamp}"
    results_folder = os.path.join(results_dir, folder_name)
    os.makedirs(results_folder)

    config = {
        "dataset": dataset_name,
        "data": [item['question'] for item in selected_data],
        "methods": [method.__name__ for method in methods],
        "lm": lm_name,
        "budget": budget,
        "num_samples": num_samples,
        "seed": seed,
    }
    
    with open(os.path.join(results_folder, "config.json"), "w") as f:
        json.dump(config, f)

    logging.basicConfig(
        filename=os.path.join(results_folder, "log.log"),
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    for method in methods:
        os.makedirs(os.path.join(results_folder, method.__name__))

    import tqdm
    
    print(f"\nProcessing {len(selected_data)} problems from {dataset_name}...")
    for problem_idx, problem in enumerate(tqdm.tqdm(selected_data)):
        if budget <= 0.0:
            print(f"Budget has been depleted, stopping.")
            logging.error(f"Budget has been depleted, stopping.")
            break
            
        for method in methods:
            if budget <= 0.0:
                break
                
            retries = 3  
            for retry in range(retries):
                try:
                    if lm_name == "llama-3.1-together":
                        lm = ATogetherAI(
                            os.path.join(
                                os.path.dirname(__file__),
                                "../../graph_of_thoughts/language_models/config.json"
                            ),
                            model_name=lm_name,
                            cache=True,
                        )
                    else:
                        lm = Mistral(
                            os.path.join(
                                os.path.dirname(__file__),
                                "../../graph_of_thoughts/language_models/config.json"
                            ),
                            model_name=lm_name,
                            cache=True,
                        )

                    operations_graph = method()
                    executor = controller.Controller(
                        lm,
                        operations_graph,
                        GSM8KPrompter(),
                        GSM8KParser(),
                        {
                            "original": problem['question'],
                            "current": "",
                            "method": method.__name__,
                            "problem": problem,
                        },
                    )
                    
                    print(f"\nExecuting method: {method.__name__} (attempt {retry + 1})")
                    result = executor.run()
                    print(f"Completed {method.__name__}")
                    
                    path = os.path.join(
                        results_folder,
                        method.__name__,
                        f"{problem_idx}.json"
                    )
                    executor.output_graph(path)
                    budget -= lm.cost
                    
                    break
                    
                except Exception as e:
                    print(f"\nError in {method.__name__} (attempt {retry + 1}): {str(e)}")
                    logging.error(f"Exception in method {method.__name__} (attempt {retry + 1}): {e}")
                    
                    if retry < retries - 1:
                        delay = (retry + 1) * 30
                        print(f"Retrying in {delay} seconds...")
                        time.sleep(delay)
                    else:
                        print(f"Failed all {retries} attempts for this problem, moving to next...")
                        continue


if __name__ == "__main__":
    budget = 30
    approaches = [tot, got]
    
    # spent_gsm8k = run(
    #     num_samples=150,
    #     methods=approaches,
    #     budget=budget,
    #     lm_name="llama-3.1-together", 
    #     dataset_name="gsm8k",
    #     seed=42 
    # )
    # logging.info(f"Spent {spent_gsm8k} out of {budget} budget on GSM8K.")
    
    spent_symbolic = run(
        num_samples=150,
        methods=approaches,
        budget=budget,
        lm_name="llama-3.1-together",
        dataset_name="gsm-symbolic",
        seed=42 
    )
    logging.info(f"Spent {spent_symbolic} out of {budget} budget on GSM-Symbolic.")