import os
import logging
import datetime
import json
from typing import Dict, List, Callable, Union
from graph_of_thoughts import controller, language_models, operations, prompter, parser
from graph_of_thoughts.language_models.a_claude import AClaude
from graph_of_thoughts.language_models.mistral_models import Mistral
from graph_of_thoughts.language_models.together_ai import ATogetherAI

try:
    from . import utils
except ImportError:
    import utils

class MMLUPrompter(prompter.Prompter):
    """
    MMLUPrompter provides the generation of prompts specific to the MMLU
    example for the language models.
    """

    def generate_prompt(
        self, num_branches: int, original: Dict, current: str, method: str, **kwargs
    ) -> str:
        """
        Generate a prompt for the language model.

        :param num_branches: The number of responses the prompt should ask the LM to generate.
        :param original: Original problem dictionary.
        :param current: Intermediate solution.
        :param method: Method for which the prompt is generated.
        :return: The generate prompt.
        """
        prompt = f"Question: {original['question']}\n\n"
        prompt += f"A: {original['A']}\n"
        prompt += f"B: {original['B']}\n"
        prompt += f"C: {original['C']}\n"
        prompt += f"D: {original['D']}\n\n"
        prompt += "Please provide your answer as a single letter (A, B, C, or D) followed by a brief explanation.\n"
        prompt += "Answer: "
        return prompt

    def aggregation_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        pass

    def improve_prompt(self, **kwargs) -> str:
        pass

    def validation_prompt(self, **kwargs) -> str:
        pass

    def score_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        pass

class MMLUParser(parser.Parser):
    """
    MMLUParser provides the parsing of language model responses specific to
    the MMLU example.
    """

    def __init__(self) -> None:
        self.cache = {}

    def parse_generate_answer(self, state: Dict, texts: List[str]) -> List[Dict]:
        new_states = []
        for text in texts:
            new_state = state.copy()
            new_state['current'] = text
            new_states.append(new_state)
        return new_states

    def parse_aggregation_answer(
        self, states: List[Dict], texts: List[str]
    ) -> Union[Dict, List[Dict]]:
        pass

    def parse_improve_answer(self, state: Dict, texts: List[str]) -> Dict:
        pass

    def parse_validation_answer(self, state: Dict, texts: List[str]) -> bool:
        pass

    def parse_score_answer(self, states: List[Dict], texts: List[str]) -> List[float]:
        pass

def extract_answer(solution: str) -> str:
    """Extracts the answer letter from a solution string."""
    return solution.strip()[0].upper()

def test_mmlu(problem: Dict, solution: str) -> bool:
    """Tests if the solution to an MMLU problem is correct."""
    extracted_answer = extract_answer(solution)
    return extracted_answer == problem['answer']

def score_solution(problem: Dict, solution: str) -> float:
    """Scores a solution based on its correctness."""
    return 1.0 if test_mmlu(problem, solution) else 0.0

def cot() -> operations.GraphOfOperations:
    operations_graph = operations.GraphOfOperations()
    operations_graph.append_operation(operations.Generate(1, 1))
    operations_graph.append_operation(operations.Score(1, False, utils.extract_answer))
    operations_graph.append_operation(operations.GroundTruth(utils.test_mmlu))
    return operations_graph

def tot() -> operations.GraphOfOperations:
    operations_graph = operations.GraphOfOperations()

    # Generate initial solutions
    generate_op = operations.Generate(1, 2)
    operations_graph.append_operation(generate_op)

    # Score initial solutions
    score_op = operations.Score(1, False, utils.extract_answer)
    score_op.add_predecessor(generate_op)
    operations_graph.append_operation(score_op)

    # Keep the best initial solution
    keep_best = operations.KeepBestN(1, False)
    keep_best.add_predecessor(score_op)
    operations_graph.append_operation(keep_best)

    # Generate improvements
    improve_generate = operations.Generate(1, 2)
    improve_generate.add_predecessor(keep_best)
    operations_graph.append_operation(improve_generate)

    # Score improvements
    improve_score = operations.Score(1, False, utils.extract_answer)
    improve_score.add_predecessor(improve_generate)
    operations_graph.append_operation(improve_score)

    # Keep the best final solution
    final_keep_best = operations.KeepBestN(1, False)
    final_keep_best.add_predecessor(improve_score)
    operations_graph.append_operation(final_keep_best)

    # Evaluate against ground truth
    operations_graph.append_operation(operations.GroundTruth(utils.test_mmlu))

    return operations_graph

def got() -> operations.GraphOfOperations:
    """
    Generates a minimal Graph of Operations for the GoT method with a single thought.
    """
    operations_graph = operations.GraphOfOperations()

    logging.info("Creating minimal GoT graph")

    # Generate a single solution
    generate_op = operations.Generate(1, 1)
    operations_graph.append_operation(generate_op)
    logging.info("Added Generate operation")

    # Evaluate against ground truth
    ground_truth_op = operations.GroundTruth(test_mmlu)
    ground_truth_op.add_predecessor(generate_op)
    operations_graph.append_operation(ground_truth_op)
    logging.info("Added GroundTruth operation")

    return operations_graph

def run(
    data_ids: List[int],
    methods: List[Callable[[], operations.GraphOfOperations]],
    budget: float,
    lm_name: str,
) -> float:
    orig_budget = budget
    data_path = os.path.join(os.path.dirname(__file__), "MMLU_test.json")
    
    # Load the entire JSON file
    with open(data_path, "r") as f:
        data = json.load(f)
    
    # Ensure data is a list
    if not isinstance(data, list):
        raise ValueError("The JSON file should contain an array of questions")

    if data_ids is None or len(data_ids) == 0:
        data_ids = list(range(len(data)))
    selected_data = [data[i] for i in data_ids]

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    extra_info = f"{lm_name}_{'-'.join([method.__name__ for method in methods])}"
    folder_name = f"{extra_info}_{timestamp}"
    results_folder = os.path.join(results_dir, folder_name)
    os.makedirs(results_folder)

    logging.basicConfig(
        filename=os.path.join(results_folder, "log.log"),
        filemode="w",
        format="%(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    for method in methods:
        os.makedirs(os.path.join(results_folder, method.__name__))

    lm = ATogetherAI(
        os.path.join(
            os.path.dirname(__file__),
            "../../graph_of_thoughts/language_models/config.json",
        ),
        model_name=lm_name,
        cache=True,
    )

    for problem in selected_data:
        logging.info(f"Running problem: {problem['question'][:50]}...")
        if budget <= 0.0:
            logging.error(f"Budget has been depleted, stopping.")
            break
        for method in methods:
            logging.info(f"Running method {method.__name__}")
            logging.info(f"Budget left: {budget}")
            if budget <= 0.0:
                logging.error(f"Budget has been depleted, stopping. Method {method.__name__} has not been run.")
                break
            
            operations_graph = method()
            executor = controller.Controller(
                lm,
                operations_graph,
                MMLUPrompter(),
                MMLUParser(),
                {
                    "original": problem,
                    "current": "",
                    "method": method.__name__,
                },
            )
            try:
                result = executor.run()
                logging.info(f"Result: {result}")
            except Exception as e:
                logging.error(f"Exception in method {method.__name__}: {e}")
            
            path = os.path.join(
                results_folder,
                method.__name__,
                f"{selected_data.index(problem)}.json",
            )
            executor.output_graph(path)
            budget -= lm.cost

    return orig_budget - budget

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    budget = 30
    samples = list(range(50))
    approaches = [cot, tot, got]

    spent = run(samples, approaches, budget, "mixtral-together")

    logging.info(f"Spent {spent} out of {budget} budget.")