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
    
class AQUAPrompter(prompter.Prompter):
    """
    AQUAPrompter provides the generation of prompts specific to the AQUA
    dataset for the language models.
    """

    solve_prompt = """<Instruction> Answer the following multiple-choice question. Provide your reasoning step by step, then give your final answer prefixed with "Answer: ". </Instruction>

<Examples>
Question: {question}
Options:
{options}
</Examples>
"""

    cot_prompt = """<Instruction> Solve the following multiple-choice question step by step. Show your reasoning for each step. Provide your final answer at the end, prefixed with "Answer: ". </Instruction>

<Examples>
Question: {question}
Options:
{options}
</Examples>
"""

    tot_improve_prompt = """<Instruction> Review the following solution to a multiple-choice question. Identify any errors or areas for improvement in the reasoning. Then, provide an improved solution with corrected steps and calculations. </Instruction>

<Examples>
Question: {question}
Options:
{options}

Original Solution:
{solution}

Improved Solution:
</Examples>
"""

    def generate_prompt(
        self, num_branches: int, original: Dict, current: str, method: str, **kwargs
    ) -> str:
        """
        Generate a prompt for the language model.

        :param num_branches: The number of responses the prompt should ask the LM to generate.
        :param original: Original problem statement (a dictionary containing 'question' and 'options').
        :param current: Intermediate solution.
        :param method: Method for which the generate prompt is generated.
        :return: The generate prompt.
        """
        options_text = "\n".join(original['options'])
        if method.startswith("io"):
            return self.solve_prompt.format(question=original['question'], options=options_text)
        elif method.startswith("cot"):
            return self.cot_prompt.format(question=original['question'], options=options_text)
        elif method.startswith("tot"):
            if current is None or current == "":
                return self.cot_prompt.format(question=original['question'], options=options_text)
            return self.tot_improve_prompt.format(question=original['question'], options=options_text, solution=current)
        elif method.startswith("got"):
            return self.cot_prompt.format(question=original['question'], options=options_text)

    def aggregation_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        problem = state_dicts[0]['original']
        solutions = [state['current'] for state in state_dicts]
        
        prompt = f"""<Instruction> Review the following solutions to the given multiple-choice question. Analyze each solution, identifying strengths and weaknesses. Then, synthesize the best elements from each solution to create a comprehensive, step-by-step final solution. Provide your final answer at the end, prefixed with "Answer: ". </Instruction>

Question: {problem['question']}
Options:
{chr(10).join(problem['options'])}

Solution 1:
{solutions[0]}

Solution 2:
{solutions[1]}

Synthesized Solution:
"""
        return prompt

    def improve_prompt(self, **kwargs) -> str:
        pass

    def validation_prompt(self, **kwargs) -> str:
        pass

    def score_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        pass

class AQUAParser(parser.Parser):
    """
    AQUAParser provides the parsing of language model responses specific to
    the AQUA dataset.
    """

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
        new_states = []
        for text in texts:
            new_state = states[0].copy()
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
    operations_graph.append_operation(operations.GroundTruth(utils.test_aqua))

    return operations_graph

def cot() -> operations.GraphOfOperations:
    operations_graph = operations.GraphOfOperations()
    operations_graph.append_operation(operations.Generate(1, 1))
    operations_graph.append_operation(operations.Score(1, False, utils.extract_answer))
    operations_graph.append_operation(operations.GroundTruth(utils.test_aqua))
    return operations_graph

def tot() -> operations.GraphOfOperations:
    operations_graph = operations.GraphOfOperations()

    generate_op = operations.Generate(1, 2)
    operations_graph.append_operation(generate_op)

    score_op = operations.Score(1, False, utils.extract_answer)
    score_op.add_predecessor(generate_op)
    operations_graph.append_operation(score_op)

    keep_best = operations.KeepBestN(1, False)
    keep_best.add_predecessor(score_op)
    operations_graph.append_operation(keep_best)

    improve_generate = operations.Generate(1, 2)
    improve_generate.add_predecessor(keep_best)
    operations_graph.append_operation(improve_generate)

    improve_score = operations.Score(1, False, utils.extract_answer)
    improve_score.add_predecessor(improve_generate)
    operations_graph.append_operation(improve_score)

    final_keep_best = operations.KeepBestN(1, False)
    final_keep_best.add_predecessor(improve_score)
    operations_graph.append_operation(final_keep_best)

    operations_graph.append_operation(operations.GroundTruth(utils.test_aqua))

    return operations_graph

def got() -> operations.GraphOfOperations:
    operations_graph = operations.GraphOfOperations()

    # Generate a single solution
    generate_op = operations.Generate(1, 1)
    operations_graph.append_operation(generate_op)

    # Evaluate against ground truth
    ground_truth_op = operations.GroundTruth(utils.test_aqua)
    ground_truth_op.add_predecessor(generate_op)
    operations_graph.append_operation(ground_truth_op)

    return operations_graph

def run(
    data_ids: List[int],
    methods: List[Callable[[], operations.GraphOfOperations]],
    budget: float,
    lm_name: str,
) -> float:
    orig_budget = budget
    data_path = os.path.join(os.path.dirname(__file__), "test.jsonl")
    data = []
    with open(data_path, "r") as f:
        for line in f:
            data.append(json.loads(line.strip()))

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

    config = {
        "data": [item['question'] for item in selected_data],
        "methods": [method.__name__ for method in methods],
        "lm": lm_name,
        "budget": budget,
    }
    with open(os.path.join(results_folder, "config.json"), "w") as f:
        json.dump(config, f)

    logging.basicConfig(
        filename=os.path.join(results_folder, "log.log"),
        filemode="w",
        format="%(name)s - %(levelname)s - %(message)s",
        level=logging.DEBUG,
    )

    for method in methods:
        os.makedirs(os.path.join(results_folder, method.__name__))

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
            lm = ATogetherAI(
                os.path.join(
                    os.path.dirname(__file__),
                    "../../graph_of_thoughts/language_models/config.json",
                ),
                model_name=lm_name,
                cache=True,
            )
            operations_graph = method()
            executor = controller.Controller(
                lm,
                operations_graph,
                AQUAPrompter(),
                AQUAParser(),
                {
                    "original": problem,
                    "current": "",
                    "method": method.__name__,
                },
            )
            try:
                logging.info(f"Starting execution for method {method.__name__}")
                result = executor.run()
                logging.info(f"Execution completed for method {method.__name__}")
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
    budget = 30
    samples = list(range(50))
    approaches = [cot, tot, got]

    spent = run(samples, approaches, budget, "mixtral-together")

    logging.info(f"Spent {spent} out of {budget} budget.")