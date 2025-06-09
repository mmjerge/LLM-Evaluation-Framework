import os
import os.path
import re
import logging
import datetime
import json
import traceback
from statistics import fmean
import csv
from typing import Dict, List, Callable, Set, Union
from graph_of_thoughts import controller, language_models, operations, prompter, parser
from graph_of_thoughts.language_models.a_claude import AClaude
from graph_of_thoughts.language_models.mistral_models import Mistral
from graph_of_thoughts.language_models.together_ai import ATogetherAI


class MEDQAPrompter(prompter.Prompter):
    """
    MEDQAPrompter provides the generation of prompts specific to the MEDQA
    example for the language models.

    Inherits from the Prompter class and implements its abstract methods.
    """

    qa_prompt_start = """Answer the following medical question. Choose the most appropriate answer from the given options.

Question: {question}

Options:
A: {option_a}
B: {option_b}
C: {option_c}
D: {option_d}
E: {option_e}

Provide your answer as a single letter (A, B, C, D, or E) between the tags <Answer> and </Answer>, without any additional text.
"""

    qa_prompt_cot_start = """Answer the following medical question. Choose the most appropriate answer from the given options.

Question: {question}

Options:
A: {option_a}
B: {option_b}
C: {option_c}
D: {option_d}
E: {option_e}

Think through this step by step before answering. You can generate any intermediate thoughts you want, but the final output should be a single letter (A, B, C, D, or E), placed between the two tags <Answer> and </Answer>.
"""

    improve_answer_prompt = """Reconsider the following medical question and your previous answer.

Question: {question}

Options:
A: {option_a}
B: {option_b}
C: {option_c}
D: {option_d}
E: {option_e}

Your previous answer was: {current}

Reconsider your answer and provide your final answer as a single letter (A, B, C, D, or E) between the tags <Answer> and </Answer>, without any additional text.
"""

    # Renamed from score_prompt to score_prompt_template to avoid the naming conflict
    score_prompt_template = """Evaluate how confident you are in the following answer to a medical question.

Question: {question}

Options:
A: {option_a}
B: {option_b}
C: {option_c}
D: {option_d}
E: {option_e}

Proposed answer: {answer}

Score the confidence on a scale of 0 to 10, where 0 means completely uncertain and 10 means absolutely certain.
Provide your reasoning for the score, and then put the final confidence score between the tags <Confidence> and </Confidence>, without any additional text within those tags.
"""

    aggregate_prompt_base = """Consider the following answers to a medical question.

Question: {question}

Options:
A: {option_a}
B: {option_b}
C: {option_c}
D: {option_d}
E: {option_e}

Answers from different sources:
"""

    aggregate_prompt_answer = """Source {num}: {answer}
"""

    aggregate_prompt_end = """
Based on these answers, provide your final answer as a single letter (A, B, C, D, or E) between the tags <Answer> and </Answer>, without any additional text.
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
        logging.info(f"Creating aggregation prompt for {len(state_dicts)} state dicts")
        
        prompt = self.aggregate_prompt_base.format(
            question=state_dicts[0]["question"],
            option_a=state_dicts[0]["options"]["A"],
            option_b=state_dicts[0]["options"]["B"],
            option_c=state_dicts[0]["options"]["C"],
            option_d=state_dicts[0]["options"]["D"],
            option_e=state_dicts[0]["options"]["E"],
        )
        
        for i, state_dict in enumerate(state_dicts):
            prompt += self.aggregate_prompt_answer.format(
                num=i+1, answer=state_dict["current"]
            )
            
        prompt += self.aggregate_prompt_end
        return prompt

    def generate_prompt(
        self,
        num_branches: int,
        method: str,
        question: str,
        options: Dict[str, str],
        current: str = None,
        **kwargs,
    ) -> str:
        """
        Generate a generate prompt for the language model.

        :param num_branches: The number of responses the prompt should ask the LM to generate.
        :type num_branches: int
        :param method: Method for which the generate prompt is generated.
        :type method: str
        :param question: The medical question to be answered.
        :type question: str
        :param options: The multiple choice options.
        :type options: Dict[str, str]
        :param current: The intermediate solution.
        :type current: str
        :param kwargs: Additional keyword arguments.
        :return: The generate prompt.
        :rtype: str
        :raise AssertionError: If method is not implemented yet.
        """
        logging.info(f"Creating generate prompt for method {method}, current answer: {current}")
        
        if method.startswith("io") or method.startswith("cot"):
            if method.startswith("io"):
                prompt = self.qa_prompt_start.format(
                    question=question,
                    option_a=options["A"],
                    option_b=options["B"],
                    option_c=options["C"],
                    option_d=options["D"],
                    option_e=options["E"],
                )
            else:
                prompt = self.qa_prompt_cot_start.format(
                    question=question,
                    option_a=options["A"],
                    option_b=options["B"],
                    option_c=options["C"],
                    option_d=options["D"],
                    option_e=options["E"],
                )
            return prompt
        elif method.startswith("tot") or method.startswith("got"):
            if current is None or current == "":
                if method.startswith("tot"):
                    prompt = self.qa_prompt_start.format(
                        question=question,
                        option_a=options["A"],
                        option_b=options["B"],
                        option_c=options["C"],
                        option_d=options["D"],
                        option_e=options["E"],
                    )
                else:
                    prompt = self.qa_prompt_cot_start.format(
                        question=question,
                        option_a=options["A"],
                        option_b=options["B"],
                        option_c=options["C"],
                        option_d=options["D"],
                        option_e=options["E"],
                    )
                return prompt
            else:
                prompt = self.improve_answer_prompt.format(
                    question=question,
                    option_a=options["A"],
                    option_b=options["B"],
                    option_c=options["C"],
                    option_d=options["D"],
                    option_e=options["E"],
                    current=current,
                )
                return prompt
        else:
            assert False, "Method not implemented yet."

    def score_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        """
        Generate a score prompt for the language model.

        :param state_dicts: The thought states that should be scored.
        :type state_dicts: List[Dict]
        :param kwargs: Additional keyword arguments.
        :return: The score prompt.
        :rtype: str
        :raise AssertionError: If more than one thought state is supplied.
        """
        assert len(state_dicts) == 1, "Only one state is allowed for scoring."
        
        logging.info(f"Creating score prompt for answer: {state_dicts[0]['current']}")
        
        prompt = self.score_prompt_template.format(
            question=state_dicts[0]["question"],
            option_a=state_dicts[0]["options"]["A"],
            option_b=state_dicts[0]["options"]["B"],
            option_c=state_dicts[0]["options"]["C"],
            option_d=state_dicts[0]["options"]["D"],
            option_e=state_dicts[0]["options"]["E"],
            answer=state_dicts[0]["current"],
        )
        return prompt

    def improve_prompt(self, **kwargs) -> str:
        """
        Generate an improve prompt for the language model.

        :param kwargs: Additional keyword arguments.
        :return: The improve prompt.
        :rtype: str
        """
        pass

    def validation_prompt(self, **kwargs) -> str:
        """
        Generate a validation prompt for the language model.

        :param kwargs: Additional keyword arguments.
        :return: The validation prompt.
        :rtype: str
        """
        pass


class MEDQAParser(parser.Parser):
    """
    MEDQAParser provides the parsing of language model responses specific to the
    MEDQA example.

    Inherits from the Parser class and implements its abstract methods.
    """

    def __init__(self) -> None:
        """
        Inits the response cache.
        """
        self.cache = {}

    def strip_answer_helper(self, text: str, tag: str = "") -> str:
        """
        Helper function to remove tags from a text.

        :param text: The input text.
        :type text: str
        :param tag: The tag to be stripped. Defaults to "".
        :type tag: str
        :return: The stripped text.
        :rtype: str
        """
        text = text.strip()
        if "Output:" in text:
            text = text[text.index("Output:") + len("Output:") :].strip()
        if tag != "":
            start = text.rfind(f"<{tag}>")
            end = text.rfind(f"</{tag}>")
            if start != -1 and end != -1:
                text = text[start + len(f"<{tag}>") : end].strip()
            elif start != -1:
                logging.warning(
                    f"Only found the start tag <{tag}> in answer: {text}. Returning everything after the tag."
                )
                text = text[start + len(f"<{tag}>") :].strip()
            elif end != -1:
                logging.warning(
                    f"Only found the end tag </{tag}> in answer: {text}. Returning everything before the tag."
                )
                text = text[:end].strip()
            else:
                logging.warning(
                    f"Could not find any tag {tag} in answer: {text}. Returning the full answer."
                )
        
        if tag == "Answer" and len(text) > 1:
            option_match = re.search(r'[A-E]', text)
            if option_match:
                text = option_match.group(0)
                logging.info(f"Extracted single letter answer: {text}")
                
        return text

    def parse_aggregation_answer(
        self, states: List[Dict], texts: List[str]
    ) -> Union[Dict, List[Dict]]:
        """
        Parse the response from the language model for an aggregation prompt.

        :param states: The thought states used to generate the prompt.
        :type states: List[Dict]
        :param texts: The responses to the prompt from the language model.
        :type texts: List[str]
        :return: The new thought states after parsing the responses from the language model.
        :rtype: Union[Dict, List[Dict]]
        """
        new_states = []
        logging.info(f"Parsing aggregation answers, got {len(texts)} responses from {len(states)} input states")
        
        for i, text in enumerate(texts):
            logging.info(f"Parsing aggregation response {i+1}: {text[:100]}...")
            text = self.strip_answer_helper(text, "Answer")
            logging.info(f"Extracted aggregated answer: {text}")
            new_state = states[0].copy()
            new_state["current"] = text
            new_states.append(new_state)
        
        logging.info(f"Aggregated to {len(new_states)} new states with answers: {[s['current'] for s in new_states]}")
        return new_states

    def parse_generate_answer(self, state: Dict, texts: List[str]) -> List[Dict]:
        """
        Parse the response from the language model for a generate prompt.

        :param state: The thought state used to generate the prompt.
        :type state: Dict
        :param texts: The responses to the prompt from the language model.
        :type texts: List[str]
        :return: The new thought states after parsing the responses from the language model.
        :rtype: List[Dict]
        """
        new_states = []
        logging.info(f"Parsing generate answers, got {len(texts)} responses")
        
        for i, text in enumerate(texts):
            logging.info(f"Parsing generate response {i+1}: {text[:100]}...")
            text = self.strip_answer_helper(text, "Answer")
            logging.info(f"Extracted answer: {text}")
            new_state = state.copy()
            new_state["current"] = text
            new_states.append(new_state)
        
        logging.info(f"Generated {len(new_states)} new states with answers: {[s['current'] for s in new_states]}")
        return new_states

    def parse_score_answer(self, states: List[Dict], texts: List[str]) -> List[float]:
        """
        Parse the response from the language model for a score prompt.

        :param states: The thought states used to generate the prompt.
        :type states: List[Dict]
        :param texts: The responses to the prompt from the language model.
        :type texts: List[str]
        :return: The scores for the thought states.
        :rtype: List[float]
        :raise AssertionError: If the number of thought states is not one.
        """
        assert len(states) == 1, "Only one state is allowed for scoring."
        
        confidence_scores = []
        logging.info(f"Parsing score answer for state with current answer: {states[0]['current']}")
        
        for i, text in enumerate(texts):
            logging.info(f"Parsing response {i+1}/{len(texts)}: {text[:100]}...")
            
            confidence_answer = self.strip_answer_helper(text, "Confidence")
            logging.info(f"Extracted confidence answer: {confidence_answer}")
            
            confidence_res = re.findall(r"\d+\.?\d*", confidence_answer)
            
            if confidence_res:
                confidence_scores.append(float(confidence_res[-1]))
                logging.info(f"Found confidence score: {confidence_res[-1]}")
            else:
                logging.warning("No confidence score in tagged section, searching whole response")
                all_numbers = re.findall(r"\d+\.?\d*", text)
                if all_numbers:
                    valid_scores = [float(n) for n in all_numbers if float(n) <= 10]
                    if valid_scores:
                        confidence_scores.append(valid_scores[-1])
                        logging.info(f"Found fallback confidence score: {valid_scores[-1]}")
                    else:
                        logging.warning("No valid confidence scores found, using default 5.0")
                        confidence_scores.append(5.0) 
                else:
                    logging.warning("No numbers found in response, using default 5.0")
                    confidence_scores.append(5.0) 
        
        if not confidence_scores:
            logging.error("Could not extract any confidence scores, using default 5.0")
            return [5.0]
        
        mean_confidence = fmean(confidence_scores) / 10.0
        answer = states[0]["current"].strip()
        correct_answer = states[0]["answer_idx"]
        
        if answer == correct_answer:
            score = mean_confidence
        else:
            score = (1 - mean_confidence) * 0.2
        
        logging.info(f"Answer: {answer}, Correct: {correct_answer}, Mean confidence: {mean_confidence}, Final score: {score}")
        return [score]

    def parse_improve_answer(self, state: Dict, texts: List[str]) -> Dict:
        """
        Parse the response from the language model for an improve prompt.

        :param state: The thought state used to generate the prompt.
        :type state: Dict
        :param texts: The responses to the prompt from the language model.
        :type texts: List[str]
        :return: The new thought state after parsing the responses from the language model.
        :rtype: Dict
        """
        pass

    def parse_validation_answer(self, state: Dict, texts: List[str]) -> bool:
        """
        Parse the response from the language model for a validation prompt.

        :param state: The thought state used to generate the prompt.
        :type state: Dict
        :param texts: The responses to the prompt from the language model.
        :type texts: List[str]
        :return: Whether the thought state is valid or not.
        :rtype: bool
        """
        pass


def io() -> operations.GraphOfOperations:
    """
    Generates the Graph of Operations for the IO method.

    :return: Graph of Operations
    :rtype: GraphOfOperations
    """
    operations_graph = operations.GraphOfOperations()

    generate_op = operations.Generate(1, 1)
    operations_graph.append_operation(generate_op)
    
    score_op = operations.Score(3, False)
    score_op.add_predecessor(generate_op)
    operations_graph.append_operation(score_op)

    return operations_graph


def cot() -> operations.GraphOfOperations:
    """
    Generates the Graph of Operations for the CoT method.

    :return: Graph of Operations
    :rtype: GraphOfOperations
    """
    operations_graph = operations.GraphOfOperations()

    generate_op = operations.Generate(1, 1)
    operations_graph.append_operation(generate_op)
    
    score_op = operations.Score(3, False)
    score_op.add_predecessor(generate_op)
    operations_graph.append_operation(score_op)

    return operations_graph


def tot() -> operations.GraphOfOperations:
    """
    Generates the Graph of Operations for the ToT method.

    :return: Graph of Operations
    :rtype: GraphOfOperations
    """
    operations_graph = operations.GraphOfOperations()

    branch_factor = 5

    generate_op = operations.Generate(1, branch_factor)
    operations_graph.append_operation(generate_op)
    
    score_op = operations.Score(3, False)
    score_op.add_predecessor(generate_op)
    operations_graph.append_operation(score_op)
    
    keep_best_1 = operations.KeepBestN(1, True)
    keep_best_1.add_predecessor(score_op)
    operations_graph.append_operation(keep_best_1)

    for _ in range(2):
        generate_op = operations.Generate(1, branch_factor)
        generate_op.add_predecessor(keep_best_1)
        operations_graph.append_operation(generate_op)
        
        score_op = operations.Score(3, False)
        score_op.add_predecessor(generate_op)
        operations_graph.append_operation(score_op)
        
        keep_best_2 = operations.KeepBestN(1, True)
        keep_best_2.add_predecessor(score_op)
        keep_best_2.add_predecessor(keep_best_1)
        operations_graph.append_operation(keep_best_2)
        keep_best_1 = keep_best_2

    return operations_graph


def got() -> operations.GraphOfOperations:
    """
    Generates the Graph of Operations for the GoT method.
    Simplified version to avoid potential infinite loops.

    :return: Graph of Operations
    :rtype: GraphOfOperations
    """
    operations_graph = operations.GraphOfOperations()

    generate_op = operations.Generate(1, 5)
    operations_graph.append_operation(generate_op)
    
    score_op = operations.Score(3, False)
    score_op.add_predecessor(generate_op)
    operations_graph.append_operation(score_op)
    
    keep_best = operations.KeepBestN(1, True)
    keep_best.add_predecessor(score_op)
    operations_graph.append_operation(keep_best)
    
    return operations_graph

# def run(
#     data_ids: List[int],
#     methods: List[Callable[[], operations.GraphOfOperations]],
#     budget: float,
#     lm_name: str,
# ) -> float:
#     """
#     Controller function that executes each specified method for each specified
#     sample while the budget is not exhausted.

#     :param data_ids: Indices of the sample to be run.
#     :type data_ids: List[int]
#     :param methods: List of functions to generate Graphs of Operations.
#     :type methods: Each function generates a Graph of Operation.
#     :param budget: Language model budget for the execution in dollars.
#     :type budget: float
#     :param lm_name: Name of the language model to be used.
#     :type lm_name: str
#     :return: Spent budget in dollars.
#     :rtype: float
#     """

#     orig_budget = budget
#     data_path = os.path.join(os.path.dirname(__file__), "test.jsonl")
#     data = []
    
#     with open(data_path, "r", encoding="utf8") as f:
#         for i, line in enumerate(f):
#             item = json.loads(line)
#             data.append([i, item["meta_info"], item["question"], item["options"], item["answer_idx"], item["answer"]])

#     if data_ids is None or len(data_ids) == 0:
#         data_ids = list(range(len(data)))
#     selected_data = [data[i] for i in data_ids]

#     results_dir = os.path.join(os.path.dirname(__file__), "results")

#     if not os.path.exists(results_dir):
#         os.makedirs(results_dir)
#     timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#     extra_info = f"{lm_name}_{'-'.join([method.__name__ for method in methods])}"
#     folder_name = f"{extra_info}_{timestamp}"
#     results_folder = os.path.join(results_dir, folder_name)
#     os.makedirs(results_folder)

#     config = {
#         "data": [item[0] for item in selected_data],
#         "methods": [method.__name__ for method in methods],
#         "lm": lm_name,
#         "budget": budget,
#     }
#     with open(os.path.join(results_folder, "config.json"), "w") as f:
#         json.dump(config, f)

#     # Configure logging to both file and console
#     log_file = os.path.join(results_folder, "log.log")
#     logging.basicConfig(
#         level=logging.INFO,
#         format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
#         handlers=[
#             logging.FileHandler(log_file, mode="w"),
#             logging.StreamHandler()  # This will also log to console
#         ]
#     )

#     for method in methods:
#         os.makedirs(os.path.join(results_folder, method.__name__))

#     for data_item in selected_data:
#         logging.info(f"Running data {data_item[0]}: {data_item[1]}")
#         if budget <= 0.0:
#             logging.error(f"Budget has been depleted, stopping.")
#             break
            
#         for method in methods:
#             logging.info(f"Running method {method.__name__}")
#             logging.info(f"Budget left: {budget}")
#             if budget <= 0.0:
#                 logging.error(f"Budget has been depleted, stopping.")
#                 break
            
#             # Use absolute path to config or try to find it relative to the script
#             try:
#                 # Try direct path first
#                 config_path = "/scratch/mj6ux/Projects/LLM-Evaluation-Framework/evaluated_methods/graph-of-thoughts/config.json"
#                 if not os.path.exists(config_path):
#                     # Fallback to constructing path
#                     base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#                     config_path = os.path.join(base_dir, "config.json")
                    
#                 logging.info(f"Using config file at: {config_path}")
                
#                 if not os.path.exists(config_path):
#                     raise FileNotFoundError(f"Config file not found at {config_path}")
                
#                 lm = language_models.ChatGPT(
#                     config_path,
#                     model_name=lm_name,
#                     cache=True,
#                 )
#                 operations_graph = method()
                
#                 # Log the operations graph structure for debugging
#                 logging.info("Operations graph structure:")
#                 for i, op in enumerate(operations_graph.operations):
#                     predecessors = [operations_graph.operations.index(p) for p in op.predecessors] if hasattr(op, 'predecessors') else []
#                     logging.info(f"  Operation {i}: {op.__class__.__name__}, Predecessors: {predecessors}")
                
#                 # Create the initial state
#                 initial_state = {
#                     "question": data_item[2],
#                     "options": data_item[3],
#                     "answer_idx": data_item[4],
#                     "answer_text": data_item[5],
#                     "current": "",
#                     "method": method.__name__,
#                 }
                
#                 executor = controller.Controller(
#                     lm,
#                     operations_graph,
#                     MEDQAPrompter(),
#                     MEDQAParser(),
#                     initial_state,
#                 )
                
#                 executor.run()
                
#                 # Debug output after execution
#                 logging.info("Execution completed. Operation results:")
#                 for i, op in enumerate(operations_graph.operations):
#                     logging.info(f"  Operation {i}: {op.__class__.__name__}, Thoughts: {len(op.thoughts)}")
                    
#                 path = os.path.join(
#                     results_folder,
#                     method.__name__,
#                     f"{data_item[0]}.json",
#                 )
#                 executor.output_graph(path)
#                 budget -= lm.cost
                
#             except Exception as e:
#                 logging.error(f"Exception in {method.__name__} for data {data_item[0]}: {e}")
#                 logging.error(traceback.format_exc())

#     return orig_budget - budget

def run(
    data_ids: List[int],
    methods: List[Callable[[], operations.GraphOfOperations]],
    budget: float,
    lm_name: str,
) -> float:
    """
    Controller function that executes each specified method for each specified
    sample while the budget is not exhausted.

    :param data_ids: Indices of the sample to be run.
    :type data_ids: List[int]
    :param methods: List of functions to generate Graphs of Operations.
    :type methods: Each function generates a Graph of Operation.
    :param budget: Language model budget for the execution in dollars.
    :type budget: float
    :param lm_name: Name of the language model to be used.
    :type lm_name: str
    :return: Spent budget in dollars.
    :rtype: float
    """

    orig_budget = budget
    data_path = os.path.join(os.path.dirname(__file__), "test.jsonl")
    data = []
    
    method_results = {method.__name__: {"correct": 0, "total": 0} for method in methods}
    all_predictions = {method.__name__: [] for method in methods}
    
    with open(data_path, "r", encoding="utf8") as f:
        for i, line in enumerate(f):
            item = json.loads(line)
            data.append([i, item["meta_info"], item["question"], item["options"], item["answer_idx"], item["answer"]])

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
        "data": [item[0] for item in selected_data],
        "methods": [method.__name__ for method in methods],
        "lm": lm_name,
        "budget": budget,
    }
    with open(os.path.join(results_folder, "config.json"), "w") as f:
        json.dump(config, f)

    log_file = os.path.join(results_folder, "log.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w"),
            logging.StreamHandler()
        ]
    )

    for method in methods:
        os.makedirs(os.path.join(results_folder, method.__name__))

    for data_item in selected_data:
        logging.info(f"Running data {data_item[0]}: {data_item[1]}")
        if budget <= 0.0:
            logging.error(f"Budget has been depleted, stopping.")
            break
            
        for method in methods:
            method_name = method.__name__
            logging.info(f"Running method {method_name}")
            logging.info(f"Budget left: {budget}")
            if budget <= 0.0:
                logging.error(f"Budget has been depleted, stopping.")
                break
            
            try:
                config_path = "/scratch/mj6ux/Projects/LLM-Evaluation-Framework/evaluated_methods/graph-of-thoughts/config.json"
                if not os.path.exists(config_path):
                    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                    config_path = os.path.join(base_dir, "config.json")
                    
                logging.info(f"Using config file at: {config_path}")
                
                if not os.path.exists(config_path):
                    raise FileNotFoundError(f"Config file not found at {config_path}")
                
                lm = language_models.ChatGPT(
                    config_path,
                    model_name=lm_name,
                    cache=True,
                )
                operations_graph = method()
                
                logging.info("Operations graph structure:")
                for i, op in enumerate(operations_graph.operations):
                    predecessors = [operations_graph.operations.index(p) for p in op.predecessors] if hasattr(op, 'predecessors') else []
                    logging.info(f"  Operation {i}: {op.__class__.__name__}, Predecessors: {predecessors}")
                
                initial_state = {
                    "question": data_item[2],
                    "options": data_item[3],
                    "answer_idx": data_item[4],
                    "answer_text": data_item[5],
                    "current": "",
                    "method": method_name,
                }
                
                executor = controller.Controller(
                    lm,
                    operations_graph,
                    MEDQAPrompter(),
                    MEDQAParser(),
                    initial_state,
                )
                
                executor.run()
                
                logging.info("Execution completed. Operation results:")
                for i, op in enumerate(operations_graph.operations):
                    logging.info(f"  Operation {i}: {op.__class__.__name__}, Thoughts: {len(op.thoughts)}")
                
                final_answer = None
                for op in reversed(operations_graph.operations):
                    if isinstance(op, operations.KeepBestN) and op.thoughts:
                        final_answer = op.thoughts[0].state["current"]
                        break
                
                if final_answer:
                    method_results[method_name]["total"] += 1
                    correct_answer = data_item[4] 
                    prediction = {"id": data_item[0], "correct": correct_answer, "predicted": final_answer}
                    all_predictions[method_name].append(prediction)
                    
                    if final_answer == correct_answer:
                        method_results[method_name]["correct"] += 1
                        logging.info(f"✓ CORRECT: Question {data_item[0]} - {method_name} predicted {final_answer}")
                    else:
                        logging.info(f"✗ INCORRECT: Question {data_item[0]} - {method_name} predicted {final_answer}, correct was {correct_answer}")
                else:
                    logging.warning(f"No final answer found for question {data_item[0]} with method {method_name}")
                
                path = os.path.join(
                    results_folder,
                    method_name,
                    f"{data_item[0]}.json",
                )
                executor.output_graph(path)
                budget -= lm.cost
                
            except Exception as e:
                logging.error(f"Exception in {method_name} for data {data_item[0]}: {e}")
                logging.error(traceback.format_exc())
                method_results[method_name]["total"] += 1 

    summary = {
        "model": lm_name,
        "timestamp": timestamp,
        "methods": {},
        "budget": {
            "initial": orig_budget,
            "spent": orig_budget - budget,
            "remaining": budget
        },
        "details": all_predictions
    }
    
    logging.info("\n" + "="*50)
    logging.info("ACCURACY SUMMARY")
    logging.info("="*50)
    
    for method_name, results in method_results.items():
        correct = results["correct"]
        total = results["total"]
        accuracy = (correct / total * 100) if total > 0 else 0
        
        summary["methods"][method_name] = {
            "correct": correct,
            "total": total,
            "accuracy": accuracy
        }
        
        logging.info(f"{method_name}: {correct}/{total} = {accuracy:.2f}%")
    
    logging.info("="*50)
    
    # Save detailed summary to a JSON file
    summary_path = os.path.join(results_folder, "accuracy_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    # Also save a simple CSV for easy analysis
    csv_path = os.path.join(results_folder, "accuracy_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "Correct", "Total", "Accuracy"])
        for method_name, results in method_results.items():
            correct = results["correct"]
            total = results["total"]
            accuracy = (correct / total * 100) if total > 0 else 0
            writer.writerow([method_name, correct, total, f"{accuracy:.2f}%"])

    return orig_budget - budget


if __name__ == "__main__":
    """
    Input: Medical questions with multiple choice options
    Output: Predicted answer (A, B, C, D, or E)
    Evaluation: Accuracy of predictions
    """
    budget = 5000
    samples = [item for item in range(0, 150)]
    
    approaches = [got]
    
    spent = run(samples, approaches, budget, "chatgpt")
    
    logging.info(f"Spent {spent} out of {budget} budget.")