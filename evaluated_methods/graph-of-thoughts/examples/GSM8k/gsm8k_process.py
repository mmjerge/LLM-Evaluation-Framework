import os
import logging
import datetime
import json
from typing import Dict, List, Callable, Union
import sys
import torch
import gc

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from graph_of_thoughts import controller, language_models, operations, prompter, parser
from graph_of_thoughts.language_models.a_claude import AClaude
# from graph_of_thoughts.language_models.mistral_models import Mistral
# from graph_of_thoughts.language_models.together_ai import ATogetherAI
from graph_of_thoughts.language_models.llama_vllm import Llama3VLLM

try:
    from . import utils
except ImportError:
    import utils

# API call tracking
api_call_tracking = {
    "io": {"calls": 0, "tokens_in": 0, "tokens_out": 0},
    "cot": {"calls": 0, "tokens_in": 0, "tokens_out": 0},
    "tot": {"calls": 0, "tokens_in": 0, "tokens_out": 0},
    "got": {"calls": 0, "tokens_in": 0, "tokens_out": 0}
}

def track_api_calls(method_name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            self = args[0]  # The instance
            
            # Extract state from arguments
            state = None
            # Check kwargs first
            if 'state' in kwargs:
                state = kwargs['state']
            # Check the second argument (typically the prompt/query which might contain state info)
            elif len(args) > 1 and isinstance(args[1], dict) and 'method' in args[1]:
                state = args[1]
            # If state still not found, try to look in the instance
            elif hasattr(self, 'state') and self.state and 'method' in self.state:
                state = self.state
            
            # Determine the method type (io, cot, tot, got)
            method_type = "unknown"
            if state and 'method' in state:
                method = state['method']
                if method.startswith('io'):
                    method_type = 'io'
                elif method.startswith('cot'):
                    method_type = 'cot'
                elif method.startswith('tot'):
                    method_type = 'tot'
                elif method.startswith('got'):
                    method_type = 'got'
            else:
                # Default to query's method_name if we can't detect from state
                method_type = method_name
            
            # Update API call counter
            if method_type in api_call_tracking:
                api_call_tracking[method_type]["calls"] += 1
                logging.info(f"API call made for method {method_type}: {api_call_tracking[method_type]['calls']} total calls")
            
            # Call the original function
            result = func(*args, **kwargs)
            
            # Track tokens if available in result
            if method_type in api_call_tracking and result is not None:
                # Try to get token counts from different result formats
                tokens_in = 0
                tokens_out = 0
                
                if isinstance(result, dict):
                    tokens_in = result.get('tokens_in', 0) or 0
                    tokens_out = result.get('tokens_out', 0) or 0
                elif hasattr(result, 'tokens_in') and hasattr(result, 'tokens_out'):
                    tokens_in = getattr(result, 'tokens_in', 0) or 0
                    tokens_out = getattr(result, 'tokens_out', 0) or 0
                
                api_call_tracking[method_type]["tokens_in"] += tokens_in
                api_call_tracking[method_type]["tokens_out"] += tokens_out
                logging.info(f"Tracked tokens for {method_type}: +{tokens_in} in, +{tokens_out} out")
            
            return result
        return wrapper
    return decorator

def apply_tracking_to_models():
    models = [Llama3VLLM, AClaude]
    
    for model_class in models:
        original_query = model_class.query
        model_class.query = track_api_calls("query")(original_query)
        logging.info(f"Applied API tracking to {model_class.__name__}.query")

apply_tracking_to_models()

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

def clean_gpu_memory():
    """Clean up GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

# Patch Llama3VLLM.query to track API calls
original_query = Llama3VLLM.query
@track_api_calls("query")
def tracked_query(self, query, num_responses=1):
    return original_query(self, query, num_responses)
Llama3VLLM.query = tracked_query

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

    # Reset API call tracking before starting
    for method in api_call_tracking:
        api_call_tracking[method] = {"calls": 0, "tokens_in": 0, "tokens_out": 0}

    # Create the model only once to avoid memory issues
    logging.info(f"Initializing model {lm_name}...")
    lm = Llama3VLLM(
        os.path.join(
            os.path.dirname(__file__),
            "../../config.json",
        ),
        model_name="llama3-vllm",  # Use the vLLM config
        cache=True,
    )
    logging.info(f"Model initialized successfully")

    # Reduce memory usage
    config_kwargs = {
        "gpu_memory_utilization": 0.7,  # Use only 70% of GPU memory
        "max_model_len": 2048           # Limit context length to save memory
    }

    # Process all problems with all methods
    try:
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
                
                # Clean GPU memory before each run
                clean_gpu_memory()
                
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
                        "problem": problem,  # Add the entire problem dict to the state
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
                
                # Clean GPU memory after each run
                clean_gpu_memory()
    
    finally:
        # Clean up at the end
        del lm
        clean_gpu_memory()
    
    # Save API call tracking data
    with open(os.path.join(results_folder, "api_call_tracking.json"), "w") as f:
        json.dump(api_call_tracking, f, indent=2)
    
    # Log API call tracking information
    logging.info("API Call Tracking Stats:")
    for method, stats in api_call_tracking.items():
        logging.info(f"  {method}: {stats['calls']} calls, {stats['tokens_in']} tokens in, {stats['tokens_out']} tokens out")
    
    return orig_budget - budget

if __name__ == "__main__":
    # Process fewer samples at a time to avoid memory issues
    budget = 30
    samples = list(range(150)) 
    approaches = [cot, tot, got]
    
    # Set environment variables for PyTorch to reduce memory fragmentation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"

    # Run with the vLLM model
    spent = run(samples, approaches, budget, "llama3-vllm")

    logging.info(f"Spent {spent} out of {budget} budget.")
    
    # Print final API call tracking stats
    print("\nAPI Call Tracking Stats:")
    for method, stats in api_call_tracking.items():
        print(f"  {method}: {stats['calls']} calls, {stats['tokens_in']} tokens in, {stats['tokens_out']} tokens out")