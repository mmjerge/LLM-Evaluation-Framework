import json
import random
import time
import argparse
from tqdm import tqdm
import os
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.callbacks import get_openai_callback
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.callbacks import BaseCallbackHandler
from datasets import load_dataset

class APITracker(BaseCallbackHandler):
    def __init__(self):
        super().__init__()
        self.api_calls = 0
        
    def on_llm_start(self, serialized, prompts, **kwargs):
        self.api_calls += 1

# Set up command line arguments
parser = argparse.ArgumentParser(description='Run ReAct agent with API call tracking on MMLU')
parser.add_argument('--model', type=str, default='claude', 
                    choices=['claude', 'openai', 'vllm'],
                    help='Model provider (claude, openai, or vllm)')
parser.add_argument('--model_name', type=str, default=None,
                    help='Specific model name to use (e.g., gpt-4o, vicuna-13b, etc.)')
parser.add_argument('--num_questions', type=int, default=150,
                    help='Number of questions to run (default: 150)')
parser.add_argument('--vllm_url', type=str, default='http://localhost:8000/v1',
                    help='URL for vLLM API (default: http://localhost:8000/v1)')
parser.add_argument('--mmlu_subject', type=str, default='all',
                    help='MMLU subject to evaluate (default: all)')
args = parser.parse_args()

# Set default model names if not provided
if args.model_name is None:
    if args.model == 'claude':
        args.model_name = "claude-3-5-sonnet-20240620"
    elif args.model == 'openai':
        args.model_name = "gpt-4o"
    elif args.model == 'vllm':
        args.model_name = "mixtral-8x7b-instruct"

print(f"Loading MMLU dataset with subject: {args.mmlu_subject}")
# Load the MMLU dataset
try:
    mmlu_data = load_dataset("cais/mmlu", args.mmlu_subject)
    # Use the test split by default
    mmlu_test_data = mmlu_data["test"]
    print(f"Loaded {len(mmlu_test_data)} questions from MMLU {args.mmlu_subject}")
except Exception as e:
    print(f"Error loading MMLU dataset: {str(e)}")
    raise

# Sample random questions if needed
if args.num_questions < len(mmlu_test_data):
    random_indices = random.sample(range(len(mmlu_test_data)), args.num_questions)
    random_entries = [mmlu_test_data[i] for i in random_indices]
else:
    random_entries = mmlu_test_data
    print(f"Using all {len(random_entries)} questions from the dataset")

# Create a tracker instance
tracker = APITracker()

# Initialize the LLM based on the selected model
if args.model == 'claude':
    llm = ChatAnthropic(
        model=args.model_name,
        temperature=0,
        max_tokens_to_sample=4000,
        callbacks=[tracker]
    )
    print(f"Using Claude model: {args.model_name}")
    
elif args.model == 'openai':
    llm = ChatOpenAI(
        model=args.model_name,
        temperature=0,
        max_tokens=4000,
        callbacks=[tracker]
    )
    print(f"Using OpenAI model: {args.model_name}")
    
else:  # vllm
    llm = ChatOpenAI(
        model=args.model_name,
        temperature=0,
        max_tokens=4000,
        base_url=args.vllm_url,
        api_key="not-needed",  # dummy API key for vLLM
        callbacks=[tracker]
    )
    print(f"Using vLLM model: {args.model_name} at {args.vllm_url}")

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
wikipedia_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

def delay_request(attempt, base_delay=10):
    delay_time = base_delay * (2 ** attempt)
    time.sleep(delay_time)

def sanitize_input(input_str):
    return input_str.replace(" ", "")

def calculator_tool(input_str):
    input_str = sanitize_input(input_str)
    try:
        result = eval(input_str)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

tools = [
    wikipedia_tool,
    Tool(name="Calculator", description="Performs basic arithmetic operations", func=calculator_tool),
]

custom_prompt = hub.pull("hwchase17/react")

agent = create_react_agent(llm, tools, custom_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

max_attempts = 5

# Set output directory based on script location
script_dir = "/scratch/mj6ux/Projects/LLM-Evaluation-Framework/evaluated_methods/ReAct/evaluations/scripts"
# Sanitize the model name for the filename by replacing problematic characters
safe_model_name = args.model_name.replace('/', '_').replace('-', '_').replace('.', '_')
output_file = os.path.join(script_dir, f"mmlu_{args.mmlu_subject}_react_results_{args.model}_{safe_model_name}.jsonl")

# Map answer indices to letter choices
answer_choices = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}

# Open the file once and keep it open
with open(output_file, "w") as f:
    for entry in tqdm(random_entries):
        question = entry["question"]
        choices = [entry["choices"][i] for i in range(4)]
        correct_answer_idx = entry["answer"]
        correct_answer = answer_choices[correct_answer_idx]
        
        # Format the question with multiple choices
        formatted_question = f"{question}\n\nA) {choices[0]}\nB) {choices[1]}\nC) {choices[2]}\nD) {choices[3]}"
        
        if question:
            attempt_counter = 0
            
            # Reset the tracker before each question
            tracker.api_calls = 0
            
            while attempt_counter < max_attempts:
                try:
                    # Execute the agent with the tracker already attached to the LLM
                    response = agent_executor.invoke({
                        "input": f"{formatted_question}\n\nRespond with the letter of your choice (A, B, C, or D) and a brief explanation."
                    })
                    model_answer = response['output']
                    
                    print(f"\nQuestion: {question}")
                    print(f"API Calls: {tracker.api_calls}")
                    break
                except Exception as e:
                    print(f"Error on attempt {attempt_counter + 1}: {str(e)}")
                    attempt_counter += 1
                    delay_request(attempt_counter)
            
            if attempt_counter == max_attempts:
                model_answer = "Error occurred. Best guess: A"
                
            result = {
                "question": question,
                "model_response": model_answer,
                "correct_answer": correct_answer,
                "correct_answer_idx": correct_answer_idx,
                "choices": {
                    "A": choices[0],
                    "B": choices[1],
                    "C": choices[2],
                    "D": choices[3]
                },
                "api_calls": tracker.api_calls,
                "model_provider": args.model,
                "model_name": args.model_name,
                "subject": entry.get("subject", args.mmlu_subject)
            }
            
            json.dump(result, f)
            f.write("\n")
            f.flush() 

print(f"Results saved to {output_file}")