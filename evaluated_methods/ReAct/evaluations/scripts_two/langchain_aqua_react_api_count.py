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

parser = argparse.ArgumentParser(description='Run ReAct agent with API call tracking on AQuA')
parser.add_argument('--model', type=str, default='claude', 
                    choices=['claude', 'openai', 'vllm'],
                    help='Model provider (claude, openai, or vllm)')
parser.add_argument('--model_name', type=str, default=None,
                    help='Specific model name to use (e.g., gpt-4o, vicuna-13b, etc.)')
parser.add_argument('--num_questions', type=int, default=150,
                    help='Number of questions to run (default: 150)')
parser.add_argument('--vllm_url', type=str, default='http://localhost:8000/v1',
                    help='URL for vLLM API (default: http://localhost:8000/v1)')
parser.add_argument('--resume', action='store_true',
                    help='Resume from previous run')
args = parser.parse_args()

if args.model_name is None:
    if args.model == 'claude':
        args.model_name = "claude-3-5-sonnet-20240620"
    elif args.model == 'openai':
        args.model_name = "gpt-4o"
    elif args.model == 'vllm':
        args.model_name = "mixtral-8x7b-instruct"

print("Loading AQuA dataset")
try:
    aqua_data = load_dataset("deepmind/aqua_rat", "raw")
    # Use the test split by default
    aqua_test_data = aqua_data["test"]
    print(f"Loaded {len(aqua_test_data)} questions from AQuA-RAT test set")
except Exception as e:
    print(f"Error loading AQuA-RAT dataset: {str(e)}")
    raise

random.seed(42)
all_indices = list(range(len(aqua_test_data)))
if args.num_questions < len(aqua_test_data):
    selected_indices = random.sample(all_indices, args.num_questions)
else:
    selected_indices = all_indices
    print(f"Using all {len(selected_indices)} questions from the dataset")

tracker = APITracker()

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
    
else:  
    llm = ChatOpenAI(
        model=args.model_name,
        temperature=0,
        max_tokens=4000,
        base_url=args.vllm_url,
        api_key="not-needed", 
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
    Tool(name="Wikipedia", description="Search Wikipedia for information", func=wikipedia_tool.run),
    Tool(name="Calculator", description="Performs basic arithmetic operations", func=calculator_tool),
]

custom_prompt = hub.pull("hwchase17/react")

agent = create_react_agent(llm, tools, custom_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

max_attempts = 5

script_dir = "/scratch/mj6ux/Projects/LLM-Evaluation-Framework/evaluated_methods/ReAct/evaluations/scripts"
safe_model_name = args.model_name.replace('/', '_').replace('-', '_').replace('.', '_')
output_file = os.path.join(script_dir, f"aqua_rat_react_results_{args.model}_{safe_model_name}.jsonl")

checkpoint_file = os.path.join(script_dir, f"checkpoint_{args.model}_{safe_model_name}.json")

def load_processed_questions():
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Error reading checkpoint file. Starting from scratch.")
            return []
    return []

def save_checkpoint(processed):
    with open(checkpoint_file, "w") as f:
        json.dump(processed, f)

def count_existing_entries():
    if not os.path.exists(output_file):
        return 0
    
    count = 0
    try:
        with open(output_file, "r") as f:
            for line in f:
                if line.strip(): 
                    count += 1
    except Exception as e:
        print(f"Error counting existing entries: {str(e)}")
        
    return count

processed_indices = []
file_mode = "w"  
start_idx = 0

if args.resume:
    processed_indices = load_processed_questions()
    
    if not processed_indices and os.path.exists(output_file):
        entry_count = count_existing_entries()
        if entry_count > 0:
            processed_indices = selected_indices[:entry_count]
            save_checkpoint(processed_indices)
            print(f"Resuming from entry {entry_count} based on output file count")
    
    if processed_indices:
        start_idx = len(processed_indices)
        file_mode = "a" 
        print(f"Resuming from question {start_idx} of {args.num_questions}")

questions_to_process = [
    selected_indices[i] for i in range(start_idx, min(args.num_questions, len(selected_indices)))
]

with open(output_file, file_mode) as f:
    for idx in tqdm(questions_to_process, initial=start_idx, total=args.num_questions):
        entry = aqua_test_data[idx]
        question = entry["question"]
        options = entry["options"]
        correct_answer = entry["correct"]
        
        formatted_question = f"{question}\n\n"
        
        option_letters = ['A', 'B', 'C', 'D', 'E']
        for i, option in enumerate(options):
            if i < len(option_letters):  
                formatted_question += f"{option_letters[i]}) {option.strip()}\n"
        
        if question:
            attempt_counter = 0
            
            tracker.api_calls = 0
            
            while attempt_counter < max_attempts:
                try:
                    response = agent_executor.invoke({
                        "input": f"{formatted_question}\n\nRespond with the letter of your choice ({', '.join(option_letters[:len(options)])}) and a brief explanation."
                    })
                    model_answer = response['output']
                    
                    print(f"\nQuestion: {question[:100]}...")
                    print(f"API Calls: {tracker.api_calls}")
                    break
                except Exception as e:
                    print(f"Error on attempt {attempt_counter + 1}: {str(e)}")
                    attempt_counter += 1
                    delay_request(attempt_counter)
            
            if attempt_counter == max_attempts:
                model_answer = "Error occurred. Best guess: A"
                
            options_dict = {}
            for i, option in enumerate(options):
                if i < len(option_letters):
                    options_dict[option_letters[i]] = option.strip()
            
            result = {
                "question": question,
                "model_response": model_answer,
                "correct_answer": correct_answer,
                "options": options_dict,
                "api_calls": tracker.api_calls,
                "model_provider": args.model,
                "model_name": args.model_name,
                "rationale": entry.get("rationale", ""),
                "question_index": idx 
            }
            
            json.dump(result, f)
            f.write("\n")
            f.flush()
            
            processed_indices.append(idx)
            save_checkpoint(processed_indices)

print(f"Results saved to {output_file}")