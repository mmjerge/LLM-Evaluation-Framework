import json
import random
import time
import argparse
from tqdm import tqdm
import os
from langchain_core.prompts import PromptTemplate
import re
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain.tools import Tool
from langchain_core.messages import AIMessage, HumanMessage
from langchain.agents.format_scratchpad import format_log_to_str
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.callbacks import get_openai_callback
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI, OpenAI
from langchain_together import ChatTogether
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFacePipeline
from langchain_core.callbacks import BaseCallbackHandler
from langchain.agents.output_parsers import ReActJsonSingleInputOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents.react.output_parser import ReActOutputParser
from langchain_core.exceptions import OutputParserException
from langchain.schema import AgentAction, AgentFinish


class APITracker(BaseCallbackHandler):
    def __init__(self):
        super().__init__()
        self.api_calls = 0
        
    def on_llm_start(self, serialized, prompts, **kwargs):
        self.api_calls += 1

parser = argparse.ArgumentParser(description='Run ReAct agent with API call tracking')
parser.add_argument('--model', type=str, default='claude', 
                    choices=['claude', 'openai', 'vllm'],
                    help='Model provider (claude, openai, or vllm)')
parser.add_argument('--model_name', type=str, default=None,
                    help='Specific model name to use (e.g., gpt-4o, vicuna-13b, etc.)')
parser.add_argument('--num_questions', type=int, default=150,
                    help='Number of questions to run (default: 150)')
parser.add_argument('--vllm_url', type=str, default='http://localhost:8000/v1',
                    help='URL for vLLM API (default: http://localhost:8000/v1)')
args = parser.parse_args()

if args.model_name is None:
    if args.model == 'claude':
        args.model_name = "claude-3-5-sonnet-20240620"
    elif args.model == 'openai':
        args.model_name = "gpt-4o"
    elif args.model == 'vllm':
        args.model_name = "/scratch/mj6ux/.cache/models/mixtral-8x22b"

possible_paths = [
    "gsm8k_test.jsonl",  # Current directory
    "data/gsm8k_test.jsonl",  # data subdirectory
    "evaluations/data/gsm8k_test.jsonl",  # evaluations/data subdirectory
    "ReAct/evaluations/data/gsm8k_test.jsonl",  # ReAct/evaluations/data subdirectory
    "../evaluations/data/gsm8k_test.jsonl",  # Parent directory's evaluations/data
    "../data/gsm8k_test.jsonl",  # Parent directory's data
]

gsm8k_path = None
for path in possible_paths:
    if os.path.exists(path):
        gsm8k_path = path
        break

if not gsm8k_path:
    # If file not found, ask for manual input
    print("Could not find the GSM8K dataset file. Please enter the path:")
    gsm8k_path = input()
    if not os.path.exists(gsm8k_path):
        raise FileNotFoundError(f"The file {gsm8k_path} does not exist.")

print(f"Using GSM8K dataset from: {gsm8k_path}")

# Load the dataset
with open(gsm8k_path, "r") as f:
    gsm8k_data = [json.loads(line) for line in f]

random_entries = random.sample(gsm8k_data, args.num_questions)

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
    def format_mistral_messages(messages):
        formatted_text = ""
        for message in messages:
            if message.type == "human":
                formatted_text += f"[INST] {message.content} [/INST]\n"
            elif message.type == "ai":
                formatted_text += f" {message.content}</s>\n"
        return formatted_text
    
    llm = OpenAI(
        model=args.model_name,
        temperature=0,
        max_tokens=200,
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
    wikipedia_tool,
    Tool(name="Calculator", description="Performs basic arithmetic operations", func=calculator_tool),
]

react_prompt = hub.pull("hwchase17/react")

# Create a custom parser that's more tolerant of formatting issues
class CustomReActOutputParser(ReActOutputParser):
    def parse(self, text):
        # First, look for Final Answer pattern - highest priority
        final_answer_match = re.search(r"Final Answer:?\s*(.*?)(?:\n|$)", text, re.DOTALL)
        if final_answer_match or "Final Answer" in text:
            final_answer = final_answer_match.group(1).strip() if final_answer_match else ""
            
            # If final_answer is empty but we found "Final Answer", extract from text
            if not final_answer and "Final Answer" in text:
                final_answer = text.split("Final Answer:")[-1].strip().split("\n")[0].strip()
            
            # Get the actual letter (A, B, C, D) from the answer if present
            letter_match = re.search(r"^([A-D])", final_answer)
            if letter_match:
                final_answer = letter_match.group(1)
            
            return AgentFinish(
                return_values={"output": final_answer},
                log=text
            )
        
        # Next, try to extract action and action input
        action_match = re.search(r"Action:?\s*(\w+)", text)
        input_match = re.search(r"Action\s*Input:?\s*(.*?)(?:\n|$)", text, re.DOTALL)
        
        if action_match and input_match:
            tool = action_match.group(1).strip()
            tool_input = input_match.group(1).strip()
            
            # Clean calculation input to remove any narrative text
            if tool.lower() == "calculator":
                # Extract just the calculation expression
                calc_match = re.search(r"([0-9\+\-\*\/\(\)\.\s]+)", tool_input)
                if calc_match:
                    tool_input = calc_match.group(1).strip()
            
            return AgentAction(
                tool=tool,
                tool_input=tool_input,
                log=text
            )
            
        # If all parsing fails, use the parent parser as fallback
        try:
            return super().parse(text)
        except OutputParserException:
            # If we can't parse properly, raise a helpful error
            raise OutputParserException(
                f"Could not parse LLM output: `{text}`",
                observation="Please format your response as either:\n1. Action: [tool name]\nAction Input: [input]\nor\n2. Final Answer: [answer]",
                llm_output=text,
                send_to_llm=True
            )

# Use the custom prompt to avoid any example injections 
custom_prompt = """Answer the following multiple choice question as best you can. You have access to the following tools:

{tools}

Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action (just the calculation or search query, nothing else)
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the letter of your choice (A, B, C, or D)

Begin!

Question: {input}
{agent_scratchpad}"""

prompt = PromptTemplate.from_template(custom_prompt)

# Create the ReAct agent with proper formatting
agent = create_react_agent(
    llm, 
    tools, 
    prompt=prompt, 
    output_parser=CustomReActOutputParser()
)

# Create the agent executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=10
)

def generate_options(correct_answer):
    options = [correct_answer]
    while len(options) < 4:
        incorrect = correct_answer * (random.uniform(0.5, 1.5))
        if incorrect not in options:
            options.append(round(incorrect, 2))
    random.shuffle(options)
    return options

max_attempts = 5

output_dir = os.path.dirname(os.path.abspath(__file__))
safe_model_name = args.model_name.replace('/', '_').replace('-', '_').replace('.', '_')
output_file = os.path.join(output_dir, f"_gsm8k_react_results_{args.model}_{safe_model_name}_multiple_choice.jsonl")

with open(output_file, "w") as f:
    for entry in tqdm(random_entries):
        question = entry.get("question", "")
        answer_str = entry.get("answer", "")

        try:
            correct_answer = float(answer_str.split()[-1])
        except (ValueError, IndexError) as e:
            print(f"Error parsing answer for question: '{question}'. Skipping entry. Error: {str(e)}")
            continue

        options = generate_options(correct_answer)
        
        if question:
            attempt_counter = 0
            
            tracker.api_calls = 0
            
            while attempt_counter < max_attempts:
                try:
                    response = agent_executor.invoke({
                        "input": f"{question}\n\nA) {options[0]}\nB) {options[1]}\nC) {options[2]}\nD) {options[3]}\n\nRespond with the letter of your choice (A, B, C, or D) and a brief explanation."
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
                "options": {
                    "A": options[0],
                    "B": options[1],
                    "C": options[2],
                    "D": options[3]
                },
                "api_calls": tracker.api_calls,
                "model_provider": args.model,
                "model_name": args.model_name
            }
            
            json.dump(result, f)
            f.write("\n")
            f.flush() 

print(f"Results saved to {output_file}")