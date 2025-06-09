from datasets import load_dataset
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

# Create a tracker for repetitive actions
class RepetitionDetector:
    def __init__(self, threshold=3):
        self.actions = []
        self.threshold = threshold
        
    def add_action(self, tool, input_str):
        self.actions.append((tool, input_str))
        
    def is_repetitive(self):
        if len(self.actions) < self.threshold:
            return False
            
        # Check the most recent actions
        recent_actions = self.actions[-self.threshold:]
        # Check if all recent actions used the same tool with similar input
        if all(action[0] == recent_actions[0][0] for action in recent_actions):
            # For simplicity, consider identical inputs or inputs with small changes
            inputs = [action[1] for action in recent_actions]
            # If all inputs are nearly identical (allow for small variations)
            if len(set(inputs)) <= 2:
                return True
        return False

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
        max_tokens=500,
        base_url=args.vllm_url,
        api_key="not-needed",
        callbacks=[tracker]
    )
    
    print(f"Using vLLM model: {args.model_name} at {args.vllm_url}")

api_wrapper = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=500)
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

# Add more specific search variations
def enhanced_wikipedia_search(query):
    try:
        # Try the original query first
        result = wikipedia_tool.run(query)
        # Check if the result is empty or "No good Wikipedia Search Result was found"
        if "No good Wikipedia Search Result was found" in result or not result.strip():
            # Try alternative query formulations
            alternatives = [
                query.replace(" ", "+"),                  # Replace spaces with plus signs
                " ".join(query.split()[:3]),              # Use just first few words
                "definition " + query,                    # Add "definition" prefix
                query + " explanation",                   # Add "explanation" suffix
                query.split("of ")[-1] if "of " in query else query  # Remove "of" prefix if present
            ]
            
            for alt_query in alternatives:
                alt_result = wikipedia_tool.run(alt_query)
                if "No good Wikipedia Search Result was found" not in alt_result and alt_result.strip():
                    return f"Search for '{alt_query}' yielded: {alt_result}"
            
            # If all alternatives fail, return a helpful message
            return f"No Wikipedia results found for '{query}' or related terms. Consider trying a different approach or a more general search term."
        return result
    except Exception as e:
        return f"Error searching Wikipedia: {str(e)}"

tools = [
    Tool(
        name="Wikipedia",
        description="A tool for searching information from Wikipedia. Use this for factual questions.",
        func=enhanced_wikipedia_search
    ),
    Tool(
        name="Calculator", 
        description="Performs basic arithmetic operations. Use this for mathematical calculations.",
        func=calculator_tool
    ),
]

# Improve the prompt with guidance for avoiding loops
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

IMPORTANT GUIDELINES TO FOLLOW:
1. Do not repeat the same search query if it doesn't yield useful results
2. Try different approaches if your initial searches don't provide relevant information
3. If Wikipedia searches aren't helpful, try to reason about the question based on your own knowledge
4. If you've tried searching multiple times without success, make your best educated guess
5. Limit to at most 5 tool uses - if you still don't have the answer, make your best educated guess
6. Focus your searches on key concepts in the question
7. If you receive the same result multiple times, stop searching and make your best guess

Begin!

Question: {input}
{agent_scratchpad}"""

prompt = PromptTemplate.from_template(custom_prompt)

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

# Create the ReAct agent with proper formatting
agent = create_react_agent(
    llm, 
    tools, 
    prompt=prompt, 
    output_parser=CustomReActOutputParser()
)

# Create a simpler approach that just uses the standard executor with early stopping
class LoopDetectionCallback(BaseCallbackHandler):
    def __init__(self, max_tool_calls=5):
        self.tool_calls = 0
        self.max_tool_calls = max_tool_calls
        self.action_history = []
        self.should_stop = False
        
    def on_agent_action(self, action, run_id=None):
        self.tool_calls += 1
        
        # Keep track of the last 3 actions
        if hasattr(action, 'tool') and hasattr(action, 'tool_input'):
            self.action_history.append((action.tool, action.tool_input))
            
            # Check for loops: are the last 3 actions repeating?
            if len(self.action_history) >= 3:
                last_three = self.action_history[-3:]
                
                # Check if they're all the same tool
                if all(a[0] == last_three[0][0] for a in last_three):
                    # Check if they have similar inputs (either identical or only 2 different values)
                    inputs = [a[1] for a in last_three]
                    if len(set(inputs)) <= 2:
                        self.should_stop = True
                        
            # Also stop if we've exceeded max tool calls
            if self.tool_calls >= self.max_tool_calls:
                self.should_stop = True
                
    def on_agent_finish(self, finish, run_id=None):
        # Reset for the next run
        self.tool_calls = 0
        self.action_history = []
        self.should_stop = False

# Create the agent executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=10,
    early_stopping_method="force",  # Force stop if callback indicates
)

# Function to run agent with loop detection
def run_agent_with_loop_detection(executor, inputs, max_tool_calls=5):
    # Create a callback to detect loops
    loop_detector = LoopDetectionCallback(max_tool_calls=max_tool_calls)
    
    try:
        # Run the agent with the loop detector
        result = executor.invoke(
            inputs,
            callbacks=[loop_detector]
        )
        
        # If we should have stopped due to loops but didn't
        if loop_detector.should_stop and 'output' not in result:
            return {"output": "B", "log": "Agent got stuck in a loop or exceeded max tool calls. Making best guess: B"}
            
        return result
    except Exception as e:
        print(f"Error executing agent: {str(e)}")
        # Return a default answer if there's an error
        return {"output": "B", "log": f"Error executing agent: {str(e)}"}

max_attempts = 3

output_dir = os.path.dirname(os.path.abspath(__file__))
safe_model_name = args.model_name.replace('/', '_').replace('-', '_').replace('.', '_')
output_file = os.path.join(output_dir, f"mmlu_react_results_{args.model}_{safe_model_name}.jsonl")

# Load the MMLU dataset
print("Loading MMLU dataset...")
dataset = load_dataset("cais/mmlu", "all")
# Get the test split
test_data = list(dataset["test"])
print(f"Loaded {len(test_data)} test questions")

# Take a random sample of questions
random_entries = random.sample(test_data, min(args.num_questions, len(test_data)))

with open(output_file, "w") as f:
    for entry in tqdm(random_entries):
        question = entry["question"]
        choices = entry["choices"]
        correct_answer_idx = entry["answer"]
        
        # Map index to letter (0->A, 1->B, etc.)
        correct_letter = chr(65 + correct_answer_idx)  # Convert 0->A, 1->B, etc.
        
        # Get the subject of the question
        subject = entry.get("subject", "unknown")
        
        if question:
            attempt_counter = 0
            tracker.api_calls = 0
            
            while attempt_counter < max_attempts:
                try:
                    # Format the question with its choices
                    formatted_question = f"{question}\n\n"
                    for i, choice in enumerate(choices):
                        letter = chr(65 + i)  # A, B, C, D
                        formatted_question += f"{letter}) {choice}\n"
                    
                    formatted_question += "\nRespond with the letter of your choice (A, B, C, or D)."
                    
                    # Use our simpler approach
                    response = run_agent_with_loop_detection(
                        agent_executor,
                        {"input": formatted_question},
                        max_tool_calls=5
                    )
                    
                    model_answer = response.get('output', 'B')  # Default to B if no output
                    
                    # Extract just the letter if there's more text
                    letter_match = re.search(r"^([A-D])", model_answer)
                    if letter_match:
                        model_answer = letter_match.group(1)
                    
                    print(f"\nQuestion: {question}")
                    print(f"Subject: {subject}")
                    print(f"Model Answer: {model_answer}")
                    print(f"Correct Answer: {correct_letter}")
                    print(f"API Calls: {tracker.api_calls}")
                    break
                except Exception as e:
                    print(f"Error on attempt {attempt_counter + 1}: {str(e)}")
                    attempt_counter += 1
                    delay_request(attempt_counter)
            
            if attempt_counter == max_attempts:
                model_answer = "B"  # Default fallback
                
            result = {
                "question": question,
                "subject": subject,
                "model_response": model_answer,
                "correct_answer": correct_letter,
                "options": {chr(65 + i): choice for i, choice in enumerate(choices)},
                "api_calls": tracker.api_calls,
                "model_provider": args.model,
                "model_name": args.model_name
            }
            
            json.dump(result, f)
            f.write("\n")
            f.flush()

print(f"Results saved to {output_file}")