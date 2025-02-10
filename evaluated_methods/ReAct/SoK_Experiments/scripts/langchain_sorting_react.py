import json
import os
import time
from tqdm import tqdm
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain.prompts import PromptTemplate
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
import pandas as pd
import ast

# Load the CSV file containing unsorted and sorted numbers
csv_file_path = "/p/llmreliability/test_repos/ReAct/data/sorting_032.csv"
data = pd.read_csv(csv_file_path)

# Limit to the first 50 entries
data = data.head(50)

# Initialize the model with adjusted parameters
model_name = "open-mixtral-8x22b"
llm = ChatMistralAI(
    model=model_name,
    max_tokens=200,
    temperature=0.2,
    safe_mode=True,
    streaming=True
)

# Wikipedia tool
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
wikipedia_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

# Define the arithmetic tools
def calculator_tool(input_str):
    try:
        result = eval(input_str)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

def addition_tool(input_str):
    try:
        parts = [float(part) for part in input_str.split('+')]
        result = sum(parts)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

def subtraction_tool(input_str):
    try:
        parts = [float(part) for part in input_str.split('-')]
        result = parts[0] - parts[1]
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

def multiplication_tool(input_str):
    try:
        parts = [float(part) for part in input_str.split('*')]
        result = parts[0] * parts[1]
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

def division_tool(input_str):
    try:
        parts = [float(part) for part in input_str.split('/')]
        if parts[1] == 0:
            return "Error: Division by zero"
        result = parts[0] / parts[1]
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

# Wrap tools as Langchain Tools
calculator = Tool(name="Calculator", description="Performs basic arithmetic operations", func=calculator_tool)
addition = Tool(name="Addition", description="Performs addition operations", func=addition_tool)
subtraction = Tool(name="Subtraction", description="Performs subtraction operations", func=subtraction_tool)
multiplication = Tool(name="Multiplication", description="Performs multiplication operations", func=multiplication_tool)
division = Tool(name="Division", description="Performs division operations", func=division_tool)

# Add all tools to the list
tools = [wikipedia_tool, calculator, addition, subtraction, multiplication, division]

# Create the custom prompt template with the required variables
prompt_template = """
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}] or 'Internal Sorting' for list sorting
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat 3 times)
Thought: I now know the final answer
Final answer: The sorted list in ascending order is [sorted list here]
STOP GENERATING AFTER PROVIDING THE FINAL ANSWER.

Begin!

Question: Can you sort this list of numbers {input} in ascending order?

Thought: {agent_scratchpad}
"""

# Convert the string prompt into a PromptTemplate object
prompt = PromptTemplate(input_variables=["input", "agent_scratchpad", "tool_names", "tools"], template=prompt_template)

# Create the agent
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# Function to process streamed LLM response
def stream_process(chunk):
    if "Final answer:" in chunk:
        return False
    return True

# Function to extract final answer from AddableDict
def extract_final_answer(response):
    if isinstance(response, dict) and 'output' in response:
        output = response['output']
        if isinstance(output, str) and "Final answer:" in output:
            return output.split("Final answer:")[-1].strip()
        return output
    return str(response)

# Prepare to store the results
results = []

# Iterate over the dataset from the CSV file
for _, row in tqdm(data.iterrows(), total=len(data)):
    unsorted_list = row['Unsorted']
    correct_sorted_list = row['Sorted']

    if unsorted_list:
        response = agent_executor.invoke({"input": unsorted_list})
        final_answer = extract_final_answer(response)

        # Convert final_answer to string if it's not already
        final_answer = str(final_answer)

        if final_answer.startswith("[") and final_answer.endswith("]"):
            try:
                model_sorted_list = ast.literal_eval(final_answer)
            except:
                model_sorted_list = final_answer  # Keep as string if parsing fails
        else:
            model_sorted_list = final_answer

        is_correct = str(model_sorted_list) == str(correct_sorted_list)
        final_thought = "The final sorted list is correct." if is_correct else f"The final sorted list is incorrect. Expected {correct_sorted_list}, but got {model_sorted_list}."

        result = {
            "id": row["ID"],
            "unsorted_list": unsorted_list,
            "model_sorted_list": str(model_sorted_list),
            "correct_sorted_list": str(correct_sorted_list),
            "final_thought": final_thought,
            "is_correct": is_correct
        }
        results.append(result)

# Output the results to a JSON file
file_path = f"{model_name}_sorting_results.json"
with open(file_path, "w") as f:
    json.dump(results, f, indent=4)

print(f"Results saved to {file_path}")