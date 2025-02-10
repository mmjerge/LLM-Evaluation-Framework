import json
import os
import time
import random
from tqdm import tqdm
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_together import ChatTogether
from datasets import load_dataset
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# List all LiveBench datasets available
livebench_datasets = {
    "data_analysis": load_dataset("livebench/data_analysis"),
    "instruction_following": load_dataset("livebench/instruction_following"),
    "math": load_dataset("livebench/math"),
    "reasoning": load_dataset("livebench/reasoning"),
    "coding": load_dataset("livebench/coding"),
    "language": load_dataset("livebench/language")
}

groq_api_key = "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" 
model_name = "llama3-groq-70b-8192-tool-use-preview"
llm = ChatGroq(
    model=model_name,
    api_key=groq_api_key,
    temperature=0.5,
    max_tokens=512,
    timeout=None,
    max_retries=2,
)

# Wikipedia tool
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
wikipedia_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

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

tools = [wikipedia_tool, calculator, addition, subtraction, multiplication, division]

prompt = hub.pull("hwchase17/react")

# Create the agent
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# Iterate over each dataset in LiveBench and run the agent
for dataset_name, dataset in livebench_datasets.items():
    print(f"Processing dataset: {dataset_name}")

    # Determine available splits and choose one (e.g., validation or test)
    split_name = "validation" if "validation" in dataset.keys() else "test"
    
    # Get the specific split
    split_dataset = dataset[split_name]

    # Prepare to store the results
    results = []

    # Iterate over the first 25 entries of the dataset and invoke the agent
    for entry in tqdm(list(split_dataset)[:25]):
        question = entry["turns"]
        if entry["category"] == "data_analysis":
            answer = entry["ground_truth"]
        elif entry["category"] == "instruction_following":
            answer = None
        elif entry["category"] == "math":
            answer = entry["ground_truth"]
        elif entry["category"] == "reasoning":
            answer = entry["ground_truth"]
        elif entry["category"] == "coding":
            answer = entry["solution"]
        else:
            answer = entry["ground_truth"]
        
        if question:
            response = agent_executor.invoke({"input": question})
            result = {
                "dataset": dataset_name,
                "question": question,
                "model_response": response,
                "correct_answer": answer
            }
            results.append(result)
            time.sleep(10)

    # Output the results to a JSON file for each dataset
    file_path = f"{model_name}_{dataset_name}_react_results.json"
    with open(file_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {file_path}")