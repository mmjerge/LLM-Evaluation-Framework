import json
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

dataset = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")

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

calculator = Tool(name="Calculator", description="Performs basic arithmetic operations", func=calculator_tool)
addition = Tool(name="Addition", description="Performs addition operations", func=addition_tool)
subtraction = Tool(name="Subtraction", description="Performs subtraction operations", func=subtraction_tool)
multiplication = Tool(name="Multiplication", description="Performs multiplication operations", func=multiplication_tool)
division = Tool(name="Division", description="Performs division operations", func=division_tool)

tools = [wikipedia_tool, calculator, addition, subtraction, multiplication, division]

prompt = hub.pull("hwchase17/react")

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

results = []

for datapoint in tqdm(list(dataset)[:50]):
    print(type(datapoint))
    if isinstance(datapoint, dict):
        question = datapoint["question"]
        answer = datapoint["best_answer"]        
        if question:
            response = agent_executor.invoke({"input": question})
            result = {
                "question": question,
                "model_response": response,
                "correct_answer": answer
            }
            results.append(result)
            time.sleep(10)
    else:
        print(f"Unexpected entry type: {type(datapoint)}, value: {datapoint}")

file_path = f"{model_name}_truthfulqa_react_results.json"
with open(file_path, "w") as f:
    json.dump(results, f, indent=4)

print(f"Results saved to {file_path}")
