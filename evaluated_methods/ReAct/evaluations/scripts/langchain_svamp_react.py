import json
import getpass
import os
import random
from tqdm import tqdm
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_together import ChatTogether
from langchain_core.prompts import ChatPromptTemplate


# Load the SVAMP dataset from the provided JSON file
with open(svamp_path, "r") as f:
    svamp_data = json.load(f)

# Randomly sample 100 questions
sampled_data = random.sample(svamp_data, 150)

openai_api_key = os.getenv('OPENAI_API_KEY')
model_name = "gpt-3.5-turbo"

llm = ChatOpenAI(model=model_name,
                 api_key=openai_api_key,
                 temperature=0.5,
                 max_tokens=512,
                 timeout=None,
                 max_retries=5)

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

# Add all tools to the list
tools = [wikipedia_tool, calculator, addition, subtraction, multiplication, division]

custom_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an AI assistant designed to solve math word problems. Your task is to analyze the given problem, perform the necessary calculations, and provide the final answer. Follow these steps:

You have access to the following tools: {tools}

1. Carefully read and understand the problem statement.
2. Identify the key information and numerical values provided in the problem.
3. Determine the appropriate mathematical operations needed to solve the problem.
4. Use the available tools to perform calculations as needed.
5. Think through the problem step by step, showing your work.
6. Provide the final numerical answer.

IMPORTANT: Always use the following format for your thoughts and actions:
Thought: [Your thought process]
Action: the action to take, should be one of [{tool_names}]
Action Input: [The input for the tool]
Observation: [The result of the action]

Repeat the Thought/Action/Action Input/Observation steps as needed.

Thought: I now know the final answer
Final Answer: [Your final numerical answer]

Remember:
- You must always provide a numerical answer, even if you're not 100% certain.
- If you can't find an exact answer, use your best judgment to make an educated guess.
- Briefly explain your reasoning for the final answer.

Begin!
Human: {input}
AI: {agent_scratchpad}""")
])

# Create the agent
agent = create_react_agent(llm, tools, custom_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

def write_to_json(results, file_path):
    with open(file_path, "w") as f:
        json.dump(results, f, indent=4)

# Prepare to store the results
results = []
file_path = f"{model_name}_svamp_react_results_random_50.json"

# Iterate over the sampled SVAMP dataset and invoke the agent
for entry in tqdm(sampled_data):
    body = entry.get("Body", "")
    question = entry.get("Question", "")
    answer = entry.get("Answer", "")
    equation = entry.get("Equation", "")
    problem_type = entry.get("Type", "")
    
    if body and question:
        full_question = f"{body} {question}"
        
        try:
            response = agent_executor.invoke({"input": full_question})
            model_answer = response['output']
        except Exception as e:
            model_answer = f"Error occurred: {str(e)}. Best guess: 0 (No explanation available due to error)"
        
        result = {
            "ID": entry.get("ID", ""),
            "Body": body,
            "Question": question,
            "model_response": model_answer,
            "correct_answer": answer,
            "Equation": equation,
            "Type": problem_type
        }
        results.append(result)
        
        # Write to JSON file after each question is answered
        write_to_json(results, file_path)

print(f"Final results saved to {file_path}")