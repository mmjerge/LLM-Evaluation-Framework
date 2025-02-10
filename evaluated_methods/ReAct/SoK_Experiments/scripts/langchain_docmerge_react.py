import json
import os
import time
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
import pandas as pd

# Load the CSV file containing the documents
csv_file_path = "/p/llmreliability/test_repos/ReAct/data/documents.csv"
data = pd.read_csv(csv_file_path)

# Initialize the model
model_name = "open-mixtral-8x22b"
llm = ChatMistralAI(model=model_name)

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

# Pull the prompt template for the ReAct agent
prompt = hub.pull("hwchase17/react")

# Create the agent
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# Prepare to store the results
results = []

# Limit the dataset to the first 25 entries
data_subset = data.iloc[10:25]

# Iterate over the dataset from the CSV file
for _, row in tqdm(data_subset.iterrows(), total=len(data_subset)):
    problem = row['problem']
    documents = f"{row['document1']} {row['document2']} {row['document3']} {row['document4']}"
    
    if problem:
        # Explicit instruction to merge the NDAs
        detailed_instruction = (
            f"You are tasked with merging four Non-Disclosure Agreements (NDAs) into a single, "
            f"coherent document. The NDAs contain overlapping and complementary information. "
            f"Please carefully combine the content from the provided NDAs to ensure that the merged "
            f"document retains all necessary legal and confidentiality clauses without redundancy.\n\n"
            f"Problem: {problem}\nDocuments: {documents}\n"
            f"Your final answer should be the final merged document and nothing more. Do not include any "
            f"explanations, headings like 'Final Answer:', or additional text. Only output the merged document content."
        )
        response = agent_executor.invoke({"input": detailed_instruction})
        # Extract only the merged document content
        merged_document = response['output'].strip()
        
        # Remove any potential "Final Answer:" prefix
        if merged_document.lower().startswith("final answer:"):
            merged_document = merged_document.split(":", 1)[1].strip()
        
        result = {
            "id": row["id"],
            "problem": problem,
            "merged_documents": merged_document
        }
        results.append(result)
        time.sleep(10)

# Output the results to a JSON file
file_path = f"{model_name}_document_merging_results.json"
with open(file_path, "w") as f:
    json.dump(results, f, indent=4)

print(f"Results saved to {file_path}")

