import json
import getpass
import os
import random
import re
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
from langchain_groq import ChatGroq
from langchain.agents import BaseSingleActionAgent
from langchain.schema import AgentAction, AgentFinish, AIMessage
import logging

logging.basicConfig(filename='mmlu_agent_output.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

mmlu_path = "/p/llmreliability/test_repos/ReAct/SoK_Experiments/data/mmlu_dataset_pretty.json"
with open(mmlu_path, "r") as f:
    mmlu_data = json.load(f)

sampled_data = random.sample(mmlu_data, 5)

together_api_key = os.getenv('TOGETHER_API_KEY')
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
llm = ChatTogether(
    model=model_name,
    api_key=together_api_key,
    temperature=0.5,
    max_tokens=512,
    timeout=None,
    max_retries=2,
    stop=["\nHuman:", "\n\nHuman:", "Human:", "Final Answer:"]
)

wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

def fix_format_errors(text):
    # Fix missing 'Action:' after 'Thought:'
    text = re.sub(r'(Thought:[^\n]+)\n(?!Action:)', r'\1\nAction: ', text)
    
    # Ensure 'Observation:' is followed by a newline
    text = re.sub(r'(Observation:[^\n]+)(?!\n)', r'\1\n', text)
    
    # Add 'Thought:' if missing after 'Observation:'
    text = re.sub(r'(Observation:[^\n]+\n)(?!Thought:)', r'\1Thought: ', text)
    
    return text

def calculator_tool(input_str):
    try:
        # Remove any text before the actual expression
        expression = input_str.split(":")[-1].strip()
        result = eval(expression, {"__builtins__": None}, {"abs": abs, "pow": pow, "round": round})
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}. Please provide a valid mathematical expression."

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

tools = [
    Tool(name="Wikipedia", description="Searches Wikipedia for information", func=wikipedia.run),
    Tool(name="Calculator", description="Performs basic arithmetic operations", func=calculator_tool),
    Tool(name="Addition", description="Performs addition operations", func=addition_tool),
    Tool(name="Subtraction", description="Performs subtraction operations", func=subtraction_tool),
    Tool(name="Multiplication", description="Performs multiplication operations", func=multiplication_tool),
    Tool(name="Division", description="Performs division operations", func=division_tool)
]

class CustomMMULAgent(BaseSingleActionAgent):
    tools: list
    llm: ChatTogether

    def get_allowed_tools(self):
        return [tool.name for tool in self.tools]

    @property
    def input_keys(self):
        return ["input", "option_a", "option_b", "option_c", "option_d"]

    async def aplan(self, intermediate_steps, **kwargs):
        return self.plan(intermediate_steps, **kwargs)

    def plan(self, intermediate_steps, **kwargs):
        thoughts = []
        for action, observation in intermediate_steps:
            thoughts.append(f"Action: {action.tool}\nAction Input: {action.tool_input}")
            thoughts.append(f"Observation: {observation}")

        thoughts_str = "\n".join(thoughts)
        
        prompt = f"""You are an AI assistant designed to answer multiple-choice questions. Analyze the problem and select the best answer from the provided options (A, B, C, D).

Question: {kwargs['input']}

Options:
A: {kwargs['option_a']}
B: {kwargs['option_b']}
C: {kwargs['option_c']}
D: {kwargs['option_d']}

Previous steps:
{thoughts_str}

Your task is to provide a final answer. Use the following format:
Thought: [Your reasoning]
Final Answer: [Letter choice (A, B, C, or D)] [Brief explanation]

If you cannot determine the answer with certainty, make your best guess based on the information available.

Your response:"""

        response = self.llm.invoke(prompt)
        
        # Extract content from AIMessage if necessary
        if isinstance(response, AIMessage):
            response_content = response.content
        elif isinstance(response, str):
            response_content = response
        else:
            raise ValueError(f"Unexpected response type: {type(response)}")
        
        # Extract Thought and Final Answer using regex
        thought_match = re.search(r"Thought: (.*?)(?:\nFinal Answer:|\Z)", response_content, re.DOTALL)
        final_answer_match = re.search(r"Final Answer: (.*?)(?:\n|\Z)", response_content, re.DOTALL)

        thought = thought_match.group(1).strip() if thought_match else ""
        final_answer = final_answer_match.group(1).strip() if final_answer_match else ""

        if final_answer:
            # Extract the letter answer (A, B, C, or D) from the final answer
            answer_match = re.search(r"^([A-D])", final_answer, re.IGNORECASE)
            if answer_match:
                letter_answer = answer_match.group(1).upper()
                explanation = final_answer[len(letter_answer):].strip(": ")
                return AgentFinish({"output": f"{letter_answer}: {explanation}"}, thought)
            else:
                return AgentFinish({"output": f"Unable to determine. Best guess: {final_answer}"}, thought)
        else:
            return AgentFinish({"output": "Unable to determine. Insufficient information to make a guess."}, thought)

# Create the custom agent
custom_agent = CustomMMULAgent(tools=tools, llm=llm)

# Create the agent executor
agent_executor = AgentExecutor(
    agent=custom_agent,
    tools=tools,
    verbose=True,
    max_iterations=1,
    early_stopping_method="force",
    return_intermediate_steps=True
)

# Main processing loop
for entry in tqdm(sampled_data):
    question = entry.get("question", "")
    answer = entry.get("answer", "")
    question_type = entry.get("type", "")
    
    if question:
        logging.info(f"Processing question: {question}")
        logging.info(f"Question type: {question_type}")
        logging.info(f"Correct answer: {answer}")
        
        try:
            response = agent_executor.invoke({
                "input": question, 
                "option_a": entry.get("A", ""),
                "option_b": entry.get("B", ""),
                "option_c": entry.get("C", ""),
                "option_d": entry.get("D", "")
            })
            
            # Log the full response
            logging.info("Agent's full response:")
            logging.info(response.get("output", "No output provided"))
            
            # Log intermediate steps
            logging.info("Agent's thought process:")
            for step in response.get("intermediate_steps", []):
                logging.info(f"Action: {step[0].log}")
                logging.info(f"Observation: {step[1]}")
            
            # Try to extract a letter answer
            full_response = response.get("output", "")
            letter_match = re.search(r"\b([A-D])\b", full_response)
            letter_answer = letter_match.group(1) if letter_match else "Unable to determine"
            
            logging.info(f"Extracted answer: {letter_answer}")
            
        except Exception as e:
            logging.error(f"Error processing question: {question}")
            logging.error(f"Error message: {str(e)}")
        
        logging.info("=" * 50)  # Separator between questions

print(f"Results saved to mmlu_agent_output.log")