import json
import os
import datetime
import random
from tqdm import tqdm
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_openai import ChatOpenAI
from langchain_together import ChatTogether
from langchain_core.prompts import ChatPromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from typing import Dict, Any
import sympy

aqua_path = "/p/llmreliability/test_repos/ReAct/SoK_Experiments/data/aqua_test.jsonl"
with open(aqua_path, "r") as f:
    aqua_data = [json.loads(line) for line in f]

sampled_data = random.sample(aqua_data, 150)

openai_api_key = os.getenv('OPENAI_API_KEY')
model_name = "gpt-3.5-turbo"

llm = ChatOpenAI(model=model_name,
                 api_key=openai_api_key,
                 temperature=0.5,
                 max_tokens=512,
                 timeout=None,
                 max_retries=10)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
wikipedia_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

def robust_calculator(expression):
    try:
        # Remove commas from large numbers
        clean_expression = expression.replace(",", "")
        
        # Detect and handle algebraic equations
        if "=" in clean_expression:
            lhs, rhs = clean_expression.split("=")
            equation = sympy.Eq(sympy.sympify(lhs), sympy.sympify(rhs))
            solution = sympy.solve(equation)
            return str(solution)
        
        # Handle regular numeric calculations
        result = sympy.sympify(clean_expression)
        return str(result.evalf())
    except Exception as e:
        return f"Error: Unable to calculate. Please provide a simpler mathematical expression."
    
calculator = Tool(name="Calculator", description="Performs basic arithmetic operations. Use this for complex calculations.", func=robust_calculator)

tools = [wikipedia_tool, calculator]

custom_prompt = hub.pull("hwchase17/react").partial(
    system_message="""You are an AI assistant designed to help with various questions, including academic topics. Your task is to analyze the given problem and then select the best answer from the provided multiple choice options. Follow these steps:

1. Carefully read and understand the question.
2. Use the available tools to gather information or perform calculations if needed.
3. Think through the problem step by step.
4. Review the multiple choice options (A, B, C, D).
5. Select the option that best matches your analysis or calculation.
6. If you're unsure, make your best educated guess based on the information available.
7. Respond with the letter (A, B, C, or D) corresponding to your chosen answer, along with a brief explanation.

IMPORTANT: Always use the following format for your thoughts and actions:
Thought: [Your thought process]
Action: the action to take, should be one of [{tool_names}]
Action Input: [The input for the tool]
Observation: [The result of the action]

Repeat the Thought/Action/Action Input/Observation steps as needed.

Thought: I now know the final answer
Final Answer: [Your final answer - the letter A, B, C, or D, followed by a brief explanation]

IMPORTANT: If an equation or algebraic expression is given, do the following:
- Try to first solve the equation symbolically using known algebraic methods.
- Only use the calculator to perform numeric operations or solve simplified expressions.
- Simplify expressions whenever possible before using the calculator.

If an error occurs (e.g., the equation cannot be simplified), break down the problem into simpler parts and solve step by step.

### Example 1: Solving Algebraic Equations
Human: Solve the equation M - N + 396c = 990 for M when N = 123 and c = 3.
A) 900
B) 880
C) 860
D) 840

Thought: I need to first substitute N = 123 and c = 3 into the equation.
Action: Calculator
Action Input: Substitute N = 123 and c = 3 into M - N + 396c = 990
Observation: M - 123 + 1188 = 990

Thought: Now I will solve for M by simplifying the equation.
Action: Calculator
Action Input: Solve M - 123 + 1188 = 990
Observation: M = -75

Final Answer: M = -75.

### Example 2: Basic Numeric Calculations
Human: What is the value of 16,000 / 20 * 30?
A) 15,000
B) 24,000
C) 32,000
D) 48,000

Thought: I need to divide 16,000 by 20, then multiply the result by 30.
Action: Calculator
Action Input: 16,000 / 20
Observation: 800

Thought: Now I will multiply the result by 30.
Action: Calculator
Action Input: 800 * 30
Observation: 24,000
Final Answer: B) 24,000.

### Example 3: Solving for a Variable
Human: Solve the equation 3x + 4 = 19 for x.
A) x = 4
B) x = 5
C) x = 6
D) x = 7

Thought: I will solve for x by first isolating x on one side of the equation.
Action: Calculator
Action Input: Solve 3x + 4 = 19 for x
Observation: x = 5
Final Answer: B) x = 5.

Remember:
- Simplify expressions when possible.
- Use algebraic methods before relying on the calculator for complex equations.
- Provide an answer even if not 100% certain.
- Always give a brief explanation of your reasoning.

Begin!
Thought: {agent_scratchpad}"""
)

agent = create_react_agent(llm, tools, custom_prompt)
class StoppingCallback(BaseCallbackHandler):
    def __init__(self):
        self.should_stop = False

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if "Final Answer:" in token:
            self.should_stop = True

    def on_llm_end(self, response, **kwargs) -> None:
        if self.should_stop:
            raise ValueError("Stop the chain")

class CustomAgentExecutor(AgentExecutor):
    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        stopping_callback = StoppingCallback()
        self.callbacks = self.callbacks or []
        self.callbacks.append(stopping_callback)
        
        try:
            return super()._call(inputs)
        except ValueError as e:
            if str(e) == "Stop the chain":
                last_output = self.agent.llm_chain.memory.buffer[-1]
                final_answer = last_output.split('Final Answer:')[-1].strip()
                return {"output": final_answer}
            else:
                if "Unable to calculate" in str(e):
                    return {"output": "Error: Failed to compute the answer. Try breaking down the problem."}
                raise

agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True, 
    handle_parsing_errors=True,
    max_iterations=10,  
    early_stopping_method="force", 
    max_execution_time=90, 
    callbacks=[StoppingCallback()]
)

results = []

for entry in tqdm(sampled_data):
    question = entry.get("question", "")
    options = entry.get("options", [])
    correct = entry.get("correct", "")
    
    if question and options:
        options_text = "\n".join([f"{chr(65 + i)}) {option}" for i, option in enumerate(options)])
        full_question = f"{question}\n\nOptions:\n{options_text}\n\nRespond with the letter of your choice (A, B, C, or D) and a brief explanation."
        
        try:
            response = agent_executor.invoke({"input": full_question})
            model_answer = response['output']
        except Exception as e:
            model_answer = f"Error occurred: {str(e)}. Best guess: A (No explanation available due to error)"
        
        result = {
            "question": question,
            "options": options,
            "model_response": model_answer,
            "correct_answer": correct
        }
        results.append(result)

file_path = f"gpt35_aqua_react_results_random_150_1.json"
with open(file_path, "w") as f:
    json.dump(results, f, indent=4)

print(f"Results saved to {file_path}")