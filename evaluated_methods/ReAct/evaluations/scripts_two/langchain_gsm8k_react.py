import json
import random
import time
from tqdm import tqdm
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.callbacks import get_openai_callback
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_together import ChatTogether
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFacePipeline

with open(gsm8k_path, "r") as f:
    gsm8k_data = [json.loads(line) for line in f]

random_entries = random.sample(gsm8k_data, 50)

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

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

custom_prompt = hub.pull("hwchase17/react") #.partial(
#     system_message="""You are an AI assistant designed to help with math problems. Your task is to solve the given problem and then select the best answer from the provided multiple choice options. Follow these steps:

# 1. Carefully analyze the question.
# 2. Use the available tools to gather information or perform calculations if needed.
# 3. Solve the problem step by step.
# 4. Review the multiple choice options (A, B, C, D).
# 5. Select the option that best matches your calculated answer.
# 6. If you're unsure, make your best educated guess based on the information available.
# 7. Respond with the letter (A, B, C, or D) corresponding to your chosen answer, along with a brief explanation.

# IMPORTANT: Always use the following format for your thoughts and actions:
# Thought: [Your thought process]
# Action: the action to take, should be one of [{tool_names}]
# Action Input: [The input for the tool]
# Observation: [The result of the action]

# Repeat the Thought/Action/Action Input/Observation steps as needed.

# Thought: I now know the final answer
# Final Answer: [Your final answer - the letter A, B, C, or D, followed by a brief explanation]

# Remember:
# - You must always provide an answer, even if you're not 100% certain.
# - If you can't find an exact answer, use your best judgment to make an educated guess.
# - Briefly explain your reasoning for choosing the answer.

# Begin!
# Thought: {agent_scratchpad}"""
# )

agent = create_react_agent(llm, tools, custom_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

def generate_options(correct_answer):
    options = [correct_answer]
    while len(options) < 4:
        incorrect = correct_answer * (random.uniform(0.5, 1.5))
        if incorrect not in options:
            options.append(round(incorrect, 2))
    random.shuffle(options)
    return options

max_attempts = 5

# Create a new JSONL file for incremental writing
output_file = "_gsm8k_react_results_multiple_choice.jsonl"

# Open the file once and keep it open
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
            while attempt_counter < max_attempts:
                try:
                    with get_openai_callback() as cb:
                        response = agent_executor.invoke({
                            "input": f"{question}\n\nA) {options[0]}\nB) {options[1]}\nC) {options[2]}\nD) {options[3]}\n\nRespond with the letter of your choice (A, B, C, or D) and a brief explanation."
                        })
                        model_answer = response['output']
                    
                    print(f"\nQuestion: {question}")
                    print(f"API Calls: {cb.successful_requests}")
                    print(f"Total Tokens: {cb.total_tokens}")
                    print(f"Prompt Tokens: {cb.prompt_tokens}")
                    print(f"Completion Tokens: {cb.completion_tokens}")
                    print(f"Total Cost (USD): ${cb.total_cost}")
                    break
                except Exception as e:
                    print(f"Error on attempt {attempt_counter + 1}: {str(e)}")
                    attempt_counter += 1
                    delay_request(attempt_counter)
            
            if attempt_counter == max_attempts:
                model_answer = "Error occurred. Best guess: A"
                cb = type('MockCallback', (), {'successful_requests': 0, 'total_tokens': 0, 'prompt_tokens': 0, 'completion_tokens': 0, 'total_cost': 0.0})()
            
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
                "api_calls": cb.successful_requests,
                "total_tokens": cb.total_tokens,
                "prompt_tokens": cb.prompt_tokens,
                "completion_tokens": cb.completion_tokens,
                "total_cost": cb.total_cost
            }
            
            json.dump(result, f)
            f.write("\n")
            f.flush() 

print(f"Results saved to {output_file}")