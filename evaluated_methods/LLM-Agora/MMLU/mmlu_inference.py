from glob import glob
import pandas as pd
from tqdm import tqdm
import json
import time
import random
import openai
import argparse
import requests
import tiktoken

def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_1", type=str)
    parser.add_argument("--model_2", type=str)
    parser.add_argument("--model_3", type=str)
    parser.add_argument(
        "--API_KEY",
        type=str,
        help="your OpenAI API key to use gpt-3.5-turbo"
    )
    parser.add_argument("--round", default=2, type=int)
    parser.add_argument(
        "--cot",
        default=False,
        action='store_true',
        help="If this is True, you can use Chain-of-Thought during inference."
    )
    parser.add_argument(
        "--output_dir",
        default="MMLU",
        type=str,
        help="Directory to save the result file"
    )

    return parser.parse_args()

def load_json(prompt_path, endpoint_path):
    with open(prompt_path, "r") as prompt_file:
        prompt_dict = json.load(prompt_file)

    with open(endpoint_path, "r") as endpoint_file:
        endpoint_dict = json.load(endpoint_file)

    return prompt_dict, endpoint_dict

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def construct_message(agent_context, instruction, idx):
    prefix_string = "Briefly summarize the following opinions in 2-3 sentences: "

    prefix_string = prefix_string + agent_context + "\n\n Provide a very concise updated answer in 1-2 sentences."

    message = [{"role": "user", "content": prefix_string}]

    try:
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=message,
            max_tokens=50,
            n=1
        )['choices'][0]['message']['content']
    except:
        print("retrying ChatGPT due to an error......")
        time.sleep(5)
        return construct_message(agent_context, instruction, idx)

    prefix_string = f"Brief summary: {completion}"
    prefix_string = prefix_string + "\n\n Provide a very concise answer in 1-2 sentences." + instruction
    return prefix_string

def summarize_message(agent_contexts, instruction, idx):
    prefix_string = "Here are a list of opinions from different agents: "

    for agent in agent_contexts:
        agent_response = agent[-1]["content"]
        response = "\n\n One agent response: ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    prefix_string = prefix_string + "\n\n Write a summary of the different opinions from each of the individual agent."
    completion = construct_message(prefix_string, instruction, idx)

    return completion

def parse_question_answer(df):
    question = f"Can you answer the following question as accurately as possible? {df['question']}: A) {df['A']}, B) {df['B']}, C) {df['C']}, D) {df['D']} Explain your answer, putting the answer in the form (X) at the end of your response."
    answer = df["answer"]
    return question, answer

def generate_mmlu(agents, question):
    agent_contexts = [[{"model": agent, "content": question}] for agent in agents]
    return agent_contexts

if __name__ == "__main__":
    args = args_parse()
    openai.api_key = args.API_KEY
    model_list = [args.model_1, args.model_2, args.model_3]

    prompt_dict, endpoint_dict = load_json("src/prompt_template.json", "src/inference_endpoint.json")

    def generate_answer(model, formatted_prompt):
        API_URL = endpoint_dict[model]["API_URL"]
        headers = endpoint_dict[model]["headers"]
        
        # Set a very low fixed number of tokens for the response
        fixed_response_tokens = 50  # This is a very short response

        # Calculate the number of tokens in the input
        input_tokens = num_tokens_from_string(formatted_prompt)
        
        # If input exceeds the limit, truncate it
        max_input_tokens = 1512 - fixed_response_tokens
        if input_tokens > max_input_tokens:
            print(f"Warning: Input exceeds token limit. Truncating input.")
            encoding = tiktoken.get_encoding("cl100k_base")
            truncated_tokens = encoding.encode(formatted_prompt)[:max_input_tokens]
            formatted_prompt = encoding.decode(truncated_tokens)
            input_tokens = num_tokens_from_string(formatted_prompt)

        payload = {
            "inputs": formatted_prompt,
            "parameters": {
                "max_new_tokens": fixed_response_tokens,
                "temperature": 0.7,  # Lower temperature for more focused outputs
                "top_p": 0.95,  # Slightly lower top_p for more focused outputs
                "do_sample": True,
                "no_repeat_ngram_size": 2  # Avoid repeating 2-grams
            }
        }

        print(f"Sending request to {API_URL}")
        print(f"Headers: {headers}")
        print(f"Payload: {json.dumps(payload, indent=2)}")
        print(f"Input tokens: {input_tokens}, Max new tokens: {fixed_response_tokens}")

        try:
            resp = requests.post(API_URL, json=payload, headers=headers)
            resp.raise_for_status()
            response = resp.json()
            print(f"Response: {json.dumps(response, indent=2)}")
            
            if not response or not isinstance(response, list) or len(response) == 0:
                raise ValueError("Unexpected response format")
            
            return {"model": model, "content": response[0].get("generated_text", "")}
        except requests.RequestException as e:
            print(f"Request error: {e}")
            print(f"Response content: {resp.text}")
        except Exception as e:
            print(f"Unexpected error: {e}")
        
        print("Retrying due to an error......")
        time.sleep(5)
        return generate_answer(model, formatted_prompt)
    
    def prompt_formatting(model, instruction, cot):
        if model == "alpaca" or model == "orca":
            prompt = prompt_dict[model]["prompt_no_input"]
        else:
            prompt = prompt_dict[model]["prompt"]
        
        if cot:
            instruction += "Let's think step by step."

        return {"model": model, "content": prompt.format(instruction=instruction)}
    
    agents = len(model_list)
    rounds = args.round

    with open("data/MMLU/MMLU_test.json", "r") as f:
        mmlu_questions = json.load(f)

    random.seed(0)
    random.shuffle(mmlu_questions)
    generated_description = []

    evaluation = 25

    for idx in tqdm(range(evaluation)):
        question, answer = parse_question_answer(mmlu_questions[idx])

        agent_contexts = generate_mmlu(model_list, question)

        print(f"# Question No.{idx+1} starts...")

        message = []

        for debate in range(rounds+1):
            # Refer to the summarized previous response
            if debate != 0:
                message.append(summarize_message(agent_contexts, question, 2 * debate - 1))
                for i in range(len(agent_contexts)):
                    agent_contexts[i].append(prompt_formatting(agent_contexts[i][-1]["model"], message, args.cot))

            for agent_context in agent_contexts:
                # Generate new response based on summarized response
                completion = generate_answer(agent_context[-1]["model"], agent_context[-1]["content"])
                agent_context.append(completion)

        print(f"# Question No.{idx+1} debate is ended.")

        models_response = {
            f"{args.model_1}": [agent_contexts[0][1]["content"], agent_contexts[0][2]["content"], agent_contexts[0][3]["content"]],
            f"{args.model_2}": [agent_contexts[1][1]["content"], agent_contexts[1][2]["content"], agent_contexts[1][3]["content"]],
            f"{args.model_3}": [agent_contexts[2][1]["content"], agent_contexts[2][2]["content"], agent_contexts[2][3]["content"]]
        }
        response_summarization = [
            message[0], message[1]
        ]
        generated_description.append({"question_id": idx, "question": question, "agent_response": models_response, "summarization": response_summarization, "answer": answer})

    if args.cot:
        file_name = "_cot.json"
    else:
        file_name = ".json"

    print(f"The result file 'mmlu_result{file_name}' is saving...")
    with open(args.output_dir + f"/mmlu_result{file_name}", "x") as f:
        json.dump(generated_description, f, indent=4)

    print("All done!!")
