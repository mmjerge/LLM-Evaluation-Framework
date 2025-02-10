from openai import OpenAI
from anthropic import Anthropic
from mistralai import Mistral
import argparse
import json
import numpy as np
import random
import os
import tqdm
from datasets import load_dataset
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# openai.api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
# anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
# mistral_api_key = os.getenv("MISTRAL_API_KEY")
# mistral_client = Mistral(api_key=mistral_api_key)

# # Initialize the OpenAI client
# client = OpenAI(
#   api_key=os.environ.get("TOGETHER_API_KEY"),
#   base_url="https://api.together.xyz/v1",
# )
# # model = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
# model = "mistralai/Mixtral-8x22B-Instruct-v0.1"

client = Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)

model = "claude-3-5-sonnet-20240620"

def construct_message(agents, question, idx, use_counterfactual=False):
    if len(agents) == 0:
        return {"role": "user", "content": "Can you double check that your answer is correct. Please reiterate your answer."}
    prefix_string = "These are the responses to the question from other agents: "
    for agent in agents:
        agent_response = agent[idx]["content"]
        response = "\n\n One agent response: ```{}```".format(agent_response)
        prefix_string = prefix_string + response
    prefix_string = prefix_string + """\n\n Using the responses from other agents as additional information, can you provide your answer to the question? \n The original question is: {}. Please provide your final answer at the end of your response.""".format(question)
    return {"role": "user", "content": prefix_string}

def construct_assistant_message(completion, model_type):
    if model_type == "openai":
        content = completion.choices[0].message.content
    elif model_type == "anthropic":
        content = completion.content[0].text
    elif model_type == "mistral":
        content = completion.choices[0].message.content
    return {"role": "assistant", "content": content}

def get_model_completion(model_type, model_name, messages):
    if model_type == "openai":
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            n=1
        )
    elif model_type == "anthropic":
        response = client.messages.create(
            model=model_name,
            messages=[{"role": m["role"], "content": m["content"]} for m in messages],
            max_tokens=1000
        )
    elif model_type == "mistral":
        response = mistral_client.chat.complete(
            model=model_name,
            messages=[{"role": m["role"], "content": m["content"]} for m in messages]
        )
    return response

def main(args):
    random.seed(args.seed)

    # Load TruthfulQA dataset
    dataset = load_dataset("truthfulqa/truthful_qa", "generation")
    
    # Convert the dataset to a list and then sample
    all_questions = list(dataset["validation"])
    selected_questions = random.sample(all_questions, args.num_of_questions)

    generated_description = {}

    for data in tqdm.tqdm(selected_questions, desc="Questions processed"):
        question = data['question']
        answer = data['best_answer']  # Using 'best_answer' as the reference answer
        agent_contexts = [[{"role": "user", "content": f"""Can you answer the following question? {question} Explain your reasoning and provide your final answer at the end of your response."""}] for _ in range(args.agents)]

        for round in range(args.rounds):
            for i, agent_context in enumerate(agent_contexts):
                if round != 0:
                    agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]
                    message = construct_message(agent_contexts_other, question, 2*round - 1, use_counterfactual=(round==args.rounds-1))
                    agent_context.append(message)

                completion = get_model_completion(args.model_type, args.model_name, agent_context)
                assistant_message = construct_assistant_message(completion, args.model_type)
                agent_context.append(assistant_message)

        generated_description[question] = (agent_contexts, answer)

    output_file = f"truthfulqa_{args.agents}_{args.rounds}_{args.model_type}.json"
    with open(output_file, "w") as f:
        json.dump(generated_description, f)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multi-agent debate for TruthfulQA")
    parser.add_argument("--model_type", type=str, choices=["openai", "anthropic", "mistral"], required=True, help="Type of model to use (openai, anthropic, or mistral)")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use (e.g., gpt-4 for OpenAI, claude-3-sonnet-20240229 for Anthropic, or mistral-large-latest for Mistral)")
    parser.add_argument("--agents", type=int, default=3, help="Number of agents")
    parser.add_argument("--rounds", type=int, default=2, help="Number of rounds")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--num_of_questions", type=int, default=50, help="Number of questions to process")

    args = parser.parse_args()
    main(args)