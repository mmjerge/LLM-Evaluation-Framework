import openai
import anthropic
from mistralai import Mistral
import argparse
import json
import numpy as np
import random
import os
import tqdm
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

openai.api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
mistral_api_key = os.getenv("MISTRAL_API_KEY")
mistral_client = Mistral(api_key=mistral_api_key)

def construct_message(agents, question, idx, use_counterfactual=False):
    if len(agents) == 0:
        return {"role": "user", "content": "Can you double check that your answer is correct. Please reiterate your answer, with your final answer a single numerical number, in the form \\boxed{{answer}}."}
    prefix_string = "These are the solutions to the problem from other agents: "
    for agent in agents:
        agent_response = agent[idx]["content"]
        response = "\n\n One agent solution: ```{}```".format(agent_response)
        prefix_string = prefix_string + response
    prefix_string = prefix_string + """\n\n Using the solutions from other agents as additional information, can you provide your answer to the math problem? \n The original math problem is {}. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response.""".format(question)
    return {"role": "user", "content": prefix_string}

def construct_assistant_message(completion, model_type):
    if model_type == "openai":
        content = completion.choices[0].message.content
    elif model_type == "anthropic":
        content = completion.content[0].text
    elif model_type == "mistral":
        content = completion.choices[0].message.content
    return {"role": "assistant", "content": content}

def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]

def get_model_completion(model_type, model_name, messages):
    global api_calls_count
    api_calls_count += 1
    if model_type == "openai":
        completion = openai.ChatCompletion.create(
            model=model_name,
            messages=messages,
            n=1
        )
    elif model_type == "anthropic":
        completion = anthropic_client.messages.create(
            model=model_name,
            messages=[{"role": m["role"], "content": m["content"]} for m in messages],
            max_tokens=1000
        )
    elif model_type == "mistral":
        completion = mistral_client.chat.complete(
            model=model_name,
            messages=[{"role": m["role"], "content": m["content"]} for m in messages]
        )
    return completion

def main(args):
    random.seed(args.seed)

    generated_description = {}
    questions = read_jsonl(args.input_file)
    random.shuffle(questions)

    global api_calls_count
    api_calls_per_problem = {}

    for data in tqdm.tqdm(questions[:args.num_of_questions], desc="Questions processed"):
        question = data['question']
        answer = data['answer']
        api_calls_count = 0
        agent_contexts = [[{"role": "user", "content": """Can you solve the following math problem? {} Explain your reasoning. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response. """.format(question)}] for _ in range(args.agents)]

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
        api_calls_per_problem[question] = api_calls_count

    output_file = f"gsm_{args.agents}_{args.rounds}_{args.model_type}.json"
    with open(output_file, "w") as f:
        json.dump({"results": generated_description, "api_calls": api_calls_per_problem}, f)
    print(f"Results saved to {output_file}")

    # Print summary of API calls
    total_calls = sum(api_calls_per_problem.values())
    avg_calls = total_calls / len(api_calls_per_problem)
    print(f"Total API calls: {total_calls}")
    print(f"Average API calls per problem: {avg_calls:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multi-agent debate for math problem solving")
    parser.add_argument("--model_type", type=str, choices=["openai", "anthropic", "mistral"], required=True, help="Type of model to use (openai, anthropic, or mistral)")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use (e.g., gpt-4 for OpenAI, claude-3-sonnet-20240229 for Anthropic, or mistral-large-latest for Mistral)")
    parser.add_argument("--agents", type=int, default=3, help="Number of agents")
    parser.add_argument("--rounds", type=int, default=2, help="Number of rounds")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--num_of_questions", type=int, default=50, help="Number of questions to process")
    parser.add_argument("--input_file", type=str, default="/p/llmreliability/test_repos/llm_multiagent_debate/gsm/grade-school-math/grade_school_math/data/test.jsonl", help="Path to input JSONL file")

    args = parser.parse_args()
    main(args)