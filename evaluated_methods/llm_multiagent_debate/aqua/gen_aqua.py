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

# Initialize the PolyJuice model and tokenizer
model_path = "uw-hai/polyjuice"
generator = pipeline(
    "text-generation",
    model=AutoModelForCausalLM.from_pretrained(model_path),
    tokenizer=AutoTokenizer.from_pretrained(model_path),
    framework="pt",
    device=-1  # Use CPU. Set to 0 for GPU
)

def generate_counterfactuals(sentence):
    prompt_text = f"{sentence} [negation] {sentence.replace('is', '[BLANK]')}"
    prompt_text = truncate_text(prompt_text, 1024 - 150)  # Adjust max length as needed
    counterfactuals = generator(
        prompt_text,
        num_beams=3,
        num_return_sequences=3,
        max_new_tokens=150
    )
    counterfactuals_text = [output["generated_text"] for output in counterfactuals]
    return counterfactuals_text

def truncate_text(text, max_length):
    tokens = generator.tokenizer.encode(text)
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
    return generator.tokenizer.decode(tokens)

def construct_message(agents, question, options, idx, use_counterfactual=False):
    if len(agents) == 0:
        return {"role": "user", "content": f"Can you double-check that your answer is correct? Please reiterate your answer, with your final answer being one of the options: {', '.join(options)}."}
    prefix_string = "These are the solutions to the problem from other agents: "
    for agent in agents:
        agent_response = agent[idx]["content"]
        response = "\n\n One agent solution: ```{}```".format(agent_response)
        prefix_string = prefix_string + response
    prefix_string = prefix_string + f"""\n\n Using the solutions from other agents as additional information, can you provide your answer to the math problem? \n The original math problem is {question}. Your final answer should be one of the options: {', '.join(options)}."""
    
    if use_counterfactual:
        counterfactuals = generate_counterfactuals(question)
        counterfactuals_text = "\n\n Here are some counterfactual scenarios to consider: " + " | ".join(counterfactuals)
        prefix_string += counterfactuals_text
    
    return {"role": "user", "content": prefix_string}

def construct_message(agents, question, options, idx, use_counterfactual=False):
    if len(agents) == 0:
        return {"role": "user", "content": f"Can you double-check that your answer is correct? Please reiterate your answer, with your final answer being one of the options: {', '.join(options)}."}
    prefix_string = "These are the solutions to the problem from other agents: "
    for agent in agents:
        agent_response = agent[idx]["content"]
        response = "\n\n One agent solution: ```{}```".format(agent_response)
        prefix_string = prefix_string + response
    prefix_string = prefix_string + f"""\n\n Using the solutions from other agents as additional information, can you provide your answer to the math problem? \n The original math problem is {question}. Your final answer should be one of the options: {', '.join(options)}."""
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

    for data in tqdm.tqdm(questions[:args.num_of_questions], desc="Questions processed"):
        question = data['question']
        options = data['options']
        answer = data['correct']
        agent_contexts = [[{"role": "user", "content": f"Can you solve the following math problem? {question}. Please select your answer from the following options: {', '.join(options)}."}] for _ in range(args.agents)]

        for round in range(args.rounds):
            for i, agent_context in enumerate(agent_contexts):
                if round != 0:
                    agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]
                    message = construct_message(agent_contexts_other, question, options, 2*round - 1, use_counterfactual=(round==args.rounds-1 and args.use_counterfactuals))
                    agent_context.append(message)

                completion = get_model_completion(args.model_type, args.model_name, agent_context)
                assistant_message = construct_assistant_message(completion, args.model_type)
                agent_context.append(assistant_message)

        generated_description[question] = {
            "contexts": agent_contexts,
            "correct_answer": answer,
            "options": options
        }

    output_file = f"aqua_{args.agents}_{args.rounds}_{args.model_type}_{'with_counterfactuals' if args.use_counterfactuals else 'without_counterfactuals'}.json"
    with open(output_file, "w") as f:
        json.dump(generated_description, f)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multi-agent debate for AQUA dataset")
    parser.add_argument("--model_type", type=str, choices=["openai", "anthropic", "mistral"], required=True, help="Type of model to use (openai, anthropic, or mistral)")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use (e.g., gpt-4 for OpenAI, claude-3-sonnet-20240229 for Anthropic, or mistral-large-latest for Mistral)")
    parser.add_argument("--agents", type=int, default=3, help="Number of agents")
    parser.add_argument("--rounds", type=int, default=2, help="Number of rounds")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--num_of_questions", type=int, default=50, help="Number of questions to process")
    parser.add_argument("--input_file", type=str, default="/p/llmreliability/test_repos/llm_multiagent_debate/aqua/test.jsonl", help="Path to input JSONL file")
    parser.add_argument("--use_counterfactuals", action="store_true", help="Whether to use counterfactuals in the experiment")

    args = parser.parse_args()
    main(args)