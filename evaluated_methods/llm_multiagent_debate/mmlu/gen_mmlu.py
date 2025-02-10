import json
import time
import random
import os
import tqdm
import argparse
from glob import glob
import pandas as pd
from mistralai import Mistral
import anthropic
import openai
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Set your API keys
openai.api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
mistral_api_key = os.getenv("MISTRAL_API_KEY")

anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
mistral_client = Mistral(api_key=mistral_api_key)

# Initialize the PolyJuice model and tokenizer using the pipeline
model_path = "uw-hai/polyjuice"
is_cuda = False  # Set this to True if you want to use GPU
generator = pipeline(
    "text-generation",
    model=AutoModelForCausalLM.from_pretrained(model_path),
    tokenizer=AutoTokenizer.from_pretrained(model_path),
    framework="pt",
    device=0 if is_cuda else -1
)

def construct_message(agents, question, round_num, use_counterfactuals=False):
    if not agents:
        return {"role": "user", "content": "Can you double check that your answer is correct. Put your final answer in the form (X) at the end of your response."}

    prefix_string = "These are the solutions to the problem from other agents: "

    for agent in agents:
        if len(agent) > round_num * 2:
            agent_response = agent[round_num * 2]["content"]
            response = "\n\n One agent solution: ```{}```".format(agent_response)
            prefix_string += response

    prefix_string += """\n\n Using the reasoning from other agents as additional advice, can you give an updated answer? Examine your solution and that of other agents step by step. Put your answer in the form (X) at the end of your response."""

    if use_counterfactuals:
        counterfactuals = generate_counterfactuals(question)
        counterfactuals_text = "\n\n Here are some counterfactual scenarios to consider: " + " | ".join(counterfactuals)
        prefix_string += counterfactuals_text

    return {"role": "user", "content": prefix_string}

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

def generate_answer(messages, model_type, model_name):
    try:
        if model_type == "openai":
            completion = openai.ChatCompletion.create(
                model=model_name,
                messages=messages,
                n=1
            )
            return completion.choices[0].message.content
        elif model_type == "anthropic":
            completion = anthropic_client.messages.create(
                model=model_name,
                messages=[{"role": m["role"], "content": m["content"]} for m in messages],
                max_tokens=1000
            )
            return completion.content[0].text
        elif model_type == "mistral":
            completion = mistral_client.chat.complete(
                model=model_name,
                messages=[{"role": m["role"], "content": m["content"]} for m in messages]
            )
            return completion.choices[0].message.content
    except Exception as e:
        print(f"Error generating answer: {e}")
        print("Retrying due to an error...")
        time.sleep(20)
        return generate_answer(messages, model_type, model_name)

def parse_question_answer(df, ix):
    question = df.iloc[ix, 0]
    a, b, c, d = df.iloc[ix, 1:5]
    question_text = f"Can you answer the following question as accurately as possible? {question}: A) {a}, B) {b}, C) {c}, D) {d} Explain your answer, putting the answer in the form (X) at the end of your response."
    answer = df.iloc[ix, 5]
    return question_text, answer

def run_experiment(use_counterfactuals, model_type, model_name, num_entries=100):
    agents = 3
    rounds = 2

    tasks = glob("/p/llmreliability/test_repos/llm_multiagent_debate/mmlu/data/test/*.csv")
    dfs = [pd.read_csv(task) for task in tasks]

    random.seed(0)
    response_dict = {}

    pbar = tqdm.tqdm(total=num_entries, desc="Processing entries")

    for _ in range(num_entries):
        df = random.choice(dfs)
        idx = random.randint(0, len(df) - 1)

        question, answer = parse_question_answer(df, idx)

        agent_contexts = [[{"role": "user", "content": question}] for _ in range(agents)]

        for round in range(rounds):
            for i, agent_context in enumerate(agent_contexts):
                if round != 0:
                    agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]
                    message = construct_message(agent_contexts_other, question, round, use_counterfactuals=use_counterfactuals)
                    agent_context.append(message)

                response = generate_answer(agent_context, model_type, model_name)
                agent_context.append({"role": "assistant", "content": response})

        response_dict[question] = (agent_contexts, answer)
        pbar.update(1)

    pbar.close()

    suffix = f"with_counterfactuals_{model_type}" if use_counterfactuals else f"without_counterfactuals_{model_type}"
    with open(f"mmlu_{agents}_{rounds}_{suffix}.json", "w") as f:
        json.dump(response_dict, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multi-agent debate experiment")
    parser.add_argument("--model_type", type=str, choices=["openai", "anthropic", "mistral"], required=True, help="Type of model to use (openai, anthropic, or mistral)")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use (e.g., gpt-4 for OpenAI, claude-2 for Anthropic, or mistral-large-latest for Mistral)")
    parser.add_argument("--use_counterfactuals", action="store_true", help="Whether to use counterfactuals in the experiment")
    parser.add_argument("--num_entries", type=int, default=100, help="Number of entries to process (default: 100)")

    args = parser.parse_args()

    run_experiment(use_counterfactuals=args.use_counterfactuals, model_type=args.model_type, model_name=args.model_name, num_entries=args.num_entries)