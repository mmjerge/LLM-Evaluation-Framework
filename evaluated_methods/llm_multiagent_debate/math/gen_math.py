import openai
import json
import numpy as np
import time
import pickle
from tqdm import tqdm
import os
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

openai.api_key = os.getenv("OPENAI_API_KEY")

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

def parse_bullets(sentence):
    bullets_preprocess = sentence.split("\n")
    bullets = []

    for bullet in bullets_preprocess:
        try:
            idx = bullet.find(next(filter(str.isalpha, bullet)))
        except Exception as e:
            print(f"Error parsing bullet: {e}")
            continue

        bullet = bullet[idx:]

        if len(bullet) != 0:
            bullets.append(bullet)

    return bullets

def generate_counterfactuals(sentence):
    prompt_text = f"{sentence} [negation] {sentence.replace('is', '[BLANK]')}"
    counterfactuals = generator(
        prompt_text,
        num_beams=3,
        num_return_sequences=3,
        max_new_tokens=150  # Adjust max_new_tokens to fit your needs
    )
    counterfactuals_text = [output["generated_text"] for output in counterfactuals]
    return counterfactuals_text

def generate_answer(answer_context):
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=answer_context,
            n=1)
    except Exception as e:
        print(f"Error generating answer: {e}")
        print("Retrying due to an error...")
        time.sleep(20)
        return generate_answer(answer_context)

    return completion

def construct_message(agents, question, idx, use_counterfactual=False):
    if len(agents) == 0:
        return {"role": "user", "content": "Can you verify that your answer is correct. Please reiterate your answer, making sure to state your answer at the end of the response."}

    prefix_string = "These are the recent/updated opinions from other agents: "

    for agent in agents:
        agent_response = agent[idx]["content"]
        response = "\n\n One agent response: ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    prefix_string = prefix_string + "\n\n Use these opinions carefully as additional advice, can you provide an updated answer? Make sure to state your answer at the end of the response.".format(question)

    if use_counterfactual:
        counterfactuals = generate_counterfactuals(question)
        counterfactuals_text = "\n\n Here are some counterfactual scenarios to consider: " + " | ".join(counterfactuals)
        prefix_string += counterfactuals_text

    return {"role": "user", "content": prefix_string}

def construct_assistant_message(completion):
    content = completion["choices"][0]["message"]["content"]
    return {"role": "assistant", "content": content}

def parse_answer(sentence):
    parts = sentence.split(" ")

    for part in parts[::-1]:
        try:
            answer = float(part)
            return answer
        except:
            continue

def most_frequent(List):
    counter = 0
    num = List[0]

    for i in List:
        current_frequency = List.count(i)
        if current_frequency > counter:
            counter = current_frequency
            num = i

    return num

def run_experiment(use_counterfactuals):
    answer = parse_answer("My answer is the same as the other agents and AI language model: the result of 12+28*19+6 is 550.")

    agents = 2
    rounds = 3
    np.random.seed(0)

    evaluation_round = 100
    scores = []

    generated_description = {}

    for round in tqdm(range(evaluation_round)):
        a, b, c, d, e, f = np.random.randint(0, 30, size=6)

        answer = a + b * c + d - e * f
        agent_contexts = [[{"role": "user", "content": """What is the result of {}+{}*{}+{}-{}*{}? Make sure to state your answer at the end of the response.""".format(a, b, c, d, e, f)}] for agent in range(agents)]

        content = agent_contexts[0][0]['content']
        question_prompt = "We seek to find the result of {}+{}*{}+{}-{}*{}?".format(a, b, c, d, e, f)

        for round in range(rounds):
            for i, agent_context in enumerate(agent_contexts):

                if round != 0:
                    agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]
                    message = construct_message(agent_contexts_other, question_prompt, 2*round - 1, use_counterfactual=use_counterfactual)
                    agent_context.append(message)

                    print("Constructed message: ", message)

                try:
                    completion = generate_answer(agent_context)
                    assistant_message = construct_assistant_message(completion)
                    agent_context.append(assistant_message)
                    print("Generated completion: ", completion)
                except Exception as e:
                    print(f"Error during generation: {e}")
                    continue

        text_answers = []

        for agent_context in agent_contexts:
            text_answer = agent_context[-1]['content']
            text_answer = text_answer.replace(",", ".")
            text_answer = parse_answer(text_answer)

            if text_answer is None:
                print("Failed to parse answer: ", agent_context[-1]['content'])
                continue

            text_answers.append(text_answer)

        generated_description[(a, b, c, d, e, f)] = (agent_contexts, answer)

        try:
            text_answer = most_frequent(text_answers)
            if text_answer == answer:
                scores.append(1)
            else:
                scores.append(0)
        except Exception as e:
            print(f"Error finding most frequent answer: {e}")
            continue

        print("Performance:", np.mean(scores), np.std(scores) / (len(scores) ** 0.5))

    suffix = "with_counterfactuals" if use_counterfactuals else "without_counterfactuals"
    pickle.dump(generated_description, open(f"math_agents_{agents}_{rounds}_{suffix}.pickle", "wb"))
    print(answer)
    print(agent_context)

if __name__ == "__main__":
    # Run experiment without counterfactuals
    run_experiment(use_counterfactuals=False)

    # Run experiment with counterfactuals
    run_experiment(use_counterfactuals=True)