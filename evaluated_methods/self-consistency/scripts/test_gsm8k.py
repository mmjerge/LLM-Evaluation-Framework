import pandas as pd
import os
import re
from multiprocessing.pool import ThreadPool
import ujson as json
from tqdm import tqdm
from openai import OpenAI

train = pd.read_json(path_or_buf='test.jsonl', lines=True)
sample = train.sample(n=100, random_state=1)


def get_response(msg, mod="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", temp=0):
    response = client.chat.completions.create(
        model=mod,
        messages=msg,
        temperature=temp
    )
    return response.choices[0].message.content

def self_consistency_solver(question, options, n_experts):
    instructions = f'''
    Imagine {n_experts} completely independent experts who reason differently
    are answering a question. The question and options are delimited by triple backticks.
    The final answer is obtained by majority vote.
    Step 1. For each of the experts, give their step-by-step
    reasoning and answer, choosing from the given options.
    Step 2. Determine the final answer by majority vote.
    Step 3. Return the final answer, obtained by majority vote,
    prefixed by 'Final answer:' and followed by the letter of the chosen option (A, B, C, D, or E).
    '''
    user_content = f'Question: {question}\nOptions: {options}'
    msg = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": user_content}
    ]
    return get_response(msg=msg, temp=0.5)

def identify_final_answer(question, options, solution):
    instructions = '''
    You will be provided with the answer to a question.
    The question and options are delimited by triple backticks,
    and the answer is delimited by triple hashtags.
    Extract the final answer from the provided solution.
    Return only the letter corresponding to the chosen option (A, B, C, D, or E),
    prefixed by 'Final answer:'
    '''
    try:
        # Attempt to split the solution by "Final answer:"
        if 'Final answer:' in solution:
            answer = solution.split('Final answer:')[1].strip()
        else:
            # If 'Final answer:' is not found, use regex to find the answer
            match = re.search(r'Final answer:\s*([A-E])', solution, re.IGNORECASE)
            if match:
                answer = match.group(1).strip()
            else:
                return 'NA'
        
        # Format the answer for the follow-up query
        answer = f'###{answer}###'
        user_content = f'```Question: {question}\nOptions: {options}```' + answer
        messages = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": user_content}
        ]
        return get_response(msg=messages, temp=0)  # Corrected argument name
    except Exception as e:
        print(f"Error in identify_final_answer: {e}")
        return 'NA'

def parse_final_answer(evaluation):
    try:
        # Adjust the regex to capture the final answer
        match = re.search(r'Final answer:\s*([A-E])', evaluation, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        # If no match, fallback to capturing the last mentioned option letter
        options = re.findall(r'\b([A-E])\)', evaluation, re.IGNORECASE)
        if options:
            return options[-1].upper()  # Return the last mentioned option
        
        return 'NA'
    except Exception as e:
        print(f"Error in parse_final_answer: {e}")
        return 'NA'

def get_best_answer(options):
    answer_count = [[x, options.count(x)] for x in set(options) if x not in ['', 'NA']]
    answer_count_sorted = sorted(answer_count, key=lambda x: x[1], reverse=True)
    if len(answer_count_sorted) > 0:
        return answer_count_sorted[0][0]
    else:
        return 'NA'

def get_true_answer(answer):
    answer = answer.split('### ')[1]
    answer = re.sub('[^\d\.]', '', answer)
    return answer

def multi_step_solver(question, options, n_experts, n_attempts):
    pool = ThreadPool(n_attempts)
    attempts = pool.starmap(self_consistency_solver, [(question, options, n_experts) for _ in range(n_attempts)])
    answers = pool.starmap(identify_final_answer, [(question, options, attempt) for attempt in attempts])
    answers_parsed = [parse_final_answer(i) for i in answers]
    best_answer = get_best_answer(answers_parsed)
    return {
        "best_answer": best_answer,
        "attempts": attempts,
        "answers": answers,
        "answers_parsed": answers_parsed
    }

try:
    df = pd.read_json('answers.ndjson', lines=True)
    start = len(df[(df['n_experts'] == 1) & (df['n_attempts'] == 1)])
except:
    start = 0

for i in tqdm(range(start, len(sample))):
    for n_attempts in [1, 3, 5, 10]:
        for n_experts in [1, 3, 5]:
            question = sample.iloc[i].question
            options = sample.iloc[i].options
            true_answer = sample.iloc[i].correct
            results = multi_step_solver(question, options, n_experts=n_experts, n_attempts=n_attempts)
            with open(f'answers.ndjson', 'a+') as f:
                json.dump({
                    "n_experts": n_experts,
                    "n_attempts": n_attempts,
                    "question": question,
                    "options": options,
                    "true_answer": true_answer,
                    "best_answer": results['best_answer'],
                    "attempts": results['attempts'],
                    "answers": results['answers'],
                    "answers_parsed": results['answers_parsed']
                }, f)
                f.write('\n')