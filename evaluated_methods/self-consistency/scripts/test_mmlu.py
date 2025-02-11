import pandas as pd
import os
import re
from multiprocessing.pool import ThreadPool
import ujson as json
from tqdm import tqdm
from anthropic import Anthropic
import random
import time

# Check for API key
api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY environment variable is not set")

def read_mmlu_dataset(file_path, sample_size=None):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        data = [data]
    
    if sample_size is not None:
        if sample_size > len(data):
            print(f"Warning: Requested sample size ({sample_size}) is larger than the dataset size ({len(data)}). Using entire dataset.")
        else:
            data = random.sample(data, sample_size)
    
    print(f"Dataset loaded. Number of samples: {len(data)}")
    print(f"First item: {data[0]}")
    
    return data

# Initialize the Anthropic API client
try:
    client = Anthropic(api_key=api_key)
except Exception as e:
    print(f"Error initializing Anthropic client: {e}")
    exit(1)

def get_response(msg, mod="claude-3-5-sonnet-20240620", temp=0):
    try:
        system_message = next((m['content'] for m in msg if m['role'] == 'system'), None)
        user_messages = [m for m in msg if m['role'] == 'user']
        
        message = client.messages.create(
            max_tokens=1024,
            model=mod,
            temperature=temp,
            system=system_message,
            messages=user_messages
        )
        return message.content[0].text
    except Exception as e:
        print(f"Error in get_response: {e}")
        return None

def self_consistency_solver(question, n_experts):
    instructions = f'''
    Imagine {n_experts} completely independent experts who reason differently
    are answering a question. The question is delimited by triple backticks.
    The final answer is obtained by majority vote.
    Step 1. For each of the experts, give their step-by-step
    reasoning and answer, choosing from the given options (a, b, c, d).
    Step 2. Determine the final answer by majority vote.
    Step 3. Return the final answer, obtained by majority vote,
    prefixed by 'Final answer:' and followed by the letter of the chosen option (a, b, c, or d).
    IMPORTANT: only respond with 'Final answer: ' followed by the letter and no additional characters.
    '''
    user_content = f'```{question}```'
    msg = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": user_content}
    ]
    return get_response(msg=msg, temp=0.5)

def identify_final_answer(question, solution):
    if solution is None:
        print("No solution provided to identify_final_answer")
        return 'NA'

    instructions = '''
    You will be provided with the answer to a question.
    The question is delimited by triple backticks,
    and the answer is delimited by triple hashtags.
    Extract the final answer from the provided solution.
    Return only the letter corresponding to the chosen option (a, b, c, or d),
    prefixed by 'Final answer:'
    IMPORTANT: only respond with 'Final answer: ' followed by the letter and no additional characters.
    '''
    try:
        # First, try to split by 'Final answer:'
        if 'Final answer:' in solution:
            answer = solution.split('Final answer:')[1].strip()
        else:
            # If 'Final answer:' is not found, use the entire solution
            answer = solution.strip()
        
        # Remove any leading/trailing whitespace and punctuation
        answer = re.sub(r'^[\s\W]+|[\s\W]+$', '', answer)
        
        # If the answer is just a single letter, use it directly
        if len(answer) == 1 and answer.lower() in 'abcd':
            return f'Final answer: {answer.lower()}'
        
        # Otherwise, wrap the answer in triple hashtags
        answer = f'###{answer}###'
        
        user_content = f'```{question}```{answer}'
        msg = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": user_content}
        ]
        return get_response(msg=msg, temp=0)
    except Exception as e:
        print(f"Error in identify_final_answer: {e}")
        return 'NA'

def parse_final_answer(evaluation):
    if evaluation is None:
        print("No evaluation provided to parse_final_answer")
        return 'NA'

    try:
        # Look for "Final answer:" or "Answer:" followed by a letter
        match = re.search(r'(?:Final answer:|Answer:)\s*([a-d])', evaluation, re.IGNORECASE)
        if match:
            return match.group(1).lower()
        
        # Look for a letter followed by a closing parenthesis
        match = re.search(r'([a-d])\)', evaluation, re.IGNORECASE)
        if match:
            return match.group(1).lower()
        
        # Look for "Option X" or "X)" where X is a, b, c, or d
        match = re.search(r'(?:Option\s*|^)([a-d])(?:\)|:|\s|$)', evaluation, re.IGNORECASE)
        if match:
            return match.group(1).lower()
        
        # If no match is found, look for any standalone letter a-d
        options = re.findall(r'\b([a-d])\b', evaluation, re.IGNORECASE)
        if options:
            return options[-1].lower()  # Return the last mentioned option
        
        # If still no match, return 'NA'
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

def multi_step_solver(question, n_experts, n_attempts):
    pool = ThreadPool(n_attempts)
    attempts = pool.starmap(self_consistency_solver, [(question, n_experts) for _ in range(n_attempts)])
    answers = pool.starmap(identify_final_answer, [(question, attempt) for attempt in attempts])
    
    # Parse the final answers from both attempts and answers
    answers_parsed_attempts = [parse_final_answer(attempt) for attempt in attempts]
    answers_parsed_answers = [parse_final_answer(answer) for answer in answers]
    
    # Combine parsed answers, preferring non-'NA' results
    answers_parsed = [a if a != 'NA' else b for a, b in zip(answers_parsed_attempts, answers_parsed_answers)]
    
    best_answer = get_best_answer(answers_parsed)
    return {
        "best_answer": best_answer,
        "attempts": attempts,
        "answers": answers,
        "answers_parsed": answers_parsed
    }

# Main execution
if __name__ == "__main__":
    try:
        file_path = '/p/llmreliability/test_repos/llmpromptboosting/dataset/mmlu570/test.json'
        sample_size = 150
        sample = read_mmlu_dataset(file_path, sample_size=sample_size)

        try:
            with open('claude_answers.ndjson', 'r') as f:
                processed = sum(1 for line in f)
            start = processed
        except FileNotFoundError:
            start = 0

        for i in tqdm(range(start, len(sample))):
            try:
                item = sample[i]
                for n_attempts in [1, 3, 5, 10]:
                    for n_experts in [1, 3, 5]:
                        question = item['question']
                        true_answer = item['answer']
                        question_type = item['type']
                        
                        print(f"Processing item {i}, n_attempts={n_attempts}, n_experts={n_experts}")
                        print(f"Question: {question}")
                        
                        results = multi_step_solver(question, n_experts=n_experts, n_attempts=n_attempts)
                        
                        print(f"Results: {results}")
                        
                        if results['best_answer'] != 'NA':
                            with open(f'claude_answers.ndjson', 'a+') as f:
                                json.dump({
                                    "n_experts": n_experts,
                                    "n_attempts": n_attempts,
                                    "question": question,
                                    "true_answer": true_answer,
                                    "best_answer": results['best_answer'],
                                    "attempts": results['attempts'],
                                    "answers": results['answers'],
                                    "answers_parsed": results['answers_parsed'],
                                    "type": question_type
                                }, f)
                                f.write('\n')
                                f.flush()
                        else:
                            print(f"Skipping write for question {i} due to NA best_answer")
                        
                        # Add a delay to avoid hitting rate limits
                        time.sleep(1)
            except Exception as e:
                print(f"Error processing item {i}: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")