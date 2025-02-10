from openai import OpenAI
import os
import re
import time
import json
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from tenacity import retry, stop_after_attempt, wait_chain, wait_fixed

client = OpenAI(
  api_key=os.environ.get("OPENAI_API_KEY")
)

@retry(wait=wait_chain(*[wait_fixed(3) for i in range(3)] +
                       [wait_fixed(5) for i in range(2)] +
                       [wait_fixed(10)]))
def completion_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)

mmlu_prompt = json.load(open('lib_prompt/mmlu-cot.json'))

mmlu_prompt.keys()

abstract_algebra = load_dataset("lukaemon/mmlu", "abstract_algebra")

abstract_algebra['test'][0]

prompt_q = mmlu_prompt['abstract_algebra'] + "\n\n" + abstract_algebra['test'][0]['input'] + '\n'
for letter in ['A', 'B', 'C', 'D']:
    prompt_q += '(' + letter + ') ' + abstract_algebra['test'][0][letter] + ' '
prompt_q += "\nA: Let's think step by step."

response = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
        {"role": "system", "content": "Follow the given examples and answer the question."},
        {"role": "user", "content": prompt_q},
    ]
)

def test_answer_mmlu(pred_str, ans_str):
    pattern = 'the answer is ('
    pred = pred_str.lower().split(pattern)
    
    if(len(pred) > 1):
        # print(pred)
        pred = pred[1][0]
        gold = ans_str.split('A:\n')[1][0].lower()
        # print('debug 1, pred %s, gold %s' % (pred, gold))
        return pred == gold
    else: 
        pred = 'C'
        gold = ans_str.split('A:\n')[1][0].lower()
        # print('debug 2, pred %s, gold %s' % (pred, gold))
        return pred == gold

def parse_pred_ans(filename):
    with open(filename) as fd: lines = fd.readlines()
    am, a = None, None
    num_q, acc = 0, 0
    current_mode = 'none'
    questions = []
    ans_pred = []
    ans_gold = []
    for l in lines:
        if(l.startswith('Q: ')):
            if(am is not None and a is not None):
                questions.append(q)
                ans_pred.append(am)
                ans_gold.append(a)
                # print(am)
                # print(a)
                if(test_answer_mmlu(am, a)):
                    acc += 1
            current_mode = 'q'
            q = l
            num_q += 1
        elif(l.startswith('A_model:')):
            current_mode = 'am'
            am = l
        elif(l.startswith('A:')):
            current_mode = 'a'
            a = l
        else:
            if(current_mode == 'q'): q += l
            elif(current_mode == 'am'): am += l
            elif(current_mode == 'a'): a += l
            else:
                raise ValueError(current_mode)
            
    questions.append(q)            
    ans_pred.append(am)
    ans_gold.append(a)
    # print(am)
    # print(a)
    if(test_answer_mmlu(am, a)):
        acc += 1
    print('num_q %d correct %d ratio %.4f' % (num_q, acc, float(acc / num_q)))
    return questions, ans_pred, ans_gold

def test_finished(ans_model):
    if('answer is' in ans_model): return True
    else: return False

def extract_ans(ans_model):
    ans_model = ans_model.split('\n')
    ans = []
    residual = []
    for li, al in enumerate(ans_model):
        ans.append(al)
        if('answer is' in al):
            break
    residual = list(ans_model[li + 1:])
    ans = '\n'.join(ans)
    residual = '\n'.join(residual)
    return ans, residual

task = 'abstract_algebra'

i = 0
with open('outputs/test_gpt_3.5_turbo_%s.txt' % task, 'w') as fd:
    for q_ in tqdm(abstract_algebra['test'], total=len(abstract_algebra['test'])):
        q = q_['input'] + '\n'
        for letter in ['A', 'B', 'C', 'D']:
            q += '(' + letter + ') ' + q_[letter] + ' '
        q += "\nA: Let's think step by step."  
            
        prompt_q = mmlu_prompt[task] + "\n\n" + q

        response = completion_with_backoff(
              model="gpt-3.5-turbo",
              messages=[
                    {"role": "system", "content": "Follow the given examples and answer the question."},
                    {"role": "user", "content": prompt_q},
                ]
            )
        ans_model = response['choices'][0]['message']['content']
        ans_, residual = extract_ans(ans_model)
            
        a = q_['target']
        fd.write('Q: %s\nA_model:\n%s\nA:\n%s\n\n' % (q, ans_, a))
        i += 1
        
_, _, _ = parse_pred_ans('outputs/test_gpt_3.5_turbo_%s.txt' % task)