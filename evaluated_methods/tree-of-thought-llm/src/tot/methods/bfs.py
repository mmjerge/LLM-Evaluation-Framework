import itertools
import numpy as np
from functools import partial
from tot.models import gpt, claude

def get_llm_function(backend):
    if backend.startswith('gpt'):
        return gpt
    elif backend.startswith('claude'):
        return claude
    else:
        raise ValueError(f"Unsupported backend: {backend}")

def get_value(task, x, y, n_evaluate_sample, llm_func, cache_value=True):
    value_prompt = task.value_prompt_wrap(x, y)
    if cache_value and value_prompt in task.value_cache:
        return task.value_cache[value_prompt]
    value_outputs = llm_func(value_prompt, n=n_evaluate_sample, stop=None)
    value = task.value_outputs_unwrap(x, y, value_outputs)
    if cache_value:
        task.value_cache[value_prompt] = value
    return value

def get_values(task, x, ys, n_evaluate_sample, llm_func, cache_value=True):
    values = []
    local_value_cache = {}
    for y in ys:  # each partial output
        if y in local_value_cache:  # avoid duplicate candidates
            value = 0
        else:    
            value = get_value(task, x, y, n_evaluate_sample, llm_func, cache_value=cache_value)
            local_value_cache[y] = value
        values.append(value)
    return values

def get_votes(task, x, ys, n_evaluate_sample, llm_func):
    vote_prompt = task.vote_prompt_wrap(x, ys)
    vote_outputs = llm_func(vote_prompt, n=n_evaluate_sample, stop=None)
    values = task.vote_outputs_unwrap(vote_outputs, len(ys))
    return values

def get_proposals(task, x, y, llm_func): 
    propose_prompt = task.propose_prompt_wrap(x, y)
    proposals = llm_func(propose_prompt, n=1, stop=None)[0].split('\n')
    return [y + _ + '\n' for _ in proposals]

def get_samples(task, x, y, n_generate_sample, prompt_sample, stop, llm_func):
    if prompt_sample == 'standard':
        prompt = task.standard_prompt_wrap(x, y)
    elif prompt_sample == 'cot':
        prompt = task.cot_prompt_wrap(x, y)
    else:
        raise ValueError(f'prompt_sample {prompt_sample} not recognized')
    samples = llm_func(prompt, n=n_generate_sample, stop=stop)
    return [y + _ for _ in samples]

def solve(args, task, idx, to_print=True):
    llm_func = get_llm_function(args.backend)
    llm_func = partial(llm_func, model=args.backend, temperature=args.temperature)
    print(llm_func)
    x = task.get_input(idx) # input
    ys = [''] # current output candidates
    infos = []
    for step in range(task.steps):
        # generation
        if args.method_generate == 'sample':
            new_ys = [get_samples(task, x, y, args.n_generate_sample, prompt_sample=args.prompt_sample, stop=task.stops[step], llm_func=llm_func) for y in ys]
        elif args.method_generate == 'propose':
            new_ys = [get_proposals(task, x, y, llm_func) for y in ys]
        new_ys = list(itertools.chain(*new_ys))
        ids = list(range(len(new_ys)))
        # evaluation
        if args.method_evaluate == 'vote':
            values = get_votes(task, x, new_ys, args.n_evaluate_sample, llm_func)
        elif args.method_evaluate == 'value':
            values = get_values(task, x, new_ys, args.n_evaluate_sample, llm_func)
        # selection
        if args.method_select == 'sample':
            ps = np.array(values) / sum(values)
            select_ids = np.random.choice(ids, size=args.n_select_sample, p=ps).tolist()
        elif args.method_select == 'greedy':
            select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[:args.n_select_sample]
        select_new_ys = [new_ys[select_id] for select_id in select_ids]
        
        # Ensure there's a clear final answer
        if step == task.steps - 1 or any("####" in y for y in select_new_ys):
            final_answer_prompt = f"Based on the reasoning above, what is the final answer? Please format your response as '#### [A/B/C/D/E]'.\n\n{select_new_ys[0]}"
            final_answer = llm_func(final_answer_prompt, n=1, stop=None)[0]
            select_new_ys[0] += f"\n{final_answer}"
        
        # log
        if to_print:
            sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True))
            print(f'-- new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}\n-- choices --: {select_new_ys}\n')
        infos.append({'step': step, 'x': x, 'ys': ys, 'new_ys': new_ys, 'values': values, 'select_new_ys': select_new_ys})
        
        # Early stopping conditions
        if 1.0 in values:
            best_index = values.index(1.0)
            if to_print:
                print(f"Correct answer found: {new_ys[best_index]}")
            return [new_ys[best_index]], {'steps': infos}
        
        if any("####" in y for y in select_new_ys):
            if to_print:
                print(f"Final answer found: {[y for y in select_new_ys if '####' in y][0]}")
            return [y for y in select_new_ys if '####' in y], {'steps': infos}
        
        ys = select_new_ys
        if to_print:
            print(ys)
    
    if to_print:
        print("Maximum steps reached without finding a definitive answer.")
    return ys, {'steps': infos}

# def solve(args, task, idx, to_print=True):
#     global gpt
#     gpt = partial(gpt, model=args.backend, temperature=args.temperature)
#     print(gpt)
#     x = task.get_input(idx)  # input
#     ys = ['']  # current output candidates
#     infos = []
#     for step in range(task.steps):
#         # generation
#         if args.method_generate == 'sample':
#             new_ys = [get_samples(task, x, y, args.n_generate_sample, prompt_sample=args.prompt_sample, stop=task.stops[step]) for y in ys]
#         elif args.method_generate == 'propose':
#             new_ys = [get_proposals(task, x, y) for y in ys]
#         new_ys = list(itertools.chain(*new_ys))
#         ids = list(range(len(new_ys)))
#         # evaluation
#         if args.method_evaluate == 'vote':
#             values = get_votes(task, x, new_ys, args.n_evaluate_sample)
#         elif args.method_evaluate == 'value':
#             values = get_values(task, x, new_ys, args.n_evaluate_sample)

#         # selection
#         if args.method_select == 'sample':
#             ps = np.array(values) / sum(values)
#             select_ids = np.random.choice(ids, size=args.n_select_sample, p=ps).tolist()
#         elif args.method_select == 'greedy':
#             select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[:args.n_select_sample]
#         select_new_ys = [new_ys[select_id] for select_id in select_ids]

#         # log
#         if to_print: 
#             sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True))
#             print(f'-- new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}\n-- choices --: {select_new_ys}\n')
        
#         infos.append({'step': step, 'x': x, 'ys': ys, 'new_ys': new_ys, 'values': values, 'select_new_ys': select_new_ys})
#         ys = select_new_ys
    
#     if to_print: 
#         print(ys)
#     return ys, {'steps': infos}

def naive_solve(args, task, idx, to_print=True):
    llm_func = get_llm_function(args.backend)
    llm_func = partial(llm_func, model=args.backend, temperature=args.temperature)
    print(llm_func)
    x = task.get_input(idx)  # input
    ys = get_samples(task, x, '', args.n_generate_sample, args.prompt_sample, stop=None, llm_func=llm_func)
    return ys, {}