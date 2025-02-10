import json
import os
import time
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from tenacity import retry, stop_after_attempt, wait_chain, wait_fixed
from utils import *
from openai import OpenAI

TASKS = [
        'abstract_algebra',
        'anatomy',
        'astronomy',
        'business_ethics',
        'clinical_knowledge',
        'college_biology',
        'college_chemistry',
        'college_computer_science',
        'college_mathematics',
        'college_medicine',
        'college_physics',
        'computer_security',
        'conceptual_physics',
        'econometrics',
        'electrical_engineering',
        'elementary_mathematics',
        'formal_logic',
        'global_facts',
        'high_school_biology',
        'high_school_chemistry',
        'high_school_computer_science',
        'high_school_european_history',
        'high_school_geography',
        'high_school_government_and_politics',
        'high_school_macroeconomics',
        'high_school_mathematics',
        'high_school_microeconomics',
        'high_school_physics',
        'high_school_psychology',
        'high_school_statistics',
        'high_school_us_history',
        'high_school_world_history',
        'human_aging',
        'human_sexuality',
        'international_law',
        'jurisprudence',
        'logical_fallacies',
        'machine_learning',
        'management',
        'marketing',
        'medical_genetics',
        'miscellaneous',
        'moral_disputes',
        'moral_scenarios',
        'nutrition',
        'philosophy',
        'prehistory',
        'professional_accounting',
        'professional_law',
        'professional_medicine',
        'professional_psychology',
        'public_relations',
        'security_studies',
        'sociology',
        'us_foreign_policy',
        'virology',
        'world_religions']

@retry(wait=wait_chain(*[wait_fixed(3) for i in range(3)] +
                       [wait_fixed(5) for i in range(2)] +
                       [wait_fixed(10)]))
def completion_with_backoff(**kwargs):
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        # base_url="https://api.together.xyz/v1",
    )
    return client.chat.completions.create(**kwargs)

def main(tasks=TASKS):
    mmlu_prompt = json.load(open('lib_prompt/mmlu-cot.json'))
    results = {}

    for task in tqdm(tasks, desc="Processing tasks"):
        print(f"Starting task: {task}")
        acc = 0
        task_data = load_dataset("lukaemon/mmlu", task)
        num_samples = len(task_data['test'])
        sample_size = int(num_samples * 0.1)
        sampled_indices = np.random.choice(num_samples, size=sample_size, replace=False)
        sampled_indices = [int(i) for i in sampled_indices]
        sampled_data = [task_data['test'][i] for i in sampled_indices]

        task_results = []
        for question_data in tqdm(sampled_data, total=sample_size, desc=f"Processing {task}"):
            print(f"Processing question for {task}")
            question = question_data['input'] + '\n'
            for letter in ['A', 'B', 'C', 'D']:
                question += f'({letter}) {question_data[letter]} '
            question += "\nA: Let's think step by step."
            prompt_question = mmlu_prompt[task] + "\n\n" + question

            try:
                response = completion_with_backoff(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "Follow the given examples and answer the question."},
                        {"role": "user", "content": prompt_question},
                    ],
                    temperature=0.5
                )
                ans_model = response.choices[0].message.content
                print(f"API Response: {ans_model}")
                ans_extracted, _ = extract_ans(ans_model)
                correct_answer = question_data['target']

                is_correct = test_answer_mmlu_(ans_extracted, correct_answer)
                if is_correct:
                    acc += 1

                question_result = {
                    'question': question,
                    'model_answer': ans_model,
                    'correct_answer': correct_answer,
                    'is_correct': is_correct
                }
                task_results.append(question_result)

            except Exception as e:
                print(f"Error processing question in {task}: {str(e)}")
                print(f"Full error: {repr(e)}")
                question_result = {
                    'question': question,
                    'error': str(e)
                }
                task_results.append(question_result)

        accuracy = acc / sample_size
        results[task] = {
            'accuracy': accuracy,
            'questions': task_results
        }
        print(f'{task} accuracy: {accuracy:.4f}')

        # Write results to JSON file after each task
        with open('mmlu_results.json', 'w') as f:
            json.dump(results, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

    # Calculate and add overall average accuracy
    average_accuracy = np.mean([task_data['accuracy'] for task_data in results.values()])
    results['average_accuracy'] = average_accuracy
    print(f'Average accuracy across all tasks: {average_accuracy:.4f}')

    # Final write to ensure all data is saved
    with open('mmlu_results.json', 'w') as f:
        json.dump(results, f, indent=2)
        f.flush()
        os.fsync(f.fileno())

if __name__ == '__main__':
    main()