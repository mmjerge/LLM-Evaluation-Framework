import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

model_name = "allenai/truthfulqa-truth-judge-llama2-7B"
info_judge = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def evaluate_answer(question, model_answer, max_new_tokens=256):
    prompt = f"Q: {question}\nA: {model_answer}\nHelpful:"
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = info_judge.generate(inputs, max_new_tokens=max_new_tokens)
    evaluation = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):].strip()
    return evaluation

# models_to_evaluate = ['Llama-3.1-8b-tot']

input_csv = '/p/llmreliability/test_repos/TruthfulQA/SoK_Experiments/scripts/Llama-3.1-8b-tot_reponses.csv'
df = pd.read_csv(input_csv)

evaluations = []
for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Evaluating llama answers"):
    evaluation = evaluate_answer(row['Question'], row[f'Llama-3.1-8b-tot_Answer'])
    evaluations.append(evaluation)

df['Evaluation'] = evaluations

df['Evaluated_Model'] = 'Llama-3.1-8b-tot'

output_csv = f'evaluation_results_Llama-3.1-8b-tot.csv'
df.to_csv(output_csv, index=False)