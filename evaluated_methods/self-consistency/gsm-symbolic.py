import os
from openai import OpenAI
from anthropic import Anthropic
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from datasets import load_dataset
import pandas as pd
import re
from multiprocessing.pool import ThreadPool
import jsonlines
from typing import Literal
import time

openai_client = OpenAI()
anthropic_client = Anthropic()
mistral_client = MistralClient(
    api_key=os.environ["MISTRAL_API_KEY"]
)

ModelProvider = Literal["openai", "anthropic", "mistral"]

def get_response(question: str, 
                provider: ModelProvider = "openai", 
                model: str = "gpt-3.5-turbo", 
                temp: float = 0.7) -> str:
    """Get response from the specified model provider."""
    instructions = '''
    Let's approach this step-by-step:
    1. First, understand what the question is asking
    2. Break down the problem into smaller parts
    3. Solve each part
    4. Combine the results to get the final answer
    
    Return your final answer in the format: "The answer is X."
    '''
    
    if provider == "openai":
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": f"```{question}```"}
            ],
            temperature=temp
        )
        return response.choices[0].message.content
        
    elif provider == "anthropic":
        response = anthropic_client.messages.create(
            model=model, 
            max_tokens=1024,
            temperature=temp,
            system=instructions,
            messages=[
                {"role": "user", "content": f"```{question}```"}
            ]
        )
        return response.content[0].text
        
    elif provider == "mistral":
        messages = [
            ChatMessage(role="system", content=instructions),
            ChatMessage(role="user", content=f"```{question}```")
        ]
        response = mistral_client.chat(
            model=model,
            messages=messages,
            temperature=temp
        )
        return response.choices[0].message.content
    
    else:
        raise ValueError(f"Unsupported provider: {provider}")

def generate_reasoning_path(question: str, provider: ModelProvider, model: str, temp: float = 0.7) -> str:
    """Generate a single reasoning path for a given question."""
    return get_response(question, provider, model, temp)

def identify_final_answer(solution: str) -> str:
    """Extract the final answer from a solution."""
    try:
        answer = solution.split('The answer is ')[1].split('.')[0]
        return clean_answer(answer)
    except:
        return 'NA'

def clean_answer(answer: str) -> str:
    """Clean the answer to only contain numeric values."""
    answer_clean = re.sub('[^\d\.]', '', answer)
    if not answer_clean:
        return 'NA'
    
    if '.' in answer_clean:
        answer_clean = answer_clean.rstrip('0').rstrip('.')
    
    return answer_clean

def get_best_answer(options: list[str]) -> str:
    """Get the most consistent answer through majority voting."""
    valid_options = [x for x in options if x not in ['', 'NA']]
    if not valid_options:
        return 'NA'
    
    answer_count = {}
    for ans in valid_options:
        answer_count[ans] = answer_count.get(ans, 0) + 1
    
    return max(answer_count.items(), key=lambda x: x[1])[0]

def get_true_answer(answer: str) -> str:
    """Extract the true answer from the ground truth."""
    answer = answer.split('### ')[1]
    return re.sub('[^\d\.]', '', answer)

def self_consistency_solver(question: str, 
                          provider: ModelProvider,
                          model: str,
                          n_samples: int = 40,
                          temp: float = 0.7,
                          max_workers: int = 5, 
                          delay: float = 1.0) -> dict:
    """
    Implementation of self-consistency method with rate limiting
    """
    pool = ThreadPool(max_workers)
    
    def delayed_generate_reasoning(q, p, m, t):
        time.sleep(delay)
        return generate_reasoning_path(q, p, m, t)
    
    reasoning_paths = pool.starmap(
        delayed_generate_reasoning,
        [(question, provider, model, temp) for _ in range(n_samples)]
    )
    pool.close()
    
    answers = [identify_final_answer(path) for path in reasoning_paths]
    best_answer = get_best_answer(answers)
    
    return {
        "best_answer": best_answer,
        "paths": reasoning_paths,
        "answers": answers
    }

def run_evaluation(provider: ModelProvider, model: str):
    """Run evaluation for a specific provider and model."""
    print(f"Loading dataset...")
    ds = load_dataset("apple/GSM-Symbolic", "main")
    train_df = pd.DataFrame(ds['test'])
    sample = train_df.sample(n=150, random_state=1)

    output_file = f"results_gsm_symbolic_{provider}_{model.replace('-', '_')}.jsonl"

    try:
        with jsonlines.open(output_file, 'r') as reader:
            df = pd.DataFrame([item for item in reader])
        start = len(df)
    except:
        start = 0
        
    print(f"Running tests starting from index {start}...")
    for i in range(start, len(sample)):
        question = sample.iloc[i].question
        true_answer = get_true_answer(sample.iloc[i].answer)
        results = self_consistency_solver(question, provider, model)
        
        with jsonlines.open(output_file, mode='a') as writer:
            writer.write({
                "question": question,
                "true_answer": true_answer,
                "best_answer": results['best_answer'],
                "paths": results['paths'],
                "answers": results['answers'],
                "provider": provider,
                "model": model
            })
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(sample)} questions")

    print("Testing completed!")

# run_evaluation("openai", "gpt-3.5-turbo")
# run_evaluation("anthropic", "claude-3-5-sonnet-20241022")
run_evaluation("mistral", "open-mixtral-8x22b")