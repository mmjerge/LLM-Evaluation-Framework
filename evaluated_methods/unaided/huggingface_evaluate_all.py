import datasets
import json
import random
import re
from tqdm import tqdm
import os
import time
import argparse
from openai import OpenAI
from huggingface_hub import InferenceClient
import pathlib
from collections import Counter

def get_api_tokens():
    """
    Get API tokens from environment variables.
    Returns a dictionary with keys for each service.
    """
    tokens = {
        "huggingface": os.getenv("HUGGINGFACE_API_KEY"),
    }
    
    missing_tokens = [service for service, token in tokens.items() if not token]
    if missing_tokens:
        print(f"WARNING: Missing API tokens for: {', '.join(missing_tokens)}")
        print("Please set the following environment variables:")
        for service in missing_tokens:
            print(f"  - {service.upper()}_API_KEY")
    
    return tokens

API_TOKENS = get_api_tokens()

vllm_client = OpenAI(
    api_key="EMPTY", 
    base_url=os.getenv("VLLM_API_BASE", "http://localhost:8000/v1"),  
)

def load_samples(dataset_name, num_samples=150):
    if dataset_name == "gsm8k":
        dataset = datasets.load_dataset('gsm8k', 'main', split='test')
    elif dataset_name == "gsm-symbolic":
        dataset = datasets.load_dataset('apple/GSM-Symbolic', 'main')
        dataset = dataset['test']
    elif dataset_name == "mmlu":
        dataset = datasets.load_dataset('cais/mmlu', 'all', split='test')
    elif dataset_name == "aqua":
        dataset = datasets.load_dataset('aqua_rat', 'raw', split='test')
    elif dataset_name == "svamp":
        dataset = datasets.load_dataset('ChilleD/SVAMP', split='test')
    elif dataset_name.startswith("legal-bench-"):
        # Extract the specific legal bench dataset name
        legal_dataset = dataset_name.replace("legal-bench-", "")
        dataset = datasets.load_dataset("nguha/legalbench", legal_dataset, split="test")
    elif dataset_name == "medqa":
        try:
            dataset = datasets.load_dataset("bigbio/med_qa", "med_qa_en_4options_bigbio_qa", split="test")
            print(f"Loaded MedQA with {len(dataset)} samples")
        except Exception as e:
            print(f"Error loading BigBIO MedQA: {e}")
            try:
                dataset = datasets.load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
                print(f"Loaded alternative MedQA with {len(dataset)} samples")
            except Exception as e2:
                print(f"Error loading alternative MedQA: {e2}")
                raise Exception("Could not load any MedQA dataset")
    
    total_samples = len(dataset)
    num_samples = min(num_samples, total_samples)
    random_indices = random.sample(range(total_samples), num_samples)
    samples = [dataset[i] for i in random_indices]
    return samples

def format_question(sample, dataset_name):
    if dataset_name == "gsm8k":
        return f"Solve this math problem:\n{sample['question']}. Please provide your final answer as a single number marked by 'Final answer: ####'"
    
    elif dataset_name == "gsm-symbolic":
        return f"Solve this math problem:\n{sample['question']}. Please provide your final answer as a single number marked by 'Final answer: ####'"
    
    elif dataset_name == "mmlu":
        return f"""Question: {sample['question']}
A) {sample['choices'][0]}
B) {sample['choices'][1]}
C) {sample['choices'][2]}
D) {sample['choices'][3]}

Please provide your answer as a single letter (A, B, C, or D) followed by a brief explanation. Please mark it as 'Final answer: #### [answer].'"""
    
    elif dataset_name == "aqua":
        options = sample['options']
        formatted_options = "\n".join([f"{i+1}) {opt}" for i, opt in enumerate(options)])
        return f"""Question: {sample['question']}
{formatted_options}

Please solve this question and provide your final answer as a single letter (A - E) at the end of your response. Please mark it as 'Final answer: #### [answer].'"""
    
    elif dataset_name == "svamp":
        return f"""Solve this math word problem and put your final numerical answer at the end of your response. Please mark your final answer as 'Final answer: #### [answer]'.
Question: {sample['Body']} {sample['Question']}"""
        
    elif dataset_name == "legal-bench-privacy_policy_qa":
        question_text = sample['question']
        policy_text = sample['text']
        return f"""Question: {question_text}

Privacy Policy Text: {policy_text}

Does this privacy policy text relate to or provide information about the question? Consider both direct answers and related information.

Final answer: ####"""
        
    elif dataset_name == "legal-bench-contract_qa":
        if 'query' in sample:
            question_text = sample['query']
        elif 'question' in sample:
            question_text = sample['question']
        else:
            question_text = str(sample)
            
        return f"""Contract Analysis Question: {question_text}

Please answer this contract-related question based on standard legal principles. If this is a yes/no question, answer with "Yes" or "No". If it requires a specific classification or short answer, provide that exact answer.

Final answer: ####"""
    
    elif dataset_name == "legal-bench-rule_qa":
        if 'query' in sample:
            question_text = sample['query']
        elif 'question' in sample:
            question_text = sample['question']
        else:
            question_text = str(sample)
            
        return f"""Legal Rule Question: {question_text}

Please answer this question about legal rules and principles. Provide a clear, direct answer. If this is a yes/no question, answer with "Yes" or "No". If it requires a classification, provide the exact classification.

Final answer: ####"""
        
    elif dataset_name.startswith("legal-bench-"):
        if 'query' in sample:
            question_text = sample['query']
        elif 'question' in sample:
            question_text = sample['question']
        else:
            print(f"Available keys in legal-bench sample: {sample.keys()}")
            question_text = str(sample)  
            
        return f"""Legal Question: {question_text}

Please provide a direct, concise answer to this legal question. If this is a yes/no question, answer with "Yes" or "No". Otherwise, provide the most appropriate short answer.

Final answer: ####"""
        
    elif dataset_name == "medqa":
        if 'question' in sample:
            question = sample['question']
        elif 'query' in sample:
            question = sample['query']
        else:
            print(f"MedQA sample keys: {list(sample.keys())}")
            question = str(sample)
        
        choices = None
        if 'choices' in sample:
            choices = sample['choices']
        elif 'options' in sample:
            choices = sample['options']
        elif 'answers' in sample:
            choices = sample['answers']
        
        if choices and len(choices) >= 4:
            formatted_options = "\n".join([f"{chr(65+i)}) {opt}" for i, opt in enumerate(choices[:4])])
            
            return f"""Medical Question: {question}
{formatted_options}

Please analyze this medical question and provide your final answer as a single letter (A, B, C, or D). Please mark it as 'Final answer: #### [answer].'"""
        else:
            print(f"Warning: Could not find proper choices for MedQA question. Available fields: {list(sample.keys())}")
            if choices:
                print(f"Choices found but unexpected format: {choices}")
            
            return f"""Medical Question: {question}

Please analyze this medical question and provide your final answer. Please mark it as 'Final answer: #### [answer].'"""

def ask_model(formatted_question, model_name, model_provider):
    """
    Query a model through OpenAI-compatible VLLM API or HuggingFace Inference API
    
    Args:
        formatted_question: The question to ask
        model_name: Name of the model to use
        model_provider: 'vllm' or 'huggingface'
    
    Returns:
        Model's response as text
    """
    try:
        if model_provider == 'vllm':
            if "llama" in model_name.lower():
                try:
                    print(f"Using chat completions API for {model_name}")
                    completion = vllm_client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": "You are a helpful AI assistant that provides accurate answers."},
                            {"role": "user", "content": formatted_question}
                        ],
                        max_tokens=800,
                        temperature=0.0 
                    )
                    return completion.choices[0].message.content
                except Exception as chat_err:
                    print(f"Chat API failed for VLLM: {chat_err}. Falling back to completions API.")
                    
            print(f"Using completions API for {model_name}")
            if "llama" in model_name.lower():
                prompt = f"<|system|>\nYou are a helpful AI assistant that provides accurate answers.\n<|user|>\n{formatted_question}\n<|assistant|>"
            else:
                prompt = f"System: You are a helpful AI assistant that provides accurate answers.\nUser: {formatted_question}\nAssistant:"
            
            try:
                completion = vllm_client.completions.create(
                    model=model_name,
                    prompt=prompt,
                    max_tokens=800,
                    temperature=0.0,
                    stop=["<|user|>", "<|system|>", "User:", "System:"]
                )
                return completion.choices[0].text
            except Exception as e:
                print(f"VLLM completions API error: {str(e)}")
                return f"Error with VLLM API: {str(e)}"
            
        elif model_provider == 'huggingface':
            if not API_TOKENS["huggingface"]:
                return "Error: HUGGINGFACE_API_KEY not set in environment variables"
            
            if not model_name.strip():
                return "Error: Empty model name provided for HuggingFace"
            
            try:
                client = InferenceClient(
                    model=model_name,
                    token=API_TOKENS["huggingface"]
                )
                
                try:
                    messages = [
                        {"role": "system", "content": "You are a helpful AI assistant that provides accurate answers."},
                        {"role": "user", "content": formatted_question}
                    ]
                    
                    response = client.chat_completion(
                        messages=messages,
                        max_tokens=800,
                        temperature=0.01
                    )
                    
                    print(f"Response from HuggingFace (chat): {response.choices[0].message.content[:500]}...")
                    return response.choices[0].message.content
                    
                except Exception as chat_err:
                    print(f"Chat completion failed: {chat_err}. Trying text_generation...")
                    if "llama" in model_name.lower():
                        prompt = f"<|system|>\nYou are a helpful AI assistant that provides accurate answers.\n<|user|>\n{formatted_question}\n<|assistant|>"
                    else:
                        prompt = f"System: You are a helpful AI assistant that provides accurate answers.\nUser: {formatted_question}\nAssistant:"
                    
                    response = client.text_generation(
                        prompt,
                        max_new_tokens=800,
                        temperature=0.01,
                        return_full_text=False
                    )
                    
                    print(f"Response from HuggingFace (text): {response[:500]}...")
                    return response
                
            except Exception as hf_error:
                print(f"HuggingFace API error: {hf_error}")
                return f"Error: HuggingFace API returned an error - {str(hf_error)}"
        
        else:
            raise ValueError(f"Unknown model provider: {model_provider}. Supported: 'vllm', 'huggingface'")
            
    except Exception as e:
        print(f"Error with {model_provider} API for model {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}"

def process_ground_truth(sample, dataset_name):
    if dataset_name == "mmlu":
        num_to_letter = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}
        return num_to_letter.get(sample['answer'], str(sample['answer']))
    elif dataset_name == "aqua":
        return str(sample['correct'])
    elif dataset_name == 'svamp':
        return str(sample['Answer'])
    elif dataset_name == "gsm8k":
        answer_text = sample['answer']
        match = re.search(r'####\s*(\d+(?:\.\d+)?)', answer_text)
        if match:
            return match.group(1)
        else:
            numbers = re.findall(r'\d+(?:\.\d+)?', answer_text)
            return numbers[-1] if numbers else str(sample['answer'])
    elif dataset_name == "gsm-symbolic":
        answer_text = sample['answer']
        match = re.search(r'####\s*(\d+(?:\.\d+)?)', answer_text)
        if match:
            return match.group(1)
        else:
            numbers = re.findall(r'\d+(?:\.\d+)?', answer_text)
            return numbers[-1] if numbers else str(sample['answer'])
    elif dataset_name == "legal-bench-privacy_policy_qa":
        answer = sample['answer'].strip()
        if answer.lower() in ['relevant', 'yes', 'true', '1']:
            return "Relevant"
        elif answer.lower() in ['irrelevant', 'no', 'false', '0']:
            return "Irrelevant" 
        else:
            if answer.lower() in ['relevant', 'irrelevant']:
                return answer.capitalize()
            else:
                print(f"Unexpected privacy_policy_qa answer format: {answer}")
                return answer
    elif dataset_name.startswith("legal-bench-"):
        if 'reference' in sample:
            answer = sample['reference']
        elif 'answer' in sample:
            answer = sample['answer']
        else:
            print(f"Available keys for ground truth in legal-bench: {sample.keys()}")
            return "unknown"
        
        answer = str(answer).strip()
        
        if answer.lower() in ['yes', 'true', '1', 'correct', 'valid']:
            return "Yes"
        elif answer.lower() in ['no', 'false', '0', 'incorrect', 'invalid']:
            return "No"
        else:
            return answer
    elif dataset_name == "medqa":
        
        print(f"\n=== DEBUG: Processing MedQA ground truth ===")
        print(f"Sample keys: {list(sample.keys())}")
        
        answer_found = False
        correct_answer = None
        
        if 'answer' in sample:
            answer_field = sample['answer']
            print(f"Found 'answer' field: {answer_field} (type: {type(answer_field)})")
            
            if isinstance(answer_field, list) and len(answer_field) > 0:
                answer_text = str(answer_field[0]).strip()
                print(f"Answer text from list: '{answer_text}'")
                
                if answer_text.upper() in ['A', 'B', 'C', 'D']:
                    correct_answer = answer_text.upper()
                    answer_found = True
                    print(f"Direct letter answer found: {correct_answer}")
                else:
                    if 'choices' in sample:
                        choices = sample['choices']
                        print(f"Matching against choices: {choices}")
                        for i, choice in enumerate(choices):
                            if str(choice).strip().lower() == answer_text.lower():
                                correct_answer = chr(65 + i)  
                                answer_found = True
                                print(f"Matched choice {i} -> {correct_answer}")
                                break
            elif isinstance(answer_field, str):
                if answer_field.upper() in ['A', 'B', 'C', 'D']:
                    correct_answer = answer_field.upper()
                    answer_found = True
                    print(f"Direct string answer: {correct_answer}")
        
        if not answer_found:
            index_fields = ['answer_idx', 'correct_answer', 'answer_index', 'correct_answer_idx']
            for field in index_fields:
                if field in sample:
                    idx_value = sample[field]
                    print(f"Found index field '{field}': {idx_value}")
                    
                    try:
                        if isinstance(idx_value, list):
                            idx = int(idx_value[0])
                        else:
                            idx = int(idx_value)
                        
                        if 0 <= idx <= 3:
                            correct_answer = chr(65 + idx)
                            answer_found = True
                            print(f"Converted index {idx} -> {correct_answer}")
                            break
                    except (ValueError, TypeError, IndexError):
                        print(f"Could not convert {idx_value} to valid index")
        
        if not answer_found and 'options' in sample:
            options = sample['options']
            if isinstance(options, list):
                for i, option in enumerate(options):
                    if isinstance(option, dict):
                        if option.get('correct') == True or option.get('is_correct') == True:
                            correct_answer = chr(65 + i)
                            answer_found = True
                            print(f"Found correct option at index {i} -> {correct_answer}")
                            break
        
        if answer_found:
            print(f"Final answer: {correct_answer}")
            return correct_answer
        else:
            print(f"ERROR: Could not determine correct answer for MedQA sample!")
            print(f"Full sample: {json.dumps(sample, indent=2, default=str)}")
            raise ValueError(f"Could not determine correct answer for MedQA sample. Available keys: {list(sample.keys())}")
    
    else:
        return str(sample['answer'])

def extract_model_answer(model_response, dataset_name):
    """
    Extract the model's answer from its response based on dataset type
    
    Args:
        model_response: The response from the model
        dataset_name: Name of the dataset
        
    Returns:
        Extracted answer (letter or number) or None if not found
    """
    if not isinstance(model_response, str):
        return None
        
    if "|<|reserved_special_token" in model_response:
        model_response = model_response.split("|<|reserved_special_token")[0]
    elif "<|system|>" in model_response:
        model_response = model_response.split("<|system|>")[0]
    elif "<|user|>" in model_response:
        model_response = model_response.split("<|user|>")[0]
    
    if "<|assistant|>" in model_response:
        parts = model_response.split("<|assistant|>")
        if len(parts) > 1:
            model_response = parts[1]  
    
    if dataset_name in ["mmlu", "medqa", "aqua"]:
        final_answer_patterns = [
            r"Final answer:\s*####\s*([A-E])",          # Final answer: #### A
            r"Final answer:\s*([A-E])[.,)]*\s*$",       # Final answer: A) or Final answer: A.
            r"Final answer:\s*([A-E])[.,)]*",           # Final answer: A anywhere
        ]
        
        for pattern in final_answer_patterns:
            match = re.search(pattern, model_response, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        
        hash_patterns = [
            r"####\s*([A-E])(?:\s|$|[.,)])",
        ]
        
        for pattern in hash_patterns:
            match = re.search(pattern, model_response)
            if match:
                return match.group(1).upper()
        
        other_patterns = [
            r"[Tt]he answer is\s*([A-E])[.,)]",
            r"[Tt]he correct answer is\s*([A-E])[.,)]",
            r"Answer:\s*([A-E])[.,)]?",
        ]
        
        for pattern in other_patterns:
            match = re.search(pattern, model_response)
            if match:
                return match.group(1).upper()
        
        last_line = model_response.strip().split('\n')[-1].strip()
        if last_line in ["A", "B", "C", "D", "E"]:
            return last_line.upper()
            
        last_letter_match = re.search(r"([A-E])[.,)]*\s*$", last_line)
        if last_letter_match:
            return last_letter_match.group(1).upper()
    
    elif dataset_name in ["gsm8k", "gsm-symbolic", "svamp"]:
        final_answer_match = re.search(r"Final answer:\s*#{0,4}\s*([-+]?\d*\.?\d+)", model_response, re.IGNORECASE)
        if final_answer_match:
            return final_answer_match.group(1)
            
        answer_is_match = re.search(r"[Tt]he answer is\s*([-+]?\d*\.?\d+)", model_response)
        if answer_is_match:
            return answer_is_match.group(1)
            
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", model_response)
        if numbers:
            return numbers[-1]
    
    return None

def collect_responses(models, dataset_name, num_samples):
    try:
        samples = load_samples(dataset_name, num_samples)
        
        if len(samples) > 0:
            print(f"\n=== Dataset: {dataset_name} ===")
            print(f"Total samples loaded: {len(samples)}")
            print(f"Sample structure:")
            print(f"Keys: {list(samples[0].keys())}")
            
            if dataset_name == "medqa":
                print(f"\n=== MedQA Validation ===")
                
                ground_truths = []
                failed_samples = 0
                
                for i, sample in enumerate(samples[:10]):
                    try:
                        gt = process_ground_truth(sample, dataset_name)
                        ground_truths.append(gt)
                        print(f"Sample {i+1} ground truth: {gt}")
                    except Exception as e:
                        print(f"Failed to process sample {i+1}: {e}")
                        failed_samples += 1
                
                if failed_samples > 0:
                    print(f"WARNING: {failed_samples}/10 samples failed ground truth processing!")
                    return {model_name: [] for model_name in models.keys()}  
                
                gt_distribution = Counter(ground_truths)
                print(f"Ground truth distribution (first 10): {gt_distribution}")
                
                if len(gt_distribution) < 3:
                    print("WARNING: Ground truth distribution seems heavily skewed!")
        
        results = {model_name: [] for model_name in models.keys()}
        
        for i, sample in enumerate(tqdm(samples, desc=f"Processing {dataset_name}")):
            try:
                formatted_question = format_question(sample, dataset_name)
                ground_truth = process_ground_truth(sample, dataset_name)
                
                for model_name, provider in models.items():
                    print(f"Querying {model_name} ({provider}) for {dataset_name} sample {i+1}/{len(samples)}...")
                    model_response = ask_model(formatted_question, model_name, provider)
                    
                    results[model_name].append({
                        'question': formatted_question,
                        'ground_truth': ground_truth,
                        'model_response': model_response
                    })
                    
                    time.sleep(1)
            except Exception as e:
                print(f"Error processing sample {i}: {str(e)}")
                if dataset_name == "medqa":
                    print(f"Aborting MedQA evaluation due to error")
                    return {model_name: [] for model_name in models.keys()}
                continue
        
        if dataset_name == "medqa":
            print(f"\n=== Final MedQA Validation ===")
            for model_name in models.keys():
                if results[model_name]:
                    all_ground_truths = [item['ground_truth'] for item in results[model_name]]
                    gt_dist = Counter(all_ground_truths)
                    print(f"Final ground truth distribution for {model_name}: {gt_dist}")
                    
                    for answer, count in gt_dist.items():
                        proportion = count / len(all_ground_truths)
                        if proportion > 0.6:  
                            print(f"WARNING: Answer '{answer}' appears in {proportion:.1%} of samples - this may indicate a problem!")
        
        for model_name in models.keys():
            model_short_name = model_name.replace(".", "-").replace("/", "-")
            filename = f'{dataset_name}_responses_{model_short_name}.json'
            
            with open(filename, 'w') as f:
                json.dump(results[model_name], f, indent=2)
        
        return results
    except Exception as e:
        print(f"Error in collect_responses for {dataset_name}: {str(e)}")
        return {model_name: [] for model_name in models.keys()}

def evaluate_accuracy(results, dataset_name):
    """Evaluate model performance based on dataset-specific criteria"""
    accuracies = {}
    
    for model, responses in results.items():
        if not responses:
            print(f"No responses for {model} on {dataset_name}")
            accuracies[model] = 0
            continue
            
        correct = 0
        total = len(responses)
        evaluation_details = []
        
        if dataset_name == "medqa":
            model_answers = []
            ground_truths = []
            incorrect_cases = []
        
        for i, item in enumerate(responses):
            ground_truth = item['ground_truth']
            model_response = item['model_response']
            
            if dataset_name == "gsm8k":
                is_correct = False
    
                gt_escaped = re.escape(ground_truth)
                
                cleaned_response = model_response.replace('$\\', '').replace('\\, '').replace(', '')
                cleaned_response = cleaned_response.replace('\\', '')
                
                if re.search(rf'final answer:\s*####\s*{gt_escaped}\b', cleaned_response, re.IGNORECASE):
                    is_correct = True
                
                elif re.search(rf'####\s*{gt_escaped}\b', cleaned_response):
                    is_correct = True
                
                elif re.search(rf'final answer:\s*{gt_escaped}\b', cleaned_response, re.IGNORECASE):
                    is_correct = True
                
                elif re.search(rf'final answer:\s*{gt_escaped}\.00?\b', cleaned_response, re.IGNORECASE):
                    is_correct = True
                elif re.search(rf'####\s*{gt_escaped}\.00?\b', cleaned_response):
                    is_correct = True
                
                elif re.search(rf'the answer is\s*{gt_escaped}\b', cleaned_response, re.IGNORECASE):
                    is_correct = True
                
                elif re.search(rf'answer:\s*{gt_escaped}\b', cleaned_response, re.IGNORECASE):
                    is_correct = True
                
                else:
                    lines = cleaned_response.strip().split('\n')
                    if lines and lines[-1].strip() == ground_truth:
                        is_correct = True
                
                if is_correct:
                    correct += 1
                    
            elif dataset_name == "gsm-symbolic":
                is_correct = False
    
                gt_escaped = re.escape(ground_truth)
                
                cleaned_response = model_response.replace('$\\', '').replace('\\, '').replace(', '')
                cleaned_response = cleaned_response.replace('\\', '')
                
                if re.search(rf'final answer:\s*####\s*{gt_escaped}\b', cleaned_response, re.IGNORECASE):
                    is_correct = True
                
                elif re.search(rf'####\s*{gt_escaped}\b', cleaned_response):
                    is_correct = True
                
                elif re.search(rf'final answer:\s*{gt_escaped}\b', cleaned_response, re.IGNORECASE):
                    is_correct = True
                
                elif re.search(rf'final answer:\s*{gt_escaped}\.00?\b', cleaned_response, re.IGNORECASE):
                    is_correct = True
                elif re.search(rf'####\s*{gt_escaped}\.00?\b', cleaned_response):
                    is_correct = True
                
                elif re.search(rf'the answer is\s*{gt_escaped}\b', cleaned_response, re.IGNORECASE):
                    is_correct = True
                
                elif re.search(rf'answer:\s*{gt_escaped}\b', cleaned_response, re.IGNORECASE):
                    is_correct = True
                
                else:
                    lines = cleaned_response.strip().split('\n')
                    if lines and lines[-1].strip() == ground_truth:
                        is_correct = True
                
                if is_correct:
                    correct += 1
                    
            elif dataset_name == "svamp":
                is_correct = False
                
                gt_escaped = re.escape(ground_truth)
                
                if re.search(rf'FINAL ANSWER:\s*####\s*{gt_escaped}\b', model_response.upper()):
                    is_correct = True
                
                elif re.search(rf'####\s*{gt_escaped}\b', model_response):
                    is_correct = True
                
                elif re.search(rf'FINAL ANSWER:\s*{gt_escaped}\b', model_response.upper()):
                    is_correct = True
                
                elif re.search(rf'THE ANSWER IS\s*{gt_escaped}(?:\s|$|\.)', model_response.upper()):
                    is_correct = True
                
                elif re.search(rf'^ANSWER:\s*{gt_escaped}(?:\s|$|\.)' , model_response.upper(), re.MULTILINE):
                    is_correct = True
                
                if is_correct:
                    correct += 1
                    
            elif dataset_name in ["mmlu", "medqa"]:
                ground_truth = ground_truth.upper()
                model_response_upper = model_response.upper()
                
                is_correct = False
                extracted_answer = None
                
                if f"FINAL ANSWER: #### {ground_truth}" in model_response_upper:
                    is_correct = True
                    extracted_answer = ground_truth
                elif f"#### {ground_truth}" in model_response_upper:
                    is_correct = True
                    extracted_answer = ground_truth
                else:
                    patterns = [
                        r'FINAL ANSWER:?\s*####?\s*([ABCD])',
                        r'####\s*([ABCD])',
                        r'FINAL ANSWER:?\s*([ABCD])',
                        r'ANSWER:?\s*([ABCD])',
                        r'(?:^|\n)\s*([ABCD])\s*(?:\.|$)', 
                        r'\b([ABCD])\)\s*(?:[A-Z]|$)',  
                    ]
                    
                    for pattern in patterns:
                        match = re.search(pattern, model_response_upper)
                        if match:
                            extracted_answer = match.group(1)
                            if extracted_answer == ground_truth:
                                is_correct = True
                            break
                
                if dataset_name == "medqa":
                    ground_truths.append(ground_truth)
                    model_answers.append(extracted_answer)
                    
                    if not is_correct:
                        incorrect_cases.append({
                            'question_num': i + 1,
                            'ground_truth': ground_truth,
                            'extracted': extracted_answer,
                            'response_snippet': model_response[:100] + "..."
                        })
                    
                    evaluation_details.append({
                        "question_index": i,
                        "ground_truth": ground_truth,
                        "extracted_answer": extracted_answer,
                        "is_correct": is_correct,
                        "response_snippet": model_response[:100] + "..." if len(model_response) > 100 else model_response
                    })
                
                if is_correct:
                    correct += 1
                    
            elif dataset_name == "legal-bench-privacy_policy_qa":
                ground_truth_clean = ground_truth.strip()
                model_response_upper = model_response.upper()
                ground_truth_upper = ground_truth_clean.upper()
                
                if f"FINAL ANSWER: #### {ground_truth_upper}" in model_response_upper:
                    correct += 1
                elif f"#### {ground_truth_upper}" in model_response_upper:
                    correct += 1
                elif f"FINAL ANSWER: {ground_truth_upper}" in model_response_upper:
                    correct += 1
                elif model_response_upper.strip().startswith(ground_truth_upper):
                    correct += 1
                elif model_response_upper.strip().endswith(ground_truth_upper):
                    correct += 1
                elif any(pattern in model_response_upper for pattern in [
                    f"ANSWER: {ground_truth_upper}",
                    f"RESPONSE: {ground_truth_upper}",
                    f"{ground_truth_upper}."
                ]):
                    correct += 1
                elif ground_truth_upper == "IRRELEVANT":
                    if any(pattern in model_response_upper for pattern in [
                        "NO,", "NO.", "NO ", "DOES NOT", "IS NOT", "NOT RELEVANT"
                    ]):
                        correct += 1
                elif ground_truth_upper == "RELEVANT":
                    if any(pattern in model_response_upper for pattern in [
                        "YES,", "YES.", "YES ", "DOES ADDRESS", "IS RELEVANT", "ADDRESSES"
                    ]):
                        correct += 1
                        
            elif dataset_name.startswith("legal-bench-"):
                ground_truth_clean = ground_truth.strip()
                model_response_upper = model_response.upper()
                ground_truth_upper = ground_truth_clean.upper()
                
                if f"FINAL ANSWER: #### {ground_truth_upper}" in model_response_upper:
                    correct += 1
                elif f"#### {ground_truth_upper}" in model_response_upper:
                    correct += 1
                elif f"FINAL ANSWER: {ground_truth_upper}" in model_response_upper:
                    correct += 1
                elif ground_truth_upper in ["YES", "NO", "RELEVANT", "IRRELEVANT"]:
                    pattern = r'\b' + re.escape(ground_truth_upper) + r'\b'
                    if re.search(pattern, model_response_upper):
                        if ("FINAL ANSWER" in model_response_upper and 
                            ground_truth_upper in model_response_upper[model_response_upper.find("FINAL ANSWER"):]):
                            correct += 1
                        elif model_response_upper.strip().endswith(ground_truth_upper):
                            correct += 1
                        elif (model_response_upper.count(ground_truth_upper) == 1):
                            correct += 1
                else:
                    sentences = model_response.split('.')
                    if len(sentences) >= 2:
                        last_sentences = '. '.join(sentences[-2:]).upper()
                        if ground_truth_upper in last_sentences:
                            correct += 1
                    else:
                        if ground_truth_upper in model_response_upper:
                            correct += 1
                            
            elif dataset_name == "aqua":
                if f"Final answer: {ground_truth}" in model_response or f"#### {ground_truth}" in model_response:
                    correct += 1
                letter_to_num = {'A': '1', 'B': '2', 'C': '3', 'D': '4', 'E': '5'}
                for letter, num in letter_to_num.items():
                    if ground_truth == num and (f"Final answer: {letter}" in model_response or f"#### {letter}" in model_response):
                        correct += 1
                        break
        
        if dataset_name == "medqa":
            print(f"\n=== {model} MedQA Results ===")
            print(f"Accuracy: {correct}/{total} = {correct/total:.3f} ({correct/total:.1%})")
            
            gt_dist = Counter(ground_truths)
            model_ans_dist = Counter([x for x in model_answers if x is not None])
            print(f"Ground truth distribution: {dict(gt_dist)}")
            print(f"Model answer distribution: {dict(model_ans_dist)}")
            print(f"No answer extracted: {model_answers.count(None)} questions")
            
            if incorrect_cases:
                print(f"\nFirst few incorrect cases:")
                for case in incorrect_cases[:3]:
                    print(f"  Q{case['question_num']}: GT={case['ground_truth']}, Extracted={case['extracted']}, Response: {case['response_snippet']}")
        
        if dataset_name == "medqa":
            model_short_name = model.replace(".", "-").replace("/", "-")
            with open(f'{dataset_name}_{model_short_name}_detailed_evaluation.json', 'w') as f:
                json.dump(evaluation_details, f, indent=2)
        
        accuracies[model] = correct / total if total > 0 else 0
        
    return accuracies

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate open source models via VLLM and HuggingFace on various benchmarks')
    
    parser.add_argument('--vllm-models', nargs='+', default=[],
                        help='Open source models to evaluate via VLLM (default: none)')
                        
    parser.add_argument('--huggingface-models', nargs='+', default=[],
                        help='Models to evaluate via HuggingFace Inference API (default: none)')
    
    parser.add_argument('--vllm-endpoint', type=str, default="http://localhost:8000/v1",
                        help='VLLM API endpoint (default: http://localhost:8000/v1)')
    
    parser.add_argument('--datasets', nargs='+', 
                       default=[
                           "gsm8k", 
                           "mmlu", 
                           "aqua",
                           "svamp",
                           "legal-bench-contract_qa",
                           "legal-bench-rule_qa",
                           "medqa"
                       ],
                       help='Datasets to evaluate on')
    
    parser.add_argument('--samples', type=int, default=150,
                        help='Number of samples to use per dataset (default: 150)')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not args.vllm_models and not args.huggingface_models:
        print("\nERROR: You must specify at least one model to evaluate.")
        print("Use --vllm-models or --huggingface-models")
        print("Example: python script.py --vllm-models meta-llama/Meta-Llama-3.1-8B-Instruct")
        exit(1)
    
    if args.huggingface_models and not API_TOKENS["huggingface"]:
        print("\nERROR: You've requested to use HuggingFace models, but HUGGINGFACE_API_KEY is not set.")
        print("Please set this environment variable before running the script.")
        print("Example: export HUGGINGFACE_API_KEY='hf_your-api-key'\n")
        exit(1)
    
    global vllm_client
    vllm_client = OpenAI(
        api_key="EMPTY",
        base_url=args.vllm_endpoint,
    )
    
    models = {}
    for model in args.vllm_models:
        models[model] = 'vllm'
        
    for model in args.huggingface_models:
        if model.strip(): 
            models[model] = 'huggingface'
    
    if not models:
        print("\nERROR: No valid models specified. Please provide model names.")
        exit(1)
    
    print(f"Evaluating {len(models)} model(s): {list(models.keys())}")
    
    all_results = {
        "accuracy": {},
        "details": {}
    }
    
    for dataset_name in args.datasets:
        print(f"\n{'='*50}")
        print(f"Evaluating on {dataset_name}")
        print(f"{'='*50}")
        
        try:
            results = collect_responses(models, dataset_name, args.samples)
            all_results["details"][dataset_name] = results
            
            dataset_accuracies = evaluate_accuracy(results, dataset_name)
            all_results["accuracy"][dataset_name] = dataset_accuracies
            
            print(f"\n=== {dataset_name} Results ===")
            for model, accuracy in dataset_accuracies.items():
                print(f"{model}: {accuracy:.3f} ({accuracy:.1%})")
        except Exception as e:
            print(f"Error evaluating {dataset_name}: {str(e)}")
            all_results["accuracy"][dataset_name] = {model: 0 for model in models.keys()}
    
    model_averages = {}
    for model in models.keys():
        accuracies = [all_results["accuracy"][dataset][model] for dataset in args.datasets 
                     if dataset in all_results["accuracy"] and model in all_results["accuracy"][dataset]]
        model_averages[model] = sum(accuracies) / len(accuracies) if accuracies else 0
    
    all_results["average_accuracy"] = model_averages
    
    print(f"\n{'='*50}")
    print("Overall Averages")
    print(f"{'='*50}")
    for model, avg_accuracy in model_averages.items():
        provider = models[model]
        print(f"{model} ({provider}): {avg_accuracy:.3f} ({avg_accuracy:.1%})")
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_file = f'opensource_model_evaluation_results_{timestamp}.json'
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nEvaluation complete! Results saved to {output_file}")

if __name__ == "__main__":
    print(f"Running script: {os.path.abspath(__file__)}")
    main()