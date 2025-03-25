import random
from datasets import load_dataset, concatenate_datasets
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI
from anthropic import Anthropic
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import torch
import pandas as pd
import time
from typing import List, Dict, Literal, Tuple, Any
from tqdm import tqdm
import re
import json
from datetime import datetime
import os
import sys
import pathlib
import argparse

ProviderType = Literal["openai", "anthropic", "mistral", "huggingface"]

def ensure_directory_exists(directory: str) -> str:
    """
    Ensures the specified directory exists, creates it if it doesn't.
    Returns the absolute path to the directory.
    """
    # Explicitly use a local directory path
    # This is a direct approach to ensure we're not using any cached paths
    local_dir = os.path.join(".", directory)
    
    # Create the directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)
    
    # Get the absolute path for reporting
    abs_path = os.path.abspath(local_dir)
    print(f"Using directory: {abs_path}")
    
    return local_dir  # Return the local path, not the absolute path

def get_few_shot_examples() -> str:
    """
    Returns few-shot examples for legal contract question answering.
    """
    return """Example 1:
Contract: 
This Agreement is entered into as of the 15th day of April, 2023, by and between ABC Corp. ("Company") and XYZ Inc. ("Contractor").

1. SERVICES: Contractor shall provide the following services to the Company: software development, testing, and deployment of a web application as described in Exhibit A.

2. COMPENSATION: Company shall pay Contractor $150 per hour, not to exceed $50,000 in total, for services rendered. Contractor shall submit invoices on a bi-weekly basis, and Company shall pay such invoices within 30 days of receipt.

3. TERM: This Agreement shall commence on April 20, 2023, and shall continue until December 31, 2023, unless earlier terminated as provided herein.

4. CONFIDENTIALITY: Contractor acknowledges that during the engagement, Contractor will have access to confidential information. Contractor agrees to maintain the confidentiality of all such information and not to disclose it to any third party without Company's prior written consent.

5. INTELLECTUAL PROPERTY: All work product created by Contractor in the course of providing services shall be the sole property of Company. Contractor hereby assigns all rights, title, and interest in such work product to Company.

Question: When does the contract term end?

Let's think step by step:
1. I need to find the clause discussing the term or duration of the contract.
2. Section 3 is titled "TERM", which is exactly what I'm looking for.
3. According to Section 3, "This Agreement shall commence on April 20, 2023, and shall continue until December 31, 2023, unless earlier terminated as provided herein."
4. The end date of the contract term is explicitly stated as December 31, 2023.

Answer: December 31, 2023

Example 2:
Privacy Policy Section:
We collect information about your device and internet connection, including the device's unique device identifier, IP address, operating system, browser type, mobile network information, and device's telephone number. We may also collect information regarding your use of the service, such as the games you play, your game scores, and your interactions with other users.

Question: Does the app collect information about my device?

Let's think step by step:
1. I need to determine if this privacy policy section mentions collecting device information.
2. The section explicitly states "We collect information about your device and internet connection, including the device's unique device identifier, IP address, operating system, browser type..."
3. This clearly indicates that they collect device information.
4. The question is asking about device information collection, which is directly addressed in this section.

Answer: Relevant

Example 3:
Privacy Policy Section:
You can access and update certain information about yourself from within the app settings. You may also send us an email to request access to, correct, or delete any personal information you have provided to us.

Question: Can I delete my account data?

Let's think step by step:
1. I need to determine if this privacy policy section addresses account data deletion.
2. The section mentions "You may also send us an email to request access to, correct, or delete any personal information you have provided to us."
3. This indicates users can request deletion of personal information.
4. The question asks about deleting account data, which falls under personal information.
5. This section is relevant to the question of account data deletion.

Answer: Relevant

Example 4:
Privacy Policy Section:
Our headquarters are located at 123 Main Street, Anytown, USA. You can contact our support team at support@example.com or by calling 1-800-555-1234 during business hours.

Question: How long do you keep my browsing history?

Let's think step by step:
1. I need to determine if this policy section addresses data retention periods for browsing history.
2. The section only provides contact information and headquarters location.
3. There is no mention of browsing history, data retention periods, or how long any user data is kept.
4. This section does not contain information relevant to the question about browsing history retention.

Answer: Irrelevant

Now solve this new problem:
"""

def create_legal_qa_prompt(contract: str, question: str) -> str:
    """
    Creates a prompt for legal contract question answering.
    Customizes the prompt based on the contract/question type.
    """
    # Check if this is likely a privacy policy relevance question
    privacy_keywords = ["privacy", "data", "information", "collect", "share", "access", "third party", 
                       "third-party", "personal", "track", "cookie", "location", "device", "camera",
                       "microphone", "profile", "account", "delete", "retain", "store", "consent"]
    
    # Determine if this is a privacy policy question based on keywords
    is_privacy_question = any(keyword in question.lower() for keyword in privacy_keywords)
    
    if is_privacy_question:
        return f"{get_few_shot_examples()}\nPrivacy Policy Section:\n{contract}\n\nQuestion: {question}\n\nLet's think step by step:"
    else:
        # Use the standard contract prompt
        return f"{get_few_shot_examples()}\nContract:\n{contract}\n\nQuestion: {question}\n\nLet's think step by step:"

def extract_final_answer(response: str, expected_answer: str = None) -> str:
    """
    Extracts the final answer from the model's response for LegalBench tasks.
    Handles different answer formats based on the expected answer type.
    """
    # Convert response to lowercase for case-insensitive matching
    response_lower = response.lower()
    
    # Check for "Answer:" pattern, which is the most common
    if "answer:" in response_lower:
        final_answer_line = response.split("Answer:", 1)[-1].strip()
        
        # If the expected answer is Relevant/Irrelevant, normalize the extracted answer
        if expected_answer in ["Relevant", "Irrelevant"]:
            if "relevant" in final_answer_line.lower() and "irrelevant" not in final_answer_line.lower():
                return "Relevant"
            elif "irrelevant" in final_answer_line.lower():
                return "Irrelevant"
        
        return final_answer_line
    
    # Special handling for relevance questions
    if expected_answer in ["Relevant", "Irrelevant"]:
        # Check for explicit relevance statements in the response
        if "is relevant" in response_lower or "section is relevant" in response_lower:
            return "Relevant"
        if "is irrelevant" in response_lower or "section is irrelevant" in response_lower or "not relevant" in response_lower:
            return "Irrelevant"
        
        # Try to extract relevance from the last sentence
        sentences = re.findall(r'([^.!?]+[.!?])(?:\s|$)', response)
        if sentences:
            last_sentence = sentences[-1].lower()
            if "relevant" in last_sentence and "not relevant" not in last_sentence and "irrelevant" not in last_sentence:
                return "Relevant"
            elif "not relevant" in last_sentence or "irrelevant" in last_sentence:
                return "Irrelevant"
    
    # Otherwise try to extract the answer from the last few lines
    lines = [line.strip() for line in response.split('\n') if line.strip()]
    if lines:
        # Take the last non-empty line as the answer
        last_line = lines[-1].lower()
        
        # For relevance questions, check the last line
        if expected_answer in ["Relevant", "Irrelevant"]:
            if "relevant" in last_line and "not relevant" not in last_line and "irrelevant" not in last_line:
                return "Relevant"
            elif "not relevant" in last_line or "irrelevant" in last_line:
                return "Irrelevant"
        
        return lines[-1]
    
    return ""

def evaluate_response(predicted: str, actual: str) -> bool:
    """
    Evaluates if the predicted answer matches the actual answer for legal QA.
    Handles different answer formats, including "Relevant"/"Irrelevant".
    """
    if not predicted or not actual:
        return False
    
    # Clean and normalize both answers
    pred_clean = predicted.lower().strip()
    actual_clean = actual.lower().strip()
    
    # Special case for "Relevant"/"Irrelevant" answers
    if actual.strip() in ["Relevant", "Irrelevant"]:
        if actual.strip() == "Relevant":
            return pred_clean == "relevant"
        else:  # Irrelevant
            return pred_clean == "irrelevant"
    
    # Perfect match
    if pred_clean == actual_clean:
        return True
    
    # Check if the actual answer is contained within the predicted answer
    if actual_clean in pred_clean:
        return True
    
    # Check for high similarity (optional, could use string similarity metrics)
    # This is a simplified approach; you might want to use more sophisticated text comparison
    common_words = set(pred_clean.split()) & set(actual_clean.split())
    if common_words and len(common_words) / len(set(actual_clean.split())) > 0.7:
        return True
    
    return False

def get_model_response(
    prompt: str,
    provider: ProviderType,
    model_name: str,
    temperature: float = 0.0,
    huggingface_pipeline = None
) -> str:
    """
    Gets response from specified model provider.
    """
    system_prompt = """You are a legal assistant specializing in contract and privacy policy analysis. 
    
For contract questions:
    1. Read the contract carefully
    2. Show your step-by-step thinking
    3. End with 'Answer: [your answer]'
    4. Be concise and base your answers solely on the information provided in the contract
    
For privacy policy questions:
    1. Read the privacy policy section carefully
    2. Determine if the section contains information relevant to the question
    3. Show your step-by-step thinking
    4. For relevance questions, end with 'Answer: Relevant' or 'Answer: Irrelevant'
    5. A section is RELEVANT if it contains information that directly or indirectly addresses the user's question
    6. A section is IRRELEVANT if it doesn't contain any information related to the user's question"""    

    if provider == "huggingface":
        if huggingface_pipeline is None:
            raise ValueError("HuggingFace pipeline must be provided for HuggingFace models")
        
        # Format prompt with system prompt for consistency
        formatted_prompt = f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:"
        
        # Generate response
        generation_config = {
            "max_new_tokens": 1000,
            "do_sample": True,
            "temperature": max(0.1, temperature),
            "top_p": 0.9,
            "num_return_sequences": 1,
            "repetition_penalty": 1.1  # Add this to prevent repetitive outputs
        }
        
        response = huggingface_pipeline(
            formatted_prompt,
            **generation_config
        )
        
        generated_text = response[0]['generated_text']
        assistant_response = generated_text.split("Assistant:")[-1].strip()
        return assistant_response
    
    if provider == "openai":
        client = OpenAI()
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature
        )
        return response.choices[0].message.content
    
    elif provider == "anthropic":
        client = Anthropic()
        response = client.messages.create(
            model=model_name,
            max_tokens=1000,
            temperature=temperature,
            system=system_prompt,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text
    
    elif provider == "mistral":
        client = MistralClient()
        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=prompt)
        ]
        response = client.chat(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=1000
        )
        return response.choices[0].message.content
    
    else:
        raise ValueError(f"Unsupported provider: {provider}")

def save_results_to_json(results: Dict, output_dir: str = "results") -> str:
    """
    Saves the test results to a JSON file with detailed comparisons.
    Returns the path to the saved file.
    """
    # Create local directory
    local_dir = ensure_directory_exists(output_dir)
    
    clean_model_name = re.sub(r'[^\w\-]', '_', results['model_name'])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{results['provider']}_{clean_model_name}_{timestamp}.json"
    
    # Use explicit local path
    output_file = os.path.join(".", local_dir, filename)
    
    json_results = {
        'metadata': {
            'model_name': results['model_name'],
            'provider': results['provider'],
            'accuracy': results['accuracy'],
            'total_samples': results['total_samples'],
            'correct_count': results['correct_count'],
            'timestamp': datetime.now().isoformat()
        },
        'questions': []
    }
    
    for _, row in results['results_df'].iterrows():
        question_result = {
            'question_id': int(row['question_id']),
            'question': row['question'],
            'contract': row['contract'][:500] + "..." if len(row['contract']) > 500 else row['contract'],
            'ground_truth': row['actual_answer'],
            'model_answer': row['predicted_answer'],
            'full_response': row['full_response'],
            'is_correct': bool(row['correct'])
        }
        json_results['questions'].append(question_result)
    
    print(f"Writing results to file: {os.path.abspath(output_file)}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed results saved to: {os.path.abspath(output_file)}")
    return output_file

def process_legalbench_format(sample: Dict[str, Any], debug_flag: bool = False) -> Tuple[str, str, str]:
    """
    Process the LegalBench dataset formats for contract_qa and privacy_policy_qa tasks.
    
    Args:
        sample: The dataset sample to process
        debug_flag: Whether to print debug information
        
    Returns:
        Tuple of (contract, question, answer)
    """
    # Debug: Print sample keys to understand structure (if needed)
    if debug_flag and hasattr(sample, 'keys'):
        print(f"Sample keys: {list(sample.keys())}")
    
    # Different formats for different tasks
    # Try to detect the dataset type based on the fields
    
    # Initialize with default values
    contract = ""
    question = ""
    answer = ""
    
    # Get all available keys
    keys = list(sample.keys()) if hasattr(sample, 'keys') else []
    
    # Process based on dataset type patterns
    
    # Case 1: contract_qa format (input, instruction, output format)
    if 'input' in keys and 'instruction' in keys and 'output' in keys:
        contract = sample.get('input', '')
        question = sample.get('instruction', '')
        answer = sample.get('output', '')
    
    # Case 2: privacy_policy_qa format (text with question and answer fields)
    elif 'question' in keys:
        # In privacy_policy_qa, sometimes the contract text is missing
        contract = sample.get('text', sample.get('segment', 'Privacy policy document'))
        question = sample.get('question', '')
        answer = sample.get('answer', sample.get('label', ''))
        
        # Special case for specific dataset formats we've observed
        if answer == "Relevant" or answer == "Irrelevant":
            # This is likely a privacy policy relevance question
            question = question.strip()
            answer = answer.strip()
    
    # Case 3: Fields specific to other tasks
    elif 'text' in keys and 'label' in keys:
        contract = sample.get('text', '')
        question = f"Based on the following text, determine the correct label: {contract}"
        answer = sample.get('label', '')
    
    # Default fallback for unknown formats
    else:
        # Try to extract fields with common names
        contract = sample.get('input', sample.get('text', sample.get('context', 'No contract text available')))
        question = sample.get('instruction', sample.get('question', 'What is the relevant information?'))
        answer = sample.get('output', sample.get('answer', sample.get('label', 'No answer available')))
    
    # Clean up and log
    contract = contract.strip() if isinstance(contract, str) else str(contract)
    question = question.strip() if isinstance(question, str) else str(question)
    answer = answer.strip() if isinstance(answer, str) else str(answer)
    
    if not contract:
        contract = "No contract text provided"
    
    if not question:
        question = "No question provided"
    
    if not answer:
        answer = "No answer provided"
    
    return contract, question, answer

def test_legalbench_contract_qa(
    num_samples: int = 150,
    provider: ProviderType = "openai",
    model_name: str = "gpt-4o",
    temperature: float = 0.0,
    output_dir: str = "results",
    huggingface_pipeline = None,
    debug: bool = True,
    tasks = None
) -> Dict:
    """
    Tests the specified model on random samples from LegalBench Contract QA dataset.
    """
    # Global debug flag to control verbosity
    debug_mode = debug
    
    # A list of LegalBench tasks that are QA-like in nature (default list)
    default_qa_tasks = [
        "contract_qa",                  # Contract QA
        "privacy_policy_qa",            # Privacy policy QA
    ]
    
    # Use custom tasks if provided, otherwise use default list
    qa_tasks = tasks if tasks else default_qa_tasks
    print(f"Using tasks: {qa_tasks}")
    
    # Load all datasets and keep track of them
    all_datasets = []
    for task in qa_tasks:
        try:
            print(f"Loading {task} dataset...")
            dataset = load_dataset("nguha/legalbench", task, trust_remote_code=True)
            
            # Use the test split (contains more examples) instead of train
            if 'test' in dataset:
                task_data = dataset['test']
                print(f"  - Added {len(task_data)} examples from {task} (test split)")
                all_datasets.append(task_data)
            elif 'train' in dataset:
                task_data = dataset['train']
                print(f"  - Added {len(task_data)} examples from {task} (train split)")
                all_datasets.append(task_data)
            else:
                print(f"  - No usable split found in {task}")
                
        except Exception as e:
            print(f"Error loading {task}: {str(e)}")
    
    # Concatenate all the datasets
    print("Concatenating all datasets...")
    ds = concatenate_datasets(all_datasets)
    
    print(f"Combined dataset size: {len(ds)} examples")
    
    print(f"Combined dataset size: {len(ds)} examples")
    
    if provider == "huggingface" and huggingface_pipeline is None:
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
        
        huggingface_pipeline = pipeline(
            "text-generation",
            model=model_name,
            tokenizer=tokenizer,
            torch_dtype=torch.float16,
            device_map="auto",
            max_new_tokens=1000,
            do_sample=False,
            temperature=0.1,
            repetition_penalty=1.2, 
            early_stopping=True
        )

    # Limit number of samples based on available data and requested sample size
    num_available = len(ds)
    actual_num_samples = min(num_samples, num_available)
    
    if actual_num_samples < num_samples:
        print(f"Warning: Requested {num_samples} samples, but only {actual_num_samples} available in the combined dataset.")
    
    # Look at some sample examples to understand structure
    if debug and num_available > 0:
        print("\n=== DEBUG: Sample Structure Examples ===")
        for i in range(min(3, num_available)):
            print(f"\nSample {i}:")
            if hasattr(ds[i], 'keys'):
                print(f"Keys: {list(ds[i].keys())}")
                for key in ds[i].keys():
                    preview = str(ds[i][key])
                    print(f"{key}: {preview[:100]}{'...' if len(preview) > 100 else ''}")
            else:
                print(f"Type: {type(ds[i])}")
    
    # Select random samples with safety check
    if num_available == 0:
        raise ValueError("No samples available in the combined dataset.")
        
    # Select random samples
    test_indices = random.sample(range(num_available), actual_num_samples)
    test_samples = [ds[i] for i in test_indices]
    
    results = []
    correct = 0
    
    for i, sample in tqdm(enumerate(test_samples), total=actual_num_samples, desc=f"Testing {model_name}"):
        contract, question, actual_answer = process_legalbench_format(sample, debug)
        
        # Skip samples with missing data
        if not contract or not question:
            print(f"Skipping sample {i} due to missing contract or question")
            continue
            
        prompt = create_legal_qa_prompt(contract, question)
        
        # Print the first prompt for debugging
        if i == 0 and debug:
            print("\n=== DEBUG: First Sample Prompt ===")
            preview_length = min(500, len(prompt))
            print(prompt[:preview_length] + ("..." if len(prompt) > preview_length else ""))
        
        try:
            response = get_model_response(
                prompt,
                provider,
                model_name,
                temperature,
                huggingface_pipeline
            )
            
            predicted_answer = extract_final_answer(response, actual_answer)
            
            is_correct = evaluate_response(predicted_answer, actual_answer)
            if is_correct:
                correct += 1
                
            results.append({
                'question_id': i,
                'contract': contract,
                'question': question,
                'actual_answer': actual_answer,
                'predicted_answer': predicted_answer,
                'full_response': response,
                'correct': is_correct
            })
            
            time.sleep(1)
            
        except Exception as e:
            print(f"Error processing question {i}: {str(e)}")
            continue
    
    accuracy = correct / actual_num_samples
    
    results_df = pd.DataFrame(results)
    
    results_dict = {
        'accuracy': accuracy,
        'total_samples': actual_num_samples,
        'correct_count': correct,
        'results_df': results_df,
        'model_name': model_name,
        'provider': provider
    }
    
    save_results_to_json(results_dict, output_dir)
    
    return results_dict

def analyze_results(results: Dict) -> None:
    """
    Analyzes and prints the test results.
    """
    print(f"\nTest Results for {results['provider']} - {results['model_name']}:")
    print(f"Accuracy: {results['accuracy']*100:.2f}%")
    print(f"Correct: {results['correct_count']}/{results['total_samples']}")
    
    # Print examples of both correct and incorrect answers
    print("\n=== SAMPLE RESULTS ===")
    
    # Show first 3 samples regardless of correctness to see what's happening
    samples_to_show = results['results_df'].head(3)
    for i, (_, row) in enumerate(samples_to_show.iterrows()):
        correct_str = "CORRECT" if row['correct'] else "INCORRECT"
        print(f"\n--- Sample {i+1} ({correct_str}) ---")
        print(f"Question: '{row['question']}'")
        print(f"Expected answer: '{row['actual_answer']}'")
        print(f"Predicted answer: '{row['predicted_answer']}'")
        contract_preview = row['contract'][:100] + "..." if len(row['contract']) > 100 else row['contract']
        print(f"Contract (preview): '{contract_preview}'")
        response_preview = row['full_response'][:100] + "..." if len(row['full_response']) > 100 else row['full_response']
        print(f"Response (preview): '{response_preview}'")
    
    # If there are any correct answers, show one as an example
    correct_samples = results['results_df'][results['results_df']['correct']].head(1)
    if len(correct_samples) > 0:
        print("\n--- Example of CORRECT answer ---")
        row = correct_samples.iloc[0]
        print(f"Question: '{row['question']}'")
        print(f"Expected answer: '{row['actual_answer']}'")
        print(f"Predicted answer: '{row['predicted_answer']}'")
    
    # Print detailed error analysis
    if results['total_samples'] - results['correct_count'] > 0:
        print("\n=== ERROR ANALYSIS ===")
        # Categorize errors by looking at the first few incorrect samples
        incorrect_samples = results['results_df'][~results['results_df']['correct']].head(3)
        for i, (_, row) in enumerate(incorrect_samples.iterrows()):
            print(f"\nError example {i+1}:")
            print(f"Question: '{row['question']}'")
            print(f"Expected: '{row['actual_answer']}'")
            print(f"Predicted: '{row['predicted_answer']}'")
            # Try to categorize the error
            if not row['predicted_answer']:
                print("Error type: No answer provided")
            elif row['predicted_answer'].lower() == "i don't know" or "sorry" in row['predicted_answer'].lower():
                print("Error type: Model indicated lack of information")
            elif len(row['predicted_answer']) > 3 * len(row['actual_answer']):
                print("Error type: Answer too verbose")
            else:
                print("Error type: Incorrect answer")

def save_comparative_results(all_results: Dict, output_dir: str = "results") -> str:
    """
    Saves comparative results to a JSON file.
    """
    # Create local directory
    local_dir = ensure_directory_exists(output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"comparative_results_{timestamp}.json"
    
    # Use explicit local path
    output_file = os.path.join(".", local_dir, filename)
    
    comparative_data = {
        'timestamp': datetime.now().isoformat(),
        'results': {}
    }
    
    for model_key, results in all_results.items():
        comparative_data['results'][model_key] = {
            'accuracy': results['accuracy'],
            'correct_count': results['correct_count'],
            'total_samples': results['total_samples'],
            'model_name': results['model_name'],
            'provider': results['provider']
        }
    
    print(f"Writing comparative results to file: {os.path.abspath(output_file)}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(comparative_data, f, indent=2)
    
    print(f"\nComparative results saved to: {os.path.abspath(output_file)}")
    return output_file

if __name__ == "__main__":
    print(f"\nPyTorch version: {torch.__version__}")
    
    parser = argparse.ArgumentParser(description='Test language models on multiple LegalBench QA datasets')
    parser.add_argument('--output-dir', type=str, default="legalbench_results",
                      help='Directory to save results (default: legalbench_results)')
    parser.add_argument('--providers', nargs='+', type=str, default=['openai', 'anthropic', 'mistral', 'huggingface'],
                      help='List of providers to test (default: all providers)')
    parser.add_argument('--gpu-ids', nargs='+', type=int, default=[0],
                      help='List of GPU IDs to use for HuggingFace models (default: [0])')
    parser.add_argument('--num-samples', type=int, default=150,
                      help='Maximum number of samples to test (default: 150)')
    parser.add_argument('--huggingface-models', nargs='+', type=str, 
                      default=['meta-llama/Llama-2-70b-chat-hf', 'mistralai/Mixtral-8x7B-Instruct-v0.1'],
                      help='List of HuggingFace model IDs to test')
    parser.add_argument('--tasks', nargs='+', type=str, 
                      help='Specific LegalBench tasks to include (default: uses a predefined list of QA-related tasks)')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug output with detailed information')
    parser.add_argument('--examine-dataset', action='store_true',
                      help='Just examine the dataset structure without running inference')
    
    args = parser.parse_args()
    
    # Print current working directory for debugging
    print(f"Current working directory: {os.getcwd()}")
    
    # Just examine the dataset structure if requested
    if args.examine_dataset:
        print("\n=== EXAMINING DATASET STRUCTURE ===")
        ds = load_dataset("nguha/legalbench", args.legalbench_config, trust_remote_code=True)
        
        print(f"Available splits: {list(ds.keys())}")
        for split_name, split in ds.items():
            print(f"\nSplit '{split_name}' has {len(split)} examples")
            if len(split) > 0:
                if hasattr(split[0], 'keys'):
                    print(f"First example keys: {list(split[0].keys())}")
                    for key in split[0].keys():
                        preview = str(split[0][key])
                        print(f"{key}: {preview[:100]}{'...' if len(preview) > 100 else ''}")
                else:
                    print(f"First example type: {type(split[0])}")
        sys.exit(0)
    
    if 'huggingface' in args.providers:
        print("\nCUDA Diagnostic Information:")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device() if torch.cuda.is_available() else 'None'}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
        print(f"NVIDIA_VISIBLE_DEVICES: {os.environ.get('NVIDIA_VISIBLE_DEVICES', 'Not set')}")
        print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")

        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpu_ids))
        
        if torch.cuda.is_available():
            print(f"\nUsing GPU: {torch.cuda.get_device_name(0)}")
            device = "cuda"
        else:
            print("\nNo GPU available, using CPU")
            device = "cpu"
    
    output_dir = args.output_dir
    
    provider_models = {
        "openai": ["gpt-3.5-turbo", "gpt-4o"],
        "anthropic": ["claude-3-5-sonnet-20241022"],
        "mistral": ["open-mixtral-8x22b"],
        "huggingface": args.huggingface_models
    }
    
    models_to_test = []
    for provider in args.providers:
        if provider in provider_models:
            models_to_test.extend([
                {"provider": provider, "model_name": model_name}
                for model_name in provider_models[provider]
            ])
        else:
            print(f"Warning: Unknown provider '{provider}' specified")
    
    all_results = {}
    
    for model in models_to_test:
        try:
            print(f"\nStarting evaluation of {model['provider']} - {model['model_name']} on LegalBench combined tasks")
            
            huggingface_pipeline = None
            if model['provider'] == "huggingface":
                huggingface_pipeline = pipeline(
                    "text-generation",
                    model=model['model_name'],
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
            
            results = test_legalbench_contract_qa(
                num_samples=args.num_samples,
                provider=model["provider"],
                model_name=model["model_name"],
                temperature=0.0,
                output_dir=output_dir,
                huggingface_pipeline=huggingface_pipeline,
                debug=args.debug,
                tasks=args.tasks
            )
            
            all_results[f"{model['provider']}_{model['model_name']}"] = results
            analyze_results(results)
            
            if huggingface_pipeline is not None:
                del huggingface_pipeline
                torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error with {model['provider']} - {model['model_name']}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    save_comparative_results(all_results, output_dir)
    
    print("\n=== Final Comparative Results ===")
    print("\nAccuracy by Model:")
    for model_key, results in all_results.items():
        print(f"{model_key}: {results['accuracy']*100:.2f}%")