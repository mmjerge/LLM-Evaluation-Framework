import random
from datasets import load_dataset, concatenate_datasets
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI
from anthropic import Anthropic
from mistralai.client import MistralClient
# from mistralai.models.chat_completion import ChatMessage
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
    Returns few-shot examples for medical question answering.
    """
    return """Example 1:
Question: A 65-year-old man presents with a 3-month history of increasing forgetfulness. His wife reports that he frequently misplaces his keys, forgets appointments, and has trouble recalling recent conversations. His past medical history is significant for hypertension, hyperlipidemia, and type 2 diabetes mellitus. On Mini-Mental State Examination (MMSE), his score is 23/30. MRI of the brain shows diffuse cortical atrophy and hippocampal volume loss. Which of the following is the most likely diagnosis?
A. Vascular dementia
B. Alzheimer's disease
C. Frontotemporal dementia
D. Dementia with Lewy bodies
E. Normal pressure hydrocephalus

Let's think step by step:
1. The patient is a 65-year-old man with progressive memory issues, particularly with recent events and conversations.
2. He has risk factors including hypertension, hyperlipidemia, and type 2 diabetes.
3. MMSE score of 23/30 indicates mild to moderate cognitive impairment.
4. MRI shows diffuse cortical atrophy and hippocampal volume loss.

Analyzing each option:
A. Vascular dementia - Usually has a more stepwise progression, often with focal neurological deficits, rather than gradual memory decline.
B. Alzheimer's disease - Characterized by progressive memory loss, particularly for recent events, with hippocampal atrophy on imaging.
C. Frontotemporal dementia - Typically presents with personality changes, disinhibition, and language problems rather than primarily memory issues.
D. Dementia with Lewy bodies - Usually includes visual hallucinations, fluctuating cognition, and parkinsonism features not mentioned here.
E. Normal pressure hydrocephalus - Classic triad of gait disturbance, urinary incontinence, and dementia; would show ventricular enlargement on imaging.

Given the gradual onset of memory impairment, particularly for recent events, and hippocampal atrophy on MRI, Alzheimer's disease is the most likely diagnosis.

Answer: B

Example 2:
Question: A 45-year-old woman presents with fatigue, weight gain, and cold intolerance for the past 6 months. Physical examination reveals dry skin, brittle hair, and delayed relaxation phase of deep tendon reflexes. Laboratory studies are most likely to show:
A. Decreased TSH, increased T4
B. Increased TSH, decreased T4
C. Decreased TSH, decreased T4
D. Increased TSH, increased T4
E. Normal TSH, decreased T4

Let's think step by step:
1. The patient has symptoms consistent with hypothyroidism: fatigue, weight gain, cold intolerance, dry skin, brittle hair, and delayed relaxation of reflexes.
2. In primary hypothyroidism, the thyroid gland produces insufficient thyroid hormones (T3 and T4).
3. When thyroid hormone levels are low, the pituitary gland responds by increasing thyroid-stimulating hormone (TSH) production to stimulate the thyroid.
4. Therefore, primary hypothyroidism typically presents with elevated TSH (due to pituitary compensation) and decreased T4.

Answer: B

Now solve this new problem:
"""

def create_medqa_prompt(question: str, options: List[str]) -> str:
    """
    Creates a prompt for medical question answering.
    
    Args:
        question: The question text
        options: List of option texts (can be in various formats)
        
    Returns:
        Formatted prompt string
    """
    formatted_options = ""
    option_letters = ["A", "B", "C", "D", "E", "F", "G", "H"]
    
    for i, option in enumerate(options):
        if i < len(option_letters):
            # Check if option is a string representation of a dictionary
            if isinstance(option, str) and option.startswith('{') and option.endswith('}'):
                try:
                    import ast
                    parsed_option = ast.literal_eval(option)
                    if isinstance(parsed_option, dict) and 'value' in parsed_option:
                        option_text = parsed_option['value']
                        formatted_options += f"{option_letters[i]}. {option_text}\n"
                    else:
                        # If can't parse properly, use the raw string
                        formatted_options += f"{option_letters[i]}. {option}\n"
                except:
                    # If parsing fails, use the raw string
                    formatted_options += f"{option_letters[i]}. {option}\n"
            else:
                # Regular string option
                formatted_options += f"{option_letters[i]}. {option}\n"
    
    return f"{get_few_shot_examples()}\nQuestion: {question}\n{formatted_options}\nLet's think step by step:"

def extract_final_answer(response: str) -> str:
    """
    Extracts the final answer (A, B, C, D, etc.) from the model's response for MedQA.
    """
    response_lower = response.lower()
    
    # Check for "Answer: X" pattern
    if "answer:" in response_lower:
        answer_section = response.split("Answer:", 1)[1].strip()
        first_word = answer_section.split()[0]
        # If the first word is just a letter (A, B, C, etc.), return it
        if len(first_word) == 1 and first_word.upper() in "ABCDEFGH":
            return first_word.upper()
    
    # Look for a standalone letter at the end of the response
    lines = [line.strip() for line in response.split('\n') if line.strip()]
    if lines:
        last_line = lines[-1]
        if len(last_line) == 1 and last_line.upper() in "ABCDEFGH":
            return last_line.upper()
    
    # Look for phrases like "the answer is X"
    match = re.search(r"the answer is ([A-Ha-h])", response_lower)
    if match:
        return match.group(1).upper()
    
    # If no clear answer format is found, try to detect any letter options mentioned in the last few sentences
    last_sentences = " ".join(lines[-3:]) if len(lines) >= 3 else " ".join(lines)
    possible_answers = re.findall(r"\b([A-Ha-h])\b", last_sentences)
    
    if possible_answers:
        # Take the last mentioned option as the answer
        return possible_answers[-1].upper()
    
    # If we can't find an answer letter, return an empty string
    return ""

def evaluate_response(predicted: str, actual_letter: str, actual_text: str, option_pairs: List[Tuple[str, str]]) -> bool:
    """
    Evaluates if the predicted answer matches the actual answer for MedQA.
    Handles both letter answers (A, B, C) and text answers.
    
    Args:
        predicted: The predicted answer (usually a letter)
        actual_letter: The actual answer letter
        actual_text: The actual answer text
        option_pairs: List of (letter, text) pairs for mapping between letters and text
        
    Returns:
        True if the prediction matches either the letter or text of the actual answer
    """
    if not predicted:
        return False
    
    # Clean up predicted answer
    predicted = predicted.strip().upper()
    
    # First, try direct letter comparison
    if predicted == actual_letter.strip().upper():
        return True
    
    # If predicted is a letter, convert to text and compare
    letter_to_text = {letter.upper(): text for letter, text in option_pairs}
    if predicted in letter_to_text:
        predicted_text = letter_to_text[predicted]
        return predicted_text.strip().lower() == actual_text.strip().lower()
    
    # If predicted is a longer string, it might be the text answer
    # Check if it contains or is similar to the actual text
    if len(predicted) > 1:
        return predicted.lower() == actual_text.strip().lower()
    
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
    system_prompt = """You are a medical expert assistant specializing in answering medical exam questions. 
    
    Your task is to:
    1. Read the question carefully
    2. Analyze each answer option in detail
    3. Show your step-by-step medical reasoning
    4. Select the single most appropriate answer
    5. End with 'Answer: [your answer letter]' (e.g., 'Answer: A')
    
    Be concise and base your answers solely on well-established medical knowledge."""    

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
        # Handle different field name structures based on what's available
        question_result = {
            'question_id': int(row['question_id']),
            'question': row['question'],
            'options': row.get('options', []),
        }
        
        # Add answer information using the new field names if available
        if 'actual_answer_letter' in row and 'actual_answer_text' in row:
            question_result['ground_truth_letter'] = row['actual_answer_letter']
            question_result['ground_truth'] = row['actual_answer_text']
        elif 'actual_answer' in row:
            # Backward compatibility with old format
            question_result['ground_truth'] = row['actual_answer']
        
        # Add model answer
        question_result['model_answer'] = row['predicted_answer']
        question_result['full_response'] = row['full_response']
        question_result['is_correct'] = bool(row['correct'])
        
        json_results['questions'].append(question_result)
    
    print(f"Writing results to file: {os.path.abspath(output_file)}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed results saved to: {os.path.abspath(output_file)}")
    return output_file

def process_medqa_format(sample: Dict[str, Any], debug_flag: bool = False) -> Tuple[str, List[str], List[Tuple[str, str]], str, str]:
    """
    Process the MedQA dataset format.
    
    Args:
        sample: The dataset sample to process
        debug_flag: Whether to print debug information
        
    Returns:
        Tuple of (question, option_texts, option_pairs, answer_letter, answer_text)
    """
    # Debug: Print sample keys to understand structure
    if debug_flag and hasattr(sample, 'keys'):
        print(f"Sample keys: {list(sample.keys())}")
    
    # Initialize with default values
    question = ""
    option_texts = []  # Plain text options for display
    option_pairs = []  # (letter, text) pairs for mapping
    answer_letter = ""  # Letter answer (A, B, C, etc.)
    answer_text = ""    # Full text answer
    
    # Process based on MedQA dataset structure
    if 'question' in sample:
        question = sample.get('question', '')
        
        # Process options based on dataset format
        if 'options' in sample:
            options_data = sample['options']
            
            # Handle dictionary format
            if isinstance(options_data, dict):
                # Some MedQA datasets store options as a dictionary with keys A, B, C, etc.
                sorted_keys = sorted(options_data.keys())
                option_texts = [options_data[key] for key in sorted_keys]
                option_pairs = [(key, options_data[key]) for key in sorted_keys]
            
            # Handle list format
            elif isinstance(options_data, list):
                # Check if the options are stored as string representations of dictionaries
                try:
                    # Try to parse as string representations of dictionaries
                    parsed_options = []
                    for opt in options_data:
                        # If it's already a dict, use it directly
                        if isinstance(opt, dict):
                            parsed_options.append(opt)
                        # If it's a string representation of a dict, try to evaluate it
                        elif isinstance(opt, str) and opt.startswith('{') and opt.endswith('}'):
                            try:
                                import ast
                                parsed_opt = ast.literal_eval(opt)
                                if isinstance(parsed_opt, dict):
                                    parsed_options.append(parsed_opt)
                                else:
                                    # If not a valid dict, just append the string
                                    parsed_options.append({'value': opt})
                            except:
                                # If parsing fails, just use the raw string
                                parsed_options.append({'value': opt})
                        else:
                            # For other formats, just use as is
                            parsed_options.append({'value': opt})
                    
                    # Extract key-value pairs
                    option_letters = "ABCDEFGH"
                    option_pairs = []
                    
                    for i, opt in enumerate(parsed_options):
                        if 'key' in opt and 'value' in opt:
                            # Use key-value format
                            option_pairs.append((opt['key'], opt['value']))
                        else:
                            # Assign letters if no key is available
                            letter = option_letters[i] if i < len(option_letters) else str(i)
                            option_pairs.append((letter, opt.get('value', str(opt))))
                    
                    # Extract just the text values for option_texts
                    option_texts = [pair[1] for pair in option_pairs]
                
                except Exception as e:
                    if debug_flag:
                        print(f"Error parsing options: {e}")
                    # Default to simple list processing
                    option_letters = "ABCDEFGH"
                    option_texts = options_data
                    option_pairs = [(option_letters[i], opt) if i < len(option_letters) else (str(i), opt) 
                                    for i, opt in enumerate(options_data)]
            
        elif 'choices' in sample:
            # Some MedQA datasets use 'choices' instead of 'options'
            choices_data = sample['choices']
            
            if isinstance(choices_data, list):
                option_letters = "ABCDEFGH"
                option_texts = choices_data
                option_pairs = [(option_letters[i], opt) if i < len(option_letters) else (str(i), opt) 
                                for i, opt in enumerate(choices_data)]
            elif isinstance(choices_data, dict):
                sorted_keys = sorted(choices_data.keys())
                option_texts = [choices_data[key] for key in sorted_keys]
                option_pairs = [(key, choices_data[key]) for key in sorted_keys]
        
        # Extract answer
        if 'answer' in sample:
            answer_data = sample['answer']
            
            # Handle different answer formats
            if isinstance(answer_data, str):
                # Check if it's a letter answer
                if len(answer_data) == 1 and answer_data.upper() in "ABCDEFGH":
                    answer_letter = answer_data.upper()
                    # Find the corresponding text answer
                    for letter, text in option_pairs:
                        if letter.upper() == answer_letter:
                            answer_text = text
                            break
                else:
                    # It's a text answer, find the corresponding letter
                    answer_text = answer_data
                    for letter, text in option_pairs:
                        if text == answer_text:
                            answer_letter = letter.upper()
                            break
            elif isinstance(answer_data, int):
                # It's an index, convert to letter
                option_letters = "ABCDEFGH"
                if 0 <= answer_data < len(option_letters):
                    answer_letter = option_letters[answer_data]
                    if answer_data < len(option_texts):
                        answer_text = option_texts[answer_data]
        
        # If we only have text answer but no letter, try to match with options
        if answer_text and not answer_letter:
            for letter, text in option_pairs:
                if text == answer_text:
                    answer_letter = letter.upper()
                    break
        
        # If we only have letter answer but no text, try to match with options
        if answer_letter and not answer_text:
            for letter, text in option_pairs:
                if letter.upper() == answer_letter:
                    answer_text = text
                    break
    
    # Clean up
    question = question.strip() if isinstance(question, str) else str(question)
    option_texts = [opt.strip() if isinstance(opt, str) else str(opt) for opt in option_texts]
    answer_letter = answer_letter.strip().upper() if isinstance(answer_letter, str) else str(answer_letter)
    answer_text = answer_text.strip() if isinstance(answer_text, str) else str(answer_text)
    
    return question, option_texts, option_pairs, answer_letter, answer_text

def test_medqa(
    num_samples: int = 150,
    provider: ProviderType = "openai",
    model_name: str = "gpt-4o",
    temperature: float = 0.0,
    output_dir: str = "results",
    huggingface_pipeline = None,
    debug: bool = True,
    languages: List[str] = None,
    input_json_file: str = None  # Added parameter for direct JSON input
) -> Dict:
    """
    Tests the specified model on samples from MedQA dataset.
    
    Args:
        num_samples: Number of samples to test
        provider: Model provider (openai, anthropic, mistral, huggingface)
        model_name: Name of the model to test
        temperature: Sampling temperature
        output_dir: Directory to save results
        huggingface_pipeline: Optional HuggingFace pipeline for local models
        debug: Whether to print debug information
        languages: Languages to test (default is English)
        input_json_file: Optional path to a JSON file with test data
        
    Returns:
        Dictionary with test results
    """
    # Global debug flag to control verbosity
    debug_mode = debug
    
    # Default to English if no languages specified
    if not languages:
        languages = ["en"]
    
    # If input JSON file is provided, load data from there
    if input_json_file:
        print(f"Loading test data from JSON file: {input_json_file}")
        try:
            with open(input_json_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                
            # Extract questions from the JSON data
            test_samples = []
            if 'questions' in json_data:
                for question_data in json_data['questions']:
                    sample = {
                        'question': question_data.get('question', ''),
                        'options': question_data.get('options', []),
                        'answer': question_data.get('ground_truth', '')
                    }
                    test_samples.append(sample)
                
                print(f"Loaded {len(test_samples)} questions from JSON file")
                actual_num_samples = min(num_samples, len(test_samples)) if num_samples > 0 else len(test_samples)
                if actual_num_samples < len(test_samples):
                    # Select a random subset if needed
                    test_samples = random.sample(test_samples, actual_num_samples)
        except Exception as e:
            print(f"Error loading JSON file: {str(e)}")
            raise ValueError(f"Could not load data from JSON file: {input_json_file}")
    else:
        # Load MedQA datasets from Hugging Face
        all_datasets = []
        
        for lang in languages:
            try:
                print(f"Loading MedQA dataset for language: {lang}...")
                
                # Try different dataset names based on language
                dataset_name = f"bigbio/med_qa" if lang == "en" else f"bigbio/med_qa_{lang}"
                try:
                    dataset = load_dataset(dataset_name, trust_remote_code=True)
                except Exception as e:
                    print(f"Error loading {dataset_name}: {str(e)}")
                    # Fallback to generic MedQA
                    if lang == "en":
                        dataset = load_dataset("GBaker/MedQA-USMLE-4-options", trust_remote_code=True)
                    else:
                        print(f"No fallback available for language {lang}")
                        continue
                
                # Use the test split preferably
                if 'test' in dataset:
                    task_data = dataset['test']
                    print(f"  - Added {len(task_data)} examples from {dataset_name} (test split)")
                    all_datasets.append(task_data)
                elif 'validation' in dataset:
                    task_data = dataset['validation']
                    print(f"  - Added {len(task_data)} examples from {dataset_name} (validation split)")
                    all_datasets.append(task_data)
                elif 'train' in dataset:
                    task_data = dataset['train']
                    print(f"  - Added {len(task_data)} examples from {dataset_name} (train split)")
                    all_datasets.append(task_data)
                else:
                    print(f"  - No usable split found in {dataset_name}")
                    
            except Exception as e:
                print(f"Error loading MedQA for {lang}: {str(e)}")
        
        # If no datasets were loaded, try a specific fallback
        if not all_datasets:
            try:
                print("No datasets loaded. Trying fallback to specific MedQA USMLE dataset...")
                dataset = load_dataset("GBaker/MedQA-USMLE-4-options", trust_remote_code=True)
                if 'test' in dataset:
                    task_data = dataset['test']
                    print(f"  - Added {len(task_data)} examples from fallback dataset (test split)")
                    all_datasets.append(task_data)
                elif 'train' in dataset:
                    task_data = dataset['train']
                    print(f"  - Added {len(task_data)} examples from fallback dataset (train split)")
                    all_datasets.append(task_data)
            except Exception as e:
                print(f"Error loading fallback dataset: {str(e)}")
                raise ValueError("Could not load any MedQA datasets. Please check dataset availability.")
        
        # Concatenate all the datasets
        print("Concatenating all datasets...")
        ds = concatenate_datasets(all_datasets)
        
        print(f"Combined dataset size: {len(ds)} examples")
        
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
    
    # Set up HuggingFace pipeline if needed
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
    
    results = []
    correct = 0
    
    for i, sample in tqdm(enumerate(test_samples), total=len(test_samples), desc=f"Testing {model_name}"):
        # Process the sample to extract question, options, and answer
        question, option_texts, option_pairs, answer_letter, answer_text = process_medqa_format(sample, debug)
        
        # Skip samples with missing data
        if not question or not option_texts:
            print(f"Skipping sample {i} due to missing question or options")
            continue
            
        # Create prompt using the text representation of options
        prompt = create_medqa_prompt(question, option_texts)
        
        # Print the first prompt for debugging
        if i == 0 and debug:
            print("\n=== DEBUG: First Sample Prompt ===")
            preview_length = min(500, len(prompt))
            print(prompt[:preview_length] + ("..." if len(prompt) > preview_length else ""))
            print(f"\nQuestion: {question}")
            print(f"Options: {option_texts}")
            print(f"Option pairs: {option_pairs}")
            print(f"Answer letter: {answer_letter}")
            print(f"Answer text: {answer_text}")
        
        try:
            response = get_model_response(
                prompt,
                provider,
                model_name,
                temperature,
                huggingface_pipeline
            )
            
            predicted_answer = extract_final_answer(response)
            
            is_correct = evaluate_response(predicted_answer, answer_letter, answer_text, option_pairs)
            if is_correct:
                correct += 1
                
            results.append({
                'question_id': i,
                'question': question,
                'options': option_texts,
                'option_pairs': option_pairs,
                'actual_answer_letter': answer_letter,
                'actual_answer_text': answer_text,
                'predicted_answer': predicted_answer,
                'full_response': response,
                'correct': is_correct
            })
            
            time.sleep(1)
            
        except Exception as e:
            print(f"Error processing question {i}: {str(e)}")
            continue
    
    accuracy = correct / len(test_samples) if test_samples else 0
    
    results_df = pd.DataFrame(results)
    
    results_dict = {
        'accuracy': accuracy,
        'total_samples': len(test_samples),
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
        
        # Handle different formats of options display
        if 'options' in row:
            print(f"Options: {row['options']}")
        
        # Handle different formats of answer fields
        if 'actual_answer_text' in row and 'actual_answer_letter' in row:
            print(f"Expected answer: '{row['actual_answer_letter']}' ({row['actual_answer_text']})")
        elif 'actual_answer' in row:
            print(f"Expected answer: '{row['actual_answer']}'")
        else:
            print("Expected answer: Not found in results")
            
        print(f"Predicted answer: '{row['predicted_answer']}'")
        
        if 'full_response' in row:
            response_preview = row['full_response'][:100] + "..." if len(row['full_response']) > 100 else row['full_response']
            print(f"Response (preview): '{response_preview}'")
    
    # If there are any correct answers, show one as an example
    correct_samples = results['results_df'][results['results_df']['correct']].head(1)
    if len(correct_samples) > 0:
        print("\n--- Example of CORRECT answer ---")
        row = correct_samples.iloc[0]
        print(f"Question: '{row['question']}'")
        
        # Handle different formats of answer fields
        if 'actual_answer_text' in row and 'actual_answer_letter' in row:
            print(f"Expected answer: '{row['actual_answer_letter']}' ({row['actual_answer_text']})")
        elif 'actual_answer' in row:
            print(f"Expected answer: '{row['actual_answer']}'")
        else:
            print("Expected answer: Not found in results")
            
        print(f"Predicted answer: '{row['predicted_answer']}'")
    
    # Print detailed error analysis
    if results['total_samples'] - results['correct_count'] > 0:
        print("\n=== ERROR ANALYSIS ===")
        # Categorize errors by looking at the first few incorrect samples
        incorrect_samples = results['results_df'][~results['results_df']['correct']].head(3)
        for i, (_, row) in enumerate(incorrect_samples.iterrows()):
            print(f"\nError example {i+1}:")
            print(f"Question: '{row['question']}'")
            
            # Handle different formats of answer fields
            if 'actual_answer_text' in row and 'actual_answer_letter' in row:
                print(f"Expected: '{row['actual_answer_letter']}' ({row['actual_answer_text']})")
            elif 'actual_answer' in row:
                print(f"Expected: '{row['actual_answer']}'")
            else:
                print("Expected: Not found in results")
                
            print(f"Predicted: '{row['predicted_answer']}'")
            
            # Try to categorize the error
            if not row['predicted_answer']:
                print("Error type: No answer provided")
            elif row['predicted_answer'].lower() == "i don't know" or "sorry" in row['predicted_answer'].lower():
                print("Error type: Model indicated lack of information")
            else:
                # Try to determine if it's a format error or a wrong answer
                if 'actual_answer_letter' in row and len(row['predicted_answer']) > 1:
                    print("Error type: Format error (model gave text instead of letter)")
                elif 'actual_answer_text' in row and len(row['predicted_answer']) == 1:
                    print("Error type: Wrong option selected")
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
    
    parser = argparse.ArgumentParser(description='Test language models on MedQA benchmark')
    parser.add_argument('--output-dir', type=str, default="medqa_results",
                      help='Directory to save results (default: medqa_results)')
    parser.add_argument('--providers', nargs='+', type=str, default=['openai', 'anthropic', 'mistral', 'huggingface'],
                      help='List of providers to test (default: all providers)')
    parser.add_argument('--gpu-ids', nargs='+', type=int, default=[0],
                      help='List of GPU IDs to use for HuggingFace models (default: [0])')
    parser.add_argument('--num-samples', type=int, default=150,
                      help='Maximum number of samples to test (default: 150)')
    parser.add_argument('--huggingface-models', nargs='+', type=str, 
                      default=['meta-llama/Llama-2-70b-chat-hf', 'mistralai/Mixtral-8x7B-Instruct-v0.1'],
                      help='List of HuggingFace model IDs to test')
    parser.add_argument('--languages', nargs='+', type=str, default=['en'],
                      help='Languages to test (default: en)')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug output with detailed information')
    parser.add_argument('--examine-dataset', action='store_true',
                      help='Just examine the dataset structure without running inference')
    parser.add_argument('--input-json', type=str, default=None,
                      help='Path to a JSON file containing test questions (alternative to using HuggingFace datasets)')
    parser.add_argument('--test-results-json', type=str, default=None,
                      help='Path to a JSON file containing previous test results to analyze (no new tests will be run)')
    
    args = parser.parse_args()
    
    # Print current working directory for debugging
    print(f"Current working directory: {os.getcwd()}")
    
    # Just examine the dataset structure if requested
    if args.examine_dataset:
        print("\n=== EXAMINING DATASET STRUCTURE ===")
        try:
            ds = load_dataset("GBaker/MedQA-USMLE-4-options", trust_remote_code=True)
            
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
        except Exception as e:
            print(f"Error examining dataset: {str(e)}")
        sys.exit(0)
    
    # If a previous test results JSON is provided, just analyze it without running new tests
    if args.test_results_json:
        print(f"\nAnalyzing previous test results from: {args.test_results_json}")
        try:
            with open(args.test_results_json, 'r', encoding='utf-8') as f:
                results_data = json.load(f)
                
            # Convert to the format expected by analyze_results
            results = {
                'accuracy': results_data.get('metadata', {}).get('accuracy', 0),
                'total_samples': results_data.get('metadata', {}).get('total_samples', 0),
                'correct_count': results_data.get('metadata', {}).get('correct_count', 0),
                'model_name': results_data.get('metadata', {}).get('model_name', 'unknown'),
                'provider': results_data.get('metadata', {}).get('provider', 'unknown'),
                'results_df': pd.DataFrame(results_data.get('questions', []))
            }
            
            analyze_results(results)
            sys.exit(0)
        except Exception as e:
            print(f"Error analyzing results file: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
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
            print(f"\nStarting evaluation of {model['provider']} - {model['model_name']} on MedQA")
            
            huggingface_pipeline = None
            if model['provider'] == "huggingface":
                huggingface_pipeline = pipeline(
                    "text-generation",
                    model=model['model_name'],
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
            
            results = test_medqa(
                num_samples=args.num_samples,
                provider=model["provider"],
                model_name=model["model_name"],
                temperature=0.0,
                output_dir=output_dir,
                huggingface_pipeline=huggingface_pipeline,
                debug=args.debug,
                languages=args.languages,
                input_json_file=args.input_json
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