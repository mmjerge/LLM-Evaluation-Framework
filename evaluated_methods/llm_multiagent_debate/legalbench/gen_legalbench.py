#!/usr/bin/env python

import openai
from openai import OpenAI
import anthropic
from mistralai.client import MistralClient
import argparse
import json
import numpy as np
import random
import os
import tqdm
import sys
from datasets import load_dataset
import re
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Set up API clients
openai.api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
mistral_api_key = os.getenv("MISTRAL_API_KEY")
mistral_client = MistralClient(api_key=mistral_api_key)
# vLLM client will be initialized with custom URL when needed

# Initialize global API calls counter
api_calls_count = 0

def construct_message(agents, question, policy_text, idx, use_counterfactual=False):
    """Construct a message for agent interaction using other agents' responses."""
    if len(agents) == 0:
        return {"role": "user", "content": "Can you double check that your answer is correct. Please reiterate your answer, with your final answer as either 'Relevant' or 'Irrelevant', in the form \\boxed{Relevant} or \\boxed{Irrelevant}."}
    
    prefix_string = "These are the solutions to the problem from other agents: "
    for agent in agents:
        agent_response = agent[idx]["content"]
        response = "\n\n One agent solution: ```{}```".format(agent_response)
        prefix_string = prefix_string + response
    
    prefix_string = prefix_string + f"""\n\n Using the solutions from other agents as additional information, can you provide your answer to the legal question? \n 

The original question is: {question} 

Privacy Policy Clause:
{policy_text}

Task: Determine if the privacy policy clause contains enough information to answer the question.

Your final answer should be either 'Relevant' (if the clause contains enough information to answer the question) or 'Irrelevant' (if the clause does not contain enough information), in the form \\boxed{{Relevant}} or \\boxed{{Irrelevant}}, at the end of your response."""
    
    return {"role": "user", "content": prefix_string}

def construct_assistant_message(completion, model_type):
    """Extract content from model completion based on the model type."""
    if model_type == "openai" or model_type == "vllm":
        content = completion.choices[0].message.content
    elif model_type == "anthropic":
        content = completion.content[0].text
    elif model_type == "mistral":
        content = completion.choices[0].message.content
    return {"role": "assistant", "content": content}

def get_model_completion(model_type, model_name, messages, vllm_url=None, vllm_api_key=None):
    """Get completion from the specified model."""
    global api_calls_count
    api_calls_count += 1
    
    try:
        if model_type == "openai":
            # Use the new OpenAI client API syntax for v1.0.0+
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            completion = client.chat.completions.create(
                model=model_name,
                messages=[{"role": m["role"], "content": m["content"]} for m in messages],
                temperature=0.7,
                max_tokens=2000  # Increased for legal texts
            )
            # Extract content from the new response format
            return type('obj', (object,), {
                'choices': [type('obj', (object,), {
                    'message': type('obj', (object,), {
                        'content': completion.choices[0].message.content
                    })
                })]
            })
        elif model_type == "vllm":
            # Use vLLM's OpenAI-compatible API
            vllm_client = OpenAI(
                base_url=vllm_url or "http://localhost:8000/v1",
                api_key=vllm_api_key or "dummy-key"
            )
            completion = vllm_client.chat.completions.create(
                model=model_name,
                messages=[{"role": m["role"], "content": m["content"]} for m in messages],
                temperature=0.7,
                max_tokens=2000  # Increased for legal texts
            )
            # Extract content from the response format
            return type('obj', (object,), {
                'choices': [type('obj', (object,), {
                    'message': type('obj', (object,), {
                        'content': completion.choices[0].message.content
                    })
                })]
            })
        elif model_type == "anthropic":
            completion = anthropic_client.messages.create(
                model=model_name,
                messages=[{"role": m["role"], "content": m["content"]} for m in messages],
                max_tokens=2000,
                temperature=0.7
            )
        elif model_type == "mistral":
            completion = mistral_client.chat(
                model=model_name,
                messages=[{"role": m["role"], "content": m["content"]} for m in messages],
                temperature=0.7,
                max_tokens=2000
            )
        return completion
    except Exception as e:
        print(f"Error in API call: {str(e)}")
        # Return a minimal completion structure to avoid crashing
        if model_type in ["openai", "vllm"]:
            return type('obj', (object,), {
                'choices': [type('obj', (object,), {
                    'message': type('obj', (object,), {
                        'content': f"ERROR: {str(e)}"
                    })
                })]
            })
        elif model_type == "anthropic":
            return type('obj', (object,), {
                'content': [type('obj', (object,), {
                    'text': f"ERROR: {str(e)}"
                })]
            })
        elif model_type == "mistral":
            return type('obj', (object,), {
                'choices': [type('obj', (object,), {
                    'message': type('obj', (object,), {
                        'content': f"ERROR: {str(e)}"
                    })
                })]
            })

def extract_answer(response_text):
    """Extract Relevant/Irrelevant answer from the response."""
    # Check for boxed answers
    boxed_pattern = r'\\boxed\{(Relevant|Irrelevant)\}'
    matches = re.findall(boxed_pattern, response_text, re.IGNORECASE)
    if matches:
        return matches[-1].lower()
    
    # Check for alternative formats
    alt_patterns = [
        r'\(\\boxed\{(Relevant|Irrelevant)\}\)',
        r'answer is[:\s]*(Relevant|Irrelevant)',
        r'final answer[:\s]*(Relevant|Irrelevant)',
        r'conclusion[:\s]*(Relevant|Irrelevant)',
        r'the answer is[:\s]*(Relevant|Irrelevant)',
        r'my answer is[:\s]*(Relevant|Irrelevant)'
    ]
    
    for pattern in alt_patterns:
        matches = re.findall(pattern, response_text, re.IGNORECASE)
        if matches:
            return matches[-1].lower()
    
    # Look for Relevant/Irrelevant in the last 50 words of the text
    words = response_text.split()
    last_words = ' '.join(words[-50:])
    if re.search(r'\brelevant\b', last_words, re.IGNORECASE) and not re.search(r'\birrelevant\b', last_words, re.IGNORECASE):
        return 'relevant'
    elif re.search(r'\birrelevant\b', last_words, re.IGNORECASE) and not re.search(r'\brelevant\b', last_words, re.IGNORECASE):
        return 'irrelevant'
    
    return None  # Cannot determine the answer

def main(args):
    random.seed(args.seed)
    
    # Load Privacy Policy QA dataset from LegalBench
    try:
        dataset = load_dataset(args.dataset, name=args.dataset_name)
        print(f"Successfully loaded dataset from {args.dataset} with name={args.dataset_name}")
    except Exception as e:
        print(f"Error loading primary dataset: {str(e)}")
        # Try alternative dataset loading approaches
        try:
            print("Trying to load from original PrivacyQA...")
            dataset = load_dataset("polisis/privacy_qa")
            print("Successfully loaded dataset from polisis/privacy_qa")
        except Exception as e2:
            print(f"Error loading polisis/privacy_qa: {str(e2)}")
            try:
                print("Trying to load from legacy Privacy QA dataset...")
                dataset = load_dataset("privacy_qa", "privacy_qa")
                print("Successfully loaded legacy privacy_qa dataset")
            except Exception as e3:
                print(f"Error loading legacy dataset: {str(e3)}")
                print("Could not load any version of the Privacy Policy QA dataset.")
                return
    
    # Print dataset structure and size information
    print(f"Dataset splits: {list(dataset.keys())}")
    for split_name in dataset.keys():
        print(f"Split '{split_name}' contains {len(dataset[split_name])} examples")
    
    # Print dataset structure example
    print("Dataset structure example:")
    first_example = dataset[args.split][0] if args.split in dataset else dataset['train'][0]  # Fallback to train if split doesn't exist
    for key in first_example:
        print(f"Key: {key}, Type: {type(first_example[key])}")
    
    # Determine which split to use
    if args.split not in dataset:
        print(f"Warning: Split '{args.split}' not found. Available splits: {list(dataset.keys())}")
        split_to_use = list(dataset.keys())[0]  # Default to first available split
        print(f"Using '{split_to_use}' split instead.")
    else:
        split_to_use = args.split
    
    # Get test data
    test_data = dataset[split_to_use]
    total_questions = len(test_data)
    
    # Ensure we don't request more questions than are available
    if args.full_evaluation:
        args.num_of_questions = min(total_questions, args.max_questions)
        print(f"Full evaluation requested. Using {args.num_of_questions} questions (total available: {total_questions})")
    else:
        args.num_of_questions = min(args.num_of_questions, total_questions)
        print(f"Using {args.num_of_questions} questions (total available: {total_questions})")
    
    # Shuffle and limit to the specified number of questions
    indices = list(range(total_questions))
    random.shuffle(indices)
    selected_indices = indices[:args.num_of_questions]
    
    print(f"Running evaluation on {len(selected_indices)} questions from the {split_to_use} split")
    print(f"Using model: {args.model_type} - {args.model_name}")
    if args.model_type == "vllm":
        print(f"vLLM server URL: {args.vllm_url}")
    
    generated_description = {}
    global api_calls_count
    api_calls_per_problem = {}
    
    # Print dataset structure
    print("Dataset structure example:")
    first_example = dataset[args.split][0] if args.split in dataset else dataset['train'][0]  # Fallback to train if split doesn't exist
    for key in first_example:
        print(f"Key: {key}, Type: {type(first_example[key])}")
    
    # Determine which split to use
    if args.split not in dataset:
        print(f"Warning: Split '{args.split}' not found. Available splits: {list(dataset.keys())}")
        split_to_use = 'train'  # Default to train
    else:
        split_to_use = args.split
    
    # Get test data
    test_data = dataset[split_to_use]
    
    # Shuffle and limit to the specified number of questions
    indices = list(range(len(test_data)))
    random.shuffle(indices)
    selected_indices = indices[:args.num_of_questions]
    
    print(f"Running evaluation on {len(selected_indices)} questions from the {split_to_use} split")
    print(f"Using model: {args.model_type} - {args.model_name}")
    if args.model_type == "vllm":
        print(f"vLLM server URL: {args.vllm_url}")
    
    generated_description = {}
    global api_calls_count
    api_calls_per_problem = {}
    
    for idx in tqdm.tqdm(selected_indices, desc="Questions processed"):
        data = test_data[idx]
        
        # Extract question, policy text, and answer 
        try:
            question = data["question"]
            
            # Try different field names for the policy text
            if "clause" in data:
                policy_text = data["clause"]
            elif "text" in data:
                policy_text = data["text"]  # Use "text" field if "clause" is not available
            else:
                # If neither field exists, list available fields and raise an error
                available_fields = ", ".join(data.keys())
                raise KeyError(f"Could not find policy text field. Available fields: {available_fields}")
            
            # Map label to "relevant" or "irrelevant"
            if "label" in data:
                # Original expected format with numeric labels
                correct_answer = "relevant" if data["label"] == 1 else "irrelevant"
            elif "answer" in data:
                # Handle string answers
                answer = data["answer"].lower()
                if answer in ["yes", "true", "1", "relevant"]:
                    correct_answer = "relevant"
                else:
                    correct_answer = "irrelevant"
            else:
                raise KeyError("Could not find answer field (neither 'label' nor 'answer' exist)")
            
            # Truncate policy text if too long
            if len(policy_text) > 4000:
                policy_text = policy_text[:4000] + "..."
            
            api_calls_count = 0
            
            # Initialize agent contexts with the question
            initial_prompt = f"""Can you answer the following question about a privacy policy? 

Question: {question}

Privacy Policy Clause:
{policy_text}

Task: Determine if the privacy policy clause contains enough information to answer the question.

Analyze the privacy policy clause to determine if it contains enough information to answer the question. If the clause provides enough information to answer the question, classify it as 'Relevant'. If the clause does not provide enough information to answer the question, classify it as 'Irrelevant'.

Explain your reasoning step-by-step and provide your final answer in the form \\boxed{{Relevant}} or \\boxed{{Irrelevant}} at the end of your response."""
            
            agent_contexts = [[{"role": "user", "content": initial_prompt}] for _ in range(args.agents)]
            
            for round in range(args.rounds):
                for i, agent_context in enumerate(agent_contexts):
                    if round != 0:
                        agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]
                        message = construct_message(agent_contexts_other, question, policy_text, 2*round - 1, use_counterfactual=(round==args.rounds-1))
                        agent_context.append(message)
                    
                    completion = get_model_completion(
                        args.model_type, 
                        args.model_name, 
                        agent_context,
                        vllm_url=args.vllm_url,
                        vllm_api_key=args.vllm_api_key
                    )
                    assistant_message = construct_assistant_message(completion, args.model_type)
                    agent_context.append(assistant_message)
            
            generated_description[question] = (agent_contexts, correct_answer)
            api_calls_per_problem[question] = api_calls_count
        
        except Exception as e:
            print(f"Error processing question at index {idx}: {str(e)}")
    
    # Save results to a JSON file
    # Clean the model name to create a valid filename
    safe_model_name = args.model_name.replace('/', '_').replace('-', '_')
    
    # Create the output directory if it doesn't exist
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(output_dir, f"results_legalbench_privacy_{args.agents}_{args.rounds}_{args.model_type}_{safe_model_name}_{len(selected_indices)}q.json")
    
    with open(output_file, "w") as f:
        json.dump({"results": generated_description, "api_calls": api_calls_per_problem}, f)
    print(f"Results saved to {output_file}")
    
    # Calculate accuracy
    correct_count = 0
    total_predictions = 0
    
    for question, (agent_contexts, correct_answer) in generated_description.items():
        for agent_context in agent_contexts:
            total_predictions += 1
            # Get the final answer from the last message
            if len(agent_context) > 0 and "content" in agent_context[-1]:
                last_message = agent_context[-1]["content"]
                extracted_answer = extract_answer(last_message)
                if extracted_answer == correct_answer:
                    correct_count += 1
    
    accuracy = correct_count / total_predictions if total_predictions > 0 else 0
    print(f"Overall accuracy: {accuracy:.2%} ({correct_count}/{total_predictions})")
    
    # Print summary of API calls
    total_calls = sum(api_calls_per_problem.values())
    avg_calls = total_calls / len(api_calls_per_problem) if api_calls_per_problem else 0
    print(f"Total API calls: {total_calls}")
    print(f"Average API calls per problem: {avg_calls:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multi-agent debate for LegalBench Privacy Policy QA")
    parser.add_argument("--model_type", type=str, choices=["openai", "anthropic", "mistral", "vllm"], default="openai", help="Type of model to use (openai, anthropic, mistral, or vllm)")
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo", help="Name of the model to use (e.g., gpt-4 for OpenAI, claude-3-sonnet-20240229 for Anthropic, mistral-large-latest for Mistral, or NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO for vLLM)")
    parser.add_argument("--vllm_url", type=str, default="http://localhost:8000/v1", help="URL for the vLLM OpenAI-compatible server")
    parser.add_argument("--vllm_api_key", type=str, default="dummy-key", help="API key for the vLLM server (can be any string)")
    parser.add_argument("--agents", type=int, default=3, help="Number of agents")
    parser.add_argument("--rounds", type=int, default=2, help="Number of rounds")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--num_of_questions", type=int, default=5, help="Number of questions to process (default: 5 for testing)")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use (train, validation, or test)")
    parser.add_argument("--full_evaluation", action="store_true", help="Run on as many questions as possible (up to 150)")
    parser.add_argument("--dataset", type=str, default="nguha/legalbench", help="HuggingFace dataset repository to use")
    parser.add_argument("--dataset_name", type=str, default="privacy_policy_qa", help="Name/configuration of the dataset to use")
    parser.add_argument("--max_questions", type=int, default=150, help="Maximum number of questions to use with --full_evaluation (default: 150)")
    
    args = parser.parse_args()
    
    main(args)