#!/usr/bin/env python

import openai
from openai import OpenAI  # Import the OpenAI client
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
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Set up API clients
openai.api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
mistral_api_key = os.getenv("MISTRAL_API_KEY")
mistral_client = MistralClient(api_key=mistral_api_key)
# vLLM client will be initialized with custom URL when needed

def construct_message(agents, question, options, idx, use_counterfactual=False):
    """Construct a message for agent interaction using other agents' responses."""
    if len(agents) == 0:
        return {"role": "user", "content": "Can you double check that your answer is correct. Please reiterate your answer, with your final answer as a single letter corresponding to one of the provided options, in the form \\boxed{A} or \\boxed{B} etc."}
    
    prefix_string = "These are the solutions to the problem from other agents: "
    for agent in agents:
        agent_response = agent[idx]["content"]
        response = "\n\n One agent solution: ```{}```".format(agent_response)
        prefix_string = prefix_string + response
    
    options_str = "\n".join([f"{option['option_id']}. {option['option_text']}" for option in options])
    prefix_string = prefix_string + f"""\n\n Using the solutions from other agents as additional information, can you provide your answer to the medical question? \n The original medical question is: {question} \n\nOptions:\n{options_str}\n\nYour final answer should be a single letter corresponding to one of the provided options, in the form \\boxed{{A}} or \\boxed{{B}} etc., at the end of your response."""
    
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
                max_tokens=1000
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
                max_tokens=1000
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
                max_tokens=1000,
                temperature=0.7
            )
        elif model_type == "mistral":
            completion = mistral_client.chat(
                model=model_name,
                messages=[{"role": m["role"], "content": m["content"]} for m in messages],
                temperature=0.7,
                max_tokens=1000
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

def format_options(options):
    """Format options for display in the prompt."""
    formatted = []
    for i, option in enumerate(options):
        # Use A, B, C, D, etc. as option IDs
        option_id = chr(65 + i)  # ASCII A=65, B=66, etc.
        formatted.append({
            "option_id": option_id,
            "option_text": option
        })
    return formatted

def main(args):
    random.seed(args.seed)
    
    # Load MedQA dataset from Hugging Face with the correct configuration
    dataset = load_dataset("bigbio/med_qa", name="med_qa_en_4options_bigbio_qa")
    
    # Filter to use only the test split
    test_data = dataset[args.split]
    
    # Print the first example to see its structure
    print("Dataset structure example:")
    first_example = test_data[0]
    for key in first_example:
        print(f"Key: {key}, Type: {type(first_example[key])}")
        if isinstance(first_example[key], list) and len(first_example[key]) > 0:
            print(f"  List content example: {first_example[key][0]}")
    
    # Shuffle and limit to the specified number of questions
    indices = list(range(len(test_data)))
    random.shuffle(indices)
    selected_indices = indices[:args.num_of_questions]
    
    print(f"Running evaluation on {len(selected_indices)} questions from the {args.split} split")
    print(f"Using model: {args.model_type} - {args.model_name}")
    if args.model_type == "vllm":
        print(f"vLLM server URL: {args.vllm_url}")
    
    generated_description = {}
    global api_calls_count
    api_calls_per_problem = {}
    
    for idx in tqdm.tqdm(selected_indices, desc="Questions processed"):
        data = test_data[idx]
        
        # Extract question, options, and answer from MedQA format
        question = data["question"]
        
        # The choices are already strings in a list
        options = data["choices"]
        
        # The answer is a list containing the correct answer(s)
        # Find the index of the correct answer in the choices
        correct_answer = None
        try:
            # Try to find the first answer in the choices
            if "answer" in data and isinstance(data["answer"], list) and len(data["answer"]) > 0:
                answer_text = data["answer"][0]
                if answer_text in options:
                    correct_answer_idx = options.index(answer_text)
                    correct_answer = chr(65 + correct_answer_idx)  # Convert to A, B, C, etc.
                else:
                    # If not found, default to the first option
                    correct_answer = "A"
            else:
                # If no answer field, default to the first option
                correct_answer = "A"
        except Exception as e:
            print(f"Error finding correct answer for question {idx}: {str(e)}")
            correct_answer = "A"  # Default to first option
        
        formatted_options = format_options(options)
        
        api_calls_count = 0
        
        # Initialize agent contexts with the question
        options_str = "\n".join([f"{opt['option_id']}. {opt['option_text']}" for opt in formatted_options])
        initial_prompt = f"""Can you solve the following medical question? 

{question}

Options:
{options_str}

Explain your reasoning. Your final answer should be a single letter corresponding to one of the provided options, in the form \\boxed{{A}} or \\boxed{{B}} etc., at the end of your response."""
        
        agent_contexts = [[{"role": "user", "content": initial_prompt}] for _ in range(args.agents)]
        
        for round in range(args.rounds):
            for i, agent_context in enumerate(agent_contexts):
                if round != 0:
                    agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]
                    message = construct_message(agent_contexts_other, question, formatted_options, 2*round - 1, use_counterfactual=(round==args.rounds-1))
                    agent_context.append(message)
                
                completion = get_model_completion(args.model_type, args.model_name, agent_context)
                assistant_message = construct_assistant_message(completion, args.model_type)
                agent_context.append(assistant_message)
        
        generated_description[question] = (agent_contexts, correct_answer)
        api_calls_per_problem[question] = api_calls_count
    
    # Save results to a JSON file
    # Clean the model name to create a valid filename
    safe_model_name = args.model_name.replace('/', '_').replace('-', '_')
    
    # Create the output directory if it doesn't exist
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(output_dir, f"results_medqa_{args.agents}_{args.rounds}_{args.model_type}_{safe_model_name}_{len(selected_indices)}q.json")
    
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
            last_message = agent_context[-1]["content"]
            # Look for boxed answers like \boxed{A}
            import re
            matches = re.findall(r'\\boxed\{([A-E])\}', last_message)
            if matches and matches[-1] == correct_answer:
                correct_count += 1
    
    accuracy = correct_count / total_predictions if total_predictions > 0 else 0
    print(f"Overall accuracy: {accuracy:.2%} ({correct_count}/{total_predictions})")
    
    # Print summary of API calls
    total_calls = sum(api_calls_per_problem.values())
    avg_calls = total_calls / len(api_calls_per_problem)
    print(f"Total API calls: {total_calls}")
    print(f"Average API calls per problem: {avg_calls:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multi-agent debate for MedQA")
    parser.add_argument("--model_type", type=str, choices=["openai", "anthropic", "mistral", "vllm"], default="openai", help="Type of model to use (openai, anthropic, mistral, or vllm)")
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo", help="Name of the model to use (e.g., gpt-4 for OpenAI, claude-3-sonnet-20240229 for Anthropic, mistral-large-latest for Mistral, or NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO for vLLM)")
    parser.add_argument("--vllm_url", type=str, default="http://localhost:8000/v1", help="URL for the vLLM OpenAI-compatible server")
    parser.add_argument("--vllm_api_key", type=str, default="dummy-key", help="API key for the vLLM server (can be any string)")
    parser.add_argument("--agents", type=int, default=3, help="Number of agents")
    parser.add_argument("--rounds", type=int, default=2, help="Number of rounds")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--num_of_questions", type=int, default=5, help="Number of questions to process (default: 5 for testing, use 150 for full evaluation)")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to use (train, validation, or test)")
    parser.add_argument("--full_evaluation", action="store_true", help="Run on 150 questions instead of 5")
    
    args = parser.parse_args()
    
    # Override num_of_questions if full_evaluation flag is set
    if args.full_evaluation:
        args.num_of_questions = 150
        
    main(args)