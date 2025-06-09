import os
import json
import time
import random
import re
from typing import List, Dict, Any, Optional
from tqdm import tqdm

# LangChain Imports
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI

from langchain_anthropic import ChatAnthropic
from langchain_mistralai import ChatMistralAI

# HuggingFace Datasets
from datasets import load_dataset

# Tools implementation
def llm_math(query: str) -> str:
    """Tool for performing mathematical calculations."""
    try:
        expression = query.strip()
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Error in calculation: {str(e)}"

def wikipedia_search(query: str) -> str:
    """Tool for searching Wikipedia for information."""
    try:
        return f"Wikipedia results for: {query}\n[This is a mock response that would contain actual information from Wikipedia in a real implementation]"
    except Exception as e:
        return f"Error searching Wikipedia: {str(e)}"

# Load datasets
def load_legalbench_dataset(config_name="privacy_policy_qa", num_samples=150):
    """Load samples from the LegalBench dataset."""
    print(f"Loading LegalBench dataset with config: {config_name}")
    
    try:
        dataset = load_dataset("nguha/legalbench", config_name)
        available_splits = list(dataset.keys())
        print(f"Available splits: {available_splits}")
        
        # Use all available splits until we get enough samples
        processed_data = []
        for split in available_splits:
            data = dataset[split]
            
            if len(data) > 0 and len(processed_data) == 0:
                print(f"Sample data fields in {split}: {list(data[0].keys())}")
            
            # Process items in this split
            for i, item in enumerate(data):
                # Skip if we have enough samples
                if len(processed_data) >= num_samples:
                    break
                    
                # Extract fields with flexible field names
                question = None
                for field in ['question', 'input', 'query']:
                    if field in item:
                        question = item[field]
                        break
                
                answer = None
                for field in ['answer', 'target', 'label']:
                    if field in item:
                        answer = item[field]
                        break
                
                text = None
                for field in ['text', 'document', 'context', 'clause']:
                    if field in item:
                        text = item[field]
                        break
                
                # Skip if missing critical fields
                if not question or not answer:
                    continue
                
                entry = {
                    'id': f"legalbench_{len(processed_data)}",
                    'question': question,
                    'document': text if text else "[No document provided]",
                    'answer': answer,
                    'task': config_name,
                    'options': ["Relevant", "Irrelevant"]
                }
                
                processed_data.append(entry)
            
            # If we have enough data, stop processing splits
            if len(processed_data) >= num_samples:
                break
        
        # If we don't have enough samples, just return what we have
        print(f"Loaded {len(processed_data)} examples from LegalBench")
        return processed_data
        
    except Exception as e:
        print(f"Error loading LegalBench dataset: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return []

def load_medqa_dataset(num_samples=150):
    """Load samples from the MedQA dataset."""
    print(f"Loading MedQA dataset")
    
    try:
        # Try different configurations
        for config_name in ["med_qa_en_4options_bigbio_qa", "med_qa_en_bigbio_qa", "med_qa_en_source"]:
            try:
                dataset = load_dataset("bigbio/med_qa", name=config_name)
                print(f"Successfully loaded MedQA with config: {config_name}")
                
                available_splits = list(dataset.keys())
                print(f"Available splits: {available_splits}")
                
                # Try each split until we have enough samples
                processed_data = []
                for split in ['test', 'validation', 'train']:
                    if split not in available_splits:
                        continue
                        
                    data = dataset[split]
                    
                    if len(data) > 0 and len(processed_data) == 0:
                        print(f"Sample data fields in {split}: {list(data[0].keys())}")
                    
                    for i, item in enumerate(data):
                        # Skip if we have enough samples
                        if len(processed_data) >= num_samples:
                            break
                            
                        # More flexible field checking
                        question = None
                        for field in ['question', 'query']:
                            if field in item:
                                question = item[field]
                                break
                                
                        if not question:
                            continue
                            
                        # Get choices with flexible field names
                        choices = None
                        for field in ['choices', 'options', 'answer_choices']:
                            if field in item and item[field]:
                                choices = item[field]
                                break
                                
                        if not choices:
                            continue
                            
                        # Get answer with flexible field names
                        answer = None
                        for field in ['answer', 'label', 'target']:
                            if field in item:
                                answer = item[field]
                                break
                                
                        if not answer:
                            continue
                        
                        entry = {
                            'id': item.get('id', f"medqa_{len(processed_data)}"),
                            'question': question,
                            'options': choices,
                            'answer': answer,
                            'task': 'MedQA'
                        }
                        
                        processed_data.append(entry)
                    
                    # If we have enough data, stop processing splits
                    if len(processed_data) >= num_samples:
                        break
                
                # If we have examples, return them
                if processed_data:
                    print(f"Loaded {len(processed_data)} examples from MedQA")
                    return processed_data
                
            except Exception as e:
                print(f"Error with config {config_name}: {str(e)}")
                continue
                
        print("Failed to load MedQA dataset with any configuration")
        return []
        
    except Exception as e:
        print(f"Error loading MedQA dataset: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return []

# Format questions - IMPROVED PROMPTING FOR LEGALBENCH
def format_legalbench_question(item):
    """Format a LegalBench privacy policy question - improved version."""
    prompt = f"""I need your help determining if a privacy policy clause is RELEVANT or IRRELEVANT to answering a specific question.

PRIVACY POLICY CLAUSE:
{item['document']}

QUESTION:
{item['question']}

A clause is RELEVANT if it contains information that directly helps answer the question.
A clause is IRRELEVANT if it does not provide information needed to answer the question.

Please analyze carefully and explain your reasoning.

IMPORTANT: End your response with exactly one of these two phrases:
"FINAL ANSWER: RELEVANT" or "FINAL ANSWER: IRRELEVANT"
"""
    
    return prompt

def format_medqa_question(item):
    """Format a MedQA question."""
    prompt = f"Question: {item['question']}\n\n"
    
    prompt += "Options:\n"
    options = ["A", "B", "C", "D", "E"]
    for i, option in enumerate(item['options']):
        if i < len(options):
            prompt += f"{options[i]}. {option}\n"
    
    prompt += "\nPlease select the correct answer and explain your reasoning. Make sure to clearly state 'The answer is: [letter]' at the end."
    
    return prompt

# IMPROVED ANSWER EXTRACTION FOR LEGALBENCH
def extract_legalbench_answer(response_text):
    """Enhanced extraction for LegalBench answers."""
    if not response_text:
        return None
    
    # Convert to uppercase for case-insensitive matching
    upper_response = response_text.upper()
    
    # Check for the explicit final answer format first
    if "FINAL ANSWER: RELEVANT" in upper_response:
        return "Relevant"
    if "FINAL ANSWER: IRRELEVANT" in upper_response:
        return "Irrelevant"
    
    # Other common formats
    if "THE ANSWER IS: RELEVANT" in upper_response or "ANSWER: RELEVANT" in upper_response:
        return "Relevant"
    if "THE ANSWER IS: IRRELEVANT" in upper_response or "ANSWER: IRRELEVANT" in upper_response:
        return "Irrelevant"
    
    # Check for letter-based answers (A = Relevant, B = Irrelevant)
    letter_patterns = [
        r"(?:the\s+)?(?:final\s+)?answer\s+is:?\s*([AB])",
        r"(?:the\s+)?(?:correct\s+)?answer\s+is:?\s*([AB])",
        r"therefore,?\s+(?:the\s+)?(?:correct\s+)?answer\s+is:?\s*([AB])",
    ]
    
    for pattern in letter_patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            answer_letter = match.group(1).upper()
            if answer_letter == "A":
                return "Relevant"
            elif answer_letter == "B":
                return "Irrelevant"
    
    # Last resort, check for the words relevant/irrelevant in the final sentence
    sentences = response_text.split('.')
    if sentences:
        last_sentences = ' '.join(sentences[-3:]).upper()  # Check last 3 sentences
        if "RELEVANT" in last_sentences and "IRRELEVANT" not in last_sentences:
            return "Relevant"
        if "IRRELEVANT" in last_sentences and "RELEVANT" not in last_sentences[-10:]:
            return "Irrelevant"
    
    # If no conclusive answer found
    return None

def extract_medqa_letter(response_text):
    """Extract just the letter answer from MedQA response."""
    if not response_text:
        return None
    
    normalized_response = ' '.join(response_text.split())
    
    # Specific pattern for "The answer is: D" style
    specific_pattern = r"(?:the\s+)?answer\s+is:?\s*([A-E])"
    match = re.search(specific_pattern, response_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # More comprehensive answer patterns
    answer_patterns = [
        # Direct letter extraction
        r"the answer is:?\s*([A-E])[.\s]",
        r"the correct answer is:?\s*([A-E])[.\s]",
        r"therefore,? the (?:correct )?answer is:?\s*([A-E])[.\s]",
        
        # Letter with variations
        r"selected answer[:\s]*([A-E])[.\s]",
        r"final answer[:\s]*([A-E])[.\s]",
        r"answer [is ]*choice ([A-E])[.\s]",
        r"the answer is choice ([A-E])[.\s]",
        r"answer is ([A-E])[.\s]",
        r"my answer is ([A-E])[.\s]",
        
        # Bolded or marked letters
        r"the answer is:?\s*\*\*?([A-E])\*\*?[.\s]",
        r"therefore,? the (?:correct )?answer is:?\s*\*\*?([A-E])\*\*?[.\s]",
        
        # Conclusion-based patterns
        r"conclusion:?\s*([A-E])[.\s]",
        r"therefore,?\s*([A-E])[.\s]"
    ]
    
    # Try regex patterns first
    for pattern in answer_patterns:
        match = re.search(pattern, normalized_response, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    # Fallback parsing
    # Parse lines that might contain the answer
    lines = response_text.split('\n')
    for line in lines:
        stripped_line = line.strip().upper()
        
        # Exact single letter match
        if re.match(r'^([A-E])$', stripped_line):
            return stripped_line
        
        # Variations with context
        for prefix in ['ANSWER IS:', 'THE ANSWER:', 'MY ANSWER', 'CHOICE', 'OPTION']:
            if prefix in stripped_line:
                potential_answer = stripped_line.replace(prefix, '').strip()
                if re.match(r'^[A-E]$', potential_answer):
                    return potential_answer
        
        # Claude-style reasoning patterns
        letter_match = re.search(r'\b([A-E])\.\s', line)
        if letter_match:
            return letter_match.group(1)
    
    # More aggressive parsing
    # Look for patterns like "Choice A is correct" or "Option B explains..."
    for letter in 'ABCDE':
        if f"choice {letter}" in response_text.lower() or f"option {letter}" in response_text.lower():
            return letter
    
    return None

class ReactAgent:
    """A simple React agent implementation without LangGraph."""
    
    def __init__(self, llm, tools):
        """Initialize the agent with a language model and tools."""
        self.llm = llm
        self.tools = tools
        self.tool_map = {tool.name: tool for tool in tools}
        
        self.system_prompt = """You are a helpful AI assistant solving questions. 
        
        IMPORTANT: For questions about LAW, MEDICINE, SCIENCE, MATH, HISTORY, or MEDICAL topics, you should verify 
        information using Wikipedia or calculate answers using the llm_math tool.
        
        Please analyze the question carefully and select the best answer among the options.
        If you're uncertain about factual details, use the wikipedia tool to look up relevant facts.
        If a question involves calculations, use the llm_math tool to ensure accuracy.
        
        To use a tool, explicitly state: "I need to use the [tool_name] tool with input: [query]"
        
        Explain your reasoning step by step and clearly indicate your final answer.
        Always end your response with a clear final answer in the requested format.
        """
    
    def run(self, user_input):
        """Run the agent on user input."""
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_input)
        ]
        
        max_steps = 10
        
        for step in range(max_steps):
            response = self.llm.invoke(messages)
            
            tool_pattern = r"I need to use the (\w+) tool with input:?\s*(.+?)(?=$|\.|\n)"
            match = re.search(tool_pattern, response.content, re.DOTALL)
            
            if match:
                tool_name = match.group(1)
                tool_input = match.group(2).strip()
                
                if tool_name in self.tool_map:
                    tool = self.tool_map[tool_name]
                    try:
                        tool_result = tool.invoke(tool_input)
                    except Exception as e:
                        tool_result = f"Error in {tool_name} tool: {str(e)}"
                    
                    messages.append(response)
                    messages.append(AIMessage(content=f"Tool {tool_name} returned: {tool_result}"))
                else:
                    messages.append(response)
                    messages.append(AIMessage(content=f"Error: Tool '{tool_name}' not found. Please use one of: {list(self.tool_map.keys())}"))
            else:
                messages.append(response)
                return messages
        
        messages.append(AIMessage(content="I've reached the maximum number of reasoning steps. Here's my answer based on what I've learned so far."))
        return messages

# LLM Provider Factory
def create_llm(provider: str, model: str, temperature: float = 0.0, base_url: str = None):
    """
    Create a language model based on the specified provider.
    
    :param provider: 'openai', 'anthropic', 'mistral', or 'vllm'
    :param model: Model name/ID
    :param temperature: Sampling temperature
    :param base_url: Base URL for API (used for vLLM)
    :return: Configured language model
    """
    if provider == 'openai':
        # Ensure OpenAI API key is set
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable not found")
        return ChatOpenAI(model=model, temperature=temperature)
    
    elif provider == 'anthropic':
        # Check if Anthropic is imported and API key is set
        if ChatAnthropic is None:
            raise ImportError("Anthropic integration not installed. Install with 'pip install langchain-anthropic'")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise ValueError("ANTHROPIC_API_KEY environment variable not found")
        return ChatAnthropic(model=model, temperature=temperature)
    
    elif provider == 'mistral':
        # Check if Mistral is imported and API key is set
        if ChatMistralAI is None:
            raise ImportError("Mistral integration not installed. Install with 'pip install langchain-mistralai'")
        if not os.environ.get("MISTRAL_API_KEY"):
            raise ValueError("MISTRAL_API_KEY environment variable not found")
        return ChatMistralAI(model=model, temperature=temperature)
    
    elif provider == 'vllm':
        # For vLLM served models (like Llama)
        if not base_url:
            base_url = "http://localhost:8000/v1"  # Default vLLM server URL
        
        # We can use the ChatOpenAI interface since vLLM exposes an OpenAI-compatible API
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            base_url=base_url,
            api_key="token-abc123"  # Dummy token for vLLM
        )
    
    else:
        raise ValueError(f"Unsupported provider: {provider}")

# Evaluation function - ADDED DEBUGGING FOR LEGALBENCH
def evaluate_agent(agent, dataset, format_question_fn, task_type, output_file):
    """Evaluate the agent on a dataset."""
    results = []
    correct_count = 0
    total_count = 0
    
    start_time = time.time()
    
    for i, item in enumerate(tqdm(dataset, desc=f"Evaluating {task_type}")):
        print(f"\nQuestion {i+1}/{len(dataset)} | Task: {item.get('task', 'Unknown')}")
        print(f"Q: {item['question'][:150]}...")
        if 'document' in item and item['document']:
            print(f"Policy clause: {item['document'][:150]}...")
        print(f"Correct answer: {item['answer']}")
        
        # Format the question
        formatted_question = format_question_fn(item)
        
        # Run the agent
        try:
            print("Running agent...")
            # Ensure the run method returns a valid response
            response = agent.run(formatted_question)
            
            # Handle different response types
            if isinstance(response, list):
                # If it's a list of messages, find the last AI message
                ai_messages = [m for m in response if hasattr(m, 'content')]
                final_response = ai_messages[-1].content if ai_messages else ""
            elif hasattr(response, 'content'):
                # If it's a single message object
                final_response = response.content
            elif isinstance(response, str):
                # If it's already a string
                final_response = response
            else:
                print(f"Unexpected response type: {type(response)}")
                final_response = str(response)
            
            # Debugging for LegalBench (first 5 samples)
            if task_type == 'LegalBench' and i < 5:
                print("\n==== DEBUGGING OUTPUT ====")
                print(f"FULL RESPONSE (last 500 chars):\n{final_response[-500:]}")
                
                # Test extraction
                extracted = extract_legalbench_answer(final_response)
                print(f"CORRECT ANSWER: {item['answer']}")
                print(f"EXTRACTED ANSWER: {extracted}")
                
                # Test if common patterns exist in the response
                key_phrases = [
                    "FINAL ANSWER: RELEVANT",
                    "FINAL ANSWER: IRRELEVANT",
                    "THE ANSWER IS: RELEVANT",
                    "THE ANSWER IS: IRRELEVANT",
                    "ANSWER: RELEVANT",
                    "ANSWER: IRRELEVANT"
                ]
                
                print("\nKEY PHRASE DETECTION:")
                for phrase in key_phrases:
                    if phrase in final_response.upper():
                        print(f"Found: {phrase}")
                print("==== END DEBUGGING ====\n")
            
            # Extract the answer based on task type
            if task_type == 'MedQA':
                # For MedQA, extract letter and map to full text
                letter_answer = extract_medqa_letter(final_response)
                full_text_answer = None
                
                if letter_answer:
                    # Convert letter (A, B, C, etc.) to index (0, 1, 2, etc.)
                    index = ord(letter_answer) - ord('A')
                    
                    # Make sure index is valid
                    if 0 <= index < len(item['options']):
                        full_text_answer = item['options'][index]
                
                print(f"Extracted answer: {letter_answer} -> {full_text_answer}")
                selected_answer = full_text_answer
            else:
                # For LegalBench
                selected_answer = extract_legalbench_answer(final_response)
                print(f"Extracted answer: {selected_answer}")
            
            # Check if the answer is correct
            is_correct = False
            if selected_answer is not None:
                # Ensure the correct answer is always a list
                correct_answers = item['answer'] if isinstance(item['answer'], list) else [item['answer']]
                
                # Check if the selected answer is in the list of correct answers
                is_correct = any(
                    selected_answer.lower() == correct_ans.lower() 
                    for correct_ans in correct_answers
                )
            
            # Update statistics
            if is_correct:
                correct_count += 1
                print("✓ CORRECT")
            else:
                print("✗ INCORRECT")
                print(f"Expected: {item['answer']}")
            
            total_count += 1
            
            # Store the result
            result = {
                'id': item.get('id', f"q{i}"),
                'question': item['question'],
                'correct_answer': item['answer'],
                'selected_answer': selected_answer,
                'is_correct': is_correct,
                'agent_response': final_response,
                'task': item.get('task', 'Unknown')
            }
            
            # Add letter answer for MedQA
            if task_type == 'MedQA' and letter_answer:
                result['letter_answer'] = letter_answer
            
            # Add document if available
            if 'document' in item and item['document']:
                result['document'] = item['document']
            
            results.append(result)
        
        except Exception as e:
            print(f"Error during agent execution: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Calculate overall accuracy
    accuracy = correct_count / total_count if total_count > 0 else 0
    print(f"\nOverall accuracy: {accuracy:.4f} ({correct_count}/{total_count})")
    
    # Duration
    duration = time.time() - start_time
    print(f"Total evaluation time: {duration:.2f} seconds")
    
    # Save results to file
    with open(output_file, 'w') as f:
        json.dump({
            'results': results,
            'overall_accuracy': accuracy,
            'correct_count': correct_count,
            'total_count': total_count,
            'duration': duration,
            'test_timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)
    
    print(f"Results saved to {output_file}")
    
    return accuracy

def main():
    # Parse arguments
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate React agent on datasets")
    parser.add_argument("--output_dir", type=str, default="./results", 
                        help="Output directory for results")
    parser.add_argument("--num_questions", type=int, default=150, 
                        help="Number of questions to evaluate per dataset")
    parser.add_argument("--provider", type=str, default="openai", 
                        choices=['openai', 'anthropic', 'mistral', 'vllm'],
                        help="LLM provider to use")
    parser.add_argument("--model_id", type=str, 
                        help="Model ID to use (will use provider-specific default if not specified)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Temperature for the model")
    parser.add_argument("--base_url", type=str, default=None,
                        help="Base URL for API endpoint (used for vLLM)")
    parser.add_argument("--eval_legalbench", action="store_true", 
                        help="Run LegalBench evaluation")
    parser.add_argument("--eval_medqa", action="store_true", 
                        help="Run MedQA evaluation")
    args = parser.parse_args()
    
    # If neither is specified, run both
    if not args.eval_legalbench and not args.eval_medqa:
        args.eval_legalbench = True
        args.eval_medqa = True
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set default model IDs if not specified
    if not args.model_id:
        default_models = {
            'openai': 'gpt-3.5-turbo',
            'anthropic': 'claude-3-5-sonnet-20240620',
            'mistral': 'open-mixtral-8x22b',
            'vllm': 'NousResearch/Meta-Llama-3-8B-Instruct'
        }
        args.model_id = default_models.get(args.provider, default_models['openai'])
    
    # Initialize LLM
    print(f"Initializing {args.provider.upper()} model: {args.model_id}")
    try:
        llm = create_llm(args.provider, args.model_id, args.temperature, args.base_url)
    except (ValueError, ImportError) as e:
        print(f"Error initializing LLM: {e}")
        return
    
    # Create tools
    tools = [
        Tool(
            name="llm_math",
            func=llm_math,
            description="A tool for performing mathematical calculations and solving equations."
        ),
        Tool(
            name="wikipedia",
            func=wikipedia_search,
            description="A tool for searching Wikipedia for information on a topic."
        )
    ]
    
    # Build the agent - using our simple implementation
    print("Building React agent...")
    agent = ReactAgent(llm, tools)
    
    # Evaluation results tracking
    overall_results = {}
    
    # Load and evaluate on LegalBench
    if args.eval_legalbench:
        print("\n=== Evaluating on LegalBench ===")
        legalbench_data = load_legalbench_dataset(num_samples=args.num_questions)
        print(f"Actually loaded {len(legalbench_data)} LegalBench questions")
        
        if legalbench_data:
            legalbench_output = os.path.join(args.output_dir, f"{args.provider}_{args.model_id.replace('/', '_')}_legalbench_results.json")
            legalbench_accuracy = evaluate_agent(
                agent,
                legalbench_data,
                format_legalbench_question,
                'LegalBench',
                legalbench_output
            )
            overall_results['LegalBench'] = legalbench_accuracy
    
    # Load and evaluate on MedQA
    if args.eval_medqa:
        print("\n=== Evaluating on MedQA ===")
        medqa_data = load_medqa_dataset(num_samples=args.num_questions)
        print(f"Actually loaded {len(medqa_data)} MedQA questions")
        
        if medqa_data:
            medqa_output = os.path.join(args.output_dir, f"{args.provider}_{args.model_id.replace('/', '_')}_medqa_results.json")
            medqa_accuracy = evaluate_agent(
                agent,
                medqa_data,
                format_medqa_question,
                'MedQA',
                medqa_output
            )
            overall_results['MedQA'] = medqa_accuracy
    
    # Print overall results summary
    print("\n=== Evaluation Summary ===")
    for dataset, accuracy in overall_results.items():
        print(f"{dataset} Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()