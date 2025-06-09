from langchain import hub
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_community.agent_toolkits.load_tools import load_tools
from langgraph.prebuilt import create_react_agent
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import load_dataset
import json
import time
import argparse
from tqdm import tqdm
import random
import torch
import re
import datetime
from typing import Dict, List, Optional, Any

class DetailedCallbackHandler(BaseCallbackHandler):
    """Custom callback handler for tracking detailed information about LLM calls
    in a React agent."""
    
    def __init__(self):
        """Initialize callback handler with storage for tracking calls."""
        self.llm_calls = []
        self.tool_calls = []
        self.current_question_id = None
        self.start_time = None
        self.total_tokens_in = 0
        self.total_tokens_out = 0
        
    def set_question_id(self, question_id: str):
        """Set the ID of the current question being processed."""
        self.current_question_id = question_id
        
    def on_llm_start(
        self, 
        serialized: Dict[str, Any], 
        prompts: List[str], 
        **kwargs: Any
    ) -> None:
        """Log when an LLM starts generating."""
        if self.start_time is None:
            self.start_time = time.time()
            
        call_info = {
            "question_id": self.current_question_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "type": "llm_call",
            "model": serialized.get("name", "unknown_model"),
            "prompt_length": sum(len(p) for p in prompts),
            "start_time": time.time()
        }
        self.llm_calls.append(call_info)
    
    def on_llm_end(
        self, 
        response, 
        **kwargs: Any
    ) -> None:
        """Log when an LLM finishes generating."""
        if not self.llm_calls:
            return
            
        call_info = self.llm_calls[-1]
        call_info["end_time"] = time.time()
        call_info["duration"] = call_info["end_time"] - call_info["start_time"]
        
        # Extract token usage information if available
        if hasattr(response, "llm_output") and response.llm_output:
            token_usage = response.llm_output.get("token_usage", {})
            call_info["token_usage"] = token_usage
            
            # Update total token counts
            self.total_tokens_in += token_usage.get("prompt_tokens", 0)
            self.total_tokens_out += token_usage.get("completion_tokens", 0)
            
    def on_tool_start(
        self, 
        serialized: Dict[str, Any], 
        input_str: str, 
        **kwargs: Any
    ) -> None:
        """Log when a tool starts being used."""
        call_info = {
            "question_id": self.current_question_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "type": "tool_call",
            "tool_name": serialized.get("name", "unknown_tool"),
            "input": input_str,
            "start_time": time.time()
        }
        self.tool_calls.append(call_info)
    
    def on_tool_end(
        self, 
        output: str, 
        **kwargs: Any
    ) -> None:
        """Log when a tool finishes being used."""
        if not self.tool_calls:
            return
            
        call_info = self.tool_calls[-1]
        call_info["end_time"] = time.time()
        call_info["duration"] = call_info["end_time"] - call_info["start_time"]
        call_info["output"] = output
        
    def on_chain_start(
        self, 
        serialized: Dict[str, Any], 
        inputs: Dict[str, Any], 
        **kwargs: Any
    ) -> None:
        """Log when a chain starts running."""
        # You could track chain executions here if needed
        pass
    
    def on_chain_end(
        self, 
        outputs: Dict[str, Any], 
        **kwargs: Any
    ) -> None:
        """Log when a chain finishes running."""
        # You could track chain completions here if needed
        pass
    
    def get_summary(self):
        """Generate a summary of all tracked calls."""
        end_time = time.time()
        duration = end_time - self.start_time if self.start_time else 0
        
        # Group calls by question ID
        calls_by_question = {}
        for call in self.llm_calls:
            question_id = call.get("question_id", "unknown")
            if question_id not in calls_by_question:
                calls_by_question[question_id] = {
                    "llm_calls": 0,
                    "tool_calls": 0,
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "total_duration": 0
                }
            calls_by_question[question_id]["llm_calls"] += 1
            calls_by_question[question_id]["total_duration"] += call.get("duration", 0)
            
            # Add token information if available
            token_usage = call.get("token_usage", {})
            calls_by_question[question_id]["tokens_in"] += token_usage.get("prompt_tokens", 0)
            calls_by_question[question_id]["tokens_out"] += token_usage.get("completion_tokens", 0)
        
        # Add tool calls to the summary
        for call in self.tool_calls:
            question_id = call.get("question_id", "unknown")
            if question_id in calls_by_question:
                calls_by_question[question_id]["tool_calls"] += 1
                calls_by_question[question_id]["total_duration"] += call.get("duration", 0)
        
        return {
            "total_llm_calls": len(self.llm_calls),
            "total_tool_calls": len(self.tool_calls),
            "total_tokens_in": self.total_tokens_in,
            "total_tokens_out": self.total_tokens_out,
            "total_tokens": self.total_tokens_in + self.total_tokens_out,
            "total_duration": duration,
            "calls_per_second": len(self.llm_calls) / duration if duration > 0 else 0,
            "calls_per_question": calls_by_question
        }
    
    def clear(self):
        """Clear all tracked data."""
        self.llm_calls = []
        self.tool_calls = []
        self.start_time = None
        self.total_tokens_in = 0
        self.total_tokens_out = 0

# Add this class to track model API calls
class ModelCallTracker:
    def __init__(self):
        self.total_calls = 0
        self.total_tokens_in = 0
        self.total_tokens_out = 0
        self.calls_per_question = {}
        self.start_time = None
        self.timestamps = []
    
    def start_tracking(self):
        self.start_time = datetime.datetime.now()
    
    def log_call(self, question_id, tokens_in=0, tokens_out=0):
        self.total_calls += 1
        self.total_tokens_in += tokens_in
        self.total_tokens_out += tokens_out
        
        # Log timestamp
        timestamp = datetime.datetime.now()
        self.timestamps.append(timestamp)
        
        # Track calls per question
        if question_id not in self.calls_per_question:
            self.calls_per_question[question_id] = {
                'count': 0,
                'tokens_in': 0,
                'tokens_out': 0
            }
        
        self.calls_per_question[question_id]['count'] += 1
        self.calls_per_question[question_id]['tokens_in'] += tokens_in
        self.calls_per_question[question_id]['tokens_out'] += tokens_out
    
    def get_summary(self):
        end_time = datetime.datetime.now()
        duration = (end_time - self.start_time).total_seconds() if self.start_time else 0
        
        return {
            'total_calls': self.total_calls,
            'total_tokens_in': self.total_tokens_in,
            'total_tokens_out': self.total_tokens_out,
            'total_tokens': self.total_tokens_in + self.total_tokens_out,
            'duration_seconds': duration,
            'calls_per_second': self.total_calls / duration if duration > 0 else 0,
            'calls_per_question': self.calls_per_question
        }

# Set device for loading model
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Device set to use {device}")

# # Initialize model and tokenizer from local path
# local_model_path = "/scratch/mj6ux/.cache/models/Mixtral-8x22B-Instruct-v0.1"  # Adjust to your model path
# print(f"Loading model from {local_model_path}")

# tokenizer = AutoTokenizer.from_pretrained(
#     local_model_path,
#     local_files_only=True
# )

# model = AutoModelForCausalLM.from_pretrained(
#     local_model_path,
#     device_map="auto",
#     torch_dtype=torch.float16,
#     local_files_only=True
# )

# # Fix the pad token issue if needed
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token

# # Create the pipeline with sampling enabled
# text_pipe = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     max_new_tokens=512,
#     do_sample=True,
#     temperature=0.1,
#     top_p=0.9,
#     return_full_text=False
# )

model_id = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map=device,
    torch_dtype=torch.float16
)

# Fix the pad token issue
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    
# Update the pipeline creation to fix the warnings
text_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    do_sample=True,  
    temperature=0.1,
    top_p=0.9,
    return_full_text=False
)

# Create a wrapped version of your text generation pipeline
class TrackedPipeline:
    def __init__(self, pipeline, tracker):
        self.pipeline = pipeline
        self.tracker = tracker
        self.current_question_id = 'unknown'
        
        # Directly expose the model attribute
        self.model = pipeline.model
        
        # Directly expose the tokenizer
        self.tokenizer = pipeline.tokenizer
        
    def __call__(self, *args, **kwargs):
        # Add debug print statement
        print(f"TrackedPipeline called with question_id: {self.current_question_id}")
        try:
            # Get the prompt in a safe way
            prompt = kwargs.get('prompt', args[0] if args else "")
            
            # Handle different input types safely
            if isinstance(prompt, (list, tuple)):
                # If it's a list/tuple, convert to string for token counting
                prompt_str = str(prompt)
            else:
                prompt_str = prompt
                
            # Safely estimate input tokens
            try:
                input_tokens = len(self.pipeline.tokenizer.encode(prompt_str))
            except Exception:
                # Fallback if encoding fails
                input_tokens = len(str(prompt_str)) // 4  # Rough estimate
            
            # Call the original pipeline
            result = self.pipeline(*args, **kwargs)
            
            # Safely estimate output tokens
            try:
                if isinstance(result, list) and result and 'generated_text' in result[0]:
                    output_text = result[0]['generated_text']
                else:
                    output_text = str(result)
                    
                output_tokens = len(self.pipeline.tokenizer.encode(output_text))
            except Exception:
                # Fallback if encoding fails
                output_tokens = len(str(result)) // 4  # Rough estimate
            
            # Log the call
            self.tracker.log_call(
                question_id=self.current_question_id,
                tokens_in=input_tokens,
                tokens_out=output_tokens
            )
            
            return result
            
        except Exception as e:
            # Log the error but still pass through the call
            print(f"Error in TrackedPipeline: {str(e)}")
            return self.pipeline(*args, **kwargs)
    
    def __getattr__(self, name):
        # Forward any other attribute access to the underlying pipeline
        return getattr(self.pipeline, name)
    
    def set_question_id(self, question_id):
        self.current_question_id = question_id

# Create a tracker instance for token counting
tracker = ModelCallTracker()
detailed_handler = DetailedCallbackHandler()

# Create the tracked pipeline
tracked_pipe = TrackedPipeline(text_pipe, tracker)
llm = HuggingFacePipeline(pipeline=tracked_pipe)
chat_model = ChatHuggingFace(llm=llm)

tools = load_tools(
    ["llm-math", "wikipedia"], 
    llm=llm
)

REACT_PROMPT = """You are a helpful AI assistant with reasoning abilities.
When faced with a question or task, use a step-by-step approach to solve it.

{tools}

Use the following format:
Thought: Think about how to solve the problem.
Action: The action to take, should be one of [{tool_names}]
Action Input: The input to the action
Observation: The result of the action
... (repeat Thought/Action/Observation as needed)
Thought: I know the final answer.
Final Answer: The final answer to the original input question.

Question: {input}

{agent_scratchpad}
"""

# Create the LangGraph React agent
agent = create_react_agent(
    chat_model,
    tools,
    prompt=REACT_PROMPT
)

# Function to load MMLU dataset from Hugging Face - just 10 questions
def load_mmlu_dataset(total_questions=10, split="test"):
    """Load MMLU dataset from Hugging Face datasets with limited samples."""
    dataset = load_dataset("cais/mmlu", "all")
    test_data = dataset[split]
    
    # Get unique subjects
    subjects = test_data["subject"]
    unique_subjects = list(set(subjects))
    
    # Sample 5 subjects max
    if len(unique_subjects) > 5:
        sampled_subjects = random.sample(unique_subjects, 5)
    else:
        sampled_subjects = unique_subjects
    
    # Create a pool of candidate questions
    question_pool = []
    for subject in sampled_subjects:
        # Get indices for this subject
        indices = [i for i, s in enumerate(subjects) if s == subject]
        
        # Sample 2 questions per subject
        if indices:
            if len(indices) > 2:
                indices = random.sample(indices, 2)
            
            # Extract questions for this subject
            for idx in indices:
                options = [
                    test_data["choices"][idx][0],
                    test_data["choices"][idx][1],
                    test_data["choices"][idx][2],
                    test_data["choices"][idx][3]
                ]
                
                question_pool.append({
                    'question': test_data["question"][idx],
                    'options': options,
                    'answer': ["A", "B", "C", "D"][test_data["answer"][idx]],
                    'category': subject
                })
    
    # Sample the final set of questions
    if len(question_pool) > total_questions:
        return random.sample(question_pool, total_questions)
    else:
        return question_pool

# Format a question for the agent
def format_question(item):
    prompt = f"Question: {item['question']}\n\n"
    prompt += "Options:\n"
    options = ["A", "B", "C", "D"]
    for i, option in enumerate(item['options']):
        prompt += f"{options[i]}. {option}\n"
    prompt += "\nPlease select the correct answer (A, B, C, or D) and explain your reasoning. Use tools if helpful."
    return prompt

# Function to print streaming output
def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            try:
                if hasattr(message, "pretty_print"):
                    message.pretty_print()
                else:
                    print(message.content)
            except:
                print(message)

# Improved function to extract answer from response
def extract_answer(response_text):
    """Extract the selected answer (A, B, C, or D) from the agent's response with better accuracy."""
    if not response_text:
        return None
        
    response_text = response_text.upper()
    
    # Look for direct answer statements (most reliable)
    direct_patterns = [
        r"THE CORRECT ANSWER IS ([ABCD])",
        r"THE ANSWER IS ([ABCD])",
        r"ANSWER: ([ABCD])",
        r"SELECTED ANSWER: ([ABCD])",
        r"I CHOOSE ([ABCD])",
        r"MY ANSWER IS ([ABCD])",
        r"OPTION ([ABCD]) IS CORRECT"
    ]
    
    for pattern in direct_patterns:
        match = re.search(pattern, response_text)
        if match:
            return match.group(1)
    
    # Look for standalone answers by line
    for line in response_text.split("\n"):
        line = line.strip()
        if line in ["A", "B", "C", "D"]:
            return line
        
        # Check for line starts like "A.", "A)", etc.
        for letter in ["A", "B", "C", "D"]:
            if line.startswith(f"{letter}.") or line.startswith(f"{letter})"):
                return letter
    
    # Word and phrase frequency analysis (last resort)
    option_scores = {"A": 0, "B": 0, "C": 0, "D": 0}
    
    # Check for phrases indicating correctness near option mentions
    correct_phrases = [
        "IS CORRECT", "IS THE ANSWER", "IS TRUE", "IS RIGHT",
        "CORRECT OPTION", "RIGHT OPTION", "BEST OPTION"
    ]
    
    for option in ["A", "B", "C", "D"]:
        # Check for standalone option mentions
        option_scores[option] += response_text.count(f" {option} ")
        option_scores[option] += response_text.count(f" {option}.")
        option_scores[option] += response_text.count(f" {option},")
        option_scores[option] += response_text.count(f" {option})")
        
        # Check for correct phrases near options
        for phrase in correct_phrases:
            option_scores[option] += response_text.count(f"{option} {phrase}") * 5
            option_scores[option] += response_text.count(f"OPTION {option} {phrase}") * 5
    
    # Get the option with the highest score
    max_score = max(option_scores.values())
    if max_score > 0:
        max_options = [k for k, v in option_scores.items() if v == max_score]
        if len(max_options) == 1:
            return max_options[0]
    
    # If no clear winner, return None
    return None
            
def evaluate_mmlu(output_file, total_questions=10, split="test"):
    # Create our detailed callback handler
    detailed_handler = DetailedCallbackHandler()
    
    # Start tracking
    detailed_handler.start_time = time.time()
    tracker.start_tracking()
    
    # Load the dataset
    dataset = load_mmlu_dataset(total_questions, split)
    print(f"Loaded {len(dataset)} questions from {len(set(item['category'] for item in dataset))} categories")
    
    results = []
    correct_count = 0
    total_count = 0
    
    # Process each question
    for i, item in enumerate(tqdm(dataset, desc="Evaluating")):
        try:
            # Set current question ID for tracking
            question_id = f"q{i+1}_{item['category']}"
            tracked_pipe.set_question_id(question_id)
            detailed_handler.set_question_id(question_id)
            
            # Format the question
            formatted_question = format_question(item)
            
            # Print question info
            print(f"\nQuestion {i+1}/{len(dataset)} | Category: {item['category']}")
            print(f"Q: {item['question']}")
            print(f"Options: A. {item['options'][0]} | B. {item['options'][1]} | C. {item['options'][2]} | D. {item['options'][3]}")
            print(f"Correct answer: {item['answer']}")
            print("\n---- Agent Reasoning Process: ----")
            
            # Stream the agent's reasoning process with our callback
            inputs = {"messages": [HumanMessage(content=formatted_question)]}
            full_response = ""
            stream_chunks = []
            
            # Use the stream mode with our callback handler
            for chunk in agent.stream(
                inputs, 
                stream_mode="values",
                config={"callbacks": [detailed_handler]}
            ):
                stream_chunks.append(chunk)
                message = chunk["messages"][-1]
                if hasattr(message, "content"):
                    full_response = message.content
                    # Print incrementally to see the process
                    print(message.content, end="", flush=True)
            
            print("\n---- End of Agent Reasoning ----")
            
            # Extract the selected answer
            selected_answer = extract_answer(full_response)
            print(f"Selected answer: {selected_answer}")
            
            # Calculate accuracy
            is_correct = selected_answer == item['answer']
            
            if is_correct:
                correct_count += 1
                print("✓ CORRECT")
            else:
                print("✗ INCORRECT")
            
            total_count += 1
            
            # Get call tracking info from both trackers
            detailed_call_data = {}
            if question_id in detailed_handler.get_summary().get("calls_per_question", {}):
                detailed_call_data = detailed_handler.get_summary()["calls_per_question"][question_id]
            
            tracker_call_data = tracker.calls_per_question.get(question_id, {})
            
            # Save the result including the full reasoning process and tracking info
            result = {
                'question': item['question'],
                'options': item['options'],
                'correct_answer': item['answer'],
                'selected_answer': selected_answer,
                'is_correct': is_correct,
                'agent_response': full_response,
                'category': item['category'],
                'model_calls': {
                    'tracked_pipeline_calls': tracker_call_data.get('count', 0),
                    'llm_calls': detailed_call_data.get('llm_calls', 0),
                    'tool_calls': detailed_call_data.get('tool_calls', 0),
                    'tokens_in': detailed_call_data.get('tokens_in', 0) or tracker_call_data.get('tokens_in', 0),
                    'tokens_out': detailed_call_data.get('tokens_out', 0) or tracker_call_data.get('tokens_out', 0),
                    'total_duration': detailed_call_data.get('total_duration', 0)
                }
            }
            results.append(result)
            
            # Print progress
            print(f"Current accuracy: {correct_count/total_count:.2f} ({correct_count}/{total_count})")
            print(f"Model pipeline calls for this question: {tracker_call_data.get('count', 0)}")
            print(f"LLM calls from callback for this question: {detailed_call_data.get('llm_calls', 0)}")
            print(f"Tool calls for this question: {detailed_call_data.get('tool_calls', 0)}")
            print("-" * 80)
            
        except Exception as e:
            print(f"Error processing question: {e}")
            import traceback
            traceback.print_exc()
            
            result = {
                'question': item['question'],
                'options': item['options'],
                'correct_answer': item['answer'],
                'selected_answer': None,
                'is_correct': False,
                'agent_response': f"ERROR: {str(e)}",
                'category': item['category'],
                'model_calls': {
                    'tracked_pipeline_calls': 0,
                    'llm_calls': 0,
                    'tool_calls': 0,
                    'tokens_in': 0,
                    'tokens_out': 0,
                    'total_duration': 0
                }
            }
            results.append(result)
            total_count += 1
    
    # Get the complete tracking summaries
    detailed_tracking_summary = detailed_handler.get_summary()
    pipeline_tracking_summary = tracker.get_summary()
    
    # Combine tracking data
    tracking_summary = {
        'detailed_tracking': detailed_tracking_summary,
        'pipeline_tracking': pipeline_tracking_summary,
        'comparison': {
            'detailed_llm_calls': detailed_tracking_summary.get('total_llm_calls', 0),
            'pipeline_calls': pipeline_tracking_summary.get('total_calls', 0),
            'total_tool_calls': detailed_tracking_summary.get('total_tool_calls', 0)
        }
    }
    
    # Calculate overall accuracy
    accuracy = correct_count / total_count if total_count > 0 else 0
    print(f"Overall accuracy: {accuracy:.4f} ({correct_count}/{total_count})")
    
    # Calculate per-category accuracy
    category_results = {}
    for item in results:
        category = item['category']
        if category not in category_results:
            category_results[category] = {
                'correct': 0, 
                'total': 0, 
                'pipeline_calls': 0, 
                'llm_calls': 0, 
                'tool_calls': 0, 
                'tokens': 0
            }
        
        category_results[category]['total'] += 1
        if item['is_correct']:
            category_results[category]['correct'] += 1
        
        # Add tracking data by category
        model_calls = item.get('model_calls', {})
        category_results[category]['pipeline_calls'] += model_calls.get('tracked_pipeline_calls', 0)
        category_results[category]['llm_calls'] += model_calls.get('llm_calls', 0)
        category_results[category]['tool_calls'] += model_calls.get('tool_calls', 0)
        category_results[category]['tokens'] += model_calls.get('tokens_in', 0) + model_calls.get('tokens_out', 0)
    
    print("\nPer-category results:")
    category_accuracy = {}
    for category, counts in category_results.items():
        cat_accuracy = counts['correct'] / counts['total'] if counts['total'] > 0 else 0
        category_accuracy[category] = cat_accuracy
        print(f"{category}: {cat_accuracy:.4f} ({counts['correct']}/{counts['total']}) - Pipeline calls: {counts['pipeline_calls']}, LLM calls: {counts['llm_calls']}")
    
    # Sort categories by accuracy for better readability
    sorted_categories = sorted(category_accuracy.items(), key=lambda x: x[1], reverse=True)
    
    # Save results to file with better formatting
    with open(output_file, 'w') as f:
        json.dump({
            'results': results,
            'overall_accuracy': accuracy,
            'correct_count': correct_count,
            'total_count': total_count,
            'category_results': {
                cat: {
                    'accuracy': res['correct']/res['total'] if res['total'] > 0 else 0, 
                    'correct': res['correct'], 
                    'total': res['total'],
                    'pipeline_calls': res['pipeline_calls'],
                    'llm_calls': res['llm_calls'],
                    'tool_calls': res['tool_calls'],
                    'tokens': res['tokens'],
                    'avg_pipeline_calls': res['pipeline_calls']/res['total'] if res['total'] > 0 else 0,
                    'avg_llm_calls': res['llm_calls']/res['total'] if res['total'] > 0 else 0
                }
                for cat, res in category_results.items()
            },
            'sorted_categories': [{"category": cat, "accuracy": acc} for cat, acc in sorted_categories],
            'tracking_stats': tracking_summary,
            'metadata': {
                'model': model_id,
                'split': split,
                'num_questions': total_questions,
                'timestamp': datetime.datetime.now().isoformat()
            }
        }, f, indent=2)
    
    print(f"Results saved to {output_file}")
    
    # Generate a summary report
    print("\nSummary Report:")
    print("=" * 80)
    print(f"Model: {model_id}")
    print(f"Questions: {total_count}")
    print(f"Overall Accuracy: {accuracy:.4f} ({correct_count}/{total_count})")
    
    # Add tracking statistics to summary
    print("\nModel API Call Statistics:")
    print(f"Pipeline API calls: {pipeline_tracking_summary['total_calls']}")
    print(f"LLM calls (from callback): {detailed_tracking_summary.get('total_llm_calls', 0)}")
    print(f"Tool calls: {detailed_tracking_summary.get('total_tool_calls', 0)}")
    print(f"Total tokens: {pipeline_tracking_summary['total_tokens']} (in: {pipeline_tracking_summary['total_tokens_in']}, out: {pipeline_tracking_summary['total_tokens_out']})")
    print(f"Average pipeline calls per question: {pipeline_tracking_summary['total_calls']/len(dataset):.2f}")
    print(f"Average LLM calls per question: {detailed_tracking_summary.get('total_llm_calls', 0)/len(dataset):.2f}")
    print(f"Duration: {pipeline_tracking_summary['duration_seconds']:.2f} seconds")
    print(f"Calls per second: {pipeline_tracking_summary['calls_per_second']:.2f}")
    
    print("\nTop Categories:")
    for cat, acc in sorted_categories[:min(3, len(sorted_categories))]:
        count = category_results[cat]
        print(f"  {cat}: {acc:.4f} ({count['correct']}/{count['total']}) - Avg pipeline calls: {count['pipeline_calls']/count['total']:.2f}, Avg LLM calls: {count['llm_calls']/count['total']:.2f}")
    
    print("\nBottom Categories:")
    for cat, acc in sorted_categories[-min(3, len(sorted_categories)):]:
        count = category_results[cat]
        print(f"  {cat}: {acc:.4f} ({count['correct']}/{count['total']}) - Avg pipeline calls: {count['pipeline_calls']/count['total']:.2f}, Avg LLM calls: {count['llm_calls']/count['total']:.2f}")
    print("=" * 80)
    
    # Return the summary data for further analysis if needed
    return {
        'accuracy': accuracy,
        'tracking': tracking_summary,
        'categories': category_results
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LangGraph React agent on MMLU")
    parser.add_argument("--output_file", type=str, default="mmlu_results.json", help="Output file for results")
    parser.add_argument("--num_questions", type=int, default=10, help="Total number of questions to evaluate")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to use")
    parser.add_argument("--model_path", type=str, default="/scratch/mj6ux/.cache/models/Mixtral-8x22B-Instruct-v0.1", 
                        help="Path to locally downloaded model")
    args = parser.parse_args()
    
    # Update model path if provided
    if args.model_path:
        local_model_path = args.model_path
    
    evaluate_mmlu(args.output_file, args.num_questions, args.split)