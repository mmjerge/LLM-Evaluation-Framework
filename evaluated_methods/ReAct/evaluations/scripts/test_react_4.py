from langchain_mistralai import ChatMistralAI
from langchain_core.tools import tool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from datasets import load_dataset
from langchain_core.callbacks import BaseCallbackHandler
import re
import random

# Create a custom callback handler for tracing
class ReactAgentTracer(BaseCallbackHandler):
    def __init__(self):
        super().__init__()
        self.ai_message_count = 0
        self.tool_calls_count = 0
        self.tool_usage = {}
        
    def on_chat_model_start(self, *args, **kwargs):
        print("\n[AI thinking...]")
        
    def on_llm_end(self, response, *args, **kwargs):
        self.ai_message_count += 1
        if hasattr(response, "generations"):
            message = response.generations[0][0].message
            print(f"\n[AI Message {self.ai_message_count}] {message.content}")
            
    def on_tool_start(self, serialized, input_str, **kwargs):
        tool_name = serialized.get("name", "unknown_tool")
        self.tool_calls_count += 1
        self.tool_usage[tool_name] = self.tool_usage.get(tool_name, 0) + 1
        print(f"\n[Tool Call {self.tool_calls_count}] {tool_name}({input_str})")
        
    def on_tool_end(self, output, **kwargs):
        print(f"[Tool Response] {output}")
        
    def on_chain_end(self, outputs, **kwargs):
        if "output" in outputs:
            print(f"\n[Chain Output] {outputs['output']}")

# Initialize tracer
tracer = ReactAgentTracer()

# Initialize model with appropriate temperature and max_tokens
model = ChatMistralAI(
    model="open-mixtral-8x22b", 
    temperature=0,
    max_tokens=2048
)

# Initialize counters
tool_call_count = 0
tool_usage = {}  # Track which tools are being used

# Custom calculator tool
def calculator(expression: str):
    """Evaluate a mathematical expression."""
    global tool_call_count, tool_usage
    tool_call_count += 1
    
    # Track calculator usage
    tool_usage['calculator'] = tool_usage.get('calculator', 0) + 1
    print(f"Tool call #{tool_call_count}: calculator({expression})")
    
    try:
        # Safely evaluate the expression
        math_functions = {
            "abs": abs, "round": round, "min": min, "max": max, "pow": pow
        }
        result = eval(expression, {"__builtins__": {}}, math_functions)
        return f"Result: {result}"
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"

# Set up the wikipedia tool
wikipedia = WikipediaAPIWrapper(top_k_results=2)

def wikipedia_search(query: str):
    """Search Wikipedia for information."""
    global tool_call_count, tool_usage
    tool_call_count += 1
    
    # Track wikipedia usage
    tool_usage['wikipedia'] = tool_usage.get('wikipedia', 0) + 1
    print(f"Tool call #{tool_call_count}: Wikipedia({query})")
    
    try:
        result = wikipedia.run(query)
        return result[:1000] if len(result) > 1000 else result
    except Exception as e:
        return f"Error searching Wikipedia: {str(e)}"

# Dictionary of available tools
available_tools = {
    "calculator": calculator,
    "wikipedia_search": wikipedia_search
}

# Define a very clear system prompt for using the tools
system_prompt = """You are a helpful assistant that can use tools to answer questions.

When you need to use a tool, format your response exactly like this:

I need to use a tool to answer this question.
TOOL_NAME: calculator
TOOL_INPUT: 2 + 2

After you receive the tool response, provide your final answer.

Available tools:

1. calculator: Use this tool for ALL math calculations. The input should be a mathematical expression.
   Example: calculator with input "2 + 2" or "300 / 5"

2. wikipedia_search: Use this tool to search for factual information. The input should be a search query.
   Example: wikipedia_search with input "Albert Einstein" or "Solar System"

ALWAYS use the calculator tool for ANY mathematical operations, even simple ones.
ALWAYS use the wikipedia_search tool when you need to look up factual information.
ALWAYS format your tool usage EXACTLY as shown above with TOOL_NAME: and TOOL_INPUT: on separate lines.

VERY IMPORTANT: After receiving a tool response, use that information to provide a direct, concise final answer. 
DO NOT call the same tool with the same input multiple times.
"""

def run_agent_conversation(question):
    """Run a conversation with the agent to answer a question."""
    global tool_call_count, tool_usage
    
    # Reset counters
    tool_call_count = 0
    tool_usage = {}
    
    # Initialize messages with system prompt
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=question)
    ]
    
    # Number of turns to allow (to prevent infinite loops)
    max_turns = 5
    turns = 0
    
    # Keep track of previous tool calls to avoid repetition
    previous_tool_calls = set()
    
    # Chat until we get a final answer or reach max turns
    while turns < max_turns:
        turns += 1
        
        # Get model response
        response = model.invoke(messages)
        content = response.content
        messages.append(AIMessage(content=content))
        
        # Check if the response contains a tool call
        tool_pattern = r"TOOL_NAME:\s*(\w+)\s*\nTOOL_INPUT:\s*(.*?)(?=\n\n|$)"
        tool_match = re.search(tool_pattern, content, re.DOTALL)
        
        if tool_match:
            tool_name = tool_match.group(1).strip()
            tool_input = tool_match.group(2).strip()
            
            # Create a unique identifier for this tool call
            tool_call_id = f"{tool_name}:{tool_input}"
            
            # Check if we've seen this exact tool call before (avoid loops)
            if tool_call_id in previous_tool_calls:
                # Add a message to break the loop
                messages.append(HumanMessage(content=f"You've already called this tool with this input. Please provide your final answer using the information you've already gathered."))
                continue
            
            # Add this tool call to our tracking set
            previous_tool_calls.add(tool_call_id)
            
            # Check if the tool exists
            if tool_name in available_tools:
                # Call the tool
                tool_result = available_tools[tool_name](tool_input)
                
                # Add the tool result to the messages with clear formatting
                tool_message = f"Tool Result: {tool_result}"
                messages.append(HumanMessage(content=tool_message))
            else:
                # Tool doesn't exist, inform the model
                error_message = f"Error: Tool '{tool_name}' not found. Available tools are: {', '.join(available_tools.keys())}"
                messages.append(HumanMessage(content=error_message))
        else:
            # No tool call found, assume this is the final answer
            break
    
    # Extract the final answer (last AI message)
    final_message = messages[-1].content if isinstance(messages[-1], AIMessage) else ""
    
    # Clear out the TOOL_NAME and TOOL_INPUT formatting from the final answer
    final_answer = re.sub(tool_pattern, "", final_message, flags=re.DOTALL)
    final_answer = final_answer.replace("I need to use a tool to answer this question.", "")
    final_answer = re.sub(r'\n+', '\n', final_answer).strip()  # Clean up excessive newlines
    
    # Force another turn if we hit the max turns but still have a tool call pattern
    if turns >= max_turns and re.search(tool_pattern, final_message):
        messages.append(HumanMessage(content="Please provide your final answer based on the tool results you've already received."))
        response = model.invoke(messages)
        final_answer = response.content
    
    return {
        "answer": final_answer,
        "messages": messages,
        "tool_calls": tool_call_count,
        "tool_usage": tool_usage
    }

def categorize_question(question):
    """Determine if a question requires calculator or Wikipedia."""
    # Check for math indicators
    math_indicators = ["calculate", "compute", "solve", "arithmetic", "what is", "equals", "divided by", 
                        "multiplied by", "+", "-", "*", "/", "square", "cube", "root", "%"]
    
    # Check for knowledge indicators
    knowledge_indicators = ["who", "what", "when", "where", "why", "how", "history", "define", "explain", 
                           "describe", "born", "discovered", "invented", "created"]
    
    # Simple heuristic check
    needs_calculator = any(indicator in question.lower() for indicator in math_indicators)
    needs_wikipedia = any(indicator in question.lower() for indicator in knowledge_indicators)
    
    return needs_calculator, needs_wikipedia

def run_agent(question):
    global tool_call_count, tool_usage, tracer
    
    # Reset tool counters for each run
    tool_call_count = 0
    tool_usage = {}
    
    # Reset tracer
    tracer = ReactAgentTracer()
    
    # Categorize the question
    needs_calculator, needs_wikipedia = categorize_question(question)
    
    # Add specific instructions based on question type
    if needs_calculator:
        augmented_question = f"Use the calculator tool for ALL calculations and then provide the final answer. Question: {question}"
    elif needs_wikipedia:
        augmented_question = f"Use the wikipedia_search tool to look up relevant information and then provide the final answer. Question: {question}" 
    else:
        augmented_question = question
        
    # Run the agent conversation
    result = run_agent_conversation(augmented_question)
    
    # Count AI messages from the conversation
    ai_message_count = sum(1 for msg in result["messages"] if isinstance(msg, AIMessage))
    
    # Calculate total calls
    total_calls = ai_message_count + tool_call_count
    
    # Print execution summary
    print("\n--- Execution Summary ---")
    print(f"AI Messages: {ai_message_count}")
    print(f"Tool Calls: {tool_call_count}")
    print(f"Tool Usage: {tool_usage}")
    print(f"Total Calls: {total_calls}")
    
    return {
        "answer": result["answer"],
        "ai_messages": ai_message_count,  
        "tool_calls": tool_call_count,    
        "total_calls": total_calls,
        "tool_usage": tool_usage          
    }

def evaluate_benchmark(dataset_name, dataset_subset, num_examples=5):
    # Load dataset
    print(f"\nEvaluating on {dataset_name}...")
    
    if dataset_name == "gsm8k":
        dataset = load_dataset(dataset_name, dataset_subset)["test"]
    elif dataset_name == "gsm-symbolic":
        # Fixed dataset loading for GSM-Symbolic
        dataset = load_dataset("apple/GSM-Symbolic", "p1")["test"]
    elif dataset_name == "mmlu":
        dataset = load_dataset("cais/mmlu", dataset_subset)["test"]
    
    # Sample examples
    if len(dataset) > num_examples:
        indices = random.sample(range(len(dataset)), num_examples)
        examples = [dataset[i] for i in indices]
    else:
        examples = dataset[:num_examples]
    
    # Metrics
    total_correct = 0
    total_ai_messages = 0
    total_tool_calls = 0
    total_combined_calls = 0
    
    # Tool-specific metrics
    tool_type_counts = {}
    
    # Evaluate each example
    for i, example in enumerate(examples):
        print(f"Processing example {i+1}/{len(examples)}")
        
        if dataset_name in ["gsm8k", "gsm-symbolic"]:
            question = example["question"]
            correct_answer = example["answer"]
            
            # Run agent
            result = run_agent(question)
            
            # Extract numerical answer
            answer_match = re.search(r'(\d+(\.\d+)?)', result["answer"])
            model_answer = answer_match.group(1) if answer_match else ""
            
            # Check if correct
            is_correct = False
            if model_answer:
                correct_match = re.search(r'(\d+(\.\d+)?)', correct_answer)
                correct_value = correct_match.group(1) if correct_match else ""
                is_correct = model_answer == correct_value
            
        elif dataset_name == "mmlu":
            question = example["question"] + "\n"
            for i, choice in enumerate(example["choices"]):
                question += f"{chr(65+i)}. {choice}\n"
            
            # Run agent
            result = run_agent(question)
            
            # Look for option A, B, C, D in answer
            answer_match = re.search(r'[ABCD]', result["answer"])
            model_answer = answer_match.group(0) if answer_match else ""
            
            # Check if correct (convert to index)
            correct_index = example["answer"]
            correct_letter = chr(65 + correct_index)
            is_correct = model_answer == correct_letter
        
        # Update metrics
        if is_correct:
            total_correct += 1
        total_ai_messages += result["ai_messages"]
        total_tool_calls += result["tool_calls"]
        total_combined_calls += result["total_calls"]
        
        # Update tool-specific metrics
        for tool_name, count in result["tool_usage"].items():
            tool_type_counts[tool_name] = tool_type_counts.get(tool_name, 0) + count
        
        # Print per-example results for debugging
        print(f"  AI Messages: {result['ai_messages']}, Tool Calls: {result['tool_calls']}, " 
              f"Total Calls: {result['total_calls']}, Tool Usage: {result['tool_usage']}, Correct: {is_correct}")
        
    # Calculate accuracy
    accuracy = total_correct / len(examples) if examples else 0
    
    # Print results
    print(f"\nResults for {dataset_name}:")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Total examples: {len(examples)}")
    print(f"Total correct: {total_correct}")
    print(f"Total AI messages: {total_ai_messages}")
    print(f"Total tool calls: {total_tool_calls}")
    print(f"Total combined calls: {total_combined_calls}")
    print(f"Tool usage breakdown: {tool_type_counts}")
    print(f"Average AI messages per example: {total_ai_messages/len(examples):.2f}")
    print(f"Average tool calls per example: {total_tool_calls/len(examples):.2f}")
    print(f"Average combined calls per example: {total_combined_calls/len(examples):.2f}")
    
    return {
        "accuracy": accuracy,
        "examples": len(examples),
        "correct": total_correct,
        "ai_messages": total_ai_messages,
        "tool_calls": total_tool_calls,
        "total_calls": total_combined_calls,
        "tool_usage": tool_type_counts
    }

# Set to 5 for quick debugging
debug_examples = 150

# Evaluate on benchmarks
results = {}

# GSM8K - math problems, expect calculator usage
results["gsm8k"] = evaluate_benchmark("gsm8k", "main", num_examples=debug_examples)

# GSM-Symbolic - math problems, expect calculator usage
results["gsm-symbolic"] = evaluate_benchmark("gsm-symbolic", None, num_examples=debug_examples)

# MMLU - use a history subject which should benefit from Wikipedia
results["mmlu"] = evaluate_benchmark("mmlu", "astronomy", num_examples=debug_examples)

# Print overall results
print("\nOverall Results:")
print("===============")
for benchmark, result in results.items():
    print(f"{benchmark}: {result['accuracy']:.2%} accuracy, {result['ai_messages']} AI messages, "
          f"{result['tool_calls']} tool calls, {result['total_calls']} total calls")
    print(f"  Tool usage: {result['tool_usage']}")

print("\nDebug mode complete. Set debug_examples to a higher number for a more comprehensive evaluation.")