# First we initialize the model we want to use
from langchain_mistralai import ChatMistralAI
from langchain_core.tools import tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AIMessage, SystemMessage
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

# Initialize model with appropriate temperature
model = ChatMistralAI(model="open-mixtral-8x22b", temperature=0)

# Initialize counters
tool_call_count = 0
tool_usage = {}  # Track which tools are being used

# Custom calculator tool
@tool
def calculator(expression: str):
    """Use this tool to evaluate mathematical expressions."""
    global tool_call_count, tool_usage
    tool_call_count += 1
    
    # Track calculator usage
    tool_usage['calculator'] = tool_usage.get('calculator', 0) + 1
    print(f"Tool call #{tool_call_count}: calculator({expression})")
    
    try:
        # Safely evaluate the expression
        result = eval(expression, {"__builtins__": {}}, {"abs": abs, "round": round})
        return f"Result: {result}"
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"

# Set up the wikipedia tool with a counter wrapper
wikipedia = WikipediaAPIWrapper()

class CountingWikipedia(WikipediaQueryRun):
    def _run(self, query: str) -> str:
        global tool_call_count, tool_usage
        tool_call_count += 1
        
        # Track wikipedia usage
        tool_usage['wikipedia'] = tool_usage.get('wikipedia', 0) + 1
        print(f"Tool call #{tool_call_count}: Wikipedia({query})")
        
        return super()._run(query)

# Create instance of the counting Wikipedia tool
wikipedia_tool = CountingWikipedia(api_wrapper=wikipedia)

# Set up the tools
tools = [calculator, wikipedia_tool]

# Define the graph without system prompt
graph = create_react_agent(model, tools=tools)

def run_agent(question):
    global tool_call_count, tool_usage, tracer
    
    # Reset tool counters for each run
    tool_call_count = 0
    tool_usage = {}
    
    # Reset tracer
    tracer = ReactAgentTracer()
    
    # Add a hint about tools in the question to encourage tool usage
    enhanced_question = (
        f"Use calculator for math and Wikipedia for facts. Question: {question}"
    )
    
    # Run the agent with the tracer
    inputs = {"messages": [("user", enhanced_question)]}
    result = graph.invoke(inputs, {"callbacks": [tracer]})
    
    # Count AI messages in the result for verification
    ai_message_count = 0
    for message in result["messages"]:
        if isinstance(message, AIMessage):
            ai_message_count += 1
    
    # Get the final answer
    final_message = result["messages"][-1]
    answer = final_message.content if hasattr(final_message, "content") else str(final_message)
    
    # Calculate total calls
    total_calls = ai_message_count + tool_call_count
    
    # Print comparison of counter vs tracer
    print("\n--- Execution Summary ---")
    print(f"AI Messages (result): {ai_message_count}")
    print(f"AI Messages (tracer): {tracer.ai_message_count}")
    print(f"Tool Calls: {tool_call_count}")
    print(f"Tool Usage: {tool_usage}")
    print(f"Total Calls: {total_calls}")
    
    return {
        "answer": answer,
        "ai_messages": ai_message_count,  # Using the count from result
        "tool_calls": tool_call_count,    # Using the original counter
        "total_calls": total_calls,
        "tool_usage": tool_usage          # Using the original tool usage
    }

def evaluate_benchmark(dataset_name, dataset_subset, num_examples=150):
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
        examples = dataset
    
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

# For debugging, start with a smaller number to verify counting works
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

# Once counting is verified, set this to 150 for the full evaluation
full_examples = 150
print("\nDebug mode complete. Set debug_examples to full_examples (150) to run the full evaluation.")