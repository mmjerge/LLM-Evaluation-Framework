from typing import Annotated, Dict, Sequence, TypedDict, List, Any, Optional
import json
import time
import datetime
import re
from tqdm import tqdm
import random
import requests
from urllib.parse import quote
import math
import numpy as np
from sympy import symbols, sympify, solve, Eq

# LangChain Imports
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import BaseTool, tool
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableConfig

# LangGraph Imports
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# OpenAI Import
from langchain_openai import ChatOpenAI

# HuggingFace Datasets Import
from datasets import load_dataset

# Add debug flag for easier troubleshooting
DEBUG = True
DEBUG_LEVEL = 1

def debug_print(message, level=1):
    """Print debug messages when DEBUG is enabled with different verbosity levels.
    Level 1: Essential info (LLM counts, tool usage)
    Level 2: Process details (node activations, routing)
    Level 3: Full details (model responses, token usage)
    """
    if DEBUG and level <= DEBUG_LEVEL:
        print(f"[DEBUG] {message}")

class SimpleCallbackHandler(BaseCallbackHandler):
    """A simplified callback handler that is compatible with OpenAI LangChain integration."""
    
    def __init__(self):
        """Initialize callback handler with storage for tracking calls."""
        self.llm_calls = []
        self.tool_calls = []
        self.current_question_id = None
        self.start_time = None
        self.total_tokens_in = 0
        self.total_tokens_out = 0
        # Add counters for each question
        self.question_counters = {}
        debug_print("SimpleCallbackHandler initialized")
        
    def set_question_id(self, question_id: str):
        """Set the ID of the current question being processed."""
        self.current_question_id = question_id
        # Initialize counters for this question if not already present
        if question_id not in self.question_counters:
            self.question_counters[question_id] = {
                "llm_calls": 0,
                "tool_calls": 0,
                "tokens_in": 0,
                "tokens_out": 0,
                "total_duration": 0
            }
        debug_print(f"Set current question_id to: {question_id}")
    
    # Only implement the required methods without extra functionality that might cause errors
    def on_llm_start(self, serialized, prompts, **kwargs):
        debug_print("LLM call started")
        try:
            if self.start_time is None:
                self.start_time = time.time()
                
            # Increment counter for current question
            if self.current_question_id and self.current_question_id in self.question_counters:
                self.question_counters[self.current_question_id]["llm_calls"] += 1
        except Exception as e:
            debug_print(f"Error in on_llm_start: {e}")
    
    def on_llm_end(self, response, **kwargs):
        debug_print("LLM call completed")
        try:
            # Try to extract token usage information
            if hasattr(response, "llm_output") and response.llm_output:
                token_usage = response.llm_output.get("token_usage", {})
                prompt_tokens = token_usage.get("prompt_tokens", 0)
                completion_tokens = token_usage.get("completion_tokens", 0)
                
                self.total_tokens_in += prompt_tokens
                self.total_tokens_out += completion_tokens
                
                # Update token counts for the current question
                if self.current_question_id and self.current_question_id in self.question_counters:
                    self.question_counters[self.current_question_id]["tokens_in"] += prompt_tokens
                    self.question_counters[self.current_question_id]["tokens_out"] += completion_tokens
        except Exception as e:
            debug_print(f"Error in on_llm_end: {e}")
    
    def on_tool_start(self, serialized, input_str, **kwargs):
        debug_print(f"Tool call started: {serialized.get('name', 'unknown_tool')}")
        try:
            # Increment counter for current question
            if self.current_question_id and self.current_question_id in self.question_counters:
                self.question_counters[self.current_question_id]["tool_calls"] += 1
        except Exception as e:
            debug_print(f"Error in on_tool_start: {e}")
    
    def on_tool_end(self, output, **kwargs):
        debug_print("Tool call completed")
        try:
            pass  # Simplified to avoid errors
        except Exception as e:
            debug_print(f"Error in on_tool_end: {e}")
    
    def get_summary(self):
        """Generate a summary of all tracked calls."""
        end_time = time.time()
        duration = end_time - self.start_time if self.start_time else 0
        
        # Use the question-specific counters
        calls_by_question = {}
        for question_id, counters in self.question_counters.items():
            calls_by_question[question_id] = {
                "llm_calls": counters["llm_calls"],
                "tool_calls": counters["tool_calls"],
                "tokens_in": counters["tokens_in"],
                "tokens_out": counters["tokens_out"],
                "total_duration": counters["total_duration"]
            }
        
        # Calculate global totals
        total_llm_calls = sum(data["llm_calls"] for data in calls_by_question.values())
        total_tool_calls = sum(data["tool_calls"] for data in calls_by_question.values())
        
        debug_print(f"Generated summary: {total_llm_calls} LLM calls, {total_tool_calls} tool calls")
        
        return {
            "total_llm_calls": total_llm_calls,
            "total_tool_calls": total_tool_calls,
            "total_tokens_in": self.total_tokens_in,
            "total_tokens_out": self.total_tokens_out,
            "total_tokens": self.total_tokens_in + self.total_tokens_out,
            "total_duration": duration,
            "calls_per_second": total_llm_calls / duration if duration > 0 else 0,
            "calls_per_question": calls_by_question
        }
    
    def clear(self):
        """Clear all tracked data."""
        self.llm_calls = []
        self.tool_calls = []
        self.start_time = None
        self.total_tokens_in = 0
        self.total_tokens_out = 0
        self.question_counters = {}
        debug_print("Cleared callback handler data")

class AgentState(TypedDict):
    """The state of our custom React agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]  # Using the add_messages reducer
    next_step: str  # Can be "call_model", "use_tools", or "end"

def call_model(state: AgentState, config: Optional[RunnableConfig] = None) -> Dict:
    """Node that calls the LLM to get the next action."""
    debug_print("call_model node activated")

    # Prepare the messages for the model
    messages = list(state["messages"])
    
    # Add system message if it's not already included
    system_content = """You are a helpful AI assistant solving questions. 
        
    IMPORTANT: For questions about LAW, MEDICINE, SCIENCE, MATH, HISTORY, or MEDICAL topics, you should verify 
    information using Wikipedia or calculate answers using the llm_math tool.
    
    Please analyze the question carefully and select the best answer among the options.
    If you're uncertain about factual details, USE THE WIKIPEDIA TOOL to look up relevant facts.
    If a question involves calculations, USE THE LLM_MATH TOOL to ensure accuracy.
    
    To use a tool, explicitly state: "I need to use the [tool_name] tool with input: [query]"
    
    Explain your reasoning step by step and clearly indicate your final answer.
    """
    
    has_system_message = False
    for msg in messages:
        if hasattr(msg, 'type') and msg.type == 'system':
            has_system_message = True
            break
    
    if not has_system_message:
        from langchain_core.messages import SystemMessage
        messages = [SystemMessage(content=system_content)] + messages
    
    debug_print(f"Prepared {len(messages)} messages for the model")
    
    try:
        # Call the model directly
        debug_print("Invoking model with messages")
        response = model.invoke(messages)
        debug_print(f"Model response received: {type(response)}")
        
        # Determine the next step based on whether the response has tool calls
        has_tool_calls = False
        
        # Check for tool calls in various formats
        if hasattr(response, "tool_calls") and response.tool_calls:
            has_tool_calls = True
            debug_print(f"Tool calls found in response.tool_calls")
        elif isinstance(response, dict) and "tool_calls" in response and response["tool_calls"]:
            has_tool_calls = True
            debug_print(f"Tool calls found in response dict")
        elif hasattr(response, "additional_kwargs") and "tool_calls" in response.additional_kwargs:
            tool_calls = response.additional_kwargs.get("tool_calls", [])
            has_tool_calls = len(tool_calls) > 0
            debug_print(f"Tool calls found in additional_kwargs")
        
        if has_tool_calls:
            debug_print("Routing to use_tools node")
            return {
                "messages": [response],
                "next_step": "use_tools"
            }
        else:
            debug_print("No tool calls found, routing to end")
            return {
                "messages": [response],
                "next_step": "end"
            }
    except Exception as e:
        # Log any errors that occur during model invocation
        debug_print(f"Error in call_model: {str(e)}")
        import traceback
        debug_print(traceback.format_exc())
        
        # Return an error message and end the agent's execution
        from langchain_core.messages import AIMessage
        error_msg = AIMessage(content=f"Error in agent: {str(e)}")
        return {
            "messages": [error_msg],
            "next_step": "end"
        }

def use_tools(state: AgentState) -> Dict:
    """Node that uses tools based on the model's request."""
    debug_print("use_tools node activated")
    
    # Get the last message which should contain tool calls
    last_message = state["messages"][-1]
    
    debug_print(f"Processing tool calls from message type: {type(last_message)}")
    
    tool_outputs = []
    tool_calls = []
    
    # Handle different message formats to extract tool calls
    if hasattr(last_message, "tool_calls"):
        tool_calls = last_message.tool_calls
        debug_print(f"Found tool_calls attribute with {len(tool_calls)} calls")
    elif isinstance(last_message, dict) and "tool_calls" in last_message:
        tool_calls = last_message["tool_calls"]
        debug_print(f"Found tool_calls in dict with {len(tool_calls)} calls")
    elif hasattr(last_message, "additional_kwargs") and "tool_calls" in last_message.additional_kwargs:
        tool_calls = last_message.additional_kwargs["tool_calls"]
        debug_print(f"Found tool_calls in additional_kwargs with {len(tool_calls)} calls")
    else:
        debug_print(f"No tool calls found in message: {last_message}")
    
    # Execute each tool call
    for tool_call in tool_calls:
        tool_name = None
        tool_input = None
        tool_call_id = None
        
        # Extract tool information from different formats
        if isinstance(tool_call, dict):
            tool_name = tool_call.get("name")
            tool_input = tool_call.get("args")
            tool_call_id = tool_call.get("id")
        elif hasattr(tool_call, "name"):
            tool_name = tool_call.name
            tool_input = tool_call.args if hasattr(tool_call, "args") else None
            tool_call_id = tool_call.id if hasattr(tool_call, "id") else None
        
        debug_print(f"Processing tool call: {tool_name} with input: {tool_input}")
        
        # Find the tool by name
        if tool_name in tools_by_name:
            tool = tools_by_name[tool_name]
            try:
                # Simplified tool invocation without complex callbacks
                debug_print(f"Invoking tool: {tool_name}")
                tool_output = tool.invoke(tool_input)
                debug_print(f"Tool output: {tool_output}")
                
                # Create a tool message with the result
                tool_outputs.append(
                    ToolMessage(
                        content=json.dumps(tool_output) if not isinstance(tool_output, str) else tool_output,
                        name=tool_name,
                        tool_call_id=tool_call_id
                    )
                )
            except Exception as e:
                # Handle tool execution errors
                error_msg = f"Error executing tool {tool_name}: {str(e)}"
                debug_print(error_msg)
                tool_outputs.append(
                    ToolMessage(
                        content=error_msg,
                        name=tool_name,
                        tool_call_id=tool_call_id
                    )
                )
        else:
            debug_print(f"Tool {tool_name} not found in tools_by_name")
            # Handle unknown tool
            tool_outputs.append(
                ToolMessage(
                    content=f"Error: Tool '{tool_name}' not found",
                    name=tool_name if tool_name else "unknown_tool",
                    tool_call_id=tool_call_id if tool_call_id else "unknown_id"
                )
            )
    
    debug_print(f"Generated {len(tool_outputs)} tool outputs, routing back to call_model")
    return {
        "messages": tool_outputs,
        "next_step": "call_model"  # Always go back to the model after using tools
    }

@tool
def llm_math(question: str) -> str:
    """
    Answer math questions by performing calculations.
    
    Args:
        question: A string containing a mathematical question or expression
        
    Returns:
        The calculated result with steps shown
    """
    debug_print(f"llm_math tool called with question: {question}")
    
    # Clean up the input
    cleaned_question = question.strip()
    debug_print(f"Cleaned question: {cleaned_question}")
    
    try:
        # Check for specific question types
        if "solve" in cleaned_question.lower() and "equation" in cleaned_question.lower():
            return solve_equation(cleaned_question)
        elif "calculate" in cleaned_question.lower() or "compute" in cleaned_question.lower() or "evaluate" in cleaned_question.lower():
            return calculate_expression(cleaned_question)
        elif "convert" in cleaned_question.lower():
            return convert_units(cleaned_question)
        else:
            # Default to trying to extract and evaluate expressions
            return extract_and_evaluate(cleaned_question)
            
    except Exception as e:
        debug_print(f"Error in llm_math: {str(e)}")
        return f"Error calculating result: {str(e)}. Please check the format of your math question."

def solve_equation(question):
    """Solve algebraic equations."""
    debug_print(f"Attempting to solve equation: {question}")
    
    # Look for equation patterns (e.g., "2x + 3 = 7")
    equation_pattern = r'([^=]+)=([^=]+)'
    match = re.search(equation_pattern, question)
    
    if not match:
        return "Could not identify an equation in the format 'expression = expression'. Please restate the equation."
    
    left_side = match.group(1).strip()
    right_side = match.group(2).strip()
    
    debug_print(f"Parsed equation: '{left_side}' = '{right_side}'")
    
    try:
        # Define the variable (assuming x as default)
        x = symbols('x')
        
        # Parse the equation using sympy
        left_expr = sympify(left_side)
        right_expr = sympify(right_side)
        
        # Solve the equation
        solution = solve(Eq(left_expr, right_expr), x)
        
        if not solution:
            return "No solution found."
        elif len(solution) == 1:
            return f"Solving the equation {left_side} = {right_side}:\nSolution: x = {solution[0]}"
        else:
            solutions_str = ", ".join([str(sol) for sol in solution])
            return f"Solving the equation {left_side} = {right_side}:\nSolutions: x = {solutions_str}"
            
    except Exception as e:
        debug_print(f"Error solving equation: {str(e)}")
        
        # Try a simpler approach with regular expressions and basic algebra
        try:
            # Handle simple linear equations like "2x + 3 = 7"
            # Rearrange to ax + b = c format
            if 'x' in left_side and not 'x' in right_side:
                # Parse coefficients using regex
                coefficient_pattern = r'([+-]?\s*\d*)\s*x'
                constant_pattern = r'([+-]?\s*\d+)(?!\s*x)'
                
                a_match = re.search(coefficient_pattern, left_side)
                b_match = re.search(constant_pattern, left_side)
                
                a = 1  # Default coefficient
                if a_match and a_match.group(1):
                    a_str = a_match.group(1).replace(' ', '')
                    if a_str == '+': a = 1
                    elif a_str == '-': a = -1
                    else: a = float(a_str) if a_str else 1
                
                b = 0  # Default constant
                if b_match:
                    b_str = b_match.group(1).replace(' ', '')
                    b = float(b_str)
                
                c = float(right_side)  # Right side constant
                
                # Solve for x: ax + b = c => x = (c - b) / a
                x_value = (c - b) / a
                return f"Solving the equation {left_side} = {right_side}:\nSolution: x = {x_value}"
                
            return "Could not solve the equation using the simplified method. Please try a different format."
            
        except Exception as inner_e:
            debug_print(f"Error in simplified equation solving: {str(inner_e)}")
            return f"Error solving the equation: {str(e)}. Please check the format."

def calculate_expression(question):
    """Calculate the result of a mathematical expression."""
    debug_print(f"Attempting to calculate expression from: {question}")
    
    # Extract the expression from the question
    expression_pattern = r'calculate\s+(.+)$|compute\s+(.+)$|evaluate\s+(.+)$'
    match = re.search(expression_pattern, question, re.IGNORECASE)
    
    if not match:
        # If no explicit keywords found, try to extract the expression directly
        return extract_and_evaluate(question)
    
    # Get the matched group (whichever one is not None)
    expression = next(filter(None, match.groups()))
    
    debug_print(f"Extracted expression: {expression}")
    
    # Handle special functions
    if "sin" in expression or "cos" in expression or "tan" in expression or "log" in expression:
        try:
            # Replace text versions of functions with python math functions
            for func in ["sin", "cos", "tan", "log", "sqrt"]:
                expression = expression.replace(func, f"math.{func}")
            
            # Make the math module available to eval
            result = eval(expression, {"math": math, "np": np})
            return f"Calculating {expression}:\nResult: {result}"
        except Exception as e:
            debug_print(f"Error evaluating special function: {str(e)}")
            return f"Error evaluating expression with special functions: {str(e)}"
    
    # For basic arithmetic, use a safer approach
    try:
        # Remove any unsafe functions or operations
        safe_expr = re.sub(r'[^0-9+\-*/^().\s]', '', expression)
        
        # Replace ^ with ** for exponentiation
        safe_expr = safe_expr.replace('^', '**')
        
        # Calculate the result
        result = eval(safe_expr)
        return f"Calculating {expression}:\nResult: {result}"
    except Exception as e:
        debug_print(f"Error calculating expression: {str(e)}")
        return f"Error calculating the expression: {str(e)}. Please check the format."

def convert_units(question):
    """Convert between different units of measurement."""
    debug_print(f"Attempting unit conversion: {question}")
    
    # Define common conversion factors
    conversions = {
        # Length
        "meters_to_feet": 3.28084,
        "feet_to_meters": 0.3048,
        "kilometers_to_miles": 0.621371,
        "miles_to_kilometers": 1.60934,
        "centimeters_to_inches": 0.393701,
        "inches_to_centimeters": 2.54,
        
        # Weight/Mass
        "kilograms_to_pounds": 2.20462,
        "pounds_to_kilograms": 0.453592,
        "grams_to_ounces": 0.035274,
        "ounces_to_grams": 28.3495,
        
        # Volume
        "liters_to_gallons": 0.264172,
        "gallons_to_liters": 3.78541,
        "cubic_meters_to_cubic_feet": 35.3147,
        "cubic_feet_to_cubic_meters": 0.0283168,
        
        # Temperature
        "celsius_to_fahrenheit": lambda c: c * 9/5 + 32,
        "fahrenheit_to_celsius": lambda f: (f - 32) * 5/9,
        "celsius_to_kelvin": lambda c: c + 273.15,
        "kelvin_to_celsius": lambda k: k - 273.15
    }
    
    # Extract value and unit information
    value_pattern = r'(\d+\.?\d*)\s*([\w]+)'
    matches = re.findall(value_pattern, question)
    
    if len(matches) < 1:
        return "Could not identify a value and unit to convert. Please specify in format like '5 meters to feet'."
    
    # Try to determine the conversion requested
    from_unit = None
    to_unit = None
    value = None
    
    # Check for simple conversion phrases
    if "to" in question:
        parts = question.split("to")
        from_part = parts[0]
        to_part = parts[1]
        
        # Extract from value and unit
        from_match = re.search(value_pattern, from_part)
        if from_match:
            value = float(from_match.group(1))
            from_unit = from_match.group(2).lower()
        
        # Extract to unit
        to_match = re.search(r'([\w]+)', to_part)
        if to_match:
            to_unit = to_match.group(1).lower()
    
    if not (from_unit and to_unit and value is not None):
        return "Could not identify the conversion units. Please specify in format like '5 meters to feet'."
    
    debug_print(f"Conversion request: {value} {from_unit} to {to_unit}")
    
    # Check if we have this conversion
    conversion_key = f"{from_unit}_to_{to_unit}"
    if conversion_key in conversions:
        conversion_factor = conversions[conversion_key]
        if callable(conversion_factor):
            result = conversion_factor(value)
        else:
            result = value * conversion_factor
        return f"Converting {value} {from_unit} to {to_unit}:\nResult: {result} {to_unit}"
    
    # Try reverse lookup
    conversion_key = f"{to_unit}_to_{from_unit}"
    if conversion_key in conversions:
        conversion_factor = conversions[conversion_key]
        if callable(conversion_factor):
            # For functions like temperature conversion, we need the direct function
            return f"This conversion direction is not directly supported."
        else:
            result = value / conversion_factor
            return f"Converting {value} {from_unit} to {to_unit}:\nResult: {result} {to_unit}"
    
    return f"Conversion from {from_unit} to {to_unit} is not supported."

def extract_and_evaluate(question):
    """Extract mathematical expressions from the question and evaluate them."""
    debug_print(f"Attempting to extract and evaluate expressions from: {question}")
    
    # Try to identify expressions using parentheses first
    paren_expressions = re.findall(r'\(([^()]+)\)', question)
    
    if paren_expressions:
        debug_print(f"Found expressions in parentheses: {paren_expressions}")
        results = []
        for expr in paren_expressions:
            try:
                # Remove any unsafe functions or operations
                safe_expr = re.sub(r'[^0-9+\-*/^().\s]', '', expr)
                
                # Replace ^ with ** for exponentiation
                safe_expr = safe_expr.replace('^', '**')
                
                # Calculate the result
                result = eval(safe_expr)
                results.append(f"({expr}) = {result}")
            except Exception as e:
                debug_print(f"Error evaluating expression '{expr}': {str(e)}")
                results.append(f"({expr}) - Error: {str(e)}")
        
        return "Evaluating expressions:\n" + "\n".join(results)
    
    # If no parenthesized expressions, look for equation patterns
    if "=" in question and not "==" in question:
        return solve_equation(question)
    
    # Last resort: try to evaluate the entire question as an expression
    try:
        # Remove any unsafe functions or operations
        safe_expr = re.sub(r'[^0-9+\-*/^().\s]', '', question)
        
        # Replace ^ with ** for exponentiation
        safe_expr = safe_expr.replace('^', '**')
        
        if len(safe_expr.strip()) == 0:
            return "Could not identify a mathematical expression in the question."
        
        # Calculate the result
        result = eval(safe_expr)
        return f"Evaluating the expression:\nResult: {result}"
    except Exception as e:
        debug_print(f"Error evaluating entire question: {str(e)}")
        return "Could not identify a clear mathematical expression or equation to evaluate."

@tool
def wikipedia(query: str) -> str:
    """Search Wikipedia for information on a topic.
    
    Args:
        query: The topic to search for on Wikipedia
        
    Returns:
        A summary of the information found on Wikipedia
    """
    debug_print(f"wikipedia tool called with query: {query}")
    
    try:
        # URL encode the query
        encoded_query = quote(query)
        
        # First search for the page
        search_url = f"https://en.wikipedia.org/w/api.php?action=opensearch&search={encoded_query}&limit=1&namespace=0&format=json"
        search_response = requests.get(search_url)
        search_data = search_response.json()
        
        # Check if we got any results
        if len(search_data[1]) == 0:
            return f"No Wikipedia articles found for '{query}'."
        
        # Get the title of the first result
        title = search_data[1][0]
        debug_print(f"Found Wikipedia article: {title}")
        
        # Now get the summary of the page
        summary_url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exintro=true&explaintext=true&titles={quote(title)}&format=json"
        summary_response = requests.get(summary_url)
        summary_data = summary_response.json()
        
        # Extract the page ID and content
        pages = summary_data["query"]["pages"]
        page_id = list(pages.keys())[0]
        
        # Check if the page exists
        if page_id == "-1":
            return f"No detailed information found for '{query}' on Wikipedia."
        
        # Get the extract (summary)
        extract = pages[page_id].get("extract", "No summary available.")
        
        # Limit the summary to a reasonable length (1000 characters)
        if len(extract) > 1000:
            extract = extract[:997] + "..."
        
        # Get the page URL
        page_url = f"https://en.wikipedia.org/wiki/{quote(title)}"
        
        # Return the formatted result
        result = f"Wikipedia article: {title}\n\n{extract}\n\nSource: {page_url}"
        debug_print(f"wikipedia returning summary with length: {len(extract)}")
        return result
    
    except Exception as e:
        error_msg = f"Error accessing Wikipedia: {str(e)}"
        debug_print(error_msg)
        return error_msg

# Dictionary to look up tools by name
tools = [llm_math, wikipedia]
tools_by_name = {tool.name: tool for tool in tools}
debug_print(f"Registered tools: {list(tools_by_name.keys())}")

def build_react_agent(model, detailed_handler):
    """Build a custom React agent from scratch that tracks all LLM and tool calls."""
    debug_print("Building React agent")
    
    # Create the workflow graph
    workflow = StateGraph(AgentState)
    
    # Add the nodes
    workflow.add_node("call_model", call_model)
    workflow.add_node("use_tools", use_tools)
    
    # Define the edges between nodes
    workflow.add_conditional_edges(
        "call_model",
        lambda state: state["next_step"],
        {
            "use_tools": "use_tools",
            "end": END
        }
    )
    
    # Add a direct edge from use_tools to call_model
    workflow.add_edge("use_tools", "call_model")
    
    # Set the entry point
    workflow.set_entry_point("call_model")
    
    # Compile the graph
    agent = workflow.compile()
    debug_print("React agent built successfully")
    
    return agent

def load_legalbench_dataset(total_questions=150):
    """Load LegalBench privacy_policy_qa dataset with limited samples."""
    debug_print(f"Loading LegalBench dataset with {total_questions} questions")
    
    try:
        # First check what splits are available
        dataset = load_dataset("nguha/legalbench", "privacy_policy_qa")
        available_splits = list(dataset.keys())
        debug_print(f"Available splits in dataset: {available_splits}")
        
        # Use the first available split (likely 'test' or 'train')
        primary_split = available_splits[0]
        data = dataset[primary_split]
        
        debug_print(f"Using '{primary_split}' split for LegalBench")
        
        # Debug the actual fields in the dataset
        if len(data) > 0:
            debug_print(f"Available fields in LegalBench data: {list(data[0].keys())}")
        
        # Convert to list of dictionaries for easier processing
        questions = []
        for idx in range(len(data)):
            # Map to the actual field names in the dataset
            question_field = 'question' if 'question' in data[idx] else 'input'
            answer_field = 'answer' if 'answer' in data[idx] else 'target'
            text_field = 'text' if 'text' in data[idx] else 'document'
            
            # Skip if required fields are missing
            if question_field not in data[idx] or answer_field not in data[idx]:
                continue
                
            question_data = {
                'question': data[idx][question_field],
                'answer': data[idx][answer_field],
                'task': 'privacy_policy_qa',
                'category': 'LegalBench',
                'options': ["Relevant", "Irrelevant"]  # Binary classification task
            }
            
            # Add document/text (privacy policy clause) if available
            if text_field in data[idx]:
                question_data['document'] = data[idx][text_field]
            
            questions.append(question_data)
        
        # Sample questions if needed
        if len(questions) > total_questions:
            sampled_questions = random.sample(questions, total_questions)
        else:
            sampled_questions = questions
            
        debug_print(f"Loaded {len(sampled_questions)} questions from LegalBench privacy_policy_qa")
        return sampled_questions
        
    except Exception as e:
        debug_print(f"Error loading LegalBench dataset: {e}")
        import traceback
        debug_print(traceback.format_exc())
        return []

def load_medqa_dataset(total_questions=150):
    """Load MedQA dataset with limited samples from bigbio/med_qa."""
    debug_print(f"Loading MedQA dataset with {total_questions} questions")
    
    try:
        # First check what configs are available
        configs = ["med_qa_en_4options_bigbio_qa", "med_qa_en_bigbio_qa"]
        
        # Try configs in order until one works
        dataset = None
        for config in configs:
            try:
                dataset = load_dataset("bigbio/med_qa", name=config)
                debug_print(f"Successfully loaded MedQA with config: {config}")
                break
            except ValueError as ve:
                debug_print(f"Failed to load with config {config}: {ve}")
                continue
        
        if dataset is None:
            debug_print("Could not load MedQA dataset with any of the attempted configs")
            return []
            
        # Check available splits
        available_splits = list(dataset.keys())
        debug_print(f"Available splits in dataset: {available_splits}")
        
        # Prefer test split, fallback to others
        test_split = 'test' if 'test' in available_splits else available_splits[0]
        test_data = dataset[test_split]
        
        # Convert to list of dictionaries for easier processing
        questions = []
        for idx in range(len(test_data)):
            # Debug field names to help understand structure
            if idx == 0:
                debug_print(f"Sample data fields: {list(test_data[idx].keys())}")
            
            # Check if the question has the required fields - adapt to actual structure
            required_fields = ["question", "choices", "answer"]
            if not all(field in test_data[idx] for field in required_fields):
                continue
                
            # Extract options from choices
            options = test_data[idx]["choices"]
            
            # Skip if no options or invalid number of options
            if not options or len(options) < 2:
                continue
                
            # Create question data
            question_data = {
                'question': test_data[idx]['question'],
                'options': options,
                'answer': test_data[idx]['answer'],
                'category': 'MedQA',
                'id': test_data[idx]['id'] if 'id' in test_data[idx] else f"medqa_{idx}"
            }
            
            questions.append(question_data)
        
        # Sample questions if needed
        if len(questions) > total_questions:
            sampled_questions = random.sample(questions, total_questions)
        else:
            sampled_questions = questions
            
        debug_print(f"Loaded {len(sampled_questions)} questions from MedQA")
        return sampled_questions
        
    except Exception as e:
        debug_print(f"Error loading MedQA dataset: {e}")
        import traceback
        debug_print(traceback.format_exc())
        return []

def format_legalbench_question(item):
    """Format a LegalBench question for the agent."""
    prompt = f"I'll show you a legal question. Please provide your answer based on the given information.\n\n"
    prompt += f"Task type: {item['task']}\n\n"
    prompt += f"Question: {item['question']}\n\n"
    
    # Add options if present
    if item['options'] and len(item['options']) > 0:
        prompt += "Options:\n"
        options = ["A", "B", "C", "D", "E"]
        for i, option in enumerate(item['options']):
            if i < len(options):  # Limit to available option letters
                prompt += f"{options[i]}. {option}\n"
        
        # Ask to select from options
        option_letters = options[:len(item['options'])]
        prompt += f"\nPlease select the correct answer ({', '.join(option_letters)}) and explain your reasoning. Use tools if helpful."
    else:
        # Free-form answer
        prompt += "\nPlease provide your answer and explain your reasoning. Use tools if helpful."
    
    debug_print(f"Formatted LegalBench question: {prompt[:100]}...")
    return prompt

def format_medqa_question(item):
    """Format a MedQA question for the agent."""
    prompt = f"Question: {item['question']}\n\n"
    
    # Add options if present
    if 'options' in item and item['options']:
        prompt += "Options:\n"
        options = ["A", "B", "C", "D", "E"]
        for i, option in enumerate(item['options']):
            if i < len(options):  # Limit to available option letters
                prompt += f"{options[i]}. {option}\n"
        
        # Ask to select from options
        option_letters = options[:len(item['options'])]
        prompt += f"\nPlease select the correct answer ({', '.join(option_letters)}) and explain your reasoning. Use tools if helpful."
    else:
        # No options provided
        prompt += "\nPlease provide your answer and explain your reasoning. Use tools if helpful."
        
    debug_print(f"Formatted MedQA question: {prompt[:100]}...")
    return prompt

def extract_legalbench_answer(response_text, item):
    """Extract the answer from the agent's response for LegalBench questions."""
    if not response_text:
        debug_print("No response text to extract answer from")
        return None
        
    response_text = response_text.upper()
    debug_print(f"Extracting answer from response text: {response_text[:100]}...")
    
    # If the item has options, try to extract a letter answer
    if item['options'] and len(item['options']) > 0:
        # Look for direct answer statements with letters
        direct_patterns = [
            r"THE CORRECT ANSWER IS ([A-E])",
            r"THE ANSWER IS ([A-E])",
            r"ANSWER: ([A-E])",
            r"ANSWER IS ([A-E])",
            r"SELECTED ANSWER: ([A-E])",
            r"I CHOOSE ([A-E])",
            r"I SELECT ([A-E])",
            r"MY ANSWER IS ([A-E])",
            r"OPTION ([A-E]) IS CORRECT",
            r"FINAL ANSWER: ([A-E])"
        ]
        
        # Try the most direct patterns with the complete text
        for pattern in direct_patterns:
            match = re.search(pattern, response_text)
            if match:
                debug_print(f"Found direct answer match with pattern '{pattern}': {match.group(1)}")
                answer_letter = match.group(1)
                
                # Convert the letter to the index and get the corresponding option
                idx = ord(answer_letter) - ord('A')
                if idx < len(item['options']):
                    return item['options'][idx]
                else:
                    return answer_letter  # Return the letter if we can't map to an option
        
        # Try to extract the option text directly
        for i, option in enumerate(item['options']):
            option_upper = option.upper()
            # Check if the option text appears in the response with emphasis
            if f'"{option_upper}"' in response_text or f"'{option_upper}'" in response_text:
                return option
    
    # For free-form answers, try to extract the direct answer
    answer_patterns = [
        r"MY ANSWER IS[:\s]+(.+?)(?:\.|\n|$)",
        r"FINAL ANSWER[:\s]+(.+?)(?:\.|\n|$)",
        r"THE ANSWER IS[:\s]+(.+?)(?:\.|\n|$)",
        r"I CONCLUDE THAT[:\s]+(.+?)(?:\.|\n|$)"
    ]
    
    for pattern in answer_patterns:
        match = re.search(pattern, response_text)
        if match:
            answer = match.group(1).strip()
            debug_print(f"Found free-form answer: {answer}")
            return answer
    
    # If still no clear winner, look for the expected answer directly
    expected_answer = item['answer'].upper()
    if expected_answer in response_text:
        return item['answer']
    
    # If all else fails, return the last sentence as the answer
    sentences = response_text.split('.')
    if sentences:
        last_sentence = sentences[-1].strip()
        if last_sentence:
            debug_print(f"Using last sentence as answer: {last_sentence}")
            return last_sentence
    
    debug_print("Could not extract a clear answer from the response")
    return None

def extract_medqa_answer(response_text, item):
    """Extract the selected answer from the agent's response for MedQA questions."""
    if not response_text:
        debug_print("No response text to extract answer from")
        return None
        
    response_text = response_text.upper()
    debug_print(f"Extracting answer from response text: {response_text[:100]}...")
    
    # For multiple choice, try to extract letter
    if 'options' in item and item['options']:
        # Determine the maximum option letter based on number of options
        max_option = chr(64 + min(len(item['options']), 5))  # A-E
        letter_pattern = f"[A-{max_option}]"
        
        # Look for direct answer statements
        direct_patterns = [
            f"THE CORRECT ANSWER IS ({letter_pattern})",
            f"THE ANSWER IS ({letter_pattern})",
            f"ANSWER: ({letter_pattern})",
            f"ANSWER IS ({letter_pattern})",
            f"SELECTED ANSWER: ({letter_pattern})",
            f"I CHOOSE ({letter_pattern})",
            f"I SELECT ({letter_pattern})",
            f"MY ANSWER IS ({letter_pattern})",
            f"OPTION ({letter_pattern}) IS CORRECT",
            f"FINAL ANSWER: ({letter_pattern})"
        ]
        
        # Try the most direct patterns with the complete text
        for pattern in direct_patterns:
            match = re.search(pattern, response_text)
            if match:
                debug_print(f"Found direct answer match with pattern '{pattern}': {match.group(1)}")
                return match.group(1)
        
        # Look for standalone answers by line
        for line in response_text.split("\n"):
            line = line.strip()
            if re.match(f"^{letter_pattern}$", line):
                debug_print(f"Found standalone answer: {line}")
                return line
        
        # Try to extract the option text directly
        for i, option in enumerate(item['options']):
            if i >= 5:  # Only check up to option E
                break
                
            option_letter = chr(65 + i)  # A, B, C, D, E
            option_upper = option.upper()
            
            # Check if the option text appears in the response with emphasis
            if f'"{option_upper}"' in response_text or f"'{option_upper}'" in response_text:
                debug_print(f"Found option text match: {option_letter}")
                return option_letter
    
    # For free-form answers, try to extract the direct answer
    answer_patterns = [
        r"MY ANSWER IS[:\s]+(.+?)(?:\.|\n|$)",
        r"FINAL ANSWER[:\s]+(.+?)(?:\.|\n|$)",
        r"THE ANSWER IS[:\s]+(.+?)(?:\.|\n|$)",
        r"I CONCLUDE THAT[:\s]+(.+?)(?:\.|\n|$)"
    ]
    
    for pattern in answer_patterns:
        match = re.search(pattern, response_text)
        if match:
            answer = match.group(1).strip()
            debug_print(f"Found free-form answer: {answer}")
            return answer
    
    # If still no clear winner, look for the expected answer directly
    if isinstance(item['answer'], str):
        expected_answer = item['answer'].upper()
        if expected_answer in response_text:
            return item['answer']
    
    debug_print("Could not extract a clear answer from the response")
    return None

def evaluate_task(task_name, dataset, format_question_fn, extract_answer_fn, output_file, model, detailed_handler):
    """Evaluate a specific task using our custom React agent."""
    debug_print(f"Starting evaluation for {task_name} with {len(dataset)} questions")
    
    # Build our custom React agent with tracking
    agent = build_react_agent(model, detailed_handler)
    
    # Start tracking time
    detailed_handler.start_time = time.time()
    
    results = []
    correct_count = 0
    total_count = 0
    
    # Process each question
    for i, item in enumerate(tqdm(dataset, desc=f"Evaluating {task_name}")):
        try:
            # Set current question ID for tracking
            question_id = f"{task_name.lower()}_q{i+1}"
            detailed_handler.set_question_id(question_id)
            
            # Format the question
            formatted_question = format_question_fn(item)
            
            # Print question info
            print(f"\nQuestion {i+1}/{len(dataset)} | Category: {item.get('category', task_name)}")
            print(f"Q: {item['question'][:200]}...")
            if 'options' in item and item['options']:
                options_str = " | ".join([f"{chr(65+i)}. {opt[:30]}..." for i, opt in enumerate(item['options'])])
                print(f"Options: {options_str}")
            if 'task' in item:
                print(f"Task type: {item['task']}")
            print(f"Correct answer: {item['answer']}")
            print("\n---- Agent Reasoning Process: ----")
            
            # Initialize the agent state with the question
            initial_state = {
                "messages": [HumanMessage(content=formatted_question)],
                "next_step": "call_model"
            }
            
            # Set up the config with callbacks
            config = {
                "callbacks": [detailed_handler],
                "configurable": {
                    "thread_id": f"thread_{question_id}"
                }
            }
            
            # Stream the agent's execution to see the full process
            full_response = ""
            all_messages = []
            final_response = None
            
            try:
                # Run agent without streaming to get complete final state
                final_state = agent.invoke(initial_state, config=config)
                
                # Get final messages list from complete run
                if final_state and "messages" in final_state:
                    all_messages = final_state["messages"]
                    
                    # Find the last message which should be the answer
                    if all_messages:
                        final_response = all_messages[-1]
                        
                # Also run with streaming for visualization
                for step in agent.stream(initial_state, config=config):
                    # Show messages during streaming for user feedback
                    if "messages" in step and step["messages"]:
                        messages = step["messages"]
                        for msg in messages:
                            # Print message content based on message type
                            if hasattr(msg, "content"):
                                content = msg.content
                                print(f"Agent: {content[:200]}...")
                            elif isinstance(msg, dict) and "content" in msg:
                                content = msg["content"]
                                print(f"Agent: {content[:200]}...")
                            elif hasattr(msg, "tool_name") and hasattr(msg, "content"):
                                # This is a tool message
                                print(f"Tool ({msg.tool_name}): {msg.content[:200]}...")
                
                # Use the final message content as the full response
                if final_response:
                    if hasattr(final_response, "content"):
                        full_response = final_response.content
                    elif isinstance(final_response, dict) and "content" in final_response:
                        full_response = final_response["content"]
                    else:
                        full_response = str(final_response)
            except Exception as e:
                print(f"Error during agent processing: {str(e)}")
                import traceback
                print(traceback.format_exc())
            
            print("\n---- End of Agent Reasoning ----")
            
            # Extract the selected answer
            selected_answer = extract_answer_fn(full_response, item)
            print(f"Selected answer: {selected_answer}")
            
            # Calculate accuracy - for both string matching and numeric approximation
            is_correct = False
            
            # Try to handle different answer formats
            if selected_answer is not None and item['answer'] is not None:
                # Direct string comparison
                if str(selected_answer).strip() == str(item['answer']).strip():
                    is_correct = True
                
                # Try to handle numeric answers with tolerance
                elif (isinstance(selected_answer, (int, float)) or str(selected_answer).replace('.', '', 1).isdigit()) and \
                     (isinstance(item['answer'], (int, float)) or str(item['answer']).replace('.', '', 1).isdigit()):
                    try:
                        selected_num = float(selected_answer)
                        answer_num = float(item['answer'])
                        # Allow for small relative difference
                        if abs(selected_num - answer_num) / max(1, abs(answer_num)) < 0.01:
                            is_correct = True
                    except (ValueError, TypeError):
                        pass
                
                # Try ignoring case and whitespace for text answers
                elif isinstance(selected_answer, str) and isinstance(item['answer'], str):
                    if selected_answer.lower().strip() == item['answer'].lower().strip():
                        is_correct = True
                
                # For yes/no questions, check for equivalence
                elif selected_answer.lower() in ['yes', 'true', 'correct'] and item['answer'].lower() in ['yes', 'true', 'correct']:
                    is_correct = True
                elif selected_answer.lower() in ['no', 'false', 'incorrect'] and item['answer'].lower() in ['no', 'false', 'incorrect']:
                    is_correct = True
            
            if is_correct:
                correct_count += 1
                print("✓ CORRECT")
            else:
                print("✗ INCORRECT")
            
            total_count += 1
            
            # Get call tracking info directly from the callback handler
            call_data = {}
            if question_id in detailed_handler.question_counters:
                call_data = detailed_handler.question_counters[question_id]
            
            # Save the result including the full reasoning process and tracking info
            result = {
                'question': item['question'],
                'options': item.get('options', []),
                'correct_answer': item['answer'],
                'selected_answer': selected_answer,
                'is_correct': is_correct,
                'agent_response': full_response,
                'category': item.get('category', task_name),
                'model_calls': {
                    'llm_calls': call_data.get('llm_calls', 0),
                    'tool_calls': call_data.get('tool_calls', 0),
                    'tokens_in': call_data.get('tokens_in', 0),
                    'tokens_out': call_data.get('tokens_out', 0),
                    'total_duration': call_data.get('total_duration', 0)
                }
            }
            
            # Add task type for LegalBench
            if 'task' in item:
                result['task'] = item['task']
                
            results.append(result)
            
            # Print progress
            print(f"Current accuracy: {correct_count/total_count:.2f} ({correct_count}/{total_count})")
            print(f"LLM calls for this question: {call_data.get('llm_calls', 0)}")
            print(f"Tool calls for this question: {call_data.get('tool_calls', 0)}")
            print("-" * 80)
            
        except Exception as e:
            print(f"Error processing question: {e}")
            import traceback
            traceback.print_exc()
            
            result = {
                'question': item['question'],
                'options': item.get('options', []),
                'correct_answer': item['answer'],
                'selected_answer': None,
                'is_correct': False,
                'agent_response': f"ERROR: {str(e)}",
                'category': item.get('category', task_name),
                'model_calls': {
                    'llm_calls': 0,
                    'tool_calls': 0,
                    'tokens_in': 0,
                    'tokens_out': 0,
                    'total_duration': 0
                }
            }
            
            # Add task type for LegalBench
            if 'task' in item:
                result['task'] = item['task']
                
            results.append(result)
            total_count += 1
    
    # Get the complete tracking summary
    tracking_summary = detailed_handler.get_summary()
    
    # Calculate overall accuracy
    accuracy = correct_count / total_count if total_count > 0 else 0
    print(f"Overall accuracy: {accuracy:.4f} ({correct_count}/{total_count})")
    
    # Save results to file with better formatting
    with open(output_file, 'w') as f:
        json.dump({
            'results': results,
            'overall_accuracy': accuracy,
            'correct_count': correct_count,
            'total_count': total_count,
            'tracking_stats': tracking_summary,
            'metadata': {
                'model': model_id,
                'task': task_name,
                'num_questions': total_count,
                'timestamp': datetime.datetime.now().isoformat()
            }
        }, f, indent=2)
    
    print(f"Results saved to {output_file}")
    
    # Generate a summary report
    print("\nSummary Report:")
    print("=" * 80)
    print(f"Model: {model_id}")
    print(f"Task: {task_name}")
    print(f"Questions: {total_count}")
    print(f"Overall Accuracy: {accuracy:.4f} ({correct_count}/{total_count})")
    
    # Add tracking statistics to summary
    print("\nModel API Call Statistics:")
    print(f"LLM calls: {tracking_summary.get('total_llm_calls', 0)}")
    print(f"Tool calls: {tracking_summary.get('total_tool_calls', 0)}")
    print(f"Total tokens: {tracking_summary.get('total_tokens', 0)} (in: {tracking_summary.get('total_tokens_in', 0)}, out: {tracking_summary.get('total_tokens_out', 0)})")
    print(f"Average LLM calls per question: {tracking_summary.get('total_llm_calls', 0)/len(dataset):.2f}")
    print(f"Duration: {tracking_summary.get('total_duration', 0):.2f} seconds")
    print(f"Calls per second: {tracking_summary.get('calls_per_second', 0):.2f}")
    print("=" * 80)
    
    # Return the summary data for further analysis if needed
    return {
        'accuracy': accuracy,
        'tracking': tracking_summary
    }

def main():
    # Set up the model and tracking
    global model, model_id, detailed_handler, config
    
    # Parse command line arguments
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Evaluate Custom ReAct Agent on Tasks")
    parser.add_argument("--output_dir", type=str, default="./results", 
                        help="Output directory for results")
    parser.add_argument("--num_questions", type=int, default=150, 
                        help="Total number of questions to evaluate per task")
    parser.add_argument("--model_id", type=str, default="gpt-3.5-turbo", 
                        help="OpenAI model ID to use")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Temperature for the model")
    parser.add_argument("--api_key", type=str, required=False,
                        help="OpenAI API key (if not set in environment variable OPENAI_API_KEY)")
    args = parser.parse_args()
    
    # Set model ID
    model_id = args.model_id
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check for API key in environment or command line
    api_key = args.api_key
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")
        
    if not api_key:
        parser.error("OpenAI API key must be provided either via --api_key argument or OPENAI_API_KEY environment variable")
        
    # Set OpenAI API key for the library
    os.environ["OPENAI_API_KEY"] = api_key
    
    # IMPORTANT: Use the OpenAI module directly rather than LangChain
    from openai import OpenAI
    
    # Create a direct OpenAI client
    openai_client = OpenAI(api_key=api_key)
    
    # Create a custom wrapper class for the OpenAI client
    class DirectOpenAIModel:
        def __init__(self, client, model_id, temperature=0.1):
            self.client = client
            self.model_id = model_id
            self.temperature = temperature
            self.tools = []
            
        def bind_tools(self, tools_list):
            """Store tools for potential use"""
            self.tools = tools_list
            return self
            
        def invoke(self, messages):
            """Call the OpenAI API directly"""
            try:
                debug_print(f"Calling OpenAI API with model {self.model_id}")
                
                # Convert LangChain message types to OpenAI format
                openai_messages = []
                for msg in messages:
                    content = msg.content if hasattr(msg, 'content') else str(msg)
                    role = 'system'
                    if hasattr(msg, 'type'):
                        if msg.type == 'human':
                            role = 'user'
                        elif msg.type == 'ai':
                            role = 'assistant'
                    openai_messages.append({"role": role, "content": content})
                
                # Prepare the tools for OpenAI format if needed
                openai_tools = None
                if self.tools:
                    openai_tools = []
                    for tool in self.tools:
                        openai_tools.append({
                            "type": "function",
                            "function": {
                                "name": tool.name,
                                "description": tool.description,
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "input": {
                                            "type": "string",
                                            "description": "The input to the tool"
                                        }
                                    },
                                    "required": ["input"]
                                }
                            }
                        })
                
                # Make the API call
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=openai_messages,
                    temperature=self.temperature,
                    tools=openai_tools
                )
                
                # Convert the response to a LangChain-like message for compatibility
                message_content = response.choices[0].message.content or ""
                
                # Check for tool calls
                tool_calls = []
                if hasattr(response.choices[0].message, 'tool_calls') and response.choices[0].message.tool_calls:
                    for tool_call in response.choices[0].message.tool_calls:
                        if tool_call.type == 'function':
                            tool_calls.append({
                                "name": tool_call.function.name,
                                "args": tool_call.function.arguments,
                                "id": tool_call.id
                            })
                
                # Create a compatible message object
                from langchain_core.messages import AIMessage
                ai_message = AIMessage(content=message_content)
                
                # Add tool calls if present
                if tool_calls:
                    ai_message.additional_kwargs = {"tool_calls": tool_calls}
                
                debug_print(f"Received response from OpenAI API")
                return ai_message
                
            except Exception as e:
                debug_print(f"Error calling OpenAI API: {str(e)}")
                from langchain_core.messages import AIMessage
                return AIMessage(content=f"Error calling model: {str(e)}")
    
    # Create our direct model instance
    model = DirectOpenAIModel(openai_client, model_id, args.temperature)
    
    # Bind tools to the model
    model = model.bind_tools(tools)
    
    # Create the simplified callback handler (just for tracking)
    detailed_handler = SimpleCallbackHandler()
    
    # Set the default config
    config = {}
    
    # Evaluate on LegalBench
    legalbench_data = load_legalbench_dataset(args.num_questions)
    if legalbench_data:
        legalbench_output = os.path.join(args.output_dir, "legalbench_results.json")
        evaluate_task(
            "LegalBench", 
            legalbench_data, 
            format_legalbench_question, 
            extract_legalbench_answer, 
            legalbench_output,
            model,
            detailed_handler
        )
        
        # Reset the callback handler for the next task
        detailed_handler.clear()
    
    # Evaluate on MedQA
    medqa_data = load_medqa_dataset(args.num_questions)
    if medqa_data:
        medqa_output = os.path.join(args.output_dir, "medqa_results.json")
        evaluate_task(
            "MedQA", 
            medqa_data, 
            format_medqa_question, 
            extract_medqa_answer, 
            medqa_output,
            model,
            detailed_handler
        )

if __name__ == "__main__":
    main()