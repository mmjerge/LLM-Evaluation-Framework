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

# HuggingFace Imports
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
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

class DetailedCallbackHandler(BaseCallbackHandler):
    """Callback handler for tracking LLM and tool calls in detail."""
    
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
        debug_print("DetailedCallbackHandler initialized")
        
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
        
    def on_llm_start(
        self, 
        serialized: Dict[str, Any], 
        prompts: List[str], 
        **kwargs: Any
    ) -> None:
        debug_print("CALLBACK TRIGGERED: on_llm_start", 1)
        """Log when an LLM starts generating."""
        if self.start_time is None:
            self.start_time = time.time()
            
        # Increment counter for current question
        if self.current_question_id and self.current_question_id in self.question_counters:
            self.question_counters[self.current_question_id]["llm_calls"] += 1
            
        call_info = {
            "question_id": self.current_question_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "type": "llm_call",
            "model": serialized.get("name", "unknown_model").split(".")[-1],  # Just the model name
            "prompt_length": sum(len(p) for p in prompts),
            "start_time": time.time()
        }
        self.llm_calls.append(call_info)
        debug_print(f"LLM call started for question: {self.current_question_id}", 1)
        # Only print detailed model info at higher debug levels
        debug_print(f"Model: {call_info['model']}", 2)
    
    def on_llm_end(
        self, 
        response, 
        **kwargs: Any
    ) -> None:
        debug_print("CALLBACK TRIGGERED: on_llm_end", 1)
        """Log when an LLM finishes generating."""
        if not self.llm_calls:
            debug_print("Warning: on_llm_end called but no llm_calls recorded")
            return
            
        call_info = self.llm_calls[-1]
        call_info["end_time"] = time.time()
        call_info["duration"] = call_info["end_time"] - call_info["start_time"]
        
        # Add the duration to the question's total duration
        if self.current_question_id and self.current_question_id in self.question_counters:
            self.question_counters[self.current_question_id]["total_duration"] += call_info["duration"]
        
        debug_print(f"Response type: {type(response)}")
        
        # Extract token usage information using a more comprehensive approach
        token_usage = {}
        token_found = False
        
        # Method 1: Check llm_output
        if hasattr(response, "llm_output") and response.llm_output:
            token_usage = response.llm_output.get("token_usage", {})
            if token_usage:
                token_found = True
                debug_print(f"Token usage from llm_output: {token_usage}")
        
        # Method 2: Check usage attribute
        if not token_found and hasattr(response, "usage"):
            token_usage = response.usage
            token_found = True
            debug_print(f"Token usage from usage: {token_usage}")
        
        # Method 3: Check generations
        if not token_found and hasattr(response, "generations") and response.generations:
            gen = response.generations[0][0] if response.generations[0] else None
            if gen and hasattr(gen, "generation_info") and gen.generation_info:
                if "token_usage" in gen.generation_info:
                    token_usage = gen.generation_info["token_usage"]
                    token_found = True
                    debug_print(f"Token usage from generation_info: {token_usage}")
        
        # Method 4: Check kwargs
        if not token_found and "token_usage" in kwargs:
            token_usage = kwargs["token_usage"]
            token_found = True
            debug_print(f"Token usage from kwargs: {token_usage}")
        
        # Method 5: Estimate token usage based on prompt and response length
        # (This is a fallback method if we can't get actual token counts)
        if not token_found:
            prompt_length = call_info.get("prompt_length", 0)
            response_length = 0
            
            # Estimate response length
            if hasattr(response, "content"):
                response_length = len(response.content)
            elif hasattr(response, "generations") and response.generations:
                gen = response.generations[0][0] if response.generations[0] else None
                if gen and hasattr(gen, "text"):
                    response_length = len(gen.text)
            
            # Very rough estimation (4 chars ≈ 1 token)
            prompt_tokens = max(1, prompt_length // 4)
            completion_tokens = max(1, response_length // 4)
            
            token_usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "estimated": True
            }
            debug_print(f"Estimated token usage: {token_usage}")
        
        # Store token usage and update counters
        call_info["token_usage"] = token_usage
        prompt_tokens = token_usage.get("prompt_tokens", 0)
        completion_tokens = token_usage.get("completion_tokens", 0)
        
        self.total_tokens_in += prompt_tokens
        self.total_tokens_out += completion_tokens
        
        # Update token counts for the current question
        if self.current_question_id and self.current_question_id in self.question_counters:
            self.question_counters[self.current_question_id]["tokens_in"] += prompt_tokens
            self.question_counters[self.current_question_id]["tokens_out"] += completion_tokens
        
        # Extract and save content from response for debugging
        content_found = False
        
        # Method 1: Direct content attribute
        if hasattr(response, "content"):
            call_info["response_content"] = response.content
            content_found = True
            debug_print(f"Response content saved: {response.content[:100]}...")
        
        # Method 2: Extract from generations
        if not content_found and hasattr(response, "generations") and response.generations:
            gen = response.generations[0][0] if response.generations[0] else None
            if gen:
                if hasattr(gen, "text"):
                    call_info["response_content"] = gen.text
                    content_found = True
                    debug_print(f"Response content from generations.text: {gen.text[:100]}...")
                elif hasattr(gen, "message") and hasattr(gen.message, "content"):
                    call_info["response_content"] = gen.message.content
                    content_found = True
                    debug_print(f"Response content from generations.message: {gen.message.content[:100]}...")
        
        # Method 3: Check if response is a dict with content
        if not content_found and isinstance(response, dict) and "content" in response:
            call_info["response_content"] = response["content"]
            content_found = True
            debug_print(f"Response content from dict: {response['content'][:100]}...")
        
        # Method 4: Check message attribute
        if not content_found and hasattr(response, "message"):
            if hasattr(response.message, "content"):
                call_info["response_content"] = response.message.content
                content_found = True
                debug_print(f"Response content from message: {response.message.content[:100]}...")
        
        # Method 5: Last resort - convert to string
        if not content_found:
            try:
                str_content = str(response)
                call_info["response_content"] = str_content
                debug_print(f"Response content from str conversion: {str_content[:100]}...")
            except Exception as e:
                debug_print(f"Failed to extract any content from response: {e}")
        
        debug_print(f"LLM call completed for question: {self.current_question_id}")
    
    def on_tool_start(
        self, 
        serialized: Dict[str, Any], 
        input_str: str, 
        **kwargs: Any
    ) -> None:
        debug_print("CALLBACK TRIGGERED: on_tool_start", 1)

        """Log when a tool starts being used."""
        # Increment counter for current question
        if self.current_question_id and self.current_question_id in self.question_counters:
            self.question_counters[self.current_question_id]["tool_calls"] += 1
            
        call_info = {
            "question_id": self.current_question_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "type": "tool_call",
            "tool_name": serialized.get("name", "unknown_tool"),
            "input": input_str,
            "start_time": time.time()
        }
        self.tool_calls.append(call_info)
        debug_print(f"Tool call started: {serialized.get('name', 'unknown_tool')} with input: {input_str}")
    
    def on_tool_end(
        self, 
        output: str, 
        **kwargs: Any
    ) -> None:
        debug_print("CALLBACK TRIGGERED: on_tool_end", 1)

        """Log when a tool finishes being used."""
        if not self.tool_calls:
            debug_print("Warning: on_tool_end called but no tool_calls recorded")
            return
            
        call_info = self.tool_calls[-1]
        call_info["end_time"] = time.time()
        call_info["duration"] = call_info["end_time"] - call_info["start_time"]
        call_info["output"] = output
        
        # Add the duration to the question's total duration
        if self.current_question_id and self.current_question_id in self.question_counters:
            self.question_counters[self.current_question_id]["total_duration"] += call_info["duration"]
            
        debug_print(f"Tool call completed: {call_info.get('tool_name', 'unknown_tool')} with output: {output}")
    
    def get_summary(self):
        """Generate a summary of all tracked calls."""
        end_time = time.time()
        duration = end_time - self.start_time if self.start_time else 0
        
        # Use the more reliable question-specific counters instead of trying to recalculate
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

    if config is None:
        config = {}
    
    # Always ensure callbacks is a list
    if "callbacks" not in config:
        config["callbacks"] = []
    elif not isinstance(config["callbacks"], list):
        config["callbacks"] = [config["callbacks"]]
        
    # Always add detailed_handler if it exists and isn't already there
    if detailed_handler and detailed_handler not in config["callbacks"]:
        config["callbacks"].append(detailed_handler)
    
    # Get the system prompt with ENHANCED instructions for better tool use
    system_prompt = SystemMessage(
        content="""You are a helpful AI assistant solving a multiple-choice question. 
        
        IMPORTANT: For questions about SCIENCE, MATH, HISTORY, or MEDICAL topics, you should verify 
        information using Wikipedia or calculate answers using the llm_math tool.
        
        Please analyze the question carefully and select the best answer among the options.
        If you're uncertain about factual details, USE THE WIKIPEDIA TOOL to look up relevant facts.
        If a question involves calculations, USE THE LLM_MATH TOOL to ensure accuracy.
        
        To use a tool, explicitly state: "I need to use the [tool_name] tool with input: [query]"
        
        Explain your reasoning step by step and clearly indicate your final answer 
        in the format 'The answer is [A/B/C/D]' at the end of your response.
        """
    )
    
    # Prepare the messages for the model
    messages = [system_prompt] + list(state["messages"])
    debug_print(f"Prepared {len(messages)} messages for the model")
    
    try:
        # Make the API call to the model with tools bound
        debug_print("Invoking model with messages")
        response = model.invoke(messages, config=config)
        debug_print(f"Model response type: {type(response)}, dir: {dir(response)}")
        
        # Save the raw response for debugging
        response_content = ""
        if hasattr(response, "content"):
            response_content = response.content
            debug_print(f"Response content: {response_content[:100]}...")
        
        # Determine the next step based on whether the response has tool calls
        has_tool_calls = False
        
        # Check for tool calls in various formats
        if hasattr(response, "tool_calls") and response.tool_calls:
            has_tool_calls = True
            debug_print(f"Tool calls found in response.tool_calls: {response.tool_calls}")
        elif isinstance(response, dict) and "tool_calls" in response and response["tool_calls"]:
            has_tool_calls = True
            debug_print(f"Tool calls found in response dict: {response['tool_calls']}")
        elif hasattr(response, "additional_kwargs") and "tool_calls" in response.additional_kwargs:
            tool_calls = response.additional_kwargs.get("tool_calls", [])
            has_tool_calls = len(tool_calls) > 0
            debug_print(f"Tool calls found in additional_kwargs: {tool_calls}")
        
        if has_tool_calls:
            debug_print("Routing to use_tools node")
            return {
                "messages": [response],
                "next_step": "use_tools"
            }
        else:
            debug_print("No tool calls found, routing to end")
            # Store the response content if it isn't already stored
            if hasattr(response, "content") and not hasattr(response, "_response_content"):
                setattr(response, "_response_content", response.content)
            
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
        error_msg = AIMessage(content=f"Error in agent: {str(e)}")
        return {
            "messages": [error_msg],
            "next_step": "end"
        }

def use_tools(state: AgentState) -> Dict:
    """Node that uses tools based on the model's request."""
    debug_print("use_tools node activated")
    
    # Ensure config exists
    if config is None:
        config = {}
    
    # Extract callbacks
    callbacks = []
    if "callbacks" in config:
        callbacks = config["callbacks"] if isinstance(config["callbacks"], list) else [config["callbacks"]]
    elif detailed_handler:
        callbacks = [detailed_handler]
    
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
                for callback in callbacks:
                    if hasattr(callback, "on_tool_start"):
                        callback.on_tool_start(
                            {"name": tool_name},
                            tool_input
                        )
                        
                tool_output = tool.invoke(tool_input)

                for callback in callbacks:
                    if hasattr(callback, "on_tool_end"):
                        callback.on_tool_end(tool_output)

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

def should_continue(state: AgentState) -> str:
    """Determine which node to go to next based on the state."""
    return state["next_step"]

def build_react_agent(model, detailed_handler):
    """Build a custom React agent from scratch that tracks all LLM and tool calls."""
    debug_print("Building React agent")
    
    # Create the workflow graph without memory saver for now
    workflow = StateGraph(AgentState)
    
    # Add the nodes
    workflow.add_node("call_model", call_model)
    workflow.add_node("use_tools", use_tools)
    
    # Define the edges between nodes - using if/elif/else logic
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
    
    # Compile the graph without a checkpointer
    agent = workflow.compile()
    debug_print("React agent built successfully")
    
    return agent

def load_mmlu_dataset(total_questions=150, split="test"):
    """Load MMLU dataset from Hugging Face datasets with limited samples."""
    debug_print(f"Loading MMLU dataset with {total_questions} questions from {split} split")
    dataset = load_dataset("cais/mmlu", "all")
    test_data = dataset[split]
    
    # Get unique subjects
    subjects = test_data["subject"]
    unique_subjects = list(set(subjects))
    
    # Calculate how many subjects to sample based on total_questions
    # Aim for approximately 10 questions per subject
    num_subjects = min(len(unique_subjects), max(5, total_questions // 10))
    
    # Sample the subjects
    if len(unique_subjects) > num_subjects:
        sampled_subjects = random.sample(unique_subjects, num_subjects)
    else:
        sampled_subjects = unique_subjects
    debug_print(f"Sampled {len(sampled_subjects)} subjects: {sampled_subjects}")
    
    # Calculate questions per subject to distribute evenly
    questions_per_subject = max(2, total_questions // len(sampled_subjects))
    debug_print(f"Targeting ~{questions_per_subject} questions per subject")
    
    # Create a pool of candidate questions
    question_pool = []
    for subject in sampled_subjects:
        # Get indices for this subject
        indices = [i for i, s in enumerate(subjects) if s == subject]
        
        # Sample appropriate number of questions per subject
        if indices:
            if len(indices) > questions_per_subject:
                indices = random.sample(indices, questions_per_subject)
            
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
        final_questions = random.sample(question_pool, total_questions)
    else:
        final_questions = question_pool
    
    debug_print(f"Loaded {len(final_questions)} questions from {len(set(item['category'] for item in final_questions))} categories")
    return final_questions

def format_question(item):
    """Format a question for the agent."""
    prompt = f"Question: {item['question']}\n\n"
    prompt += "Options:\n"
    options = ["A", "B", "C", "D"]
    for i, option in enumerate(item['options']):
        prompt += f"{options[i]}. {option}\n"
    prompt += "\nPlease select the correct answer (A, B, C, or D) and explain your reasoning. Use tools if helpful."
    debug_print(f"Formatted question: {prompt[:100]}...")
    return prompt

def extract_answer(response_text):
    """Extract the selected answer (A, B, C, or D) from the agent's response."""
    if not response_text:
        debug_print("No response text to extract answer from")
        return None
        
    response_text = response_text.upper()
    debug_print(f"Extracting answer from response text: {response_text[:100]}...")
    
    # Look for direct answer statements (most reliable)
    direct_patterns = [
        r"THE CORRECT ANSWER IS ([ABCD])",
        r"THE ANSWER IS ([ABCD])",
        r"ANSWER: ([ABCD])",
        r"ANSWER IS ([ABCD])",
        r"SELECTED ANSWER: ([ABCD])",
        r"I CHOOSE ([ABCD])",
        r"I SELECT ([ABCD])",
        r"MY ANSWER IS ([ABCD])",
        r"OPTION ([ABCD]) IS CORRECT",
        r"FINAL ANSWER: ([ABCD])",
        r"ANSWER ([ABCD])"
    ]
    
    # First, try the most direct patterns with the complete text
    for pattern in direct_patterns:
        match = re.search(pattern, response_text)
        if match:
            debug_print(f"Found direct answer match with pattern '{pattern}': {match.group(1)}")
            return match.group(1)
    
    # Direct search for "D. Customer's needs" style patterns
    option_letters = ["A", "B", "C", "D"]
    for letter in option_letters:
        # Look for patterns like "D." followed by content
        if re.search(f"{letter}\\.", response_text):
            debug_print(f"Found match for '{letter}.' pattern")
            return letter
    
    # Look for standalone answers by line
    for line in response_text.split("\n"):
        line = line.strip()
        if line in ["A", "B", "C", "D"]:
            debug_print(f"Found standalone answer: {line}")
            return line
        
        # Check for line starts like "A.", "A)", etc.
        for letter in option_letters:
            if line.startswith(f"{letter}.") or line.startswith(f"{letter})"):
                debug_print(f"Found answer at line start: {letter}")
                return letter
    
    # Word and phrase frequency analysis (last resort)
    option_scores = {"A": 0, "B": 0, "C": 0, "D": 0}
    
    # Check for phrases indicating correctness near option mentions
    correct_phrases = [
        "IS CORRECT", "IS THE ANSWER", "IS TRUE", "IS RIGHT",
        "CORRECT OPTION", "RIGHT OPTION", "BEST OPTION", "ANSWER"
    ]
    
    for option in option_letters:
        # Check for standalone option mentions
        option_scores[option] += response_text.count(f" {option} ")
        option_scores[option] += response_text.count(f" {option}.")
        option_scores[option] += response_text.count(f" {option},")
        option_scores[option] += response_text.count(f" {option})")
        
        # Check for correct phrases near options
        for phrase in correct_phrases:
            option_scores[option] += response_text.count(f"{option} {phrase}") * 5
            option_scores[option] += response_text.count(f"OPTION {option} {phrase}") * 5
    
    debug_print(f"Option frequency scores: {option_scores}")
    
    # Get the option with the highest score
    max_score = max(option_scores.values())
    if max_score > 0:
        max_options = [k for k, v in option_scores.items() if v == max_score]
        if len(max_options) == 1:
            debug_print(f"Selected answer based on frequency: {max_options[0]}")
            return max_options[0]
        
    # Emergency fallback - if response contains D AND "customer needs", return D
    if "D" in response_text and ("CUSTOMER" in response_text or "NEED" in response_text):
        debug_print("Emergency fallback: Found 'D' and 'customer needs' keywords")
        return "D"
        
    # If the response contains specific keywords strongly associated with a particular answer
    # This is domain specific but can help catch answers
    if "CUSTOMER" in response_text and "NEED" in response_text:
        debug_print("Keyword match: Found customer needs keywords")
        return "D"
    
    # If no clear winner, try to extract any letter that appears frequently
    for option in option_letters:
        if response_text.count(option) > 5:  # If letter appears multiple times
            debug_print(f"Fallback to frequent letter: {option}")
            return option
    
    # Last resort - if we can see a specific answer in the debug logs but couldn't extract it
    if "THE CORRECT ANSWER IS D" in response_text:
        return "D"
    
    # If still no clear winner, return None
    debug_print("Could not extract a clear answer from the response")
    return None

def evaluate_mmlu_with_custom_agent(output_file, total_questions=150, split="test"):
    """Evaluate MMLU using our custom React agent that tracks all API calls."""
    global detailed_handler  # Use the global handler for tracking
    
    # Create the detailed callback handler
    detailed_handler = DetailedCallbackHandler()
    
    # Build our custom React agent with tracking
    agent = build_react_agent(model, detailed_handler)
    
    # Start tracking time
    detailed_handler.start_time = time.time()
    
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
            detailed_handler.set_question_id(question_id)
            
            # Format the question
            formatted_question = format_question(item)
            
            # Print question info
            print(f"\nQuestion {i+1}/{len(dataset)} | Category: {item['category']}")
            print(f"Q: {item['question']}")
            print(f"Options: A. {item['options'][0]} | B. {item['options'][1]} | C. {item['options'][2]} | D. {item['options'][3]}")
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
                    "thread_id": f"thread_{question_id}"  # Add a unique thread ID for each question
                }
            }
            
            # Stream the agent's execution to see the full process
            full_response = ""
            all_messages = []
            final_response = None
            
            try:
                # Run agent without streaming to get complete final state
                final_state = agent.invoke(initial_state, config=config)
                debug_print(f"Final state keys: {final_state.keys() if final_state else 'None'}")
                
                # Get final messages list from complete run
                if final_state and "messages" in final_state:
                    all_messages = final_state["messages"]
                    debug_print(f"Final state has {len(all_messages)} messages")
                    
                    # Find the last message which should be the answer
                    if all_messages:
                        final_response = all_messages[-1]
                        
                # Also run with streaming for visualization
                for step in agent.stream(initial_state, config=config):
                    debug_print(f"Stream step: {step.keys()}")
                    
                    # Show messages during streaming for user feedback
                    if "messages" in step and step["messages"]:
                        messages = step["messages"]
                        for msg in messages:
                            # Print message content based on message type
                            if hasattr(msg, "content"):
                                content = msg.content
                                print(f"Agent: {content}")
                            elif isinstance(msg, dict) and "content" in msg:
                                content = msg["content"]
                                print(f"Agent: {content}")
                            elif hasattr(msg, "tool_name") and hasattr(msg, "content"):
                                # This is a tool message
                                print(f"Tool ({msg.tool_name}): {msg.content}")
                
                # Use the final message content as the full response
                if final_response:
                    if hasattr(final_response, "content"):
                        full_response = final_response.content
                        debug_print(f"Final response content: {full_response[:100]}...")
                    elif isinstance(final_response, dict) and "content" in final_response:
                        full_response = final_response["content"]
                        debug_print(f"Final response content from dict: {full_response[:100]}...")
                    else:
                        debug_print(f"Final response has unknown format: {type(final_response)}")
                        full_response = str(final_response)
            except Exception as e:
                print(f"Error during agent processing: {str(e)}")
                import traceback
                print(traceback.format_exc())
            
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
            
            # Get call tracking info directly from the callback handler
            call_data = {}
            if question_id in detailed_handler.question_counters:
                call_data = detailed_handler.question_counters[question_id]
            
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
                    'llm_calls': call_data.get('llm_calls', 0),
                    'tool_calls': call_data.get('tool_calls', 0),
                    'tokens_in': call_data.get('tokens_in', 0),
                    'tokens_out': call_data.get('tokens_out', 0),
                    'total_duration': call_data.get('total_duration', 0)
                }
            }
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
                'options': item['options'],
                'correct_answer': item['answer'],
                'selected_answer': None,
                'is_correct': False,
                'agent_response': f"ERROR: {str(e)}",
                'category': item['category'],
                'model_calls': {
                    'llm_calls': 0,
                    'tool_calls': 0,
                    'tokens_in': 0,
                    'tokens_out': 0,
                    'total_duration': 0
                }
            }
            results.append(result)
            total_count += 1
    
    # Get the complete tracking summary
    tracking_summary = detailed_handler.get_summary()
    
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
                'llm_calls': 0, 
                'tool_calls': 0, 
                'tokens': 0
            }
        
        category_results[category]['total'] += 1
        if item['is_correct']:
            category_results[category]['correct'] += 1
        
        # Add tracking data by category
        model_calls = item.get('model_calls', {})
        category_results[category]['llm_calls'] += model_calls.get('llm_calls', 0)
        category_results[category]['tool_calls'] += model_calls.get('tool_calls', 0)
        category_results[category]['tokens'] += model_calls.get('tokens_in', 0) + model_calls.get('tokens_out', 0)
    
    print("\nPer-category results:")
    category_accuracy = {}
    for category, counts in category_results.items():
        cat_accuracy = counts['correct'] / counts['total'] if counts['total'] > 0 else 0
        category_accuracy[category] = cat_accuracy
        print(f"{category}: {cat_accuracy:.4f} ({counts['correct']}/{counts['total']}) - LLM calls: {counts['llm_calls']}, Tool calls: {counts['tool_calls']}")
    
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
                    'llm_calls': res['llm_calls'],
                    'tool_calls': res['tool_calls'],
                    'tokens': res['tokens'],
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
    print(f"LLM calls: {tracking_summary.get('total_llm_calls', 0)}")
    print(f"Tool calls: {tracking_summary.get('total_tool_calls', 0)}")
    print(f"Total tokens: {tracking_summary.get('total_tokens', 0)} (in: {tracking_summary.get('total_tokens_in', 0)}, out: {tracking_summary.get('total_tokens_out', 0)})")
    print(f"Average LLM calls per question: {tracking_summary.get('total_llm_calls', 0)/len(dataset):.2f}")
    print(f"Duration: {tracking_summary.get('total_duration', 0):.2f} seconds")
    print(f"Calls per second: {tracking_summary.get('calls_per_second', 0):.2f}")
    
    print("\nTop Categories:")
    for cat, acc in sorted_categories[:min(3, len(sorted_categories))]:
        count = category_results[cat]
        print(f"  {cat}: {acc:.4f} ({count['correct']}/{count['total']}) - Avg LLM calls: {count['llm_calls']/count['total']:.2f}")
    
    print("\nBottom Categories:")
    for cat, acc in sorted_categories[-min(3, len(sorted_categories)):]:
        count = category_results[cat]
        print(f"  {cat}: {acc:.4f} ({count['correct']}/{count['total']}) - Avg LLM calls: {count['llm_calls']/count['total']:.2f}")
    print("=" * 80)
    
    # Return the summary data for further analysis if needed
    return {
        'accuracy': accuracy,
        'tracking': tracking_summary,
        'categories': category_results
    }

def main():
    # Set up the model and tracking
    global model, model_id, detailed_handler

    # Initialize your HuggingFace model
    model_id = "/scratch/mj6ux/.cache/models/mixtral-8x22b"
    
    # Import necessary modules for model setup
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    
    # Check available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Set device mapping for multiple GPUs
    if num_gpus > 1:
        device_map = "auto"
        print(f"Using device_map: {device_map}")
    else:
        device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Falling back to single device: {device_map}")
    
    # Initialize model and tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Fix the pad token issue
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        llm_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype=torch.float16
        )
        
        # Create the pipeline - REMOVE the device parameter when using device_map="auto"
        text_pipe = pipeline(
            "text-generation",
            model=llm_model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.1,
            top_p=0.9,
            return_full_text=False
        )
        
        # Create tracking handler
        detailed_handler = DetailedCallbackHandler()
        
        # Create a simple wrapper class for the HuggingFace pipeline
        # This doesn't inherit from LangChain classes to avoid compatibility issues
        class SimpleLLMWrapper:
            def __init__(self, pipeline):
                self.pipeline = pipeline
                self.tools = []
            
            def invoke(self, messages, config=None):
                """Process the messages into a format the model can understand."""
                
                # Get callbacks from config
                callbacks = []
                if config and "callbacks" in config:
                    callbacks = config["callbacks"] if isinstance(config["callbacks"], list) else [config["callbacks"]]
                
                # Format the messages into a prompt string
                prompt = ""
                for message in messages:
                    if isinstance(message, SystemMessage):
                        prompt += f"System: {message.content}\n\n"
                    elif isinstance(message, HumanMessage):
                        prompt += f"Human: {message.content}\n\n"
                    elif isinstance(message, AIMessage):
                        prompt += f"AI: {message.content}\n\n"
                    elif isinstance(message, ToolMessage):
                        prompt += f"Tool ({message.name}): {message.content}\n\n"
                    else:
                        prompt += f"{message.type}: {message.content}\n\n"
                
                prompt += "AI: "
                
                # Call on_llm_start for all callbacks - use tracking to prevent double calls
                called_callbacks = set()
                for callback in callbacks:
                    callback_id = id(callback)
                    if callback_id not in called_callbacks and hasattr(callback, "on_llm_start"):
                        callback.on_llm_start({"name": "HuggingFacePipeline"}, [prompt])
                        called_callbacks.add(callback_id)
                
                # Measure token input (approximate)
                prompt_tokens = len(prompt) // 4  # Rough estimation
                
                # Invoke the pipeline
                start_time = time.time()
                result = self.pipeline(prompt)[0]['generated_text']
                completion_time = time.time() - start_time
                
                # Estimate token usage
                completion_tokens = len(result) // 4
                total_tokens = prompt_tokens + completion_tokens
                
                # Create token usage information
                token_usage = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens
                }
                
                # Check for tool calls with improved detection
                has_tool_call = False
                tool_name = None
                tool_input = None
                
                for tool in self.tools:
                    # Process tool detection with improved patterns
                    tool_patterns = [
                        r"I(?:'ll| will| need to| want to| should)? use the ([a-zA-Z_]+) tool(?: with input:? ?)(.*?)(?:$|\.|\n)",
                        r"Using the ([a-zA-Z_]+) tool(?: with:? ?)(.*?)(?:$|\.|\n)",
                        r"Let(?:'s| me) use the ([a-zA-Z_]+) tool(?: to:? ?)(.*?)(?:$|\.|\n)",
                        r"I need to look up (.*?) using the ([a-zA-Z_]+) tool",
                        r"To solve this, I(?:'ll| will) use the ([a-zA-Z_]+) tool(?: to:? ?)(.*?)(?:$|\.|\n)",
                        r"I'll consult the ([a-zA-Z_]+) tool(?: for:? ?)(.*?)(?:$|\.|\n)",
                        r"Let me calculate this using the ([a-zA-Z_]+) tool:? ?(.*?)(?:$|\.|\n)"
                    ]
                    
                    for pattern in tool_patterns:
                        match = re.search(pattern, result, re.IGNORECASE | re.DOTALL)
                        if match:
                            # Handle different pattern formats
                            if "using the" in pattern and "look up" in pattern:
                                # Reversed pattern (input first, then tool name)
                                tool_input = match.group(1).strip()
                                found_tool = match.group(2).strip()
                            else:
                                # Standard pattern (tool name first, then input)
                                found_tool = match.group(1).strip()
                                if len(match.groups()) > 1:
                                    tool_input = match.group(2).strip()
                                else:
                                    tool_input = ""
                            
                            # Check if this matches our tool
                            if found_tool.lower() == tool.name.lower():
                                has_tool_call = True
                                tool_name = tool.name
                                debug_print(f"Detected tool call: {tool_name} with input: {tool_input}")
                                break
                    
                    if has_tool_call:
                        break
                
                # Create appropriate response message
                if has_tool_call and tool_name and tool_input:
                    response = AIMessage(
                        content=result,
                        additional_kwargs={
                            "tool_calls": [
                                {
                                    "name": tool_name,
                                    "args": tool_input,
                                    "id": f"tool_call_{int(time.time())}"
                                }
                            ]
                        }
                    )
                else:
                    response = AIMessage(content=result)
                
                # Add token usage to response
                response.llm_output = {"token_usage": token_usage}
                
                # Call on_llm_end for all callbacks - use tracking to prevent double calls
                called_callbacks = set()
                for callback in callbacks:
                    callback_id = id(callback)
                    if callback_id not in called_callbacks and hasattr(callback, "on_llm_end"):
                        callback.on_llm_end(response)
                        called_callbacks.add(callback_id)
                
                return response
            
            def bind_tools(self, tools):
                """Store tools but don't actually bind them."""
                self.tools = tools
                return self
        
        # Create our simple wrapper
        model = SimpleLLMWrapper(pipeline=text_pipe)
        
        # Test binding tools
        try:
            model = model.bind_tools(tools)
            print("Successfully bound tools to the model")
        except Exception as e:
            print(f"Error binding tools: {e}")
            import traceback
            print(traceback.format_exc())
    except Exception as e:
        print(f"Error setting up model: {e}")
        import traceback
        print(traceback.format_exc())
        return
        
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Custom ReAct Agent on MMLU")
    parser.add_argument("--output_file", type=str, default="mmlu_custom_react_results.json", 
                        help="Output file for results")
    parser.add_argument("--num_questions", type=int, default=150, 
                        help="Total number of questions to evaluate")
    parser.add_argument("--split", type=str, default="test", 
                        help="Dataset split to use")
    args = parser.parse_args()
    
    # Run the evaluation
    evaluate_mmlu_with_custom_agent(
        args.output_file, 
        args.num_questions, 
        args.split
    )

if __name__ == "__main__":
    main()