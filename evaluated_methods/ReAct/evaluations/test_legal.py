import os
import json
import time
import re
from typing import List, Dict, Any, Optional
from tqdm import tqdm

# LangChain Imports
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI

# HuggingFace Datasets
from datasets import load_dataset

# Tool Functions
def calculate(expression: str) -> str:
    """Calculate the result of a mathematical expression."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Error in calculation: {str(e)}"

def search_wikipedia(query: str) -> str:
    """Search Wikipedia for information on a topic."""
    # Mock implementation
    return f"Wikipedia results for: {query}\n[This is a mock response that would contain actual information from Wikipedia in a real implementation]"

# Load LegalBench Dataset
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
        
        # Print statistics
        relevant_count = sum(1 for item in processed_data if item['answer'] == 'Relevant')
        irrelevant_count = sum(1 for item in processed_data if item['answer'] == 'Irrelevant')
        print(f"\nDataset statistics:")
        print(f"Total examples: {len(processed_data)}")
        print(f"Relevant examples: {relevant_count} ({relevant_count/len(processed_data)*100:.1f}%)")
        print(f"Irrelevant examples: {irrelevant_count} ({irrelevant_count/len(processed_data)*100:.1f}%)")
        
        return processed_data
        
    except Exception as e:
        print(f"Error loading LegalBench dataset: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return []

# Format LegalBench Questions
def format_legalbench_question(item):
    """Format a LegalBench privacy policy question."""
    prompt = f"""I'll show you a question about a privacy policy clause. Please determine if the clause is RELEVANT or IRRELEVANT to answering the question.

PRIVACY POLICY CLAUSE:
{item['document']}

QUESTION:
{item['question']}

A clause is RELEVANT if it contains information that helps answer the question.
A clause is IRRELEVANT if it does not provide information needed to answer the question.

Analyze the clause and question carefully. You can use tools to assist your analysis if needed.
After your analysis, provide your final answer by clearly stating "FINAL ANSWER: RELEVANT" or "FINAL ANSWER: IRRELEVANT".
"""
    
    return prompt

# Extract answer from model response
def extract_legalbench_answer(response_text):
    """Extract the binary classification answer from the agent's response."""
    if not response_text:
        return None
    
    # Check for the explicit final answer format first
    final_answer_match = re.search(r"FINAL ANSWER:\s*(RELEVANT|IRRELEVANT)", response_text, re.IGNORECASE)
    if final_answer_match:
        answer = final_answer_match.group(1).upper()
        if answer == "RELEVANT":
            return "Relevant"
        elif answer == "IRRELEVANT":
            return "Irrelevant"
    
    # Check for other common formats
    if re.search(r"\b(THE ANSWER IS:|ANSWER:|MY ANSWER IS:)\s*(RELEVANT)", response_text, re.IGNORECASE):
        return "Relevant"
    if re.search(r"\b(THE ANSWER IS:|ANSWER:|MY ANSWER IS:)\s*(IRRELEVANT)", response_text, re.IGNORECASE):
        return "Irrelevant"
    
    # Check for plain statements at the end of the text
    last_paragraph = response_text.split('\n\n')[-1].strip().upper()
    if "RELEVANT" in last_paragraph and "IRRELEVANT" not in last_paragraph:
        return "Relevant"
    if "IRRELEVANT" in last_paragraph and "RELEVANT" not in last_paragraph[-10:]:
        return "Irrelevant"
    
    # If no conclusive answer found
    return None

# Simple ReAct Agent Implementation
class SimpleReactAgent:
    def __init__(self, model_name="gpt-4o", temperature=0.0):
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        
        # Define tools
        self.tools = [
            Tool(
                name="calculate",
                func=calculate,
                description="Calculate the result of a mathematical expression."
            ),
            Tool(
                name="search_wikipedia",
                func=search_wikipedia,
                description="Search Wikipedia for information on a topic."
            )
        ]
        
        self.tool_map = {tool.name: tool for tool in self.tools}
        
        self.system_prompt = """You are a skilled legal analyst who evaluates if privacy policy clauses are relevant to specific questions.

IMPORTANT GUIDELINES:
1. Analyze each question and clause carefully.
2. If needed, use tools to help your analysis:
   - Use the 'calculate' tool for any mathematical operations.
   - Use the 'search_wikipedia' tool to look up relevant legal concepts or terminology.
3. Structure your thinking step-by-step.
4. After analysis, clearly state your conclusion with "FINAL ANSWER: RELEVANT" or "FINAL ANSWER: IRRELEVANT".

To use a tool, use the following format:
Action: tool_name
Action Input: input to the tool

Remember, a clause is RELEVANT if it contains information that helps answer the question, and IRRELEVANT if it doesn't.
"""
    
    def run(self, query):
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=query)
        ]
        
        # Maximum number of steps
        max_steps = 10
        
        # Messages history
        conversation = []
        
        # Add system prompt and query
        conversation.append({"role": "system", "content": self.system_prompt})
        conversation.append({"role": "user", "content": query})
        
        for step in range(max_steps):
            # Get model response
            response = self.llm.invoke(messages)
            assistant_message = response.content
            
            # Check if the response contains a tool use pattern
            tool_match = re.search(r"Action: (\w+)\nAction Input: (.*?)(?=\n|$)", assistant_message, re.DOTALL)
            
            if tool_match:
                tool_name = tool_match.group(1).strip()
                tool_input = tool_match.group(2).strip()
                
                # Add assistant's message
                messages.append(response)
                conversation.append({"role": "assistant", "content": assistant_message})
                
                # Execute tool if it exists
                if tool_name in self.tool_map:
                    tool = self.tool_map[tool_name]
                    try:
                        tool_result = tool.func(tool_input)
                        tool_response = f"Tool {tool_name} result: {tool_result}"
                    except Exception as e:
                        tool_response = f"Error using {tool_name} tool: {str(e)}"
                else:
                    tool_response = f"Error: Tool '{tool_name}' not found. Available tools: {list(self.tool_map.keys())}"
                
                # Add tool response
                human_message = HumanMessage(content=tool_response)
                messages.append(human_message)
                conversation.append({"role": "user", "content": tool_response})
            else:
                # No tool use, this is the final answer
                conversation.append({"role": "assistant", "content": assistant_message})
                return {
                    "output": assistant_message,
                    "conversation": conversation
                }
        
        # If we reach max steps, return what we have
        return {
            "output": "Reached maximum number of reasoning steps without a final answer.",
            "conversation": conversation
        }

# Evaluate the agent on LegalBench
def evaluate_agent(agent, dataset, output_file):
    """Evaluate the agent on the LegalBench dataset."""
    results = []
    correct_count = 0
    total_count = 0
    
    start_time = time.time()
    
    for i, item in enumerate(tqdm(dataset, desc="Evaluating LegalBench")):
        print(f"\nQuestion {i+1}/{len(dataset)}")
        print(f"Q: {item['question'][:150]}...")
        print(f"Clause: {item['document'][:150]}...")
        print(f"Correct answer: {item['answer']}")
        
        # Format the question
        formatted_question = format_legalbench_question(item)
        
        # Run the agent
        try:
            print("Running agent...")
            result = agent.run(formatted_question)
            final_response = result["output"]
            
            print(f"Agent response (last 200 chars): {final_response[-200:] if final_response else ''}")
            
            # Extract the answer
            selected_answer = extract_legalbench_answer(final_response)
            print(f"Extracted answer: {selected_answer}")
            
            # Check if the answer is correct
            is_correct = False
            if selected_answer is not None:
                is_correct = selected_answer.lower() == item['answer'].lower()
            
            # Update statistics
            if is_correct:
                correct_count += 1
                print("✓ CORRECT")
            else:
                print("✗ INCORRECT")
            
            total_count += 1
            
            # Store the result
            result_entry = {
                'id': item.get('id', f"q{i}"),
                'question': item['question'],
                'document': item['document'],
                'correct_answer': item['answer'],
                'selected_answer': selected_answer,
                'is_correct': is_correct,
                'agent_response': final_response,
            }
            
            results.append(result_entry)
        
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
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)
    
    print(f"Results saved to {output_file}")
    
    return accuracy

def main():
    """Main function to run the evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate a SimpleReact agent on LegalBench")
    parser.add_argument("--output_dir", type=str, default="./results_simplereact", 
                        help="Output directory for results")
    parser.add_argument("--num_questions", type=int, default=150, 
                        help="Number of questions to evaluate")
    parser.add_argument("--model", type=str, default="gpt-4o", 
                        help="OpenAI model to use")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Temperature for the model")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Initializing SimpleReact agent with {args.model}")
    agent = SimpleReactAgent(args.model, args.temperature)
    
    print("\n=== Loading LegalBench Privacy Policy QA Dataset ===")
    legalbench_data = load_legalbench_dataset(num_samples=args.num_questions)
    
    if legalbench_data:
        output_file = os.path.join(args.output_dir, f"simplereact_{args.model.replace('-', '_')}_results.json")
        print("\n=== Running Evaluation ===")
        accuracy = evaluate_agent(agent, legalbench_data, output_file)
        print(f"\nFinal LegalBench accuracy: {accuracy:.4f}")
    else:
        print("Failed to load LegalBench data. Exiting.")

if __name__ == "__main__":
    main()