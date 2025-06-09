"""
Multi-Model LegalBench Privacy Policy QA Evaluator

This script evaluates 150 examples from the privacy_policy_qa subset of LegalBench
using multiple LLM models with tools:
1. Claude 3.5 Sonnet (Anthropic)
2. Mistral Open-Mistral-8x22B  
3. Llama 3.1 8B Instruct (via vLLM)
"""

import os
import json
import math
import time
import random
import wikipedia
from tqdm import tqdm
from datasets import load_dataset
from langgraph.prebuilt import create_react_agent
from langchain.agents import AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain.chat_models import init_chat_model
from langchain_anthropic import ChatAnthropic
from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI
from langchain_community.llms import VLLM
from langchain_huggingface import HuggingFaceEndpoint
from langchain_together import ChatTogether
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from langchain_huggingface import HuggingFaceEndpoint
from langchain.agents import Tool

# Define tools for the agent
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression.
    
    Args:
        expression: A string representing a mathematical expression
    
    Returns a string containing the result of evaluating the expression.
    """
    try:
        # Use Python's built-in eval function with limited scope for safety
        allowed_names = {
            "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "sqrt": math.sqrt, "log": math.log, "log10": math.log10,
            "pi": math.pi, "e": math.e, "abs": abs, "round": round,
            "max": max, "min": min, "sum": sum, "len": len,
            "int": int, "float": float, "str": str
        }
        
        # Evaluate the expression
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"Result: {result}"
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"

def wiki_search(query: str) -> str:
    """Search Wikipedia for information about privacy terms, regulations, or concepts.
    
    Args:
        query: The search query related to privacy policies or terms
    
    Returns a string with the Wikipedia article summary.
    """
    try:
        # Search Wikipedia
        search_results = wikipedia.search(query, results=3)
        
        if not search_results:
            return "No results found on Wikipedia for that query."
        
        # Get summary of the first result
        try:
            page = wikipedia.page(search_results[0])
            summary = wikipedia.summary(search_results[0], sentences=5)
            return f"Title: {page.title}\nSummary: {summary}\nURL: {page.url}"
        except wikipedia.DisambiguationError as e:
            # Handle disambiguation pages
            return f"Multiple matches found. Try one of these: {', '.join(e.options[:5])}"
    except Exception as e:
        return f"Error searching Wikipedia: {str(e)}"

# Privacy term analyzer tool
def privacy_term_analyzer(term: str) -> str:
    """Analyze common privacy terms and provide standardized definitions.
    
    Args:
        term: A privacy policy term or concept to analyze
    
    Returns a string with the definition and context for the privacy term.
    """
    # Dictionary of common privacy terms
    privacy_terms = {
        "gdpr": "The General Data Protection Regulation (GDPR) is a comprehensive EU data protection law that came into effect in May 2018. It requires businesses to protect the personal data and privacy of EU citizens. Key principles include consent, right to access, right to be forgotten, data portability, privacy by design, and potential penalties for non-compliance.",
        
        "ccpa": "The California Consumer Privacy Act (CCPA) is a state statute intended to enhance privacy rights and consumer protection for residents of California. Effective January 1, 2020, it gives California residents the right to know what personal data is collected, whether it's sold/disclosed and to whom, to opt out of the sale of this data, to access their data, and to request deletion of data.",
        
        "third party": "In privacy policies, 'third parties' typically refer to entities that are separate from the primary company and its users. These could include advertisers, analytics providers, business partners, or service providers who may receive user data. Privacy policies should disclose which third parties receive data and for what purposes.",
        
        "opt out": "An 'opt-out' mechanism allows users to choose not to participate in certain data collection or processing activities. This contrasts with 'opt-in,' where users must actively consent before their data is collected or used. Many privacy regulations require that companies provide clear opt-out options for certain types of data processing.",
        
        "data retention": "Data retention refers to how long a company keeps user data. Privacy policies should specify retention periods and the criteria used to determine these periods. Many regulations require that data not be kept longer than necessary for the purpose for which it was collected.",
        
        "encryption": "Encryption is a security method that converts information into a code to prevent unauthorized access. In privacy contexts, encryption is used to protect sensitive user data both in transit (being sent) and at rest (stored). Strong encryption is considered a best practice for data protection.",
        
        "anonymization": "Anonymization is the process of removing or modifying personally identifiable information so that individuals cannot be identified. Properly anonymized data may be exempt from certain privacy regulation requirements since it can no longer be linked to specific individuals.",
        
        "cookie": "Cookies are small text files stored on a user's device that help websites recognize users and remember their preferences. Privacy policies typically explain what cookies are used, what information they collect, and how users can manage or delete them.",
        
        "data subject": "In privacy regulations, a 'data subject' is the individual whose personal data is being collected, held, or processed. Data subjects have various rights regarding their personal data, such as the right to access, correct, or delete their information.",
        
        "personal information": "Personal information (or personal data) is any information that relates to an identified or identifiable individual. This can include names, identification numbers, location data, online identifiers, or factors specific to the physical, physiological, genetic, mental, economic, cultural, or social identity of a person."
    }
    
    # Normalize the term to lowercase and remove punctuation
    normalized_term = term.lower().strip()
    
    # Look for matches
    for key, definition in privacy_terms.items():
        if key in normalized_term or normalized_term in key:
            return f"Term: {key.upper()}\nDefinition: {definition}"
    
    # If no direct match, return a more generic response
    return f"No standard definition found for '{term}'. Consider looking up this term with the Wikipedia tool or checking privacy regulation glossaries."

def get_model_client(model_name):
    """
    Get the appropriate LLM client based on the model name.
    """
    # Define tools
    tools = [calculator, wiki_search, privacy_term_analyzer]
    
    if model_name == "claude-3-5-sonnet":
        # Claude 3.5 Sonnet from Anthropic
        model = ChatAnthropic(
            model="claude-3-5-sonnet-20240620",
            temperature=0.5,
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY")
        )
        # Bind tools to Claude
        return model.bind_tools(tools)
        
    elif model_name == "open-mixtral-8x22b":
        # Mistral AI's Open-Mixtral-8x22B
        model = ChatMistralAI(
            model="open-mixtral-8x22b",
            temperature=0.5,
            mistral_api_key=os.environ.get("MISTRAL_API_KEY")
        )
        # Bind tools to Mistral
        return model.bind_tools(tools)
        
    elif model_name == "llama-3-1-8b":
        # Try multiple methods to get a working Llama model
        together_api_key = os.environ.get("TOGETHER_API_KEY")
        hf_api_key = os.environ.get("HUGGINGFACE_API_KEY")
        
        # Try Together AI first if API key is available
        if together_api_key:
            try:                
                print("Using Together AI for Llama 3.1 model...")
                model = ChatTogether(
                    together_api_key=together_api_key,
                    model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                    temperature=0.5,
                    max_tokens=1024
                )
                # Bind tools to the model
                return model.bind_tools(tools)
            except Exception as e:
                print(f"Error using Together AI: {e}")
                print("Falling back to Hugging Face Endpoints...")
        
        # Try Hugging Face Endpoints if API key is available
        if hf_api_key:
            try:
                # Create base chat model to use as wrapper
                wrapped_model = ChatOpenAI(
                    model_name="gpt-3.5-turbo",  # Will be replaced
                    temperature=0.5
                )
                
                # Create HuggingFace endpoint
                hf_llm = HuggingFaceEndpoint(
                    repo_id="meta-llama/Llama-3.1-8B-Instruct",
                    temperature=0.5,
                    max_length=1024,
                    huggingfacehub_api_token=hf_api_key
                )
                
                # Override generation to use HF endpoint
                original_generate = wrapped_model._generate
                def custom_generate(*args, **kwargs):
                    messages = kwargs.get("messages", [])
                    combined_prompt = "\n".join([m.content for m in messages])
                    response = hf_llm(combined_prompt)
                    return {
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": response
                                }
                            }
                        ]
                    }
                
                wrapped_model._generate = custom_generate
                return wrapped_model.bind_tools(tools)
            except Exception as e:
                print(f"Error using Hugging Face Endpoints: {e}")
                print("Falling back to local transformers library...")
        
        try:
            print("Using local transformers library for Llama 3.1 model...")
            
            def evaluate_with_transformers(example, verbose=False):
                """Evaluate using transformers directly"""
                try:
                    # Load model and tokenizer
                    if verbose:
                        print("Loading model and tokenizer...")
                    
                    model_name = "meta-llama/Llama-3.1-8B-Instruct"
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.bfloat16,
                        device_map="auto"
                    )
                    
                    # Create prompt
                    prompt = f"""You are a legal assistant analyzing privacy policies.
                    
Question: {example['question']}
Clause: {example['text']}

Is this clause directly relevant to the question? Answer with ONLY "Relevant" or "Irrelevant".
"""
                    
                    # Generate response
                    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                    outputs = model.generate(**inputs, max_new_tokens=20, temperature=0.1)
                    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    # Extract prediction
                    if "relevant" in response.lower() and "irrelevant" not in response.lower():
                        prediction = "Relevant"
                    else:
                        prediction = "Irrelevant"
                    
                    if verbose:
                        print(f"Response: {response}")
                        print(f"Prediction: {prediction}")
                    
                    return {
                        "question": example["question"],
                        "clause": example["text"],
                        "true_label": example["answer"],
                        "prediction": prediction,
                        "content": response,
                        "correct": prediction == example["answer"],
                        "tool_usage": {"any_tool_used": False}
                    }
                except Exception as e:
                    if verbose:
                        print(f"Error evaluating with transformers: {e}")
                    return None
            
            # Return the evaluation function
            return evaluate_with_transformers
            
        except Exception as e:
            print(f"Error setting up transformers: {e}")
            print("All methods failed. Please provide a valid API key or fix local configuration.")
            return None
            
    else:
        # Default to GPT-3.5-Turbo as a fallback
        model = init_chat_model(
            "openai:gpt-3.5-turbo",
            temperature=0.5
        )
        # Bind tools to default model
        return model.bind_tools(tools)

def create_agent_from_llm(llm, tools):
    """
    Create a ReAct agent from a standard LLM (not a chat model).
    """
    
    prompt_template = """You are a legal assistant specialized in privacy policies who USES TOOLS to make accurate determinations.

YOU SHOULD USE TOOLS when analyzing privacy clauses - this helps provide accurate and informed analyses.

You have the following tools at your disposal:
{tools}

Your workflow should include these steps:
1. IDENTIFY key terms or concepts in the question and clause that may need clarification
2. USE TOOLS to research these terms/concepts when helpful
3. ANALYZE how the information applies to the question
4. DETERMINE if the clause is "Relevant" or "Irrelevant" to the question

Question: {question}
Clause: {text}

{agent_scratchpad}

After your analysis, respond with "Relevant" if the clause contains enough information to answer the question, or "Irrelevant" if it does not.
"""
    
    prompt = PromptTemplate.from_template(prompt_template)
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    return agent_executor

def extract_content_from_response(response, model_name):
    """
    Improved response content extraction that handles tool usage.
    """
    try:
        # Convert the response to a string for analysis
        response_str = str(response)
        
        # Extract the final message that contains the verdict
        # This is where we need to look for the actual decision after tool usage
        
        if "claude" in model_name.lower():
            # For Claude, we need to extract the final message after tool usage
            ai_messages = []
            
            # Look for AIMessage patterns in the response
            if hasattr(response, "messages"):
                # If response has messages attribute, use it
                ai_messages = [msg.content for msg in response.messages if msg.type == "ai"]
            else:
                # Otherwise parse from string
                import re
                ai_message_pattern = r"AIMessage\(content='([^']+)'"
                ai_matches = re.findall(ai_message_pattern, response_str)
                ai_messages = ai_matches
            
            # Get the last message, which should contain the verdict
            if ai_messages:
                return ai_messages[-1]  # Return the last AI message
        
        # Generic extraction (fallback)
        if "content='" in response_str:
            content_start = response_str.find("content='") + len("content='")
            content_end = response_str.find("'", content_start)
            if content_end > content_start:
                return response_str[content_start:content_end]
        
        # If still not found, try different pattern
        if 'content="' in response_str:
            content_start = response_str.find('content="') + len('content="')
            content_end = response_str.find('"', content_start)
            if content_end > content_start:
                return response_str[content_start:content_end]
                
        # Last resort: return the entire string
        return response_str
    
    except Exception as e:
        print(f"Error extracting content: {e}")
        return "Could not extract content"

def detect_tool_usage(response_str):
    """
    Detect if tools were used in the response.
    
    Args:
        response_str: The string representation of the response
        
    Returns:
        Dictionary with tool usage information
    """
    tool_usage = {
        "calculator_used": "tool_call" in response_str.lower() and "calculator" in response_str.lower(),
        "wikipedia_used": "tool_call" in response_str.lower() and "wiki_search" in response_str.lower(),
        "privacy_analyzer_used": "tool_call" in response_str.lower() and "privacy_term_analyzer" in response_str.lower()
    }
    
    # Check if any tool was used
    tool_usage["any_tool_used"] = any(tool_usage.values())
    
    return tool_usage

def create_tool_using_agent(model_name):
    """
    Create and return a React agent with tools for privacy policy analysis.
    """
    # Define tools
    tools = [calculator, wiki_search, privacy_term_analyzer]
    
    # Get the appropriate model client with tools already bound
    model = get_model_client(model_name)
    
    # Create the agent with tools
    agent = create_react_agent(
        model=model,
        tools=tools,
        prompt="""You are a legal assistant specialized in privacy policies who USES TOOLS to make accurate determinations.

YOU SHOULD USE TOOLS when analyzing privacy clauses - this helps provide accurate and informed analyses.

You have the following tools at your disposal:

1. calculator: Use this tool to perform any numerical analysis, including:
   - Counting entities mentioned in a clause
   - Analyzing time periods
   - Calculating any numbers relevant to privacy policies

2. wiki_search: Use this tool to look up information about:
   - Privacy laws and regulations (GDPR, CCPA, HIPAA, etc.)
   - Legal terminology in privacy contexts
   - Standard practices in data protection
   - Companies or services mentioned

3. privacy_term_analyzer: Use this tool to get standard definitions of:
   - Common privacy terms like "opt-out," "third party," "data retention"
   - Regulatory concepts from privacy frameworks
   - Technical terms used in privacy contexts

Your workflow should include these steps:
1. IDENTIFY key terms or concepts in the question and clause that may need clarification
2. USE TOOLS to research these terms/concepts when helpful
3. ANALYZE how the information applies to the question
4. DETERMINE if the clause is "Relevant" or "Irrelevant" to the question

After your analysis, respond with "Relevant" if the clause contains enough information to answer the question, or "Irrelevant" if it does not.
"""
    )
    
    return agent

def extract_prediction(content):
    """Extract prediction from model content with improved logic"""
    # Look for clear "Relevant" or "Irrelevant" statements
    # Check for "The clause is Relevant" or similar patterns
    
    # Look for conclusion statements
    conclusion_patterns = [
        r"conclusion:?\s*(relevant|irrelevant)",
        r"verdict:?\s*(relevant|irrelevant)",
        r"the clause is\s*(relevant|irrelevant)",
        r"(relevant|irrelevant) to the question",
        r"^\s*(relevant|irrelevant)\s*$"  # Stand-alone verdict
    ]
    
    import re
    for pattern in conclusion_patterns:
        matches = re.search(pattern, content.lower())
        if matches:
            verdict = matches.group(1)
            return "Relevant" if verdict.lower() == "relevant" else "Irrelevant"
    
    # If no clear verdict pattern, count occurrences
    relevant_count = content.lower().count("relevant")
    irrelevant_count = content.lower().count("irrelevant")
    
    # If only one type appears, use that
    if relevant_count > 0 and irrelevant_count == 0:
        return "Relevant"
    elif irrelevant_count > 0 and relevant_count == 0:
        return "Irrelevant"
    
    # If both appear, use the last one in the text
    last_relevant_pos = content.lower().rfind("relevant")
    last_irrelevant_pos = content.lower().rfind("irrelevant")
    
    if last_relevant_pos > last_irrelevant_pos:
        return "Relevant"
    elif last_irrelevant_pos > last_relevant_pos:
        return "Irrelevant"
    
    # Default based on content patterns (check for positive indicators)
    yes_patterns = ["yes", "answers", "addresses", "provides", "contains"]
    for pattern in yes_patterns:
        if pattern in content.lower():
            return "Relevant"
    
    # If still unclear, default to "Irrelevant"
    return "Irrelevant"

def evaluate_example(agent, example, model_name, verbose=False):
    """
    Evaluate a single example with the agent.
    
    Args:
        agent: The React agent
        example: Dictionary with question, text, and answer
        model_name: The name of the model being used
        verbose: Whether to print detailed output
        
    Returns:
        Dictionary with evaluation results or None if error
    """
    # Format query to encourage tool usage
    query = f"""Question: {example['question']}
Clause: {example['text']}

Determine if this clause is relevant to the question. Use available tools if they would help with your analysis.
"""
    
    try:
        # Invoke the agent
        response = agent.invoke({"messages": [{"role": "user", "content": query}]})
        
        # Convert to string for analysis and debugging
        response_str = str(response)
        
        if verbose:
            print("\nFULL RESPONSE:")
            print(response_str[:1000])  # Print first 1000 chars for debugging
            print("..." if len(response_str) > 1000 else "")
        
        # Check for tool usage
        tool_usage = detect_tool_usage(response_str)
        
        if verbose:
            # Report tool usage
            for tool, used in tool_usage.items():
                if used and tool != "any_tool_used":
                    print(f"✅ Used {tool}")
        
        # IMPROVED CONTENT EXTRACTION FOR CLAUDE
        if "claude" in model_name.lower():
            # For Claude, we need to extract the final message after tool usage
            final_content = ""
            
            # Try to access the last AI message directly from the response object
            if hasattr(response, "return_values") and "output" in response.return_values:
                final_content = response.return_values["output"]
            elif hasattr(response, "messages"):
                ai_messages = [msg for msg in response.messages if msg.type == "ai"]
                if ai_messages:
                    final_content = ai_messages[-1].content
            else:
                # Complex parsing of the response string for Claude
                import re
                # Find all AIMessage contents
                ai_message_pattern = r"AIMessage\(content='(.*?)',\s*additional_kwargs"
                ai_matches = re.findall(ai_message_pattern, response_str, re.DOTALL)
                
                # Get the last one (final verdict)
                if ai_matches:
                    final_content = ai_matches[-1]
                else:
                    # If we couldn't find AIMessage pattern, try simpler approach
                    # Look for the final "Relevant" or "Irrelevant" statement
                    relevant_pattern = r"(Relevant|Irrelevant)[.\s]*$"
                    match = re.search(relevant_pattern, response_str)
                    if match:
                        # Get 100 characters before the final verdict for context
                        start_pos = max(0, match.start() - 100)
                        final_content = response_str[start_pos:match.end()]
            
            if verbose:
                print("\nEXTRACTED CONTENT:")
                print(final_content[:200])
                print("..." if len(final_content) > 200 else "")
            
            content = final_content
        else:
            # Original extraction for other models
            content = extract_content_from_response(response, model_name)
        
        if not content or len(content) < 5:  # Too short to be valid
            if verbose:
                print("Could not extract proper content from response")
            return None
        
        # IMPROVED PREDICTION EXTRACTION
        # Look specifically for the verdict
        import re
        relevant_match = re.search(r"(?:verdict|conclusion|determination|answer):\s*(relevant|irrelevant)", content.lower())
        
        if relevant_match:
            prediction = "Relevant" if relevant_match.group(1) == "relevant" else "Irrelevant"
        else:
            # If no explicit verdict found, check for the last occurrence
            last_relevant_pos = content.lower().rfind("relevant")
            last_irrelevant_pos = content.lower().rfind("irrelevant")
            
            if last_relevant_pos > last_irrelevant_pos and last_relevant_pos > 0:
                prediction = "Relevant"
            elif last_irrelevant_pos > last_relevant_pos and last_irrelevant_pos > 0:
                prediction = "Irrelevant"
            else:
                # Default based on yes/no indicators
                yes_patterns = ["yes", "provides information", "directly addresses", "contains information"]
                for pattern in yes_patterns:
                    if pattern in content.lower():
                        prediction = "Relevant"
                        break
                else:
                    prediction = "Irrelevant"  # Default
        
        if verbose:
            print(f"Extracted prediction: {prediction}")
        
        # Check if correct
        is_correct = prediction == example["answer"]
        
        return {
            "question": example["question"],
            "clause": example["text"],
            "true_label": example["answer"],
            "prediction": prediction,
            "content": content[:100] + "..." if len(content) > 100 else content,  # Truncate for readability
            "correct": is_correct,
            "tool_usage": tool_usage
        }
    
    except Exception as e:
        if verbose:
            print(f"Error evaluating example: {e}")
            import traceback
            print(traceback.format_exc())
        return None

def load_legalbench_examples(num_examples=150):
    """
    Load examples from the privacy_policy_qa subset of LegalBench.
    
    Args:
        num_examples: Maximum number of examples to load0
        
    Returns:
        List of examples
    """
    try:
        # Load the dataset
        print("Loading LegalBench privacy_policy_qa dataset...")
        dataset = load_dataset("nguha/legalbench", "privacy_policy_qa")
        
        # Convert the dataset to a list for easier handling
        if "test" in dataset:
            examples_list = dataset["test"].to_list()
            
            # Map the dataset fields to our expected format
            formatted_examples = []
            for example in examples_list[:num_examples]:
                # For privacy_policy_qa:
                # - 'text' field contains the privacy policy clause
                # - 'question' field contains the question
                # - 'answer' field contains the label (Relevant/Irrelevant)
                formatted_examples.append({
                    "question": example.get("question", ""),
                    "text": example.get("text", ""),
                    "answer": example.get("answer", "")
                })
            
            print(f"Successfully loaded {len(formatted_examples)} examples from LegalBench")
            return formatted_examples
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        print(traceback.format_exc())
    
    # If there's an error loading the dataset, return a small set of hardcoded examples
    print("Using hardcoded examples as fallback")
    return [
        {
            "question": "will my personal details be shared with third party companies?",
            "text": "The information may be disclosed to: (i) provide joint content and our services (eg, registration, coordination of membership accounts between the Viber corporate family, transactions, analytics and customer support); (ii) help detect and prevent potentially illegal acts, violations of our policies, fraud and/or data security breaches.",
            "answer": "Relevant"
        },
        {
            "question": "how long do you keep my data?",
            "text": "We use secure server software to protect the confidentiality of your personal information.",
            "answer": "Irrelevant"
        },
        {
            "question": "does your policy comply with GDPR requirements?",
            "text": "Our privacy practices are designed to comply with applicable data protection laws, including the General Data Protection Regulation (GDPR) for users in the European Economic Area.",
            "answer": "Relevant"
        }
    ]

def evaluate_model(model_name, examples, num_examples_to_use=150):
    """
    Evaluate a model on the given examples.
    
    Args:
        model_name: The name of the model to evaluate
        examples: List of examples to evaluate
        num_examples_to_use: Maximum number of examples to use
        
    Returns:
        Dictionary with evaluation results
    """
    print(f"\n{'='*50}")
    print(f"Evaluating model: {model_name}")
    
    # Create the agent for this model
    agent = create_tool_using_agent(model_name)
    
    # Take a subset of examples if needed
    examples_to_use = examples[:min(num_examples_to_use, len(examples))]
    print(f"Using {len(examples_to_use)} examples for evaluation")
    
    # Evaluate all examples
    results = []
    correct_count = 0
    tool_usage_count = 0
    
    # Use tqdm for progress tracking
    for i, example in enumerate(tqdm(examples_to_use, desc=f"Evaluating {model_name}")):
        # Only print details for a small subset of examples
        verbose = (i < 3) or (i % 50 == 0)
        
        if verbose:
            print(f"\n{'-'*30}")
            print(f"Example {i+1}/{len(examples_to_use)}:")
            print(f"Question: {example['question']}")
            print(f"Clause: {example['text']}")
            print(f"True Label: {example['answer']}")
        
        # Evaluate the example
        result = evaluate_example(agent, example, model_name, verbose)
        
        if not result:
            continue
        
        if verbose:
            print(f"Prediction: {result['prediction']}")
            print(f"Correct: {result['correct']}")
        
        # Track tool usage
        if result["tool_usage"].get("any_tool_used", False):
            tool_usage_count += 1
        
        results.append(result)
        if result["correct"]:
            correct_count += 1
        
        # Add a small delay to avoid rate limits
        time.sleep(0.5)
    
    # Calculate metrics
    if results:
        accuracy = correct_count / len(results)
        tool_usage_rate = tool_usage_count / len(results)
        
        # Count distributions
        relevant_count = sum(1 for r in results if r["true_label"] == "Relevant")
        irrelevant_count = sum(1 for r in results if r["true_label"] == "Irrelevant")
        
        # Count specific tool usage
        calculator_usage = sum(1 for r in results if r.get("tool_usage", {}).get("calculator_used", False))
        wikipedia_usage = sum(1 for r in results if r.get("tool_usage", {}).get("wikipedia_used", False))
        privacy_analyzer_usage = sum(1 for r in results if r.get("tool_usage", {}).get("privacy_analyzer_used", False))
        
        print(f"\nModel: {model_name} - Evaluation Complete")
        print(f"Examples evaluated: {len(results)}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Tool usage rate: {tool_usage_rate:.4f}")
        print(f"Tool breakdown:")
        print(f"  - Calculator: {calculator_usage} ({calculator_usage/len(results):.2%})")
        print(f"  - Wikipedia: {wikipedia_usage} ({wikipedia_usage/len(results):.2%})")
        print(f"  - Privacy Term Analyzer: {privacy_analyzer_usage} ({privacy_analyzer_usage/len(results):.2%})")
        
        return {
            "model": model_name,
            "examples_evaluated": len(results),
            "accuracy": accuracy,
            "tool_usage_rate": tool_usage_rate,
            "tool_usage_breakdown": {
                "calculator": calculator_usage/len(results),
                "wikipedia": wikipedia_usage/len(results),
                "privacy_analyzer": privacy_analyzer_usage/len(results)
            },
            "class_distribution": {
                "relevant": relevant_count,
                "irrelevant": irrelevant_count
            },
            "results": results
        }
    else:
        print(f"No successful evaluations for {model_name}")
        return None

def main():
    """Run a multi-model evaluation on the privacy_policy_qa subset of LegalBench."""
    print("Starting multi-model LegalBench privacy_policy_qa evaluation...")
    
    # Load examples from LegalBench
    num_examples = 150  # We'll use the same examples for all models
    examples = load_legalbench_examples(num_examples)
    
    if not examples:
        print("Failed to load examples")
        return
    
    print(f"Loaded {len(examples)} examples")
    
    # Define the models to evaluate
    models = [
        "claude-3-5-sonnet",
        # "open-mixtral-8x22b"
        # "llama-3-1-8b"
    ]
    
    # Evaluate each model
    for model_name in models:
        try:
            examples_per_model = 150
            
            # Evaluate the model
            model_results = evaluate_model(model_name, examples, examples_per_model)
            
            if model_results:
                # Save individual model results
                with open(f"legalbench_results_{model_name}.json", "w") as f:
                    json.dump(model_results, f, indent=2)
                
                print(f"Results for {model_name} saved to legalbench_results_{model_name}.json")
            
        except Exception as e:
            print(f"Error evaluating model {model_name}: {e}")
            import traceback
            print(traceback.format_exc())
    
    print("\nEvaluation complete. Individual model results saved to separate files.")

if __name__ == "__main__":
    main()