import os
from openai import OpenAI
from openai import OpenAIError
from anthropic import Anthropic
import backoff
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

completion_tokens = prompt_tokens = 0

api_call_tracker = {
    'enabled': False,
    'total_calls': 0,
    'calls_by_model': {},
    'calls_by_backend': {}
}

def enable_api_tracking():
    """Enable API call tracking."""
    api_call_tracker['enabled'] = True
    print("[API] API call tracking enabled")

def disable_api_tracking():
    """Disable API call tracking."""
    api_call_tracker['enabled'] = False

def reset_api_tracking():
    """Reset API call counters."""
    api_call_tracker['total_calls'] = 0
    api_call_tracker['calls_by_model'] = {}
    api_call_tracker['calls_by_backend'] = {}

def get_api_stats():
    """Get current API call statistics."""
    return {
        'total_calls': api_call_tracker['total_calls'],
        'calls_by_model': api_call_tracker['calls_by_model'].copy(),
        'calls_by_backend': api_call_tracker['calls_by_backend'].copy(),
        'enabled': api_call_tracker['enabled']
    }

def print_api_stats():
    """Print current API call statistics."""
    if api_call_tracker['enabled']:
        print(f"\n[API] API Call Statistics:")
        print(f"  Total API calls: {api_call_tracker['total_calls']}")
        for backend, count in api_call_tracker['calls_by_backend'].items():
            print(f"  {backend}: {count} calls")
        for model, count in api_call_tracker['calls_by_model'].items():
            print(f"    {model}: {count} calls")
        print()

def _track_api_call(backend, model):
    """Internal function to track API calls."""
    if api_call_tracker['enabled']:
        api_call_tracker['total_calls'] += 1
        
        if backend not in api_call_tracker['calls_by_backend']:
            api_call_tracker['calls_by_backend'][backend] = 0
        api_call_tracker['calls_by_backend'][backend] += 1
        
        if model not in api_call_tracker['calls_by_model']:
            api_call_tracker['calls_by_model'][model] = 0
        api_call_tracker['calls_by_model'][model] += 1
        
        print(f"[API] Call #{api_call_tracker['total_calls']} to {backend} ({model})")

openai_client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)
if not openai_client.api_key:
    print("Warning: OPENAI_API_KEY is not set")

vllm_client = OpenAI(
    base_url=os.environ.get("VLLM_API_BASE", "http://localhost:8000/v1"),
    api_key=os.environ.get("VLLM_API_KEY", "token-abc123")
)
print(f"VLLM client configured with base URL: {vllm_client.base_url}")

anthropic_client = Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
)
if not anthropic_client.api_key:
    print("Warning: ANTHROPIC_API_KEY is not set")

mistral_client = MistralClient(
    api_key=os.environ.get("MISTRAL_API_KEY")
)
if not os.environ.get("MISTRAL_API_KEY"):
    print("Warning: MISTRAL_API_KEY is not set")

@backoff.on_exception(backoff.expo, OpenAIError)
def completions_with_backoff(client, **kwargs):
    return client.chat.completions.create(**kwargs)

@backoff.on_exception(backoff.expo, Exception)
def anthropic_completions_with_backoff(**kwargs):
    return anthropic_client.messages.create(**kwargs)

@backoff.on_exception(backoff.expo, Exception, max_tries=5)
def mistral_completions_with_backoff(**kwargs):
    return mistral_client.chat(**kwargs)

def claude(prompt, model="claude-3-5-sonnet-20240620", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    messages=[{"role": "user", "content": prompt}]
    return anthropic_claude(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=1, stop=None)

def gpt(prompt, model="gpt-4o", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    messages = [{"role": "user", "content": prompt}]
    return chatgpt(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)

def vllm(prompt, model="meta-llama/Meta-Llama-3.1-8B-Instruct", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    messages = [{"role": "user", "content": prompt}]
    return chat_vllm(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)

def mistral(prompt, model="open-mixtral-8x22b", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    messages = [{"role": "user", "content": prompt}]
    return chat_mistral(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)

def chatgpt(messages, model="gpt-4o", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    global completion_tokens, prompt_tokens
    outputs = []
    while n > 0:
        cnt = min(n, 20)
        n -= cnt
        
        _track_api_call("openai", model)
        
        res = completions_with_backoff(client=openai_client, model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, n=cnt, stop=stop)
        outputs.extend([choice.message.content for choice in res.choices])
        completion_tokens += res.usage.completion_tokens
        prompt_tokens += res.usage.prompt_tokens
    return outputs

def chat_vllm(messages, model="meta-llama/Meta-Llama-3.1-8B-Instruct", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    global completion_tokens, prompt_tokens
    outputs = []
    
    print(f"\n=== Debug: Starting vLLM API call ===")
    print(f"Model: {model}")
    print(f"Temperature: {temperature}")
    print(f"Max tokens: {max_tokens}")
    print(f"First message content: {messages[0]['content'][:50]}...")
    
    while n > 0:
        cnt = min(n, 20)
        n -= cnt
        try:
            import time
            print(f"Calling vLLM API (requesting {cnt} completions)...")
            start_time = time.time()
            
            _track_api_call("vllm", model)
            
            res = completions_with_backoff(
                client=vllm_client, 
                model=model, 
                messages=messages, 
                temperature=temperature, 
                max_tokens=max_tokens, 
                n=cnt
            )
            
            elapsed = time.time() - start_time
            print(f"Response received in {elapsed:.2f} seconds! Choices: {len(res.choices)}")
            
            outputs.extend([choice.message.content for choice in res.choices])
            
            if hasattr(res, 'usage') and res.usage:
                if hasattr(res.usage, 'completion_tokens'):
                    completion_tokens += res.usage.completion_tokens
                if hasattr(res.usage, 'prompt_tokens'):
                    prompt_tokens += res.usage.prompt_tokens
        except Exception as e:
            print(f"Error with vLLM call: {e}")
            outputs.extend([f"vLLM Error: {str(e)}" for _ in range(cnt)])
    return outputs

def anthropic_claude(messages, model="claude-3-5-sonnet-20240620", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    outputs = []
    
    for i in range(n):
        temp_variation = min(1.0, temperature * (0.9 + (i * 0.05))) if n > 1 else temperature
        
        try:
            print(f"Making Claude API call {i+1}/{n} with temperature {temp_variation:.2f}")
            
            _track_api_call("anthropic", model)
            
            response = anthropic_completions_with_backoff(
                model=model,
                messages=messages,
                temperature=temp_variation,
                max_tokens=max_tokens,
                stop_sequences=stop
            )
            
            if response and hasattr(response, 'content') and response.content:
                text = response.content[0].text if isinstance(response.content, list) and len(response.content) > 0 else ""
                outputs.append(text)
            else:
                print(f"Warning: Empty response from Claude API (call {i+1}/{n})")
                outputs.append("")
                
        except Exception as e:
            print(f"Error in Claude API call {i+1}/{n}: {str(e)}")
            outputs.append(f"Error: {str(e)}")
    
    print(f"Generated {len(outputs)} different Claude responses")
    
    return outputs

def chat_mistral(messages, model="open-mixtral-8x22b", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    outputs = []
    
    print(f"\n=== Debug: Starting Mistral API call ===")
    print(f"Model: {model}")
    print(f"Temperature: {temperature}")
    print(f"Max tokens: {max_tokens}")
    
    mistral_messages = [
        ChatMessage(role=msg["role"], content=msg["content"]) 
        for msg in messages
    ]
    
    print(f"First message content: {mistral_messages[0].content[:50]}...")
    
    while n > 0:
        cnt = min(n, 20)
        n -= cnt
        try:
            import time
            print(f"Calling Mistral API (requesting {cnt} completions)...")
            start_time = time.time()
            
            _track_api_call("mistral", model)
            
            response = mistral_completions_with_backoff(
                model=model,
                messages=mistral_messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            elapsed = time.time() - start_time
            print(f"Response received in {elapsed:.2f} seconds! Choices: {len(response.choices)}")
            
            for choice in response.choices:
                outputs.append(choice.message.content)
                
            if cnt > len(response.choices):
                outputs.extend([response.choices[0].message.content for _ in range(cnt - len(response.choices))])
                
        except Exception as e:
            print(f"Error with Mistral API call: {e}")
            outputs.extend([f"Mistral API Error: {str(e)}" for _ in range(cnt)])
    
    return outputs

def gpt_usage(backend="gpt-4o"):
    global completion_tokens, prompt_tokens
    if backend == "gpt-4o":
        cost = completion_tokens / 1000 * 0.06 + prompt_tokens / 1000 * 0.03
    elif backend == "gpt-3.5-turbo":
        cost = completion_tokens / 1000 * 0.002 + prompt_tokens / 1000 * 0.0015
    elif backend.startswith("mistral"):
        cost = completion_tokens / 1000 * 0.007 + prompt_tokens / 1000 * 0.007
    elif backend.startswith("claude"):
        cost = completion_tokens / 1000 * 0.03 + prompt_tokens / 1000 * 0.015
    else:
        cost = 0
    return {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "cost": cost}