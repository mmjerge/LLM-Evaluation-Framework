import os
from openai import OpenAI
from openai import OpenAIError
from anthropic import Anthropic
import backoff

completion_tokens = prompt_tokens = 0

# OpenAI and Together AI client
openai_client = OpenAI(
    api_key=os.environ.get("TOGETHER_API_KEY"),
    base_url="https://api.together.xyz/v1"
)
if not openai_client.api_key:
    print("Warning: OPENAI_API_KEY is not set")
if openai_client.base_url != "https://api.openai.com/v1":
    print(f"Warning: OPENAI_API_BASE is set to {openai_client.base_url}")

# Anthropic client
anthropic_client = Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
)
if not anthropic_client.api_key:
    print("Warning: ANTHROPIC_API_KEY is not set")

@backoff.on_exception(backoff.expo, OpenAIError)
def completions_with_backoff(**kwargs):
    return openai_client.chat.completions.create(**kwargs)

@backoff.on_exception(backoff.expo, Exception)
def anthropic_completions_with_backoff(**kwargs):
    return anthropic_client.messages.create(**kwargs)

def claude(prompt, model="claude-3-5-sonnet-20240620", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    messages=[{"role": "user", "content": prompt}]
    return anthropic_claude(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=1, stop=None)

def gpt(prompt, model="gpt-4o", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    messages = [{"role": "user", "content": prompt}]
    return chatgpt(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)

def chatgpt(messages, model="gpt-4o", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    global completion_tokens, prompt_tokens
    outputs = []
    while n > 0:
        cnt = min(n, 20)
        n -= cnt
        res = completions_with_backoff(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, n=cnt, stop=stop)
        outputs.extend([choice.message.content for choice in res.choices])
        completion_tokens += res.usage.completion_tokens
        prompt_tokens += res.usage.prompt_tokens
    return outputs

def anthropic_claude(messages, model="claude-3-5-sonnet-20240620", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    outputs = []
    while n > 0:
        cnt = min(n, 20)
        n -= cnt
        
        response = anthropic_completions_with_backoff(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop_sequences=stop
        )
        
        outputs.append(response.content[0].text)
        
        if cnt > 1:
            outputs.extend([response.content[0].text for _ in range(cnt - 1)])

    return outputs

def gpt_usage(backend="gpt-4o"):
    global completion_tokens, prompt_tokens
    if backend == "gpt-4o":
        cost = completion_tokens / 1000 * 0.06 + prompt_tokens / 1000 * 0.03
    elif backend == "gpt-3.5-turbo":
        cost = completion_tokens / 1000 * 0.002 + prompt_tokens / 1000 * 0.0015
    else:
        cost = 0  # Unknown model
    return {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "cost": cost}