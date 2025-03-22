import os
import time
import logging
from functools import wraps
from typing import Dict, List, Optional, Union, Callable, Any
from collections import Counter, defaultdict
from openai import OpenAI
from vllm import LLM, SamplingParams

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("vllm_tracker")

class VLLMCallTracker:
    """
    A class to track vLLM API calls, providing metrics on method usage and performance.
    """
    def __init__(self, log_to_file: bool = False, log_file: str = "vllm_tracking.log"):
        self.call_counter = Counter()
        self.method_latency = defaultdict(list)
        self.prompt_counter = 0 
        self.total_tokens_generated = 0
        self.total_prompt_tokens = 0
        self.log_to_file = log_to_file
        self._model_name = None
        
        if log_to_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            logger.addHandler(file_handler)
    
    def track(self, method_name: Optional[str] = None):
        """
        Decorator to track calls to vLLM API methods.
        
        Args:
            method_name: Optional override for the method name to track
        
        Returns:
            Decorated function that tracks calls
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                tracked_name = method_name or func.__name__
                start_time = time.time()
                
                self.call_counter[tracked_name] += 1
                logger.info(f"Calling vLLM method: {tracked_name}")
                
                prompt_count = 0
                if args and len(args) > 0 and isinstance(args[0], list):
                    prompts = args[0]
                    if all(isinstance(item, str) for item in prompts):
                        prompt_count = len(prompts)
                        self.prompt_counter += prompt_count
                        
                        self.call_counter[tracked_name] += (prompt_count - 1)
                        
                        logger.info(f"Processing {prompt_count} prompts in this batch (counting as {prompt_count} API calls)")
                
                result = func(*args, **kwargs)
                
                latency = time.time() - start_time
                self.method_latency[tracked_name].append(latency)
                
                if hasattr(result, 'usage'):
                    if hasattr(result.usage, 'completion_tokens'):
                        self.total_tokens_generated += result.usage.completion_tokens
                    if hasattr(result.usage, 'prompt_tokens'):
                        self.total_prompt_tokens += result.usage.prompt_tokens
                
                elif isinstance(result, list) and len(result) > 0:
                    try:
                        import transformers
                        from transformers import AutoTokenizer
                        
                        if args and isinstance(args[0], list) and all(isinstance(x, str) for x in args[0]):
                            model_name = None
                            if hasattr(self, '_model_name'):
                                model_name = self._model_name
                            elif args and hasattr(args[0], '_model_name'):
                                model_name = args[0]._model_name
                            
                            if not model_name:
                                model_name = "gpt2"
                                
                            tokenizer = AutoTokenizer.from_pretrained(model_name)
                            prompts = args[0]
                            for prompt in prompts:
                                prompt_tokens = len(tokenizer.encode(prompt))
                                self.total_prompt_tokens += prompt_tokens
                        
                        for output in result:
                            if hasattr(output, 'outputs') and output.outputs:
                                for gen_output in output.outputs:
                                    if hasattr(gen_output, 'text') and gen_output.text:
                                        if 'tokenizer' not in locals():
                                            tokenizer = AutoTokenizer.from_pretrained("gpt2")
                                        completion_tokens = len(tokenizer.encode(gen_output.text))
                                        self.total_tokens_generated += completion_tokens
                    except (ImportError, Exception) as e:
                        logger.warning(f"Failed to count tokens: {e}")
                
                logger.info(f"Completed vLLM method: {tracked_name} (latency: {latency:.4f}s)")
                return result
            
            return wrapper
        
        return decorator
    
    def get_call_count(self, method_name: Optional[str] = None) -> Union[int, Dict[str, int]]:
        """
        Get the number of calls for a specific method or all methods.
        
        Args:
            method_name: The method name to get call count for, or None for all methods
            
        Returns:
            Call count for the method or dictionary of all method counts
        """
        if method_name:
            return self.call_counter[method_name]
        return dict(self.call_counter)
    
    def get_avg_latency(self, method_name: Optional[str] = None) -> Union[float, Dict[str, float]]:
        """
        Get the average latency for a specific method or all methods.
        
        Args:
            method_name: The method name to get latency for, or None for all methods
            
        Returns:
            Average latency for the method or dictionary of all method latencies
        """
        if method_name and method_name in self.method_latency:
            latencies = self.method_latency[method_name]
            return sum(latencies) / len(latencies) if latencies else 0
        
        result = {}
        for method, latencies in self.method_latency.items():
            result[method] = sum(latencies) / len(latencies) if latencies else 0
        return result
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics about vLLM API usage.
        
        Returns:
            Dictionary containing all tracked metrics
        """
        return {
            "call_counts": dict(self.call_counter),
            "avg_latencies": self.get_avg_latency(),
            "total_calls": sum(self.call_counter.values()),
            "total_prompts": self.prompt_counter,
            "total_tokens_generated": self.total_tokens_generated,
            "total_prompt_tokens": self.total_prompt_tokens
        }
    
    def reset(self):
        """Reset all tracking metrics."""
        self.call_counter.clear()
        self.method_latency.clear()
        self.prompt_counter = 0
        self.total_tokens_generated = 0
        self.total_prompt_tokens = 0

def track_openai_client():    
    tracker = VLLMCallTracker(log_to_file=True)
    
    class TrackedOpenAI(OpenAI):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.completions.create = tracker.track("completions.create")(self.completions.create)
            self.chat.completions.create = tracker.track("chat.completions.create")(self.chat.completions.create)
            self.embeddings.create = tracker.track("embeddings.create")(self.embeddings.create)
    
    return TrackedOpenAI, tracker

def track_direct_vllm():    
    tracker = VLLMCallTracker(log_to_file=True)
    
    class TrackedLLM(LLM):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if 'model' in kwargs:
                tracker._model_name = kwargs['model']
            self.generate = tracker.track("generate")(self.generate)
            if hasattr(self, "execute_model"):
                self.execute_model = tracker.track("execute_model")(self.execute_model)
    return TrackedLLM, tracker

if __name__ == "__main__":
    TrackedOpenAI, openai_tracker = track_openai_client()
    
    client = TrackedOpenAI(
        api_key="EMPTY", 
        base_url="http://localhost:8000/v1" 
    )
    
    completion = client.completions.create(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        prompt="Write a short poem about AI",
        max_tokens=100
    )
    
    print("OpenAI Interface Metrics:")
    print(openai_tracker.get_metrics())
    
    TrackedLLM, vllm_tracker = track_direct_vllm()
    
    model = TrackedLLM(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        dtype="auto"
    )
    
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    outputs = model.generate(
        ["Tell me about the future of AI", "What is machine learning?"],
        sampling_params
    )
    
    print("\nDirect vLLM API Metrics:")
    print(vllm_tracker.get_metrics())