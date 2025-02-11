from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Protocol, Optional, Type, Callable
from enum import Enum
import json
import os
from pathlib import Path
import time
from datetime import datetime
import jsonlines
import pandas as pd
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import openai
from anthropic import Anthropic

class ModelProvider(Enum):
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    
@dataclass
class ModelConfig:
    provider: ModelProvider
    model_name: str
    api_key: str
    temperature: float = 0.7
    max_tokens: int = 500

@dataclass
class EvaluationConfig:
    batch_size: int = 32
    sample_size: int = 150
    random_seed: int = 42

class LLMClient(Protocol):
    """Protocol for model clients."""
    def get_response(self, question: str, temperature: float) -> str:
        """Get response from the model."""
        ...

class InferenceMethod(Protocol):
    """Protocol for custom inference methods."""
    def __call__(
        self,
        question: str,
        model_client: LLMClient,
        **kwargs
    ) -> str:
        """Run inference on a single question."""
        ...

class HuggingFaceClient:
    def __init__(self, config: ModelConfig):
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            device_map="auto",
            token=config.api_key
        )
        self.max_tokens = config.max_tokens
        
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto"
        )
        
    def get_response(self, question: str, temperature: float) -> str:
        try:
            prompt = f"{question}"
            outputs = self.pipe(
                prompt,
                max_new_tokens=self.max_tokens,
                temperature=temperature,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )
            return outputs[0]['generated_text'].split(prompt)[-1].strip()
        except Exception as e:
            print(f"Error getting prediction from HuggingFace: {e}")
            return None

class OpenAIClient:
    def __init__(self, config: ModelConfig):
        self.client = openai.Client(api_key=config.api_key)
        self.model = config.model_name
        self.max_tokens = config.max_tokens
        
    def get_response(self, question: str, temperature: float) -> str:
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": question}],
                temperature=temperature,
                max_tokens=self.max_tokens
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error getting prediction from OpenAI: {e}")
            return None

class AnthropicClient:
    def __init__(self, config: ModelConfig):
        self.client = Anthropic(api_key=config.api_key)
        self.model = config.model_name
        self.max_tokens = config.max_tokens
        
    def get_response(self, question: str, temperature: float) -> str:
        try:
            completion = self.client.messages.create(
                model=self.model,
                messages=[{"role": "user", "content": question}],
                temperature=temperature,
                max_tokens=self.max_tokens
            )
            return completion.content[0].text
        except Exception as e:
            print(f"Error getting prediction from Anthropic: {e}")
            return None

class Benchmark(ABC):
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def load_data(self, sample_size: int, random_seed: int) -> pd.DataFrame:
        """Load and sample benchmark data."""
        pass
    
    @abstractmethod
    def extract_answer(self, solution: str) -> str:
        """Extract the answer from model output."""
        pass
    
    @abstractmethod
    def get_ground_truth(self, row: pd.Series) -> str:
        """Get ground truth answer from dataset."""
        pass
    
    @abstractmethod
    def format_prompt(self, question: str) -> str:
        """Format the question into a prompt suitable for the model."""
        pass

class GSM8KBenchmark(Benchmark):
    def load_data(self, sample_size: int, random_seed: int) -> pd.DataFrame:
        ds = load_dataset("gsm8k", "main")
        df = pd.DataFrame(ds['test'])
        return df.sample(n=sample_size, random_state=random_seed)
    
    def format_prompt(self, question: str) -> str:
        return (
            "Solve this math problem step by step and end with 'The answer is X': "
            f"{question}"
        )
    
    def extract_answer(self, solution: str) -> str:
        try:
            answer = solution.split('The answer is ')[1].split('.')[0]
            return self.clean_answer(answer)
        except:
            return 'NA'
    
    def get_ground_truth(self, row: pd.Series) -> str:
        return self.clean_answer(row.answer.split('#### ')[1])
    
    @staticmethod
    def clean_answer(answer: str) -> str:
        import re
        answer_clean = re.sub('[^\d\.]', '', answer)
        if not answer_clean:
            return 'NA'
        if '.' in answer_clean:
            answer_clean = answer_clean.rstrip('0').rstrip('.')
        return answer_clean

class GSMSymbolicBenchmark(GSM8KBenchmark):
    def load_data(self, sample_size: int, random_seed: int) -> pd.DataFrame:
        ds = load_dataset("apple/GSM-Symbolic", "main")
        df = pd.DataFrame(ds['test'])
        return df.sample(n=sample_size, random_state=random_seed)

class BenchmarkRegistry:
    _benchmarks: Dict[str, Type[Benchmark]] = {
        "gsm8k": GSM8KBenchmark,
        "gsm-symbolic": GSMSymbolicBenchmark,
    }
    
    @classmethod
    def register(cls, name: str, benchmark_class: Type[Benchmark]):
        cls._benchmarks[name] = benchmark_class
    
    @classmethod
    def get(cls, name: str) -> Optional[Type[Benchmark]]:
        return cls._benchmarks.get(name)
    
    @classmethod
    def list_benchmarks(cls) -> List[str]:
        return list(cls._benchmarks.keys())

def direct_inference(
    question: str,
    model_client: LLMClient,
    temperature: float = 0.7
) -> str:
    """Simple direct inference method."""
    return model_client.get_response(question, temperature)

def self_consistency_inference(
    question: str,
    model_client: LLMClient,
    n_samples: int = 5,
    temperature: float = 0.7
) -> str:
    """Self-consistency inference method."""
    responses = [
        model_client.get_response(question, temperature)
        for _ in range(n_samples)
    ]
    
    response_counts = {}
    for response in responses:
        if response:
            response_counts[response] = response_counts.get(response, 0) + 1
    
    if not response_counts:
        return None
    
    return max(response_counts.items(), key=lambda x: x[1])[0]

class Evaluator:
    def __init__(
        self,
        model_config: ModelConfig,
        eval_config: EvaluationConfig,
        benchmark: Benchmark,
        inference_method: InferenceMethod,
        output_dir: str = "results",
        inference_kwargs: Dict[str, Any] = None
    ):
        self.model_config = model_config
        self.eval_config = eval_config
        self.benchmark = benchmark
        self.inference_method = inference_method
        self.inference_kwargs = inference_kwargs or {}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        if model_config.provider == ModelProvider.HUGGINGFACE:
            self.client = HuggingFaceClient(model_config)
        elif model_config.provider == ModelProvider.OPENAI:
            self.client = OpenAIClient(model_config)
        elif model_config.provider == ModelProvider.ANTHROPIC:
            self.client = AnthropicClient(model_config)
        else:
            raise ValueError(f"Unsupported provider: {model_config.provider}")
    
    def run_evaluation(self) -> pd.DataFrame:
        """Run evaluation using the specified inference method."""
        print(f"Starting evaluation on {self.benchmark.name} with {self.model_config.model_name}")
        
        sample = self.benchmark.load_data(
            self.eval_config.sample_size,
            self.eval_config.random_seed
        )
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = (
            f"{self.benchmark.name}_{self.model_config.provider.value}_"
            f"{self.model_config.model_name.replace('/', '_')}_{timestamp}.jsonl"
        )
        output_path = self.output_dir / output_file
        
        results = []
        for i in range(0, len(sample), self.eval_config.batch_size):
            batch = sample.iloc[i:i + self.eval_config.batch_size]
            
            for row in batch.itertuples():
                prompt = self.benchmark.format_prompt(row.question)
                prediction = self.inference_method(
                    prompt,
                    self.client,
                    temperature=self.model_config.temperature,
                    **self.inference_kwargs
                )
                
                predicted_answer = self.benchmark.extract_answer(prediction)
                true_answer = self.benchmark.get_ground_truth(row)
                
                result = {
                    "question": row.question,
                    "prediction": prediction,
                    "predicted_answer": predicted_answer,
                    "true_answer": true_answer,
                    "correct": predicted_answer == true_answer,
                    "provider": self.model_config.provider.value,
                    "model": self.model_config.model_name
                }
                results.append(result)
                
                with jsonlines.open(output_path, mode='a') as writer:
                    writer.write(result)
            
            print(f"Processed {min(i + self.eval_config.batch_size, len(sample))}/{len(sample)} questions")
        
        results_df = pd.DataFrame(results)
        accuracy = results_df['correct'].mean()
        print(f"Evaluation completed! Final accuracy: {accuracy:.2%}")
        print(f"Results saved to {output_path}")
        
        return results_df

def main():
    """Example usage showing how to use custom inference methods."""
    def custom_inference(
        question: str,
        model_client: LLMClient,
        temperature: float = 0.7,
        custom_param: str = "default"
    ) -> str:
        augmented_prompt = f"{custom_param}: {question}"
        return model_client.get_response(augmented_prompt, temperature)
    
    model_config = ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name="gpt-4",
        api_key="your_api_key"
    )
    
    eval_config = EvaluationConfig(
        batch_size=32,
        sample_size=150,
        random_seed=42
    )
    
    inference_methods = {
        "direct": direct_inference,
        "self_consistency": self_consistency_inference,
        "custom": custom_inference
    }
    
    inference_kwargs = {
        "direct": {},
        "self_consistency": {"n_samples": 5},
        "custom": {"custom_param": "My custom prefix"}
    }
    
    results = {}
    for benchmark_name in BenchmarkRegistry.list_benchmarks():
        benchmark_class = BenchmarkRegistry.get(benchmark_name)
        if benchmark_class:
            benchmark = benchmark_class(benchmark_name)
            
            for method_name, method in inference_methods.items():
                evaluator = Evaluator(
                    model_config=model_config,
                    eval_config=eval_config,
                    benchmark=benchmark,
                    inference_method=method,
                    inference_kwargs=inference_kwargs[method_name]
                )
                results[(benchmark_name, method_name)] = evaluator.run_evaluation()

if __name__ == "__main__":
    main()