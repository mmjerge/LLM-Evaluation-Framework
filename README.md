# LLM Evaluation Framework

A modular framework for evaluating Large Language Models (LLMs) across different benchmarks using custom inference methods. This repository contains the evaluation framework and implementations of inference-time methods studied in the paper.

## Overview

This framework allows researchers and practitioners to:
- Evaluate different LLMs (OpenAI, Anthropic, HuggingFace models)
- Test custom inference methods
- Run evaluations across multiple benchmarks
- Compare different approaches systematically

## Installation

```bash
conda env create -f environment.yaml
```

Required dependencies:
- transformers
- openai
- anthropic
- pandas
- datasets
- jsonlines
- torch

## Quick Start

```python
from framework import (
    Evaluator, ModelConfig, EvaluationConfig,
    ModelProvider, GSM8KBenchmark
)

def my_inference_method(
    question: str,
    model_client: LLMClient,
    temperature: float = 0.7,
    **kwargs
) -> str:
    return model_client.get_response(question, temperature)

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

evaluator = Evaluator(
    model_config=model_config,
    eval_config=eval_config,
    benchmark=GSM8KBenchmark("gsm8k"),
    inference_method=my_inference_method
)

results = evaluator.run_evaluation()
```

## Framework Components

### Model Providers

The framework supports multiple model providers:
- OpenAI (GPT-3.5, GPT-4)
- Anthropic (Claude)
- HuggingFace (Local models)

### Benchmarks

Adding a new benchmark:
```python
class MyBenchmark(Benchmark):
    def load_data(self, sample_size: int, random_seed: int) -> pd.DataFrame:
        # Load your dataset
        pass
    
    def format_prompt(self, question: str) -> str:
        # Format the question for the model
        pass
    
    def extract_answer(self, solution: str) -> str:
        # Extract answer from model output
        pass
    
    def get_ground_truth(self, row: pd.Series) -> str:
        # Get ground truth answer
        pass

BenchmarkRegistry.register("my-benchmark", MyBenchmark)
```

### Inference Methods

Along with the framework to evaluate inference-time methods, this repository also includes methods that were evaluated on different models and benchmarks. These can be found in the /evaluated_methods directory.

Creating a custom inference method:
```python
def custom_inference(
    question: str,
    model_client: LLMClient,
    temperature: float = 0.7,
    **kwargs
) -> str:
    # Implement your inference strategy
    # Can use any combination of:
    # - Chain-of-thought prompting
    # - Few-shot examples
    # - Multiple passes
    # - Custom prompt engineering
    # - etc.
    return model_client.get_response(question, temperature)
```

## Configuration

### ModelConfig
```python
@dataclass
class ModelConfig:
    provider: ModelProvider
    model_name: str
    api_key: str
    temperature: float = 0.7
    max_tokens: int = 500
```

### EvaluationConfig
```python
@dataclass
class EvaluationConfig:
    batch_size: int = 32
    sample_size: int = 150
    random_seed: int = 42
```

## Results Format

The evaluation results are saved in JSONL format with the following structure:
```json
{
    "question": "original question",
    "prediction": "model's full response",
    "predicted_answer": "extracted answer",
    "true_answer": "ground truth",
    "correct": true/false,
    "provider": "model provider",
    "model": "model name"
}
```

## Advanced Usage

### Comparing Multiple Methods

```python
# Define methods to compare
inference_methods = {
    "direct": direct_inference,
    "self_consistency": self_consistency_inference,
    "custom": custom_inference
}

# Configure method-specific parameters
inference_kwargs = {
    "direct": {},
    "self_consistency": {"n_samples": 5},
    "custom": {"custom_param": "value"}
}

# Run evaluations
results = {}
for benchmark_name in BenchmarkRegistry.list_benchmarks():
    benchmark = BenchmarkRegistry.get(benchmark_name)(benchmark_name)
    
    for method_name, method in inference_methods.items():
        evaluator = Evaluator(
            model_config=model_config,
            eval_config=eval_config,
            benchmark=benchmark,
            inference_method=method,
            inference_kwargs=inference_kwargs[method_name]
        )
        results[(benchmark_name, method_name)] = evaluator.run_evaluation()
```

## License

MIT License