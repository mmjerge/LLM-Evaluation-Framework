# Pitfalls in Evaluating Inference-time Methods for Improving LLM Reliability

**Official Repository for the Research Paper**

[![Paper](https://img.shields.io/badge/Paper-TMLR%202025-blue)](https://openreview.net/forum?id=xeGWsmqFS8)
[![arXiv](https://img.shields.io/badge/arXiv-coming%20soon-red)]()

This repository contains the official implementation and experimental code for the paper *"Pitfalls in Evaluating Inference-time Methods for Improving LLM Reliability"* published in Transactions on Machine Learning Research (TMLR) 2025.

## Key Findings

🔍 **Literature Analysis**: Analysis of 4,886 papers citing Chain of Thought reveals:
- 7,635 different benchmarks used across papers
- No single benchmark used by more than 25% of evaluations
- Significant fragmentation in evaluation approaches

📊 **Experimental Results**: Comprehensive evaluation across 6 inference-time methods, 5 models, and 10 benchmarks shows:
- Methods effective on weaker models (e.g., Llama-3.1-8B) often fail to improve stronger models (e.g., GPT-4o, Claude-3.5-Sonnet)
- Performance varies significantly across different benchmark domains
- High computational costs (3.5x to 54x API calls) limit practical deployment

## Repository Structure

```
├── evaluated_methods/          # Implementations of inference-time methods
├── literature_analysis/        # Automated analysis of 4,886 papers
├── data/                      # Benchmark datasets (see Data Directory section)
├── utils/                     # Utility functions and helpers
├── assets/                    # Figures and supplementary materials
└── README.md                  # This file
```

## Methods Evaluated

Our framework includes implementations of six prominent inference-time methods:

1. **Chain of Thought** (Wei et al., 2022) - Step-by-step reasoning prompts
2. **Self-Consistency** (Wang et al., 2023) - Multiple reasoning paths with majority voting
3. **ReAct** (Yao et al., 2023) - Reasoning and acting with language models
4. **Tree of Thoughts** (Yao et al., 2024) - Tree-structured problem exploration
5. **Graph of Thoughts** (Besta et al., 2024) - Graph-based reasoning networks
6. **LLM Multi-Agent Debate** (Du et al., 2024) - Collaborative multi-agent reasoning

## Models and Benchmarks

### Models Tested
- **Advanced**: GPT-4o, Claude-3.5-Sonnet
- **Widely-used**: GPT-3.5-turbo
- **Open-weights**: Llama-3.1-8B-Instruct, Mixtral-8x22B

### Benchmarks
- **Mathematical Reasoning**: GSM8K, GSM-Symbolic, AQuA, SVAMP
- **General Knowledge**: MMLU, TruthfulQA
- **Domain-Specific**: MedQA, LegalBench
- **Specialized Tasks**: Sorting, Document Merging

## Data Directory

For convenience, we have included portions of the benchmark datasets in the `data/` directory. While the complete datasets are available from their original sources (Hugging Face or open source repositories), the included data allows for quick experimentation and testing.

The `data/` directory contains the following benchmark datasets:

- **AQuA**
- **Document Merging**
- **GSM8K**
- **GSM-Symbolic**
- **LegalBench**
- **MMLU**
- **MedQA**
- **SVAMP**
- **Sorting 032**
- **TruthfulQA**

**Note**: These are partial datasets included for convenience. For complete datasets and the most up-to-date versions, please refer to:
- [Hugging Face Datasets](https://huggingface.co/datasets)
- Original dataset repositories cited in our paper
- The benchmark-specific documentation in each method's evaluation scripts

## Installation

```bash
# Clone the repository
git clone https://github.com/mmjerge/LLM-Evaluation-Framework.git
cd LLM-Evaluation-Framework

# Create conda environment
conda env create -f environment.yaml
conda activate llm-eval
```

## Running Open Source Models with vLLM

For open source models, we use vLLM to serve models locally. 

### Resource Requirements

Different models have varying resource requirements:

- **Mixtral-8x22B**: Requires 4 A100 GPUs, 8 cores, ~700 GB memory
- **Llama-3.1-8B-Instruct**: Can run on a single A100 GPU with appropriate memory

### Setting Up vLLM Server

#### For Mixtral-8x22B (Large Model)
```bash
# Serve Mixtral 8x22B with tensor parallelism across 4 GPUs
vllm serve <path-to-mixtral-8x22b> \
  --tensor-parallel-size 4 \
  --dtype bfloat16 \
  --max-model-len 4000 \
  --enforce-eager \
  --max-parallel-loading-workers 1
```

#### For Llama-3.1-8B-Instruct (Smaller Model)
```bash
# Serve Llama-3.1-8B on single GPU
vllm serve "meta-llama/Llama-3.1-8B-Instruct" \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.95
```

### Running Evaluations with vLLM

Once the vLLM server is running (default endpoint: `http://localhost:8000/v1`), you can run various inference methods:

#### ReAct Evaluation
```bash
python evaluated_methods/ReAct/evaluations/scripts/langchain_gsm8k_react_api_count.py \
  --model vllm \
  --vllm_url http://localhost:8000/v1 \
  --model_name <model-path> \
  --num_questions 150
```

#### Unaided Baseline
```bash
python evaluated_methods/unaided/scripts/huggingface_evaluate_all.py \
  --vllm-models <model-path> \
  --datasets "legal-bench-privacy_policy_qa" "medqa" \
  --samples 100 \
  --vllm-endpoint "http://localhost:8000/v1"
```

#### Chain of Thought
```bash
python evaluated_methods/chain-of-thought/scripts/medqa_cot.py \
  --providers huggingface \
  --huggingface-models <model-path> \
  --num-samples 150 \
  --debug \
  --output-dir medqa_results
```

#### Tree of Thoughts
```bash
python evaluated_methods/tree-of-thought-llm/run.py \
  --backend meta-llama/Llama-3.1-8B-Instruct \
  --temperature 0.7 \
  --task medqa \
  --method_generate sample \
  --method_evaluate value \
  --method_select greedy \
  --prompt_sample standard \
  --n_generate_sample 5 \
  --n_evaluate_sample 1 \
  --n_select_sample 1
```

## Reproducing the Literature Analysis

Our paper includes a comprehensive analysis of 4,886 papers citing Chain of Thought (Wei et al., 2022). You can reproduce this analysis using the tools in the `literature_analysis/` directory.

### Prerequisites

To run the literature analysis, you'll need:

1. **OpenAI API Token** (required) - Used for GPT-4o automated analysis of papers
2. **Semantic Scholar API Token** (optional but recommended) - For enhanced paper retrieval and metadata

### Setup API Keys

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your_openai_api_key_here"

# Set your Semantic Scholar API key (optional)
export SEMANTIC_SCHOLAR_API_KEY="your_semantic_scholar_api_key_here"
```

## Research Contributions

### 1. Comprehensive Literature Survey
- Automated analysis of 4,886 papers using GPT-4o
- Systematic categorization of evaluation practices
- Identification of evaluation fragmentation issues

### 2. Systematic Experimental Evaluation
- First comprehensive comparison across multiple state-of-the-art models
- Evaluation on diverse benchmark suite including novel domains
- Cost analysis revealing practical deployment challenges

### 3. Reproducibility Assessment
- Direct comparison with original paper results
- Documentation of reproducibility challenges
- Recommendations for standardized evaluation protocols

## Citation

If you use this work in your research, please cite:

```bibtex
@article{
  jerge2025pitfalls,
  title={Pitfalls in Evaluating Inference-time Methods for Improving {LLM} Reliability},
  author={Michael M. Jerge and David Evans},
  journal={Transactions on Machine Learning Research},
  issn={2835-8856},
  year={2025},
  url={https://openreview.net/forum?id=xeGWsmqFS8},
  note={Reproducibility Certification, Survey Certification}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions about the paper or code, please contact:
- Michael Jerge: [mj6ux@virginia.edu](mailto:mj6ux@virginia.edu)
