# Pitfalls in Evaluating Inference-time Methods for Improving LLM Reliability

**Official Repository for the Research Paper**

[![Paper](https://img.shields.io/badge/Paper-TMLR%202025-blue)](https://openreview.net/forum?id=xeGWsmqFS8)
[![arXiv](https://img.shields.io/badge/arXiv-coming%20soon-red)]()

This repository contains the official implementation and experimental code for the paper *"Pitfalls in Evaluating Inference-time Methods for Improving LLM Reliability"* published in Transactions on Machine Learning Research (TMLR) 2025.

## Abstract

Large Language Models (LLMs) have demonstrated remarkable capabilities but are still prone to outputting falsehoods using seemingly persuasive language. Many recent works attempt to address this problem by using LLMs in a framework where a single seed prompt results in a series of interactions involving augmented prompts with an otherwise unchanged LLM, and the results are aggregated with a goal of producing a more reliable output.

We consider the replicability and generalizability of evaluations of inference-time methods intended to improve the reliability of responses from base LLMs. We survey how methods have been evaluated in the literature and find a great variety of benchmarks and models in use. Motivated by this, we conduct our own evaluation to evaluate the effectiveness of a few methods across a range of benchmarks and models. **We find that while these techniques show promise in improving reliability, there is still significant variability in performance across different domains and tasks, and methods that show substantial improvements on weaker base models often do not improve reliability for better base models.**

## Key Findings

🔍 **Literature Analysis**: Analysis of 4,886 papers citing Chain of Thought reveals:
- 7,635 different benchmarks used across papers
- No single benchmark used by more than 25% of evaluations
- Significant fragmentation in evaluation approaches

📊 **Experimental Results**: Comprehensive evaluation across 6 inference-time methods, 5 models, and 10 benchmarks shows:
- Methods effective on weaker models (e.g., Llama-3.1-8B) often fail to improve stronger models (e.g., GPT-4o, Claude-3.5-Sonnet)
- Performance varies significantly across different benchmark domains
- High computational costs (3.5x to 54x API calls) limit practical deployment

⚠️ **Reproducibility Challenges**: Comparison with original papers reveals substantial discrepancies in reported vs. reproduced results

## Repository Structure

```
├── evaluated_methods/          # Implementations of inference-time methods
├── literature_analysis/        # Automated analysis of 4,886 papers
├── config/                    # Configuration files for experiments
├── tests/                     # Test suite for validation
├── utils/                     # Utility functions and helpers
├── assets/                    # Figures and supplementary materials
├── main.py                    # Main experiment runner
├── environment.yaml           # Conda environment specification
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

## Installation

```bash
# Clone the repository
git clone https://github.com/mmjerge/LLM-Evaluation-Framework.git
cd LLM-Evaluation-Framework

# Create conda environment
conda env create -f environment.yaml
conda activate llm-eval

# Install additional dependencies
pip install -r requirements.txt
```

## Reproducing Paper Results
bash# Run main experiments from the paper
python main.py --config config/paper_experiments.yaml

# Run literature analysis
python literature_analysis/analyze_papers.py

# Generate figures
python utils/generate_figures.py

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
@article{jerge2025pitfalls,
  title={Pitfalls in Evaluating Inference-time Methods for Improving LLM Reliability},
  author={Jerge, Michael and Evans, David},
  journal={Transactions on Machine Learning Research},
  year={2025},
  url={https://openreview.net/forum?id=xeGWsmqFS8}
}
```

## Acknowledgments

This work is supported in part by funds provided by the National Science Foundation, Department of Homeland Security, and IBM through the ACTION AI Institute (Award #2229876).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions about the paper or code, please contact:
- Michael Jerge: [mj6ux@virginia.edu](mailto:mj6ux@virginia.edu)

---

**Note**: This repository represents the complete experimental framework used in our TMLR 2025 paper. For the latest updates and additional resources, please check our [project page](https://github.com/mmjerge/LLM-Evaluation-Framework).
