# Empirical AI: Automated Kaggle Competition Solver

This project implements an automated system that generates and iteratively refines Python code to achieve high scores in Kaggle competitions. It combines large language models (LLMs), tree search algorithms, and sandboxed code execution to explore the solution space efficiently.

## Overview

The system consists of two main components:

1. **Core Generation Loop**: An iterative process that creates, evaluates, and refines code solutions using:
   - **TS Controller**: Selects parent solutions using PUCT algorithm
   - **LLM Rewriter**: Generates new code variations using advanced prompt engineering
   - **Sandbox**: Securely executes code against competition data
   - **Scorer**: Evaluates performance on validation sets

2. **Post-Hoc Analysis Module**: Tools for understanding the search process and solution space using code embeddings and visualization.

## Key Features

- **Intelligent Code Mutation**: Uses LLMs to rewrite and improve code based on performance feedback
- **Research Idea Integration**: Incorporates external research and expert strategies
- **Automated Recombination**: Combines successful approaches from different solution branches
- **Secure Execution**: Containerized sandbox with timeout and error handling
- **Tree Search Optimization**: PUCT-based selection for efficient exploration
- **Embedding Analysis**: Visualizes solution space and measures novelty

## Architecture

### Core Components

- **LLM Rewriter**: Handles code generation with context-aware prompts including competition details, parent code, and performance feedback
- **Code Execution Sandbox**: Docker-based environment with pre-installed ML libraries
- **Scorer**: Implements competition-specific evaluation metrics
- **Tree Search Controller**: Manages solution tree with flat selection strategy

### Advanced Strategies

- **AI-Powered Ideation**: Generates initial research ideas for seeding the search
- **Idea Recombination**: Automatically combines strengths from different approaches
- **Embedding-Based Analysis**: Uses text embeddings to analyze solution diversity

## Configuration

Key parameters to configure:
- `max_nodes`: Total solutions to generate (500-2000 recommended)
- `c_puct`: Exploration constant for PUCT formula
- Validation dataset split
- LLM models for different tasks

## Dependencies

- Python 3.x
- Docker (for sandboxed execution)
- Required ML libraries: pandas, scikit-learn, xgboost, etc.
- LLM APIs (e.g., Gemini series)

## Usage

1. Configure competition details and parameters
2. Initialize with baseline solution
3. Run the core generation loop
4. Analyze results using post-hoc tools
5. Select the highest-scoring solution

## Research Foundation

This implementation is based on advanced techniques in automated machine learning and evolutionary computation, adapting AlphaZero-style tree search for code optimization rather than game playing.

## License

[Add license information here]
