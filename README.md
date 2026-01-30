# VEGA: Verifiable Enterprise GUI Agents

VEGA is a high-performance benchmark and evaluation framework designed for Verifiable Enterprise GUI Agents. It provides a robust environment for testing agents on complex, multi-step enterprise workflows using vision-language models and accessibility tree interactions.

## Key Features
- **Multi-modal Support**: Integrated support for vision-based (SoM) and text-based (Accessibility Tree) observations.
- **Enterprise Ready**: Specialized for enterprise software interaction (CRM, ERP, Project Management).
- **Verifiable Results**: Automated correctness checking using advanced LLM-based verification.
- **Scalable Execution**: Concurrent process-based task distribution for high-throughput benchmarking.

## Project Structure
- `src/vega/agent`: Core agent logic and prompt construction.
- `src/vega/browser_env`: Playwright-based browser environment.
- `src/vega/evaluation_harness`: Custom evaluators for benchmark tasks.
- `src/vega/runner`: Orchestration pipeline for large-scale evaluation.
- `src/vega/config.py`: Unified configuration management.
