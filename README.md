# HypoLens: An Interactive System for Hypothesis Testing on Large Property Graphs

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🔎 Overview

HypoLens is an interactive system for end-to-end hypothesis testing on large property graphs. It enables analysts to construct graph hypotheses through a schema-aware visual interface, evaluate them with scalable graph analysis, and explore validated follow-up hypotheses in a unified construct-test-explore workflow.

## ✨ Highlights

- **Visual Graph Hypothesis Builder:** A schema-aware interface for selecting nodes, edges, and attribute constraints, configuring the test objective, and validating the hypothesis before execution.
- **Graph Hypothesis Testing:** Sampling-based analysis with PHASE and PHASEopt, alongside full-graph evaluation and comparison modes. The result dashboard presents statistical evidence, including p-values, confidence intervals, and agreement across repeated sampling runs. An LLM explains the computed results while statistical decisions remain grounded in the testing engine.
- **Follow-up Hypothesis Recommendation:** Context-aware generation and rule-based validation of follow-up hypotheses. Selecting a recommendation loads it into the visual builder for further analysis.

HypoLens supports hypothesis testing on the MovieLens, DBLP, and Yelp property graphs.

## 📦 1. Installation

Install all dependencies:

```bash
conda create -n hypolens python=3.10 -y
conda activate hypolens
pip install -r requirements.txt
```

## 🗂️ 2. Datasets

Download the preprocessed datasets from [OneDrive](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/carrie07_connect_hku_hk/EsIJtppoMWxGgsYpXvRb7c0B1tidq4n5XH43MvKxgSHWfw?e=v48xjz) and extract them into `datasets/`:

```text
datasets/
|-- README.md
|-- citation/
|   `-- graph.pickle
|-- movielens/
|   `-- graph.pickle
`-- yelp/
    `-- graph.pickle
```

## ⚙️ 3. LLM Configuration

HypoLens supports LLM services that provide an OpenAI-compatible Chat Completions API. We use `deepseek-chat` for result interpretation and follow-up hypothesis recommendations.

Set the base URL, model name, and API key for your service before starting HypoLens:

```bash
export LLM_BASE_URL="https://your-provider.example/v1"
export LLM_MODEL="your-model-name"
export LLM_API_KEY="your-api-key"
```

Replace the example values with your service settings. Run these commands in the same shell that you will use to start HypoLens.

## 🚀 4. Running HypoLens

Start the Web application:

```bash
conda activate hypolens
python -m hypolens.web --host 127.0.0.1 --port 8000 --dataset MovieLens
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) to explore the HypoLens system.

1. **Construct a hypothesis.** Select valid node types from the schema panel, add nodes and edges to the pattern canvas, set attribute constraints, and configure the test target. Click *Parse Pattern* to validate the hypothesis. Suggested hypotheses provide starting points for each dataset.
2. **Run the test.** Choose sampling-based approximation, full-graph analysis, or comparison mode. In sampling mode, select PHASE or PHASEopt and optionally adjust the sampling ratio and number of sampled subgraphs. Comparison mode reports the fraction of sampled decisions matching the full-graph decision, alongside sampling times and confidence intervals. Inspect the result summary and expand the technical details to examine the statistical evidence.
3. **Explore follow-up hypotheses.** With an LLM service configured, request follow-up hypotheses, select a validated recommendation, and continue the analysis from the loaded pattern and test settings.

HypoLens supports hypotheses defined by a single node, an edge, or a linear path. PHASEopt is the default sampler, providing faster execution with a modest accuracy trade-off compared with PHASE.

HypoLens treats the user's claim as the null hypothesis. A higher p-value indicates weaker evidence against the claim. Thus, HypoLens reports the claim as supported when `p >= 0.05` and rejected otherwise.

## 📁 Directory Structure

```text
HypoLens/
|-- datasets/                # Graph datasets
|   `-- README.md            # Dataset download instructions
|-- dataprep/                # Dataset loading and schema extraction
|-- ght_engine/              # Sampling and statistical testing
|   |-- executor.py          # Hypothesis execution
|   |-- PHASE.py             # PHASE sampler
|   |-- Opt_PHASE.py         # PHASEopt sampler
|   |-- sampling.py          # Sampling methods
|   |-- extraction.py        # Test target extraction
|   |-- utils.py             # Data loading and statistical tests
|   `-- paths.py             # Project and dataset paths
|-- ghtql/                   # Pattern processing and schema metadata
|   `-- schema/              # Dataset schemas and attribute summaries
|-- hypolens/                # Application and LLM integration
|   |-- llm.py               # LLM service client
|   |-- tools/               # Hypothesis validation
|   `-- web/                 # Web server and interface
|       |-- app.py           # Flask application
|       `-- templates/       # Interface pages
|-- requirements.txt         # Python dependencies
|-- README.md
`-- LICENSE
```

## 📄 License

HypoLens is released under the [MIT License](LICENSE). Datasets are subject to their respective licenses.
