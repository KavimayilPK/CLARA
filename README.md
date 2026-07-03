# CLARA - Causal Learning via Adaptive Resampling and Aggregation

Source code for the experiments reported in our BIBM paper.
This repository covers **CLARA** running two constraint variants
(**Full Graphs** and **Common Edges**), two LLM baselines
(**LLM One Shot** and **LLM One Shot + Theory Refinement**),
and two data-based baselines (**PC** and **FCI**).

---

## Overview

CLARA uses a large language model (LLM) as a hypothesis generator for Bayesian network
structure learning. The LLM is prompted `n_initial` times — each call independently samples
the graph 5 times and takes their union — producing a set of candidate graphs that are
scored and used to extract structural patterns.

The two **constraint variants** tested are:

| Variant | `--constraint` flag | What goes into the LLM prompt |
|---|---|---|
| Full Graphs | `full_graph` | The entire top-k graphs from history |
| Common Edges | `direct` | Only the most-agreed-upon direct edges from history |

---

## Repository Structure

```
CLARA/
│
├── clara_main.py               # Main entry point — orchestrates LLM calls, scoring, logging
├── validate_llm_graph.py       # Parses and validates LLM graph outputs (DAG check, variable check)
├── break_cycles.py             # Removes cycles from LLM-generated graphs to enforce DAG structure
├── domain_config.py            # Single source of truth for domain paths (prompts, expert graph, blacklist)
├── eval_clara_outputs.py       # Evaluation: compute SHD, SID, precision, recall from output CSVs
├── create_history.py           # Utility: inspect and summarise CLARA run history JSON files
│
├── domains/
│   ├── alarm/
│   │   ├── expert.graph        # Ground-truth Bayesian network (ALARM benchmark)
│   │   ├── blacklist.csv       # Forbidden edges for the ALARM domain
│   │   └── prompts/
│   │       ├── G0.txt              # Initial prompt (no constraints)
│   │       ├── G1MandatoryEdge.txt
│   │       └── LLMoneShot.txt
│   └── apo/
│       ├── expert.graph        # Ground-truth Bayesian network (APO domain)
│       ├── blacklist.csv       # Forbidden edges for the APO domain
│       └── prompts/
│           ├── G0_Prompt.txt       # Initial prompt (no constraints)
│           ├── G1_Prompt.txt       # Prompt — common-edges variant
│           └── G1_FullGraphs.txt   # Prompt — full-graphs variant
│
└── LLM-guided-CBNs/
    ├── no_refinement_score.py      # Scores LLM graphs with StructuredBIC + TreeBIC
    ├── tree_bic_score.py           # TreeBIC scoring implementation
    ├── compute_metrics.py          # Computes SHD, SID and other metrics over saved result graphs
    ├── ancestral_constraints.csv   # Reference file for ancestral constraint definitions
    ├── required_edges.csv          # Reference file for required/mandatory edges
    │
    ├── results/
    │   ├── expert.graph            # Expert graph used by compute_metrics (APO)
    │   └── alarm_expert.graph      # Expert graph used by compute_metrics (ALARM)
    │
    ├── datasets/
    │   └── create_bootstrap_samples.py   # Generates bootstrap samples from a dataset CSV
    │
    ├── initial_edge_lists/
    │   └── get_llm_graphs_from_edges.py  # Reconstructs graphs from pre-saved edge-list CSVs
    │
    └── data_based_methods/
        ├── learn_from_data_pc.py   # PC algorithm baseline (pgmpy)
        └── learn_from_data_fci.py  # FCI algorithm baseline (causal-learn)
```

> **Note:** Dataset CSV files are not included in this repository.
> Place your dataset at the path configured in `domain_config.py` before running.

---

## Running LLM Baselines

### LLM One Shot

Calls the LLM once (no iterative loop) and scores the resulting graph directly.
Uses the initial prompt with no constraints.

```bash
python clara_main.py \
    --domain alarm \
    --constraint none \
    --llm-backend openai \
    --llm-model gpt-4o \
    --n-initial 1 \
    --no-refinement
```

### LLM One Shot + Theory Refinement

Generates initial edge lists from a single LLM call, then applies theory-guided
hill-climb refinement.

**Step 1 — Generate initial edges** (same as LLM One Shot above).

**Step 2 — Run theory refinement:**

```bash
cd LLM-guided-CBNs
python refine_step_1.py --domain alarm
```

Place the initial edge CSV files (named `initial_edges_*.csv`) in
`LLM-guided-CBNs/initial_edge_lists/` before running.
Pass `--no-required-edges` to skip required-edge enforcement and run plain subtractive refinement only.

#### Files used by theory refinement

| File | Role |
|---|---|
| `refine_step_1.py` | Orchestrator — reads edge CSVs, applies blacklist, calls hill climbing |
| `structure_learning.py` | Implements `refine_bn()` and `StructuredBicScore` |
| `hill_climb.py` | Hill-climb search used by `structure_learning.py` |
| `ancestral_constraint_score.py` | Enforces ancestral constraints during hill climbing |
| `required_edges.csv` | Edges that must appear in the refined graph |
---

## Setup

### 1. Install dependencies

```bash
pip install -r LLM-guided-CBNs/requirements.txt
```

Key libraries: `pgmpy`, `causal-learn`, `cdt`, `networkx`, `openai`, `pandas`, `numpy`.

### 2. Set your API key

For OpenAI or any cloud LLM backend, set the relevant environment variable:

```bash
export OPENAI_API_KEY="sk-..."          # OpenAI
export TOGETHER_API_KEY="..."           # Together AI
export OPENROUTER_API_KEY="..."         # OpenRouter
```

### 3. Add your dataset

Place the dataset CSV at the path listed in `domain_config.py` for your domain.
For example, for the `alarm` domain:

```
domains/alarm/dataset.csv
```

---

## Running CLARA

### Full Graphs variant

```bash
python clara_main.py \
    --domain alarm \
    --constraint full_graph \
    --llm-backend openai \
    --llm-model gpt-4o \
    --n-initial 5 \
    --no-refinement
```

### Common Edges variant

```bash
python clara_main.py \
    --domain alarm \
    --constraint direct \
    --llm-backend openai \
    --llm-model gpt-4o \
    --n-initial 5 \
    --no-refinement
```

### Key CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--domain` | *(required)* | Domain to run (`alarm` or `apo`) |
| `--constraint` | *(required)* | Constraint type: `full_graph`, `direct`, `motif`, `ancestral`, `none`, etc. |
| `--llm-backend` | `openai` | LLM provider: `openai`, `together`, `openrouter`, `ollama`, `huggingface`, `groq` |
| `--llm-model` | backend default | Model name (e.g. `gpt-4o`, `meta-llama/Llama-3.3-70B-Instruct-Turbo`) |
| `--n-initial` | 5 | Number of initial graphs to generate |
| `--top-k` | 5 | Number of top graphs used for constraint extraction |
| `--no-refinement` | off | Run in non-iterative mode — scores LLM graphs directly |
| `--output-dir` | `clara_output` | Directory where prompts, summaries, and logs are saved |

---

## Running Data-Based Baselines

These scripts run independently and read data directly from the `datasets/` folder.

### PC Algorithm

```bash
cd LLM-guided-CBNs/data_based_methods
python learn_from_data_pc.py
```

### FCI Algorithm

```bash
cd LLM-guided-CBNs/data_based_methods
python learn_from_data_fci.py
```

Edit the `data_file`, `sample_id`, and `domain` variables at the top of each script
to point to your dataset and domain.

---

## Evaluation

### Compute metrics (SHD, SID, precision, recall) on CLARA outputs

Drop the output edge-list CSVs from a CLARA run into `clara_eval/`, then run:

```bash
python eval_clara_outputs.py --domain alarm
```

This compares each output graph against the expert ground-truth graph and writes a
summary CSV with all metrics.

### Compute metrics from saved result graphs

```bash
cd LLM-guided-CBNs
python compute_metrics.py
```

---

## Supported LLM Backends

| Backend | `--llm-backend` value | Notes |
|---|---|---|
| OpenAI | `openai` | Requires `OPENAI_API_KEY` |
| Together AI | `together` | Requires `TOGETHER_API_KEY` |
| OpenRouter | `openrouter` | Requires `OPENROUTER_API_KEY` |
| Groq | `groq` | Requires `GROQ_API_KEY` |
| Ollama (local) | `ollama` | Default URL: `http://localhost:11434` |
| HuggingFace TGI / vLLM | `huggingface` | Pass endpoint with `--llm-url` |

---

## Adding a New Domain

1. Create a folder `domains/<name>/` with:
   - `expert.graph` — ground-truth graph
   - `blacklist.csv` — forbidden edges (columns: `X`, `Y`)
   - `prompts/G0.txt` — the initial prompt
2. Add an entry to the `DOMAINS` dict in `domain_config.py`.
3. Add a branch in `build_state_names()` in `domain_config.py` for the variable state spaces.
