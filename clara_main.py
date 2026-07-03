"""CLARA — Causal Learning via Adaptive Refinement and Aggregation
Main orchestrator.

Usage:
  # OpenAI
  python clara_main.py --domain alarm --constraint motif \\
      --llm-backend openai --llm-model gpt-4o

  # Ollama (local, OpenAI-compatible API)
  python clara_main.py --domain alarm --constraint ancestral \\
      --llm-backend ollama --llm-model llama3

  # HuggingFace TGI / vLLM (OpenAI-compatible endpoint)
  python clara_main.py --domain alarm --constraint direct \\
      --llm-backend huggingface \\
      --llm-model mistralai/Mixtral-8x7B-Instruct-v0.1 \\
      --llm-url https://<your-tgi-endpoint>/v1

  # OpenRouter (Qwen3-32B, thinking disabled)
  python clara_main.py --domain alarm --constraint motif \\
      --llm-backend openrouter --llm-model qwen/qwen3-32b

  # Together AI (Llama-3.3-70B-Instruct-Turbo)
  python clara_main.py --domain alarm --constraint motif \\
      --llm-backend together --llm-model meta-llama/Llama-3.3-70B-Instruct-Turbo

  # Together AI (Qwen2.5-7B-Instruct-Turbo)
  python clara_main.py --domain alarm --constraint motif \\
      --llm-backend together --llm-model Qwen/Qwen2.5-7B-Instruct-Turbo

  # Together AI (Qwen3.5-9B, thinking auto-disabled)
  python clara_main.py --domain alarm --constraint motif \\
      --llm-backend together --llm-model Qwen/Qwen3.5-9B

  # Together AI (Qwen3-32B, thinking auto-disabled)
  python clara_main.py --domain alarm --constraint motif \\
      --llm-backend together --llm-model Qwen/Qwen3-32B

  # Together AI (Gemma-4-31B)
  python clara_main.py --domain alarm --constraint motif \\
      --llm-backend together --llm-model google/gemma-4-31B-it

  # Together AI (Qwen3-235B MoE, thinking auto-disabled)
  python clara_main.py --domain alarm --constraint motif \\
      --llm-backend together --llm-model Qwen/Qwen3-235B-A22B-Instruct-2507-tput

  # Together AI (Qwen3.5-397B MoE, thinking disabled)
  python clara_main.py --domain alarm --constraint motif \\
      --llm-backend together --llm-model Qwen/Qwen3.5-397B-A17B

  # Together AI (DeepSeek-V4-Pro)
  python clara_main.py --domain alarm --constraint motif \\
      --llm-backend together --llm-model deepseek-ai/DeepSeek-V4-Pro

Flow:
  Phase 0  : Generate n_initial LLM graphs (G0, no constraints).
             Each graph = union of N_RESPONSES=5 independent LLM samples.
             Refine all n_initial graphs → history.

  Loop     : Extract top-k patterns from history (constraint type flag) →
             build G1 prompt → generate 1 new graph (union of 5 samples) →
             refine → add to history → early-stop if BIC does not improve.

  Final    : Combine top-k from history into union / consensus graph.

All steps are logged to clara_output/clara_<timestamp>.log for debugging.
"""

import sys
import os
import json
import argparse
import random
import csv
import logging
import subprocess
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import networkx as nx
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT    = Path(__file__).parent
LLM_DIR = ROOT / "LLM-guided-CBNs"

sys.path.insert(0, str(ROOT))

import domain_config
import validate_llm_graph
from validate_llm_graph import construct_llm_graph, is_valid_dag, check_variables
from break_cycles import build_graph, find_cycle_edges, compute_frequencies
from break_cycles import break_cycles as _break_cycles
# from extract_motif import (
#     build_digraph,
#     aggregate_ancestral_edges,
#     aggregate_motif_counts,
#     select_motif_constraints,
#     select_ancestral_constraints,
#     MOTIF_PROMPT_SYMBOLS,
# )

# ---------------------------------------------------------------------------
# Constants — gen_llm_response.py
# ---------------------------------------------------------------------------
N_RESPONSES           = 5      # independent LLM samples per graph (union)
MAX_RETRIES           = 5      # retries per sample if validation fails
TEMPERATURE           = 0.7    # LLM temperature for sampling
MAX_COMPLETION_TOKENS = 8192   # cap on output tokens — prevents mid-response truncation

TOP_K             = 7    # top graphs used for constraint extraction each run
N_INITIAL         = TOP_K # must be >= TOP_K; set together so they stay in sync
MAX_RUNS          = 5    # max iterative runs

# File system layout
HISTORY_JSON      = LLM_DIR / "g1_refined_results.json"
BIC_SCORES_CSV    = LLM_DIR / "bic_scores.csv"
INITIAL_EDGES_DIR = LLM_DIR / "initial_edge_lists"
REQUIRED_CSV      = LLM_DIR / "required_edges.csv"
ANCESTRAL_CSV     = LLM_DIR / "ancestral_constraints.csv"

LLM_RESPONSES_DIR = ROOT / "llm_responses"
LLM_EDGES_DIR     = ROOT / "llm_edges"

# ---------------------------------------------------------------------------
# Logger — module-level so every function can use it after setup_logging()
# ---------------------------------------------------------------------------
log = logging.getLogger("clara")


def setup_logging(output_dir: Path, run_id: str) -> Path:
    """Configure console + file logging. Returns path to the log file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path  = output_dir / f"clara_{run_id}.log"

    log.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler — captures everything (DEBUG+)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    # Console handler — INFO+ only (less verbose on screen)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    log.addHandler(fh)
    log.addHandler(ch)

    log.info("Log file: %s", log_path)
    return log_path


# ---------------------------------------------------------------------------
# LLM client (OpenAI SDK — supports OpenAI, Ollama, and TGI-compatible endpoints)
# ---------------------------------------------------------------------------

class LLMClient:
    """
    Thin wrapper around the OpenAI SDK.

    Backend          base_url                         api_key
    -------          --------                         -------
    openai           https://api.openai.com/v1        OPENAI_API_KEY env var
    ollama           http://localhost:11434/v1         "ollama" (required but unused)
    huggingface      <user-supplied TGI/vLLM URL>/v1   HF_API_KEY env var
    groq             https://api.groq.com/openai/v1   GROQ_API_KEY env var
    openrouter       https://openrouter.ai/api/v1     OPENROUTER_API_KEY env var
    together         https://api.together.xyz/v1      TOGETHER_API_KEY env var
    """

    _SYSTEM_MSG = (
        "You are a helpful assistant. Think carefully and logically before you provide your answer. "
        "Only include answers you are highly confident about — do not hallucinate or speculate. "
        "Follow the output format and instructions specified by the user exactly."
    )

    # Price per 1M tokens: (input, output). Verify at platform pricing pages.
    _PRICE_PER_1M: Dict[str, Tuple[float, float]] = {
        # OpenAI
        "gpt-4o-mini-2025-04-16":  (0.15,  0.60),
        "gpt-4o-mini":             (0.15,  0.60),
        "gpt-4o":                  (2.50, 10.00),
        "gpt-5.2":                 (1.75, 14.00),
        # Google
        "gemini-2.0-flash":        (0.10,  0.40),
        "gemini-1.5-flash":        (0.075, 0.30),
        "gemini-1.5-pro":          (1.25,  5.00),
        # Groq
        "llama-3.3-70b-versatile": (0.59,  0.79),
        # OpenRouter
        "qwen/qwen3-32b":          (0.10,  0.30),
        # Together AI
        "meta-llama/Llama-3.3-70B-Instruct-Turbo":          (0.18, 0.18),
        "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8": (0.27, 0.27),
        "Qwen/Qwen2.5-7B-Instruct-Turbo":                   (0.10, 0.10),
        "Qwen/Qwen3.5-9B":                                  (0.17, 0.25),
        "Qwen/Qwen3-32B":                                   (0.30, 0.30),
        "google/gemma-4-31B-it":                            (0.20, 0.20),
        "Qwen/Qwen3-235B-A22B-Instruct-2507-tput":         (0.60, 0.60),
        "Qwen/Qwen3.5-397B-A17B":                          (0.60, 0.60),
        "deepseek-ai/DeepSeek-V4-Pro":                     (1.74, 3.48),
    }

    def __init__(self, base_url: str, api_key: str, model: str, extra_body: Optional[Dict] = None):
        from openai import OpenAI
        self.client               = OpenAI(base_url=base_url, api_key=api_key)
        self.model                = model
        self._extra_body          = extra_body or {}
        self._calls               = 0
        self._prompt_tokens       = 0
        self._completion_tokens   = 0
        log.debug("LLMClient created  base_url=%s  model=%s", base_url, model)

    def complete(
        self,
        prompt:      str,
        temperature: float         = TEMPERATURE,
        seed:        Optional[int] = None,
    ) -> str:
        kw: Dict = dict(
            model       = self.model,
            messages    = [
                {"role": "system", "content": self._SYSTEM_MSG},
                {"role": "user",   "content": prompt},
            ],
            temperature = temperature,
            max_tokens  = MAX_COMPLETION_TOKENS,
        )
        if seed is not None:
            kw["seed"] = seed
        if self._extra_body:
            kw["extra_body"] = self._extra_body
        resp = self.client.chat.completions.create(**kw)
        if resp.usage:
            self._calls             += 1
            self._prompt_tokens     += resp.usage.prompt_tokens
            self._completion_tokens += resp.usage.completion_tokens
        choice = resp.choices[0]
        if choice.finish_reason == "length":
            log.warning("Response truncated at max_tokens=%d — consider raising MAX_COMPLETION_TOKENS", MAX_COMPLETION_TOKENS)
        return choice.message.content.strip()

    def log_usage_summary(self, elapsed_s: float) -> None:
        inp_price, out_price = self._PRICE_PER_1M.get(self.model, (0.0, 0.0))
        cost  = (self._prompt_tokens / 1e6 * inp_price) + (self._completion_tokens / 1e6 * out_price)
        known = self.model in self._PRICE_PER_1M
        log.info("=" * 60)
        log.info("RUN COST SUMMARY")
        log.info("  Wall-clock time     : %.1f s  (%.1f min)", elapsed_s, elapsed_s / 60)
        log.info("  LLM API calls       : %d", self._calls)
        log.info("  Prompt tokens       : %d", self._prompt_tokens)
        log.info("  Completion tokens   : %d", self._completion_tokens)
        log.info("  Total tokens        : %d", self._prompt_tokens + self._completion_tokens)
        if known:
            log.info("  Estimated cost (USD): $%.4f  (model=%s)", cost, self.model)
        else:
            log.info("  Estimated cost (USD): N/A  (no price entry for model=%s)", self.model)
        log.info("=" * 60)


def build_client(backend: str, model: Optional[str], url: Optional[str]) -> LLMClient:
    if backend == "openai":
        client = LLMClient(
            base_url = "https://api.openai.com/v1",
            api_key  = os.getenv("OPENAI_API_KEY", ""),
            model    = model or "gpt-4o",
        )
    elif backend == "ollama":
        client = LLMClient(
            base_url = (url or "http://localhost:11434") + "/v1",
            api_key  = "ollama",
            model    = model or "llama3",
        )
    elif backend == "huggingface":
        if not model:
            raise ValueError("--llm-model is required for the huggingface backend")
        if not url:
            raise ValueError(
                "--llm-url is required for the huggingface backend "
                "(provide your TGI or vLLM OpenAI-compatible endpoint, e.g. https://host/v1)"
            )
        client = LLMClient(
            base_url = url,
            api_key  = os.getenv("HF_API_KEY") or os.getenv("HUGGINGFACE_API_KEY") or "",
            model    = model,
        )
    elif backend == "groq":
        client = LLMClient(
            base_url = "https://api.groq.com/openai/v1",
            api_key  = os.getenv("GROQ_API_KEY", ""),
            model    = model or "llama-3.3-70b-versatile",
        )
    elif backend == "openrouter":
        client = LLMClient(
            base_url   = "https://openrouter.ai/api/v1",
            api_key    = os.getenv("OPENROUTER_API_KEY", ""),
            model      = model or "qwen/qwen3-32b",
            extra_body = {"reasoning": {"effort": "none"}},
        )
    elif backend == "together":
        _together_model = model or "Qwen/Qwen3.5-397B-A17B"
        _together_extra = (
            {"chat_template_kwargs": {"enable_thinking": False}}
            if "Qwen3" in _together_model or "qwen3" in _together_model
            else {}
        )
        client = LLMClient(
            base_url   = "https://api.together.xyz/v1",
            api_key    = os.getenv("TOGETHER_API_KEY", ""),
            model      = _together_model,
            extra_body = _together_extra,
        )
    else:
        raise ValueError(f"Unknown backend: {backend!r}")

    log.info("LLM backend: %s  model: %s", backend, client.model)
    return client


# ---------------------------------------------------------------------------
# Edge parsing + validation
# ---------------------------------------------------------------------------

def _extract_edges_from_text(text: str) -> List[Tuple[str, str]]:
    """Parse edges from the 'Section 1' block of an LLM response."""
    edges, in_sec = [], False
    for line in text.splitlines():
        line = line.strip().lstrip("#*->`~_").strip()
        if line.lower().startswith("section 1"):
            in_sec = True
            continue
        if line.lower().startswith("section 2"):
            break
        if in_sec and "," in line:
            parts = line.split(",")
            if len(parts) == 2:
                c, e = parts[0].strip(), parts[1].strip()
                if c and e:
                    edges.append((c, e))
    return edges


def _load_blacklist(path: Path) -> List[Tuple[str, str]]:
    if not path.exists():
        log.warning("Blacklist file not found: %s — blacklist check skipped", path)
        return []
    import pandas as pd
    df = pd.read_csv(path)
    bl = list(zip(df["X"], df["Y"]))
    log.debug("Loaded %d blacklisted edges from %s", len(bl), path)
    return bl


def _validate_edges(
    edges:      List[Tuple[str, str]],
    black_list: List[Tuple[str, str]],
) -> Tuple[bool, Dict]:
    G             = construct_llm_graph(edges)
    invalid_nodes = check_variables(G.nodes())
    cycles        = is_valid_dag(G)
    blacklisted   = [e for e in edges if e in black_list] or None
    ok            = not invalid_nodes and not cycles and not blacklisted
    return ok, {"invalid_nodes": invalid_nodes, "cycles": cycles, "blacklisted": blacklisted}


# ---------------------------------------------------------------------------
# Graph generation: replicates generate_response() + parse_edges() from
# gen_llm_response.py, using a configurable LLMClient.
# ---------------------------------------------------------------------------

def generate_graph_union(
    client:     LLMClient,
    prompt:     str,
    run_number: int,
    black_list: List[Tuple[str, str]],
    label:      str = "",
) -> Optional[List[Tuple[str, str]]]:
    """
    Generate N_RESPONSES independent LLM samples, validate each (retrying up to
    MAX_RETRIES times), then return the union of all valid edges (with cycles
    broken).  Returns None if no valid response was obtained.

    Saves per-run artefacts under:
      llm_responses/run_NNN/response_XX.json
      llm_edges/run_NNN/response_XX_edges.csv
      llm_edges/run_NNN/initial_edges_<model>-run_NNN.csv   ← union CSV
    """
    run_tag = f"run_{run_number:03d}"
    run_dir = LLM_RESPONSES_DIR / run_tag
    csv_dir = LLM_EDGES_DIR     / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    log.info("[%s] Generating %d samples (LLM run %s) ...", label, N_RESPONSES, run_tag)

    all_edges:             Set[Tuple[str, str]] = set()
    individual_csv_paths:  List[Path]           = []

    for i in range(1, N_RESPONSES + 1):
        saved = False
        for attempt in range(1, MAX_RETRIES + 1):
            seed = random.randint(0, 2**31 - 1)
            log.debug(
                "[%s] sample %d/%d  attempt %d/%d  seed=%d",
                label, i, N_RESPONSES, attempt, MAX_RETRIES, seed,
            )
            try:
                raw   = client.complete(prompt, seed=seed)
                edges = _extract_edges_from_text(raw)
                ok, issues = _validate_edges(edges, black_list)

                if not ok:
                    log.debug(
                        "[%s] sample %d attempt %d INVALID — "
                        "invalid_nodes=%s  cycles=%s  blacklisted=%s",
                        label, i, attempt,
                        issues["invalid_nodes"],
                        bool(issues["cycles"]),
                        issues["blacklisted"],
                    )
                    continue

                # Save response JSON
                json_path = run_dir / f"response_{i:02d}.json"
                with open(json_path, "w") as f:
                    json.dump({
                        "run_number":     run_number,
                        "response_index": i,
                        "model":          client.model,
                        "seed":           seed,
                        "attempts":       attempt,
                        "timestamp":      datetime.now().isoformat(),
                        "prompt":         prompt,
                        "response":       raw,
                    }, f, indent=4)

                # Save per-response edge CSV
                csv_path = csv_dir / f"response_{i:02d}_edges.csv"
                _write_csv(csv_path, edges)
                individual_csv_paths.append(csv_path)

                all_edges.update(edges)
                log.info(
                    "[%s] sample %d/%d  VALID — %d edges  (saved: %s)",
                    label, i, N_RESPONSES, len(edges), json_path.name,
                )
                saved = True
                break

            except Exception as exc:
                log.warning("[%s] sample %d attempt %d error: %s", label, i, attempt, exc)

        if not saved:
            log.warning(
                "[%s] sample %d/%d SKIPPED after %d attempts",
                label, i, N_RESPONSES, MAX_RETRIES,
            )

    if not all_edges:
        log.error("[%s] No valid samples obtained — graph generation failed", label)
        return None

    # Break cycles in the union graph, weighted by per-sample frequencies
    g = build_graph(all_edges)
    try:
        find_cycle_edges(g)
        has_cycle = True
    except nx.exception.NetworkXNoCycle:
        has_cycle = False

    if has_cycle and individual_csv_paths:
        log.info("[%s] Cycles detected in union graph — breaking cycles ...", label)
        freq    = compute_frequencies(individual_csv_paths, has_header=True)
        removed = _break_cycles(g, freq)
        for (x, y), fr in removed:
            log.info("[%s]   Removed edge (%s → %s)  freq=%d", label, x, y, fr)
        all_edges = set(g.edges())
        log.info("[%s] After cycle-breaking: %d edges in union", label, len(all_edges))
    else:
        log.debug("[%s] Union graph has no cycles", label)

    # Save union CSV (mirrors gen_llm_response.py naming)
    model_slug = client.model.replace("/", "-").replace(":", "-")
    union_name = f"initial_edges_{model_slug}-{run_tag}.csv"
    union_path = csv_dir / union_name
    _write_csv(union_path, sorted(all_edges))
    log.info(
        "[%s] Union graph: %d unique edges → %s",
        label, len(all_edges), union_path,
    )
    return list(all_edges)


# ---------------------------------------------------------------------------
# Small file helpers
# ---------------------------------------------------------------------------

def _write_csv(path: Path, edges) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["X", "Y"])
        w.writerows(edges)


def _clear_initial_edge_lists() -> None:
    removed = []
    for p in INITIAL_EDGES_DIR.glob("initial_edges_*.csv"):
        p.unlink()
        removed.append(p.name)
    if removed:
        log.debug("Cleaned initial_edge_lists: %s", removed)


def _clear_llm_artefacts() -> None:
    import shutil
    for d in (LLM_RESPONSES_DIR, LLM_EDGES_DIR):
        if d.exists():
            shutil.rmtree(d)
            log.info("Cleared artefact directory: %s", d)
        d.mkdir(parents=True, exist_ok=True)


def _run_refinement(label: str = "", domain: str = "alarm") -> int:
    """Run refine_step_1.py as a subprocess from LLM_DIR. Streams output to log."""
    log.info("[%s] Launching refine_step_1.py ...", label)
    proc = subprocess.Popen(
        [sys.executable, "refine_step_1.py", "--domain", domain],
        cwd=str(LLM_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    for line in proc.stdout:
        line = line.rstrip()
        log.debug("[refine] %s", line)
        if "tree bic" in line.lower():
            print(f"  [refine] {line}")
    proc.wait()
    if proc.returncode != 0:
        log.error("[%s] refine_step_1.py exited with code %d", label, proc.returncode)
    else:
        log.info("[%s] refine_step_1.py finished (exit 0)", label)
    return proc.returncode


def _run_no_refinement_score(label: str = "", domain: str = "alarm", cfg: dict = None) -> int:
    """Run no_refinement_score.py to score raw LLM graphs without structure learning."""
    log.info("[%s] Launching no_refinement_score.py ...", label)
    cmd = [
        sys.executable, "no_refinement_score.py",
        "--output",    str(HISTORY_JSON),
        "--domain",    domain,
        "--data_file", str(cfg["dataset"]),
        "--black_list", str(cfg["blacklist"]),
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=str(LLM_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    for line in proc.stdout:
        line = line.rstrip()
        log.debug("[no-refine-score] %s", line)
        if "tree bic" in line.lower():
            print(f"  [no-refine-score] {line}")
    proc.wait()
    if proc.returncode != 0:
        log.error("[%s] no_refinement_score.py exited with code %d", label, proc.returncode)
    else:
        log.info("[%s] no_refinement_score.py finished (exit 0)", label)
    return proc.returncode


def _cleanup_result_files() -> None:
    result_dirs = [
        LLM_DIR / "results" / "llm" / "full",
        LLM_DIR / "results" / "llm" / "no_refinement",
        LLM_DIR / "results" / "llm" / "subtractive",
    ]
    for d in result_dirs:
        if not d.exists():
            continue
        removed = [f.name for f in d.iterdir() if f.is_file()]
        for f in d.iterdir():
            if f.is_file():
                f.unlink()
        if removed:
            log.debug("Cleaned %s: %s", d.name, removed)


def _run_llm_graph_builder(label: str = "") -> int:
    """Run get_llm_graphs_from_edges.py from INITIAL_EDGES_DIR to create .graph files."""
    log.info("[%s] Launching get_llm_graphs_from_edges.py ...", label)
    proc = subprocess.Popen(
        [sys.executable, "get_llm_graphs_from_edges.py"],
        cwd=str(INITIAL_EDGES_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    for line in proc.stdout:
        log.debug("[graph-builder] %s", line.rstrip())
    proc.wait()
    if proc.returncode != 0:
        log.error("[%s] get_llm_graphs_from_edges.py exited with code %d", label, proc.returncode)
    else:
        log.info("[%s] get_llm_graphs_from_edges.py finished (exit 0)", label)
    return proc.returncode


# ---------------------------------------------------------------------------
# History (read-only; refine_step_1.py owns the writes)
# ---------------------------------------------------------------------------

def read_history() -> List[Dict]:
    if not HISTORY_JSON.exists():
        return []
    with open(HISTORY_JSON) as f:
        return json.load(f)


def top_k_unique(history: List[Dict], k: int = 5) -> List[Dict]:
    """Return top-k unique (by edge-set) graphs ranked by tree_bic descending (higher BIC is better).
    Falls back to structured_bic for entries that pre-date tree_bic storage."""
    ranked = sorted(
        history,
        key=lambda r: float(r.get("tree_bic", r["structured_bic"])),
        reverse=True,
    )
    seen:   Set[FrozenSet] = set()
    result: List[Dict]     = []
    for r in ranked:
        key = frozenset(tuple(e) for e in r["edges"])
        if key in seen:
            continue
        seen.add(key)
        result.append(r)
        if len(result) >= k:
            break
    return result


# ---------------------------------------------------------------------------
# Constraint extraction
# ---------------------------------------------------------------------------

def _extract_direct(top_graphs: List[Dict], threshold: float) -> List[Tuple[str, str]]:
    n       = len(top_graphs)
    counts: Counter = Counter()
    for g in top_graphs:
        for e in {tuple(x) for x in g["edges"]}:
            counts[e] += 1
    result = [e for e, c in counts.items() if c / n >= threshold]
    log.debug(
        "Direct constraint extraction: %d candidate edges → %d above threshold %.2f",
        len(counts), len(result), threshold,
    )
    return result


def _extract_motifs(top_graphs: List[Dict], threshold: float) -> List[Dict]:
    dgs    = [build_digraph(g["edges"]) for g in top_graphs]
    result = select_motif_constraints(
        aggregate_motif_counts(dgs), total=len(top_graphs), threshold=threshold
    )
    by_type: Counter = Counter(c["motif_type"] for c in result)
    log.debug(
        "Motif extraction: %d constraints above threshold %.2f  %s",
        len(result), threshold, dict(by_type),
    )
    return result


def _extract_ancestral(top_graphs: List[Dict], threshold: float) -> List[Dict]:
    dgs    = [build_digraph(g["edges"]) for g in top_graphs]
    result = select_ancestral_constraints(
        aggregate_ancestral_edges(dgs), total=len(top_graphs), threshold=threshold
    )
    log.debug(
        "Ancestral extraction: %d constraints above threshold %.2f",
        len(result), threshold,
    )
    return result


# ---------------------------------------------------------------------------
# Constraint CSV writers (consumed by refine_step_1.py)
# ---------------------------------------------------------------------------

def _write_required_edges(edges: List[Tuple[str, str]]) -> None:
    _write_csv(REQUIRED_CSV, edges)
    log.debug("Wrote %d required edges → %s", len(edges), REQUIRED_CSV)


def _write_ancestral_constraints(ancs: List[Dict]) -> None:
    with open(ANCESTRAL_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["X", "Y"])
        for c in ancs:
            w.writerow([c["ancestor"], c["descendant"]])
    log.debug("Wrote %d ancestral constraints → %s", len(ancs), ANCESTRAL_CSV)


def _clear_constraint_csvs() -> None:
    _write_csv(REQUIRED_CSV, [])
    _write_csv(ANCESTRAL_CSV, [])
    log.debug("Cleared required_edges.csv and ancestral_constraints.csv")


# ---------------------------------------------------------------------------
# Prompt injection
# ---------------------------------------------------------------------------

_TASK_ADDENDUM = {
    "direct": (
        "2. Include all data driven edges.\n"
        "3. You can add any additional edges, as long as they satisfy constraints (1)."
    ),
    "direct_no_refinement": (
        "2. Include all LLM-driven edges.\n"
        "3. You can add any additional edges, as long as they satisfy constraints (1)."
    ),
    "motif": (
        "2. Every motif constraint must appear as an exact subgraph.\n"
        "3. You can add any additional edges, as long as they satisfy constraints (1)."
    ),
    "ancestral": (
        "2. Every ancestral constraint must be satisfied by a directed path.\n"
        "3. You can add any additional edges, as long as they satisfy constraints (1)."
    ),
    "ancestral_motif": (
        "2. Every ancestral constraint must be satisfied by a directed path.\n"
        "3. Every motif constraint must appear as an exact subgraph.\n"
        "4. You can add any additional edges, as long as they satisfy constraints (1)."
    ),
    "all": (
        "2. Include all data driven edges.\n"
        "3. Every motif constraint must appear as an exact subgraph.\n"
        "4. Every ancestral constraint must be satisfied by a directed path.\n"
        "5. You can add any additional edges, as long as they satisfy constraints (1)."
    ),
    "full_graph": (
        "2. You can add any additional edges, as long as they satisfy constraints (1)."
    ),
    "none": "",
}


def _block_direct(edges: List[Tuple[str, str]], title: str = "## Data driven edges") -> str:
    grouped: Dict[str, List[str]] = defaultdict(list)
    for u, v in edges:
        grouped[u].append(v)
    adj_lines = [
        f"  {src} : {', '.join(sorted(dsts))}"
        for src, dsts in sorted(grouped.items())
    ]
    lines = [
        title,
        "The causal graph must contain the following directed edges:",
    ] + adj_lines
    return "\n".join(lines)


def _block_motif(motifs: List[Dict]) -> str:
    lines = [
        "## Motif Constraints",
        "The following 3-node structural patterns must appear in the graph:",
        "- Chain X → Z → Y : X causes Y through Z",
        "- Fork X ← Z → Y : Z is a common cause of both X and Y",
        "- Collider X → Z ← Y : both X and Y independently cause Z",
    ]
    for c in motifs:
        lines.append(MOTIF_PROMPT_SYMBOLS[c["motif_type"]].format(
            x=c["triple"][0], z=c["triple"][1], y=c["triple"][2]
        ))
    return "\n".join(lines)


def _block_ancestral(ancs: List[Dict]) -> str:
    grouped: Dict[str, List[str]] = defaultdict(list)
    for c in ancs:
        grouped[c["descendant"]].append(c["ancestor"])
    lines = [
        "## Ancestral constraints",
        "The following ancestor → descendant relationships must hold",
        "(i.e. there must exist a directed path from each node in the set to the node in parentheses):",
    ] + [
        f"  Ancestor({desc}) = {{{', '.join(sorted(ancs_))}}}"
        for desc, ancs_ in sorted(grouped.items())
    ]
    return "\n".join(lines)


def _block_blacklist(black_list: List[Tuple[str, str]]) -> str:
    grouped: Dict[str, List[str]] = {}
    for u, v in black_list:
        if u not in grouped:
            grouped[u] = []
        grouped[u].append(v)
    return "\n".join(
        f"{src}: {', '.join(dsts)}"
        for src, dsts in grouped.items()
    )


def _content_direct(edges: List[Tuple[str, str]]) -> str:
    grouped: Dict[str, List[str]] = defaultdict(list)
    for u, v in edges:
        grouped[u].append(v)
    return "\n".join(
        f"  {src} : {', '.join(sorted(dsts))}"
        for src, dsts in sorted(grouped.items())
    )


def _content_ancestral(ancs: List[Dict]) -> str:
    grouped: Dict[str, List[str]] = defaultdict(list)
    for c in ancs:
        grouped[c["descendant"]].append(c["ancestor"])
    return "\n".join(
        f"  Ancestor({desc}) = {{{', '.join(sorted(ancs_))}}}"
        for desc, ancs_ in sorted(grouped.items())
    )


def _content_motif(motifs: List[Dict]) -> str:
    return "\n".join(
        MOTIF_PROMPT_SYMBOLS[c["motif_type"]].format(
            x=c["triple"][0], z=c["triple"][1], y=c["triple"][2]
        )
        for c in motifs
    )


def _content_full_graphs(top_graphs: List[Dict]) -> str:
    """Format top-k graphs as numbered adjacency lists (for template placeholder fill)."""
    parts = []
    for rank, g in enumerate(top_graphs, 1):
        grouped: Dict[str, List[str]] = defaultdict(list)
        for e in g["edges"]:
            grouped[e[0]].append(e[1])
        adj_lines = "\n".join(
            f"  {src} : {', '.join(sorted(dsts))}"
            for src, dsts in sorted(grouped.items())
        )
        parts.append(f"Graph {rank}:\n{adj_lines}")
    return "\n\n".join(parts)


def _block_full_graphs(top_graphs: List[Dict]) -> str:
    """Full section block (header + graphs) for insertion-based prompts (e.g. alarm)."""
    return "\n".join([
        "## Previously generated Causal graphs",
        f"The following are the top {len(top_graphs)} previously generated graphs "
        "sorted from best to worst based on their final structure scores.",
        "",
        _content_full_graphs(top_graphs),
    ])


_BL_PLACEHOLDERS      = ("<Forbidden edge list>", "<forbidden edges>")
_DIRECT_PLACEHOLDER   = "< data driven edges>"
_ANC_PLACEHOLDERS     = ("<Ancestral constraints>", "<Ancestral Edges>")
_MOTIF_PLACEHOLDER    = "<motif constraints>"
_GRAPHS_PLACEHOLDER   = "<Graphs list>"


def _fill_g1_template(
    template:    str,
    black_list:  List[Tuple[str, str]],
    req:         Optional[List[Tuple[str, str]]] = None,
    ancs:        Optional[List[Dict]]            = None,
    motifs:      Optional[List[Dict]]            = None,
    top_graphs:  Optional[List[Dict]]            = None,
) -> str:
    """Fill domain-specific placeholders in a G1 prompt template."""
    result  = template
    bl_text = _block_blacklist(black_list) if black_list else "  (none)"
    for ph in _BL_PLACEHOLDERS:
        result = result.replace(ph, bl_text)
    if req is not None:
        result = result.replace(_DIRECT_PLACEHOLDER, _content_direct(req))
    if ancs is not None:
        for ph in _ANC_PLACEHOLDERS:
            result = result.replace(ph, _content_ancestral(ancs))
    if motifs is not None:
        result = result.replace(_MOTIF_PLACEHOLDER, _content_motif(motifs))
    if top_graphs is not None:
        result = result.replace(_GRAPHS_PLACEHOLDER, _content_full_graphs(top_graphs))
    return result


def _build_g1_prompt(g0: str, constraint_type: str, constraint_block: str, no_refinement: bool = False) -> str:
    """Insert constraint block before '# Task' and append enforcement rules."""
    if constraint_type == "none":
        return g0
    marker = "# Task"
    if marker not in g0:
        raise ValueError("G0 prompt is missing the '# Task' section.")
    before, after = g0.split(marker, 1)

    addendum_key = f"{constraint_type}_no_refinement" if (no_refinement and f"{constraint_type}_no_refinement" in _TASK_ADDENDUM) else constraint_type
    updated, added = [], False
    for line in after.split("\n"):
        updated.append(line)
        if not added and line.strip().startswith("1. No cycles"):
            updated.append(_TASK_ADDENDUM[addendum_key])
            added = True

    return (
        before.rstrip()
        + "\n\n"
        + constraint_block
        + "\n\n"
        + marker
        + "\n".join(updated)
    )


# ---------------------------------------------------------------------------
# Human-readable run report
# ---------------------------------------------------------------------------

def _write_run_report(summary: Dict, history: List[Dict], out_dir: Path) -> Path:
    """Write a human-readable .txt report combining hyperparameters and full history."""
    run_id      = summary.get("run_id", "unknown")
    report_path = out_dir / f"clara_{run_id}_report.txt"
    W           = 80
    SEP         = "=" * W
    sep         = "-" * W

    lines: List[str] = []
    def add(s: str = "") -> None:
        lines.append(s)

    add(SEP)
    add("CLARA RUN REPORT")
    add(SEP)
    add(f"Run ID        : {run_id}")
    add(f"Generated     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    add(sep)
    add("HYPERPARAMETERS")
    add(sep)
    add(f"  Domain            : {summary.get('domain', 'N/A')}")
    add(f"  Constraint type   : {summary.get('constraint_type', 'N/A')}")
    add(f"  LLM model         : {summary.get('llm_model', 'N/A')}")
    add(f"  n_initial         : {summary.get('n_initial', 'N/A')}")
    add(f"  max_runs          : {summary.get('max_runs', 'N/A')}")
    add(f"  threshold         : {summary.get('threshold', 'N/A'):.2f}")
    add(f"  top_k             : {summary.get('top_k', 'N/A')}")
    add(f"  N_RESPONSES       : {N_RESPONSES}")
    add(f"  MAX_RETRIES       : {MAX_RETRIES}")
    add(f"  TEMPERATURE       : {TEMPERATURE}")
    add(f"  PATIENCE          : 3")
    add(sep)
    add("RUN STATISTICS")
    add(sep)
    elapsed = summary.get("elapsed_seconds", 0)
    add(f"  Elapsed time       : {elapsed:.2f} s  ({elapsed / 60:.1f} min)")
    add(f"  Budget total       : {summary.get('budget_total', 'N/A')}")
    add(f"  Budget used        : {summary.get('budget_used', 'N/A')}")
    add(f"  Early stopped      : {'Yes' if summary.get('early_stopped') else 'No'}")
    add(f"  Total history      : {summary.get('total_history', 'N/A')}")
    p0bic  = summary.get("phase0_best_bic")
    finbic = summary.get("final_best_bic")
    add(f"  Phase 0 best BIC   : {p0bic:.4f}" if p0bic is not None else "  Phase 0 best BIC   : N/A")
    add(f"  Final best BIC     : {finbic:.4f}" if finbic is not None else "  Final best BIC     : N/A")
    add(f"  Final # edges      : {summary.get('final_n_edges', 'N/A')}")
    bic_imp = summary.get("bic_improvement")
    add(f"  BIC improvement    : {bic_imp:.4f}" if bic_imp is not None else "  BIC improvement    : N/A")
    add(f"  Duplicate retries  : {summary.get('n_duplicate_retries', 'N/A')}")
    add(f"  Failed runs        : {summary.get('n_failed_runs', 'N/A')}")
    add(sep)
    add("BIC TRAJECTORY  (iterative runs)")
    add(sep)
    for entry in summary.get("runs", []):
        run_num = entry.get("run", "?")
        status  = entry.get("status", "?")
        if "bic" in entry:
            add(f"  Run {run_num:>3}: bic={entry['bic']:.4f}  history_min={entry.get('history_min_bic', float('nan')):.4f}"
                f"  edges={entry.get('n_edges', '?')}  status={status}")
        else:
            add(f"  Run {run_num:>3}: {status}")
    add(sep)
    add(f"REFINED HISTORY  (g1_refined_results.json)  —  total graphs: {len(history)}")
    add(sep)
    for idx, entry in enumerate(history, 1):
        add(f"")
        add(f"Entry {idx:>3}")
        add(f"  llm_type     : {entry.get('llm_type', 'N/A')}")
        add(f"  sample_id    : {entry.get('sample_id', 'N/A')}")
        tbic = entry.get("tree_bic")
        sbic = entry.get("structured_bic")
        add(f"  tree_bic     : {tbic:.6f}" if tbic is not None else "  tree_bic     : N/A")
        add(f"  struct_bic   : {sbic:.6f}" if sbic is not None else "  struct_bic   : N/A")
        edges = entry.get("edges", [])
        add(f"  n_edges      : {len(edges)}")
        add(f"  edges (X,Y)  :")
        for e in edges:
            add(f"    {e[0]},{e[1]}")
    add("")
    add(SEP)

    report_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Run report → %s", report_path)
    return report_path


# ---------------------------------------------------------------------------
# Final graph (union + consensus from top-5)
# ---------------------------------------------------------------------------

def _build_final_graphs(top5: List[Dict], out_dir: Path, k_consensus: int = 3, run_id: str = "") -> None:
    log.info("Building final graphs from top-%d history graphs ...", len(top5))
    for rank, g in enumerate(top5, 1):
        log.info("  Rank %d: llm_type=%s  tree_bic=%.4f  edges=%d",
                 rank, g.get("llm_type", "?"), g.get("tree_bic", g["structured_bic"]), len(g["edges"]))

    edge_lists = [[tuple(e) for e in g["edges"]] for g in top5]
    counts: Counter = Counter()
    for edges in edge_lists:
        counts.update(set(edges))

    G: nx.DiGraph = nx.DiGraph()
    for (u, v), w in counts.items():
        G.add_edge(u, v, weight=w)

    while not nx.is_directed_acyclic_graph(G):
        cycle = nx.find_cycle(G, orientation="original")
        worst = min(
            [(u, v) for u, v, _ in cycle],
            key=lambda e: (counts.get(e, 0), e[0], e[1]),
        )
        log.debug("Final graph: removed cycle edge %s (freq=%d)", worst, counts.get(worst, 0))
        G.remove_edge(*worst)

    union     = sorted(G.edges())
    consensus = sorted(e for e, c in counts.items() if c >= k_consensus and G.has_edge(*e))

    suffix = f"_{run_id}" if run_id else ""
    _write_csv(out_dir / f"final_union{suffix}.csv",     union)
    _write_csv(out_dir / f"final_consensus{suffix}.csv", consensus)
    nx.write_gml(G, str(out_dir / f"final_union{suffix}.gml"))

    log.info("Final union graph:     %d edges → %s/final_union%s.csv",     len(union),     out_dir, suffix)
    log.info("Final consensus graph: %d edges → %s/final_consensus%s.csv", len(consensus), out_dir, suffix)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_clara(
    domain:            str,
    constraint_type:   str,
    client:            LLMClient,
    runs:              int           = MAX_RUNS,
    threshold:         float         = 1.0,
    n_initial:         int           = N_INITIAL,
    top_k:             int           = TOP_K,
    output_dir:        str           = "clara_output",
    no_early_stop:     bool          = False,
    no_refinement:     bool          = False,
    run_id:            str           = "",
    initial_edges_dir: Optional[str] = None,
) -> None:
    import time
    t0  = time.time()
    cfg = domain_config.DOMAINS[domain]
    out = Path(output_dir)
    out.mkdir(exist_ok=True)
    INITIAL_EDGES_DIR.mkdir(exist_ok=True)

    # Point node validation at the domain's expert graph (skipped gracefully if absent)
    validate_llm_graph.set_expert_graph(cfg["expert_graph"])

    log.info("=" * 60)
    log.info("CLARA starting")
    log.info("  domain          : %s", domain)
    log.info("  constraint_type : %s", constraint_type)
    log.info("  llm model       : %s", client.model)
    log.info("  n_initial       : %d", n_initial)
    log.info("  max_runs        : %d", runs)
    log.info("  threshold       : %.2f", threshold)
    log.info("  top_k           : %d", top_k)
    log.info("  output_dir      : %s", out)
    log.info("=" * 60)

    black_list = _load_blacklist(cfg["blacklist"])

    with open(cfg["g0_prompt"], encoding="utf-8") as f:
        g0_prompt = f.read().replace(" ", " ")
    log.info("Loaded G0 prompt from: %s", cfg["g0_prompt"])

    g0_prompt_path = out / "prompt_g0.txt"
    g0_prompt_path.write_text(g0_prompt, encoding="utf-8")
    log.info("G0 prompt saved → %s", g0_prompt_path)

    run_log:       List[Dict] = []
    run_counter:   int        = 0
    early_stopped: bool       = False

    # -----------------------------------------------------------------------
    # Phase 0: generate n_initial graphs (G0, no constraints)
    #          — or load pre-supplied files when --initial-edges-dir is set
    # -----------------------------------------------------------------------
    log.info("")

    if HISTORY_JSON.exists():
        HISTORY_JSON.unlink()
        log.info("Cleared existing history JSON: %s", HISTORY_JSON)
    if BIC_SCORES_CSV.exists():
        BIC_SCORES_CSV.unlink()
        log.info("Cleared existing BIC scores CSV: %s", BIC_SCORES_CSV)
    _clear_constraint_csvs()
    _clear_initial_edge_lists()

    if no_refinement:
        no_early_stop = True   # BIC is unavailable without refinement

    written = 0

    if initial_edges_dir is not None:
        # ------------------------------------------------------------------
        # Pre-supplied files mode: skip LLM generation entirely.
        # Copy every CSV from the given directory into INITIAL_EDGES_DIR,
        # then run builder + refinement and jump straight to iteration.
        # ------------------------------------------------------------------
        import shutil
        src_dir   = Path(initial_edges_dir)
        csv_files = sorted(src_dir.glob("*.csv"))
        if not csv_files:
            raise RuntimeError(
                f"--initial-edges-dir '{src_dir}' contains no CSV files."
            )
        log.info(
            "PHASE 0 — loading %d pre-supplied initial edge files from %s (LLM generation skipped)",
            len(csv_files), src_dir,
        )
        for i, f in enumerate(csv_files, 1):
            dest = INITIAL_EDGES_DIR / f"initial_edges_g0_{i:02d}.csv"
            shutil.copy(f, dest)
            written += 1
            log.info("[init-preload] %s → %s", f.name, dest.name)

    else:
        # ------------------------------------------------------------------
        # LLM generation mode (original Phase 0 behaviour)
        # ------------------------------------------------------------------
        _clear_llm_artefacts()
        log.info("PHASE 0 — %d initial graphs  (G0, no constraints)", n_initial)
        log.info("  Each graph = union of %d independent LLM samples", N_RESPONSES)

        phase0_edge_lists: List[List[Tuple[str, str]]] = []
        for i in range(1, n_initial + 1):
            run_counter += 1
            label = f"init-{i}"
            log.info("--- [%s] Generating initial graph %d/%d ---", label, i, n_initial)

            edges = generate_graph_union(client, g0_prompt, run_counter, black_list, label=label)
            if edges is None:
                log.warning("[%s] Skipped — no valid union graph produced", label)
                continue

            dest = INITIAL_EDGES_DIR / f"initial_edges_g0_{i:02d}.csv"
            _write_csv(dest, edges)
            phase0_edge_lists.append(edges)
            written += 1
            log.info("[%s] Written to %s  (%d edges)", label, dest.name, len(edges))

        if written == 0:
            log.error("No valid initial graphs generated. Aborting.")
            raise RuntimeError(
                "No valid initial graphs generated. "
                "Check API key, data path, and prompt file."
            )

    if no_refinement:
        log.info(
            "Phase 0 complete (%d graphs). Scoring raw LLM graphs (no structure learning) ...",
            written,
        )
        _run_llm_graph_builder(label="phase-0")
        _run_no_refinement_score(label="phase-0", domain=domain, cfg=cfg)
        _clear_initial_edge_lists()
    else:
        log.info("Phase 0: %d graphs written to initial_edge_lists. Running refinement ...", written)
        _run_llm_graph_builder(label="phase-0")
        _run_refinement(label="phase-0", domain=domain)
        _clear_initial_edge_lists()
        _cleanup_result_files()

    history = read_history()
    log.info("History after phase 0: %d refined graphs", len(history))
    phase0_best_bic = max(
        (g.get("tree_bic", g["structured_bic"]) for g in history), default=None
    )

    # Top-up: only in LLM generation mode — if duplicates reduced unique count below n_initial.
    if initial_edges_dir is None:
        n_unique = len(top_k_unique(history, k=len(history) + 1))
        if n_unique < n_initial:
            log.info(
                "Phase 0: %d/%d initial graphs are unique — generating %d replacement G0 graphs ...",
                n_unique, n_initial, n_initial - n_unique,
            )
            needed          = n_initial - n_unique
            known_edge_sets = {frozenset(tuple(e) for e in g["edges"]) for g in history}
            max_topup_tries = needed * 5
            topup_try       = 0

            while needed > 0 and topup_try < max_topup_tries:
                topup_try   += 1
                run_counter += 1
                label        = f"init-topup-{topup_try}"
                log.info("--- [%s] Generating replacement G0 graph (need %d more unique) ---", label, needed)

                edges = generate_graph_union(client, g0_prompt, run_counter, black_list, label=label)
                if edges is None:
                    log.warning("[%s] Skipped — no valid union graph produced", label)
                    continue

                dest = INITIAL_EDGES_DIR / f"initial_edges_g0_topup_{topup_try:02d}.csv"
                _write_csv(dest, edges)
                log.info("[%s] Written → %s  (%d edges)", label, dest.name, len(edges))

                prev_len = len(history)
                if no_refinement:
                    _run_llm_graph_builder(label=label)
                    rc = _run_no_refinement_score(label=label, domain=domain, cfg=cfg)
                    _clear_initial_edge_lists()
                    if rc != 0:
                        log.warning("[%s] Scoring failed (exit %d) — skipping", label, rc)
                        continue
                else:
                    _run_llm_graph_builder(label=label)
                    rc = _run_refinement(label=label, domain=domain)
                    _clear_initial_edge_lists()
                    _cleanup_result_files()
                    if rc != 0:
                        log.warning("[%s] Refinement failed (exit %d) — skipping", label, rc)
                        continue

                history     = read_history()
                new_entries = history[prev_len:]
                if not new_entries:
                    log.warning("[%s] No new history entry after refinement", label)
                    continue

                candidate      = new_entries[-1]
                candidate_eset = frozenset(tuple(e) for e in candidate["edges"])
                if candidate_eset in known_edge_sets:
                    log.info("[%s] Replacement graph is also a duplicate — retrying", label)
                    continue

                known_edge_sets.add(candidate_eset)
                needed -= 1
                log.info("[%s] Unique graph added (need %d more)", label, needed)

            if needed > 0:
                log.warning(
                    "Phase 0 top-up: could only resolve %d/%d duplicates after %d attempts",
                    (n_initial - n_unique) - needed, n_initial - n_unique, topup_try,
                )
            else:
                log.info("Phase 0 top-up complete: %d unique replacement graphs added", n_initial - n_unique)

            phase0_best_bic = max(
                (g.get("tree_bic", g["structured_bic"]) for g in history), default=None
            )

    if len(top_k_unique(history, k=top_k)) < top_k:
        msg = (
            f"Need at least {top_k} unique refined graphs before iterating, "
            f"got {len(top_k_unique(history, k=top_k))}. "
            "Try increasing --n-initial or check for refinement errors."
        )
        log.error(msg)
        raise RuntimeError(msg)

    # -----------------------------------------------------------------------
    # Iterative loop
    # -----------------------------------------------------------------------
    PATIENCE          = 3   # stop after this many consecutive non-improving runs
    no_improve_count: int = 0
    budget = runs   # total generate+refine cycles allowed (shared across runs and dup retries)
    used   = 0      # cycles consumed so far
    run    = 0      # logical run counter (constraint extraction cycles)

    while used < budget:
        run += 1
        log.info("")
        log.info("=" * 60)
        log.info("RUN %d  |  constraint=%s  |  budget_used=%d/%d  |  history=%d graphs",
                 run, constraint_type, used, budget, len(history))
        log.info("=" * 60)

        top_graphs = top_k_unique(history, k=top_k)
        bics       = [round(g.get("tree_bic", g["structured_bic"]), 4) for g in top_graphs]
        log.info("Top-%d tree_bics: %s", top_k, bics)
        for rank, g in enumerate(top_graphs, 1):
            log.debug("  Rank %d: llm_type=%s  tree_bic=%.4f  edges=%d",
                      rank, g.get("llm_type", "?"), g.get("tree_bic", g["structured_bic"]), len(g["edges"]))

        # 1. Extract constraints and update CSV files for refine_step_1.py
        req, motifs, ancs = None, None, None

        req = _extract_direct(top_graphs, threshold)
        _write_required_edges(req)
        log.info("Extracted %d required edges", len(req))
        for u, v in sorted(req):
            log.debug("  Required: %s → %s", u, v)

        if constraint_type == "direct":
            _write_csv(ANCESTRAL_CSV, [])

        elif constraint_type == "motif":
            motifs = _extract_motifs(top_graphs, threshold)
            _write_csv(ANCESTRAL_CSV, [])
            log.info("Extracted %d motif constraints", len(motifs))
            for c in motifs:
                log.debug("  Motif %-10s  freq=%.2f  %s",
                          c["motif_type"], c["freq"], c["triple"])

        elif constraint_type == "ancestral":
            ancs = _extract_ancestral(top_graphs, threshold)
            _write_ancestral_constraints(ancs)
            log.info("Extracted %d ancestral constraints", len(ancs))
            for c in ancs:
                log.debug("  Ancestral  %s →* %s  freq=%.2f", c["ancestor"], c["descendant"], c["freq"])

        elif constraint_type == "ancestral_motif":
            motifs = _extract_motifs(top_graphs, threshold)
            ancs   = _extract_ancestral(top_graphs, threshold)
            _write_ancestral_constraints(ancs)
            log.info("Extracted %d motif, %d ancestral constraints (ancestral_motif)",
                     len(motifs), len(ancs))

        elif constraint_type == "all":
            motifs = _extract_motifs(top_graphs, threshold)
            ancs   = _extract_ancestral(top_graphs, threshold)
            _write_ancestral_constraints(ancs)
            log.info(
                "Extracted %d direct, %d motif, %d ancestral constraints (all)",
                len(req), len(motifs), len(ancs),
            )

        elif constraint_type == "full_graph":
            _clear_constraint_csvs()
            log.info("Full-graph constraint: top-%d graphs will be passed to LLM prompt (no CSV constraints)", top_k)

        elif constraint_type == "none":
            _clear_constraint_csvs()
            log.info("No constraints extracted (constraint_type=none)")

        # 2. Build G1 prompt and save for inspection
        g1_prompts_map = cfg.get("g1_prompts", {})
        if constraint_type in g1_prompts_map:
            # Template-based injection: fill domain-specific placeholders
            with open(g1_prompts_map[constraint_type], encoding="utf-8") as f:
                g1_template = f.read().replace(" ", " ")
            g1_prompt = _fill_g1_template(
                g1_template, black_list, req=req, ancs=ancs, motifs=motifs,
                top_graphs=top_graphs if constraint_type == "full_graph" else None,
            )
            log.info("G1 prompt built from template: %s", g1_prompts_map[constraint_type].name)
        else:
            # Insertion-based: inject constraint block before '# Task' (e.g. alarm)
            direct_title = "## LLM-driven edges" if no_refinement else "## Data driven edges"
            if constraint_type == "direct":
                constraint_block = _block_direct(req, title=direct_title)
            elif constraint_type == "motif":
                constraint_block = _block_motif(motifs)
            elif constraint_type == "ancestral":
                constraint_block = _block_ancestral(ancs)
            elif constraint_type == "ancestral_motif":
                blocks = []
                if motifs: blocks.append(_block_motif(motifs))
                if ancs:   blocks.append(_block_ancestral(ancs))
                constraint_block = "\n\n".join(blocks)
            elif constraint_type == "all":
                blocks = []
                if req:    blocks.append(_block_direct(req, title=direct_title))
                if motifs: blocks.append(_block_motif(motifs))
                if ancs:   blocks.append(_block_ancestral(ancs))
                constraint_block = "\n\n".join(blocks)
            elif constraint_type == "full_graph":
                constraint_block = _block_full_graphs(top_graphs)
            else:
                constraint_block = ""
            g1_prompt = _build_g1_prompt(g0_prompt, constraint_type, constraint_block, no_refinement=no_refinement)

        prompt_path = out / f"prompt_run_{run:03d}.txt"
        prompt_path.write_text(g1_prompt, encoding="utf-8")
        log.info("G1 prompt saved → %s", prompt_path)

        # Snapshot known edge-sets before this run for duplicate detection.
        known_edge_sets: Set[FrozenSet] = {
            frozenset(tuple(e) for e in g["edges"]) for g in history
        }

        # 3–4. Generate → refine loop (re-tries when refined graph is a duplicate).
        # Each attempt (including dup retries) draws from the shared budget.
        dup_try       = 0
        refined_entry = None
        while used < budget:
            dup_try     += 1
            used        += 1
            run_counter += 1
            attempt_label = f"run-{run}" if dup_try == 1 else f"run-{run}-dup{dup_try}"
            log.info("--- [%s] Generating G1 graph ---", attempt_label)

            edges = generate_graph_union(
                client, g1_prompt, run_counter, black_list, label=attempt_label,
            )
            if edges is None:
                log.warning("[%s] No valid graph — skipping attempt", attempt_label)
                continue

            dest = INITIAL_EDGES_DIR / (
                f"initial_edges_g1_{constraint_type}_run{run:03d}_dup{dup_try:02d}.csv"
            )
            _write_csv(dest, edges)
            log.info("[%s] Graph written → %s  (%d edges)", attempt_label, dest.name, len(edges))

            prev_len = len(history)
            if no_refinement:
                _run_llm_graph_builder(label=attempt_label)
                rc = _run_no_refinement_score(label=attempt_label, domain=domain, cfg=cfg)
                _clear_initial_edge_lists()
                if rc != 0:
                    log.error("[%s] Scoring failed (exit %d) — skipping run", attempt_label, rc)
                    run_log.append({"run": run, "status": "scoring_failed", "exit_code": rc})
                    break
                history = read_history()
            else:
                _run_llm_graph_builder(label=attempt_label)
                rc = _run_refinement(label=attempt_label, domain=domain)
                _clear_initial_edge_lists()
                _cleanup_result_files()

                if rc != 0:
                    log.error("[%s] Refinement failed (exit %d) — skipping run", attempt_label, rc)
                    run_log.append({"run": run, "status": "refinement_failed", "exit_code": rc})
                    break

                history = read_history()

            new_entries = history[prev_len:]
            if not new_entries:
                log.error("[%s] No new entry found in history after refinement", attempt_label)
                continue

            candidate      = new_entries[-1]
            candidate_eset = frozenset(tuple(e) for e in candidate["edges"])
            if candidate_eset in known_edge_sets:
                log.info(
                    "[%s] Refined graph is a duplicate (attempt %d, budget_used=%d/%d)"
                    " — re-generating ...",
                    attempt_label, dup_try, used, budget,
                )
                if used >= budget:
                    log.warning(
                        "[%s] Budget exhausted resolving duplicates — skipping run",
                        attempt_label,
                    )
                continue

            refined_entry = candidate
            break

        if refined_entry is None:
            run_log.append({"run": run, "status": "duplicate_or_failed"})
            continue

        current_bic = float(
            refined_entry.get("tree_bic", refined_entry["structured_bic"])
        )

        # Collect all BICs from history BEFORE this run's entry was appended.
        # history already contains the new entry, so exclude the last one.
        prior_history = history[:-1]
        prior_bics = [
            float(g.get("tree_bic", g["structured_bic"])) for g in prior_history
        ]
        min_prior_bic = min(prior_bics) if prior_bics else None

        beats_history = min_prior_bic is None or current_bic > min_prior_bic
        status = "improved" if beats_history else "no_improvement"

        log.info(
            "[run-%d] tree_bic=%.4f  history_min=%.4f  edges=%d  status=%s  patience=%d/%d",
            run, current_bic,
            min_prior_bic if min_prior_bic is not None else float("nan"),
            len(refined_entry["edges"]), status,
            no_improve_count + (0 if beats_history else 1), PATIENCE,
        )
        run_log.append({
            "run":             run,
            "bic":             current_bic,
            "history_min_bic": min_prior_bic,
            "n_edges":         len(refined_entry["edges"]),
            "status":          status,
            "edges":           refined_entry["edges"],
        })

        # 5. Early stopping — patience window
        if not no_early_stop and min_prior_bic is not None:
            if beats_history:
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= PATIENCE:
                    log.info(
                        "Early stopping at run %d: BIC %.4f did not beat any history graph "
                        "(history_min=%.4f) for %d consecutive runs",
                        run, current_bic, min_prior_bic, PATIENCE,
                    )
                    early_stopped = True
                    break

    # -----------------------------------------------------------------------
    # Final graph
    # -----------------------------------------------------------------------
    log.info("")
    log.info("=" * 60)
    log.info("FINAL GRAPH — top-%d from history (%d total refined graphs)", top_k, len(history))
    log.info("=" * 60)

    top5 = top_k_unique(history, k=top_k)
    _build_final_graphs(top5, out, run_id=run_id)

    bic_trajectory      = [e["bic"] for e in run_log if "bic" in e]
    n_duplicate_retries = sum(1 for e in run_log if e.get("status") == "duplicate_or_failed")
    n_failed_runs       = sum(1 for e in run_log if e.get("status") in ("refinement_failed", "scoring_failed"))
    final_best_bic      = float(top5[0].get("tree_bic", top5[0]["structured_bic"])) if top5 else None
    final_n_edges       = len(top5[0]["edges"])            if top5 else None
    bic_improvement     = round(final_best_bic - phase0_best_bic, 4) \
                          if (final_best_bic is not None and phase0_best_bic is not None) else None

    summary = {
        "run_id":              run_id,
        "domain":              domain,
        "constraint_type":     constraint_type,
        "llm_model":           client.model,
        "n_initial":           n_initial,
        "max_runs":            runs,
        "threshold":           threshold,
        "top_k":               top_k,
        "elapsed_seconds":     round(time.time() - t0, 2),
        "budget_total":        budget,
        "budget_used":         used,
        "early_stopped":       early_stopped,
        "total_history":       len(history),
        "phase0_best_bic":     phase0_best_bic,
        "final_best_bic":      final_best_bic,
        "final_n_edges":       final_n_edges,
        "bic_improvement":     bic_improvement,
        "bic_trajectory":      bic_trajectory,
        "n_duplicate_retries": n_duplicate_retries,
        "n_failed_runs":       n_failed_runs,
        "runs":                run_log,
    }
    summary_path = out / f"clara_{run_id}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("Run summary → %s", summary_path)

    run_history = [
        {
            "run":             e["run"],
            "bic":             e["bic"],
            "history_min_bic": e.get("history_min_bic"),
            "n_edges":         e["n_edges"],
            "status":          e["status"],
            "edges":           e["edges"],
        }
        for e in run_log
        if "edges" in e
    ]
    run_history_path = out / f"clara_{run_id}_run_history.json"
    run_history_path.write_text(json.dumps(run_history, indent=2), encoding="utf-8")
    log.info("Run history → %s", run_history_path)
    log.info("History     → %s  (%d total graphs)", HISTORY_JSON, len(history))
    _write_run_report(summary, history, out)

    if LLM_EDGES_DIR.exists():
        import shutil
        shutil.move(str(LLM_EDGES_DIR), str(out / "llm_edges"))
        log.info("Moved llm_edges → %s", out / "llm_edges")

    log.info("CLARA finished.")
    client.log_usage_summary(elapsed_s=time.time() - t0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="CLARA: Causal Learning via Adaptive Refinement and Aggregation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--domain",
                   choices=list(domain_config.DOMAINS.keys()), required=True,
                   help="Domain (determines which G0 prompt template to use)")
    p.add_argument("--constraint",
                   choices=["direct", "motif", "ancestral", "ancestral_motif", "all", "full_graph", "none"], required=True,
                   help="Constraint type to extract from history and inject into each G1 prompt")
    p.add_argument("--llm-backend",
                   choices=["openai", "ollama", "huggingface", "groq", "openrouter", "together"], default="openai")
    p.add_argument("--llm-model", default=None,
                   help=(
                       "Model name. "
                       "OpenAI default: gpt-4o. "
                       "Ollama default: llama3. "
                       "HuggingFace: full model ID required (e.g. mistralai/Mixtral-8x7B-Instruct-v0.1)."
                   ))
    p.add_argument("--llm-url", default=None,
                   help=(
                       "Base URL override. "
                       "Ollama default: http://localhost:11434. "
                       "HuggingFace: your TGI/vLLM OpenAI-compatible endpoint (required)."
                   ))
    p.add_argument("--runs",       type=int,   default=MAX_RUNS,
                   help="Max iterative runs after the initial phase")
    p.add_argument("--threshold",  type=float, default=1.0,
                   help="Min fraction of top-k graphs that must agree on a constraint")
    p.add_argument("--n-initial",  type=int,   default=N_INITIAL,
                   help="Number of initial graphs (phase 0, no constraints)")
    p.add_argument("--top-k",      type=int,   default=TOP_K,
                   help="Number of top graphs used for constraint extraction each run")
    p.add_argument("--output-dir", default="clara_output",
                   help="Directory for prompts, run summary, and log file")
    p.add_argument("--no-early-stop", action="store_true",
                   help="Disable early stopping and always run for the full --runs count")
    p.add_argument("--no-refinement", action="store_true",
                   help="Skip refine_step_1.py; use raw LLM graphs directly (CLARA0 mode). "
                        "Implies --no-early-stop since BIC is unavailable.")
    p.add_argument("--initial-edges-dir", default=None,
                   help="Directory containing pre-supplied Phase 0 initial edge CSVs. "
                        "When set, Phase 0 LLM generation is skipped; CLARA loads these "
                        "files, runs refinement, and starts directly at the iterative loop.")
    args = p.parse_args()

    client = build_client(args.llm_backend, args.llm_model, args.llm_url)

    model_slug  = client.model.replace("/", "-").replace(":", "-")
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id      = f"{args.domain}_{args.constraint}_{model_slug}_{timestamp}"
    run_folder  = f"{args.domain}-{args.constraint}-{model_slug}-{timestamp}-k{args.top_k}-t{args.runs}"
    out         = Path(args.output_dir) / run_folder
    logfile     = setup_logging(out, run_id)
    print(f"Log file: {logfile}")

    _clear_constraint_csvs()

    run_clara(
        domain             = args.domain,
        constraint_type    = args.constraint,
        client             = client,
        runs               = args.runs,
        threshold          = args.threshold,
        n_initial          = args.n_initial,
        top_k              = args.top_k,
        output_dir         = str(out),
        no_early_stop      = args.no_early_stop,
        no_refinement      = args.no_refinement,
        run_id             = run_id,
        initial_edges_dir  = args.initial_edges_dir,
    )


if __name__ == "__main__":
    main()
