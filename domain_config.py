"""domain_config.py — single source of truth for all domain-specific assets.

To add a new domain:
  1. Drop its files into  domains/<name>/
  2. Add an entry to DOMAINS below.
  3. Add a branch in build_state_names().
"""

from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).parent

DOMAINS: Dict[str, Dict] = {
    "alarm": {
        "g0_prompt":    ROOT / "domains" / "alarm" / "prompts" / "G0.txt",
        "expert_graph": ROOT / "domains" / "alarm" / "expert.graph",
        "blacklist":    ROOT / "domains" / "alarm" / "blacklist.csv",
        "dataset":      ROOT / "domains" / "alarm" / "dataset.csv",
    },
    "apo": {
        "g0_prompt":    ROOT / "domains" / "apo" / "prompts" / "G0_Prompt.txt",
        "g1_prompts": {
            "direct":          ROOT / "domains" / "apo" / "prompts" / "G1_Prompt.txt",
            "ancestral":       ROOT / "domains" / "apo" / "prompts" / "G1_Prompt_Ans.txt",
            "motif":           ROOT / "domains" / "apo" / "prompts" / "G1_Prompt_Motif.txt",
            "ancestral_motif": ROOT / "domains" / "apo" / "prompts" / "G1_Prompt_AnsMotifs.txt",
            "full_graph":      ROOT / "domains" / "apo" / "prompts" / "G1_FullGraphs.txt",
        },
        "expert_graph": ROOT / "domains" / "apo" / "expert.graph",
        "blacklist":    ROOT / "domains" / "apo" / "blacklist.csv",
        "dataset":      ROOT / "domains" / "apo" / "dataset.csv",
    },
}


def build_state_names(domain: str, columns) -> Dict[str, List]:
    """Return a pgmpy-compatible state_names dict for the given domain and column list."""
    if domain == "alarm":
        three_level = {
            "CVP", "PCWP", "TPR", "BP", "CO", "HRBP", "HREKG", "HRSAT",
            "PAP", "SAO2", "MINVOLSET", "INTUBATION",
            "LVEDVOLUME", "STROKEVOLUME", "HR", "PVSAT", "ARTCO2",
        }
        four_level = {
            "PRESS", "EXPCO2", "MINVOL",
            "VENTALV", "VENTLUNG", "VENTTUBE", "VENTMACH",
        }
        return {
            col: (
                list(range(3)) if col in three_level
                else list(range(4)) if col in four_level
                else list(range(2))
            )
            for col in columns
        }

    elif domain == "apo":
        three_level = {"Age", "BMI"}
        six_level   = {"Race"}
        return {
            col: (
                list(range(3)) if col in three_level
                else list(range(6)) if col in six_level
                else list(range(2))
            )
            for col in columns
        }

    else:
        raise ValueError(f"Unknown domain: {domain!r}. Add it to domain_config.DOMAINS.")
