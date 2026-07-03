#!/usr/bin/env python3
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Iterable, Tuple, List, Optional
import os

import networkx as nx


from pathlib import Path

# Folder containing the 5 initial edge files
BASE_DIR = Path(__file__).resolve().parent
INITIAL_FOLDER = BASE_DIR / "initial_llm_edges"

# Pattern to select initial files inside the folder
INITIAL_GLOB = "initial_edges_gpt5*.csv"

# Combined edge file (X,Y format)
COMBINED_FILE = INITIAL_FOLDER / "initial_edges_combined.csv"

# Does combined file have header?
COMBINED_HAS_HEADER = True

# Do initial files have header?
INITIALS_HAVE_HEADER = True

# Output files
OUTPUT_FILE = BASE_DIR / "cycles" / "combined_acyclic.csv"
REMOVAL_LOG = BASE_DIR / "cycles" / "removed_edges.tsv"

# Optional safety cap (None = no limit)
MAX_REMOVALS = None

# If True: frequency = number of files containing edge (recommended)
# If False: frequency = total occurrences across all files
COUNT_BY_FILE_PRESENCE = True

Edge = Tuple[str, str]


def read_edges_csv_xy(path: Path, has_header: bool) -> List[Edge]:
    edges: List[Edge] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        first = True
        for row in reader:
            if first and has_header:
                first = False
                continue
            first = False

            if not row or len(row) < 2:
                continue

            x = row[0].strip()
            y = row[1].strip()

            if x and y:
                edges.append((x, y))
    return edges


def write_edges_csv_xy(path: Path, edges: Iterable[Edge], write_header: bool):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["X", "Y"])
        for x, y in edges:
            writer.writerow([x, y])


def build_graph(edges: Iterable[Edge]) -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_edges_from(edges)
    return g


def get_initial_files(folder: Path, combined_file: Path, pattern: str) -> List[Path]:
    combined_resolved = combined_file.resolve()

    files = sorted([
        p for p in folder.glob(pattern)
        if p.is_file() and p.resolve() != combined_resolved
    ])

    if len(files) < 5:
        raise RuntimeError(f"Found only {len(files)} initial files. Need at least 5.")

    if len(files) > 5:
        print("More than 5 initial files found — using first 5 after sorting.")

    return files[:5]


def compute_frequencies(files: List[Path], has_header: bool) -> Counter[Edge]:
    freq: Counter[Edge] = Counter()

    for file in files:
        edges = read_edges_csv_xy(file, has_header)

        if COUNT_BY_FILE_PRESENCE:
            unique_edges = set(edges)
            for e in unique_edges:
                freq[e] += 1
        else:
            for e in edges:
                freq[e] += 1

    return freq


def find_cycle_edges(g: nx.DiGraph) -> List[Edge]:
    cycle = nx.find_cycle(g, orientation="original")
    return [(u, v) for (u, v, _) in cycle]


def choose_edge_to_remove(cycle_edges: List[Edge], freq: Counter[Edge]) -> Edge:
    # Remove edge with smallest frequency (ties broken lexicographically)
    return min(cycle_edges, key=lambda e: (freq.get(e, 0), e[0], e[1]))


def break_cycles(g: nx.DiGraph, freq: Counter[Edge]) -> List[Tuple[Edge, int]]:
    removed: List[Tuple[Edge, int]] = []
    count = 0

    while True:
        try:
            cycle_edges = find_cycle_edges(g)
        except nx.exception.NetworkXNoCycle:
            break

        edge = choose_edge_to_remove(cycle_edges, freq)
        g.remove_edge(*edge)

        removed.append((edge, freq.get(edge, 0)))
        count += 1

        if MAX_REMOVALS and count >= MAX_REMOVALS:
            print("Reached MAX_REMOVALS limit.")
            break

    return removed


def main():
    print("Loading initial files...")
    initial_files = get_initial_files(INITIAL_FOLDER, COMBINED_FILE, INITIAL_GLOB)

    for f in initial_files:
        print("  -", f)

    print("\nComputing edge frequencies...")
    freq = compute_frequencies(initial_files, INITIALS_HAVE_HEADER)

    print("Loading combined graph...")
    combined_edges = read_edges_csv_xy(COMBINED_FILE, COMBINED_HAS_HEADER)
    g = build_graph(combined_edges)

    print("Checking for cycles...")
    try:
        find_cycle_edges(g)
        print("Cycles detected. Breaking cycles...")
        had_cycle = True
    except nx.exception.NetworkXNoCycle:
        print("No cycles found.")
        had_cycle = False

    removed = break_cycles(g, freq)

    print("\nWriting output...")
    write_edges_csv_xy(OUTPUT_FILE, g.edges(), COMBINED_HAS_HEADER)

    REMOVAL_LOG.parent.mkdir(parents=True, exist_ok=True)
    with REMOVAL_LOG.open("w", encoding="utf-8") as f:
        f.write("removed_X\tremoved_Y\tinitial_frequency\n")
        for (x, y), fr in removed:
            f.write(f"{x}\t{y}\t{fr}\n")

    print("\n==============================")
    print("Finished")
    print("Had cycle initially:", had_cycle)
    print("Edges removed:", len(removed))
    print("Final edge count:", g.number_of_edges())
    print("==============================")


if __name__ == "__main__":
    main()