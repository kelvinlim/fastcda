# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

fastcda is a Python package for causal discovery analysis (CDA). It wraps CMU's Tetrad Java library (v7.6.3) via JPype, providing a Python API for running causal search algorithms, SEM estimation, and graph visualization.

## Prerequisites

- **Java JDK 17+** (JDK 21 LTS recommended)
- **Graphviz** (`dot` command must be on PATH)
- **Python 3.11+**

If Java/Graphviz are in non-standard locations, create `~/.fastcda.yaml` (Linux/macOS) or `%LOCALAPPDATA%\fastcda\fastcda.yaml` (Windows) with `JAVA_HOME` and `GRAPHVIZ_BIN` keys.

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

- `tests/test_fastcda.py` — Unit tests for pure-Python utilities (no JVM required, ~37 tests)
- `tests/test_integration.py` — Integration tests that start the JVM and run Tetrad models (~30 tests, auto-skipped if Java is unavailable)

Notebooks (`fastcda_demo_short.ipynb`, `fastcda_demo_long.ipynb`, etc.) also serve as manual validation.

## Architecture

The entire library is a single class `FastCDA` in `fastcda/fastcda.py`, exposed via `fastcda/__init__.py`.

### JVM / Tetrad Integration

`FastCDA.__init__` starts a JVM via JPype with the bundled JAR at `fastcda/jars/tetrad-gui-7.6.3-launch.jar` (or a user-provided JAR path). The JVM can only be started once per process. Key Tetrad Java packages are stored as instance attributes:
- `self.td` → `edu.cmu.tetrad.data`
- `self.tg` → `edu.cmu.tetrad.graph`
- `self.ts` → `edu.cmu.tetrad.search`
- `self.knowledge` → Tetrad `Knowledge` object for prior constraints

### Data Flow Pipeline

1. **Load data** → pandas DataFrame (`getEMAData()`, `getSampleData()`, `read_csv()`)
2. **Transform** → optional lagging (`add_lag_columns()`), standardization (`standardize_df_cols()`), subsampling (`subsample_df()`)
3. **Convert** → `df_to_data()` bridges pandas → Tetrad's `BoxDataSet` (with optional jitter)
4. **Search** → `run_gfci()` executes the GFCI algorithm, returns Tetrad graph as string
5. **Parse** → `extract_edges()` regex-parses numbered edge lines from Tetrad output
6. **SEM** → `edges_to_lavaan()` converts to lavaan syntax, `run_semopy()` fits via semopy
7. **Visualize** → edges + SEM results are loaded into a `DgraphFlex` graph object

### Two Main Entry Points

- **`run_model_search(df, **kwargs)`** — Single run: GFCI search → edge extraction → optional SEM → DgraphFlex graph. Returns `(result_dict, dg)`.
- **`run_stability_search(full_df, ...)`** — Bootstrapped stability: runs `run_model_search` N times on subsampled data, counts edge frequency, retains edges above `min_fraction`. The `select_edges()` method handles deduplication of directed vs undirected edge types.

### Prior Knowledge System

Temporal tier constraints (e.g., lagged variables can only be parents) are set via `load_knowledge()` which calls Tetrad's `Knowledge.addToTier()`. The knowledge dict uses `addtemporal` key with tier numbers (0-indexed) mapping to variable name lists.

### Key External Dependencies

- **JPype1** — Python↔Java bridge for Tetrad
- **semopy** — Structural equation modeling in Python
- **dgraph_flex** (`DgraphFlex`) — Graph object for edge management and Graphviz rendering
- **scikit-learn** — `StandardScaler` for data standardization

## Version Management

Version is defined in two places that must stay in sync:
- `fastcda/fastcda.py`: `__version_info__` tuple (line 26)
- `pyproject.toml`: `version` field (line 7)

## Node Styling

Nodes in causal graphs can be visually styled by name pattern using `fnmatch` glob syntax. Style rules are a list of dicts, each with a `pattern` key and any Graphviz node attributes. Rules apply in order — later rules override earlier ones for the same node.

```python
node_styles = [
    {"pattern": "*_lag",    "shape": "box",  "style": "filled", "fillcolor": "lightyellow"},
    {"pattern": "PANAS_PA*","shape": "box",  "style": "filled", "fillcolor": "lightblue", "penwidth": "2"},
    {"pattern": "PANAS_NA*","shape": "oval", "style": "filled,dotted", "fillcolor": "lightpink",
     "penwidth": "5", "color": "red"},
    {"pattern": "PHQ9",     "shape": "diamond", "style": "filled", "fillcolor": "lightsalmon"},
]
```

Pattern types: `COG*` (prefix), `*_lag` (suffix), `PHQ9` (exact), `PANAS_?A` (single-char wildcard).

Any valid Graphviz node attribute works: `shape`, `fillcolor`, `style`, `color`, `penwidth`, `fontname`, `fontsize`, `fontcolor`, `width`, `height`, etc.

**Usage** — use `show_styled_graph` / `save_styled_graph` instead of `dg.show_graph()` / `dg.save_graph()`:

```python
result, dg = fc.run_model_search(df, ...)

# Display in notebook
fc.show_styled_graph(dg, node_styles)

# Save to file
fc.save_styled_graph(dg, node_styles, "my_graph")
```

These are wrappers that call `dg.load_graph()` then apply per-node styling before rendering. This is necessary because DgraphFlex rebuilds its Graphviz object from scratch on every `show_graph()` / `save_graph()` call.

Helper methods: `get_node_names(dg)` returns all node names from a graph, `resolve_node_styles(names, rules)` previews which attributes each node will get, `set_node_styles(rules)` stores rules on the FastCDA instance.

## Edge Type Conventions

Tetrad outputs edges like `A --> B`, `A o-> B`, `A o-o B`, `A <-> B`. The code handles a known Tetrad quirk where `o--` edges are converted to `o-o`. When converting to lavaan for SEM, only directed edges (`-->`, `o->`) are used; undirected types (`---`, `<->`, `o-o`) are excluded.
