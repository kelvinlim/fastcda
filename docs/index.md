# fastcda

**fastcda** is a Python package for causal discovery analysis (CDA). It wraps CMU's [Tetrad](https://github.com/cmu-phil/tetrad) Java library (v7.6.3) via JPype, providing a Python API for running causal search algorithms, SEM estimation, and graph visualization.

---

## Prerequisites

- **Java JDK 17+** (JDK 21 LTS recommended) — [download](https://www.oracle.com/java/technologies/downloads/#java21)
- **Graphviz** (`dot` must be on PATH) — [download](https://graphviz.org/download/)
- **Python 3.11+**

If Java or Graphviz are in non-standard locations, create `~/.fastcda.yaml` (Linux/macOS) or `%LOCALAPPDATA%\fastcda\fastcda.yaml` (Windows):

```yaml
JAVA_HOME: /Library/Java/JavaVirtualMachines/jdk-21.jdk/Contents/Home
GRAPHVIZ_BIN: /opt/homebrew/bin
```

---

## Installation

```bash
pip install fastcda
```

---

## Quick Start

```python
from fastcda import FastCDA

fc = FastCDA()

# Load the built-in EMA dataset
df = fc.getEMAData()

# Add lagged columns and standardize
lag_stub = '_lag'
df_lag = fc.add_lag_columns(df, lag_stub=lag_stub)
df_lag_std = fc.standardize_df_cols(df_lag)

# Temporal prior knowledge: lag variables can only be parents
cols = df.columns
knowledge = {'addtemporal': {
    0: [col + lag_stub for col in cols],
    1: [col for col in cols],
}}

# Run GFCI causal search
result, graph = fc.run_model_search(
    df_lag_std,
    model='gfci',
    score={'sem_bic': {'penalty_discount': 1.0}},
    test={'fisher_z': {'alpha': 0.01}},
    knowledge=knowledge,
)

# Display the graph (in a Jupyter notebook)
graph.show_graph()
```

![Example Graph](https://raw.githubusercontent.com/kelvinlim/fastcda/main/assets/causal_graph_boston.png)

---

## Platforms

Tested on **Windows 11**, **macOS Sequoia**, and **Ubuntu 24.04**.

---

## Demo Notebooks

- `fastcda_demo_short.ipynb` — core workflow including node styling, multi-graph comparison, and background knowledge
- `fastcda_demo_long.ipynb` — stability search and advanced usage
- `fastcda_demo_nodestyles.ipynb` — node styling reference
