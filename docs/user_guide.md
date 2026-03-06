# User Guide

## Data Pipeline

The typical FastCDA workflow has five stages:

```
Load data → Transform → Search → SEM → Visualize
```

### 1. Load Data

```python
from fastcda import FastCDA
fc = FastCDA()

# Built-in EMA dataset
df = fc.getEMAData()

# Or load your own CSV
df = fc.read_csv("mydata.csv")
```

### 2. Transform

```python
# Add lagged copies of each column (suffix '_lag')
df_lag = fc.add_lag_columns(df, lag_stub='_lag')

# Standardize all columns (zero mean, unit variance)
df_std = fc.standardize_df_cols(df_lag)

# Optionally subsample rows
df_sub = fc.subsample_df(df_std, n=200, seed=42)
```

### 3. Run Causal Search

```python
result, graph = fc.run_model_search(
    df_std,
    model='gfci',
    score={'sem_bic': {'penalty_discount': 1.0}},
    test={'fisher_z': {'alpha': 0.01}},
    knowledge=knowledge,   # optional prior knowledge dict
)
```

`result` is a dict containing `edges`, `cda_output`, and optionally `sem_results`.
`graph` is a `DgraphFlex` object ready for visualization.

### 4. Stability Search (Bootstrap)

For robust edge discovery, run GFCI on many subsamples and keep only edges that appear frequently:

```python
result_stable = fc.run_stability_search(
    df_std,
    n_iterations=50,
    subsample_size=200,
    min_fraction=0.5,       # keep edges found in ≥50% of runs
    knowledge=knowledge,
)
```

### 5. Visualize

```python
# Show full graph
graph.show_graph()

# Directed edges only
graph.show_graph(directed_only=True)

# Save to PNG
graph.save_graph("my_graph", res=300)
```

---

## Node Styling

Nodes can be styled by name pattern using `fnmatch` glob syntax. Rules are applied in order — later rules override earlier ones for the same node.

```python
node_styles = [
    {"pattern": "*_lag",        "style": "dotted"},
    {"pattern": "PANAS_PA*",    "style": "filled", "fillcolor": "lightgreen"},
    {"pattern": "PANAS_NA*",    "style": "filled", "fillcolor": "lightpink"},
    {"pattern": "alcohol_bev*", "shape": "box", "style": "filled",
     "fillcolor": "purple", "fontcolor": "white"},
]

# Display styled graph in notebook
fc.show_styled_graph(graph, node_styles)

# Save styled graph to file
fc.save_styled_graph(graph, node_styles, "styled_graph")
```

Any valid [Graphviz node attribute](https://graphviz.org/doc/info/attrs.html) works: `shape`, `fillcolor`, `style`, `color`, `penwidth`, `fontname`, `fontsize`, `fontcolor`, `width`, `height`, etc.

**Pattern types:**

| Pattern | Matches |
|---------|---------|
| `*_lag` | suffix — any node ending in `_lag` |
| `COG*` | prefix — any node starting with `COG` |
| `PHQ9` | exact — only `PHQ9` |
| `PANAS_?A` | single wildcard — e.g. `PANAS_PA` or `PANAS_NA` |

Helper methods:

```python
# List all node names in a graph
names = fc.get_node_names(graph)

# Preview which style attributes each node will receive
resolved = fc.resolve_node_styles(names, node_styles)
```

![Node Styling Example](https://raw.githubusercontent.com/kelvinlim/fastcda/main/assets/node_styles_example.png)

---

## Multi-Graph Comparison

Compare N graphs side-by-side with nodes pinned at identical positions:

```python
result1, g1 = fc.run_model_search(df_std, model='gfci',
    score={'sem_bic': {'penalty_discount': 1.0}}, ...)
result2, g2 = fc.run_model_search(df_std, model='gfci',
    score={'sem_bic': {'penalty_discount': 2.0}}, ...)
result3, g3 = fc.run_model_search(df_std, model='gfci',
    score={'sem_bic': {'penalty_discount': 3.0}}, ...)

fc.show_n_graphs(
    [g1, g2, g3],
    node_styles=node_styles,
    gray_disconnected=True,
    directed_only=True,
    labels=["PD=1.0", "PD=2.0", "PD=3.0"],
    graph_size="10,8",
)

# Save all three to PNG files
fc.save_n_graphs(
    [g1, g2, g3],
    ["graph_pd1", "graph_pd2", "graph_pd3"],
    node_styles=node_styles,
    gray_disconnected=True,
    labels=["PD=1.0", "PD=2.0", "PD=3.0"],
    graph_size="10,8",
    res=300,
)
```

Disconnected nodes (no edges in a given graph) are grayed out by default (`gray_disconnected=True`), making structural differences immediately visible.

---

## Edge Types

Tetrad outputs PAG (Partial Ancestral Graph) edges:

| Edge | Meaning |
|------|---------|
| `A --> B` | A is a direct cause of B |
| `A o-> B` | A may be a direct cause of B (uncertain) |
| `A o-o B` | Undirected / uncertain in both directions |
| `A <-> B` | A and B share a latent common cause |

Only directed edges (`-->`, `o->`) are used when converting to lavaan for SEM estimation. Undirected types (`---`, `<->`, `o-o`) are excluded.

---

## SEM Results

After `run_model_search`, `result['sem_results']` (if available) contains semopy fit statistics. Use `summarize_estimates` to extract a clean summary table:

```python
summary = fc.summarize_estimates(result['sem_results'])
print(summary)
```
