# Changelog

## [Unreleased]

### Added
- **Node styling**: Pattern-based visual styling for graph nodes via `fnmatch` glob patterns.
  New methods on `FastCDA`:
  - `set_node_styles(rules)` — store style rules on the instance
  - `get_node_names(dg)` — extract unique node names from a DgraphFlex graph
  - `resolve_node_styles(names, rules)` — preview resolved attributes per node
  - `apply_node_styles(dg, rules)` — apply per-node Graphviz attributes after `load_graph()`
  - `show_styled_graph(dg, rules, ...)` — display styled graph in Jupyter
  - `save_styled_graph(dg, rules, ...)` — save styled graph to file
- Unit tests for all node styling methods (13 tests in `TestNodeStyling`)
- `fastcda_demo_nodestyles.ipynb` — demo notebook for node styling
- `CLAUDE.md` — project guidance for Claude Code
- `CHANGELOG.md` — this file
