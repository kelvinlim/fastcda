# Changelog

## [0.1.24] - 2026-03-21

### Changed
- **Default Tetrad version reverted to 7.6.3**: The default `tetrad_version` parameter in
  `FastCDA.__init__` and `start_jvm` is now `"7.6.3"` (was `"7.6.8"`).

## [0.1.22] - 2026-03-05

### Fixed
- **`load_knowledge` — `forbiddirect`/`requiredirect` support**: The `forbiddirect` branch in
  `load_knowledge` had an incomplete loop body (no-op). Completed the loop to call
  `self.set_forbidden(pair[0], pair[1])` for each pair, and added the missing `requiredirect`
  block that calls `self.set_required(pair[0], pair[1])` for each pair.

### Tests
- **`TestLoadKnowledgeForbidRequired`** (10 unit tests in `tests/test_fastcda.py`): mock-based
  tests verifying that `load_knowledge` dispatches correctly to `set_forbidden`/`set_required`
  for all combinations of `forbiddirect`, `requiredirect`, empty lists, missing keys, and the
  combined case with `addtemporal`.
- **`TestKnowledgeForbidRequired`** (7 integration tests in `tests/test_integration.py`):
  JVM-level tests including behavioral verification with a synthetic X→Y→Z linear chain dataset:
  - `test_forbidden_directed_edge_absent`: forbids X→Y, verifies `X --> Y` absent from GFCI output
  - `test_required_edge_present`: requires X→Z (non-natural mediated path), verifies `X --> Z`
    present in GFCI output

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
