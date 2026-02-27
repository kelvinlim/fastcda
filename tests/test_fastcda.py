"""
Tests for fastcda utility methods.

These tests exercise the pure-Python data processing and parsing utilities
that don't require a running JVM. Integration tests that require Java/Tetrad
are in test_integration.py and are skipped when Java is unavailable.
"""

import numpy as np
import pandas as pd
import pytest
from importlib.resources import files as pkg_resources_files
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def boston_df():
    """Load the bundled Boston EMA dataset."""
    csv_resource = pkg_resources_files('fastcda.data').joinpath('boston_data_raw.csv')
    return pd.read_csv(str(csv_resource))


@pytest.fixture
def boston_prior_lines():
    """Load the bundled Boston prior file as a list of strings."""
    txt_resource = pkg_resources_files('fastcda.data').joinpath('boston_prior.txt')
    with open(str(txt_resource), 'r') as f:
        return f.readlines()


@pytest.fixture
def sample_edges():
    """A small list of edges typical of GFCI output."""
    return [
        "A --> B",
        "C o-> D",
        "E o-o F",
        "G <-> H",
    ]


@pytest.fixture
def fc_no_jvm():
    """
    Create a FastCDA instance with JVM startup and environment checks mocked out.
    This allows testing pure-Python methods without Java installed.
    """
    with patch('fastcda.fastcda.FastCDA.check_graphviz_dot', return_value=(True, '12.0')), \
         patch('fastcda.fastcda.FastCDA.check_java_version', return_value=True), \
         patch('fastcda.fastcda.FastCDA.startJVM'):
        from fastcda import FastCDA
        fc = FastCDA(verbose=0)
    return fc


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

class TestDataLoading:

    def test_boston_df_shape(self, boston_df):
        """Boston dataset has expected dimensions."""
        assert boston_df.shape[0] == 641
        assert boston_df.shape[1] == 7

    def test_boston_df_columns(self, boston_df):
        expected = ['alcohol_bev', 'TIB', 'TST', 'PANAS_PA', 'PANAS_NA', 'worry_scale', 'PHQ9']
        assert list(boston_df.columns) == expected

    def test_boston_df_no_nulls(self, boston_df):
        assert not boston_df.isnull().any().any()


# ---------------------------------------------------------------------------
# add_lag_columns
# ---------------------------------------------------------------------------

class TestAddLagColumns:

    def test_lag_column_count(self, fc_no_jvm, boston_df):
        result = fc_no_jvm.add_lag_columns(boston_df, lag_stub='_lag')
        # Should have original + lagged columns
        assert result.shape[1] == boston_df.shape[1] * 2

    def test_lag_column_names(self, fc_no_jvm, boston_df):
        result = fc_no_jvm.add_lag_columns(boston_df, lag_stub='_lag')
        for col in boston_df.columns:
            assert f'{col}_lag' in result.columns

    def test_lag_drops_first_row(self, fc_no_jvm, boston_df):
        result = fc_no_jvm.add_lag_columns(boston_df, lag_stub='_lag')
        # One row lost due to shift
        assert result.shape[0] == boston_df.shape[0] - 1

    def test_lag_no_nulls(self, fc_no_jvm, boston_df):
        result = fc_no_jvm.add_lag_columns(boston_df, lag_stub='_lag')
        assert not result.isnull().any().any()

    def test_lag_values_correct(self, fc_no_jvm):
        """The lagged column value at row i should equal original column at row i."""
        df = pd.DataFrame({'x': [10, 20, 30, 40]})
        result = fc_no_jvm.add_lag_columns(df, lag_stub='_lag')
        # After dropping NaN row, result index is reset: 0,1,2
        # _lag column should have: 10, 20, 30 (shifted from original)
        assert list(result['x_lag']) == [10.0, 20.0, 30.0]
        assert list(result['x']) == [20.0, 30.0, 40.0]


# ---------------------------------------------------------------------------
# standardize_df_cols
# ---------------------------------------------------------------------------

class TestStandardize:

    def test_standardized_mean_near_zero(self, fc_no_jvm, boston_df):
        result = fc_no_jvm.standardize_df_cols(boston_df)
        for col in result.columns:
            assert abs(result[col].mean()) < 1e-10

    def test_standardized_std_near_one(self, fc_no_jvm, boston_df):
        result = fc_no_jvm.standardize_df_cols(boston_df)
        for col in result.columns:
            assert abs(result[col].std(ddof=0) - 1.0) < 1e-10

    def test_standardized_shape_unchanged(self, fc_no_jvm, boston_df):
        result = fc_no_jvm.standardize_df_cols(boston_df)
        assert result.shape == boston_df.shape


# ---------------------------------------------------------------------------
# subsample_df
# ---------------------------------------------------------------------------

class TestSubsample:

    def test_subsample_fraction(self, fc_no_jvm, boston_df):
        result = fc_no_jvm.subsample_df(boston_df, fraction=0.5)
        # Allow some tolerance since sample is random
        expected = int(boston_df.shape[0] * 0.5)
        assert result.shape[0] == expected

    def test_subsample_reproducible(self, fc_no_jvm, boston_df):
        r1 = fc_no_jvm.subsample_df(boston_df, fraction=0.8, random_state=42)
        r2 = fc_no_jvm.subsample_df(boston_df, fraction=0.8, random_state=42)
        pd.testing.assert_frame_equal(r1, r2)

    def test_subsample_preserves_columns(self, fc_no_jvm, boston_df):
        result = fc_no_jvm.subsample_df(boston_df, fraction=0.9)
        assert list(result.columns) == list(boston_df.columns)

    def test_subsample_invalid_fraction(self, fc_no_jvm, boston_df):
        with pytest.raises(ValueError):
            fc_no_jvm.subsample_df(boston_df, fraction=0)
        with pytest.raises(ValueError):
            fc_no_jvm.subsample_df(boston_df, fraction=1.5)


# ---------------------------------------------------------------------------
# extract_edges
# ---------------------------------------------------------------------------

class TestExtractEdges:

    def test_extract_from_typical_output(self, fc_no_jvm):
        text = (
            "Graph Nodes:\nA;B;C\n\n"
            "Graph Edges:\n"
            "1. A --> B\n"
            "2. B o-> C\n"
            "3. A o-o C\n\n"
            "Graph Attributes:\n"
        )
        edges = fc_no_jvm.extract_edges(text)
        assert len(edges) == 3
        assert "A --> B" in edges
        assert "B o-> C" in edges
        assert "A o-o C" in edges

    def test_extract_from_empty_output(self, fc_no_jvm):
        edges = fc_no_jvm.extract_edges("")
        assert edges == []

    def test_extract_from_no_edges(self, fc_no_jvm):
        text = "Graph Nodes:\nA;B\n\nGraph Edges:\n\nGraph Attributes:\n"
        edges = fc_no_jvm.extract_edges(text)
        assert edges == []


# ---------------------------------------------------------------------------
# edges_to_lavaan
# ---------------------------------------------------------------------------

class TestEdgesToLavaan:

    def test_directed_edge(self, fc_no_jvm):
        lavaan = fc_no_jvm.edges_to_lavaan(["A --> B"])
        assert "B ~ A" in lavaan

    def test_o_arrow_edge(self, fc_no_jvm):
        lavaan = fc_no_jvm.edges_to_lavaan(["C o-> D"])
        assert "D ~ C" in lavaan

    def test_excluded_edges(self, fc_no_jvm, sample_edges):
        lavaan = fc_no_jvm.edges_to_lavaan(sample_edges)
        # o-o and <-> should be excluded by default
        assert "F ~ E" not in lavaan
        assert "H ~ G" not in lavaan

    def test_custom_exclude(self, fc_no_jvm):
        edges = ["A --> B", "C <-> D"]
        lavaan = fc_no_jvm.edges_to_lavaan(edges, exclude_edges=['<->'])
        assert "B ~ A" in lavaan
        assert "D ~ C" not in lavaan


# ---------------------------------------------------------------------------
# extract_knowledge / read_prior_file
# ---------------------------------------------------------------------------

class TestKnowledge:

    def test_extract_knowledge_from_boston_prior(self, fc_no_jvm, boston_prior_lines):
        knowledge = fc_no_jvm.extract_knowledge(boston_prior_lines)
        assert 'addtemporal' in knowledge
        tiers = knowledge['addtemporal']
        assert 0 in tiers
        assert 1 in tiers
        # Tier 0 should have lag variables
        assert all('_lag' in var for var in tiers[0])
        # Tier 1 should have non-lag variables
        assert all('_lag' not in var for var in tiers[1])

    def test_extract_knowledge_tier_sizes(self, fc_no_jvm, boston_prior_lines):
        knowledge = fc_no_jvm.extract_knowledge(boston_prior_lines)
        tiers = knowledge['addtemporal']
        assert len(tiers[0]) == 7
        assert len(tiers[1]) == 7


# ---------------------------------------------------------------------------
# create_lag_knowledge
# ---------------------------------------------------------------------------

class TestCreateLagKnowledge:

    def test_creates_two_tiers(self, fc_no_jvm):
        cols = ['A', 'B', 'C']
        knowledge = fc_no_jvm.create_lag_knowledge(cols, lag_stub='_lag')
        tiers = knowledge['addtemporal']
        assert 0 in tiers
        assert 1 in tiers

    def test_tier_0_has_lag_suffix(self, fc_no_jvm):
        cols = ['A', 'B']
        knowledge = fc_no_jvm.create_lag_knowledge(cols, lag_stub='_lag')
        assert knowledge['addtemporal'][0] == ['A_lag', 'B_lag']

    def test_tier_1_matches_input(self, fc_no_jvm):
        cols = ['A', 'B']
        knowledge = fc_no_jvm.create_lag_knowledge(cols, lag_stub='_lag')
        assert knowledge['addtemporal'][1] == ['A', 'B']


# ---------------------------------------------------------------------------
# create_permuted_dfs
# ---------------------------------------------------------------------------

class TestCreatePermutedDfs:

    def test_returns_correct_count(self, fc_no_jvm, boston_df):
        result = fc_no_jvm.create_permuted_dfs(boston_df, n_permutations=3, seed=42)
        assert len(result) == 3

    def test_permuted_shape_matches(self, fc_no_jvm, boston_df):
        result = fc_no_jvm.create_permuted_dfs(boston_df, n_permutations=2, seed=42)
        for df in result:
            assert df.shape == boston_df.shape

    def test_permuted_values_differ(self, fc_no_jvm, boston_df):
        result = fc_no_jvm.create_permuted_dfs(boston_df, n_permutations=1, seed=42)
        # At least one column should have different ordering
        assert not result[0].equals(boston_df)

    def test_reproducible_with_seed(self, fc_no_jvm, boston_df):
        r1 = fc_no_jvm.create_permuted_dfs(boston_df, n_permutations=2, seed=99)
        r2 = fc_no_jvm.create_permuted_dfs(boston_df, n_permutations=2, seed=99)
        for df1, df2 in zip(r1, r2):
            pd.testing.assert_frame_equal(df1, df2)

    def test_empty_df_raises(self, fc_no_jvm):
        with pytest.raises(ValueError):
            fc_no_jvm.create_permuted_dfs(pd.DataFrame(), n_permutations=1)

    def test_invalid_n_raises(self, fc_no_jvm, boston_df):
        with pytest.raises(ValueError):
            fc_no_jvm.create_permuted_dfs(boston_df, n_permutations=0)


# ---------------------------------------------------------------------------
# select_edges
# ---------------------------------------------------------------------------

class TestSelectEdges:

    def test_selects_above_threshold(self, fc_no_jvm):
        edge_counts = {
            "A --> B": 0.9,
            "C --> D": 0.5,
            "E o-> F": 0.8,
        }
        selected = fc_no_jvm.select_edges(edge_counts, min_fraction=0.75)
        assert "A --> B" in selected
        assert "E o-> F" in selected

    def test_excludes_below_threshold(self, fc_no_jvm):
        edge_counts = {
            "A --> B": 0.3,
            "C --> D": 0.2,
        }
        selected = fc_no_jvm.select_edges(edge_counts, min_fraction=0.5)
        assert len(selected) == 0

    def test_undirected_edges(self, fc_no_jvm):
        edge_counts = {
            "A o-o B": 0.9,
        }
        selected = fc_no_jvm.select_edges(edge_counts, min_fraction=0.75)
        assert len(selected) == 1


# ---------------------------------------------------------------------------
# summarize_estimates
# ---------------------------------------------------------------------------

class TestSummarizeEstimates:

    def test_summarize(self, fc_no_jvm):
        df = pd.DataFrame({'Estimate': [0.5, -0.3, 0.8, -0.2]})
        result = fc_no_jvm.summarize_estimates(df)
        assert 'mean_abs_estimates' in result
        assert 'std_abs_estimates' in result
        assert abs(result['mean_abs_estimates'] - 0.45) < 1e-10


# ---------------------------------------------------------------------------
# Node styling
# ---------------------------------------------------------------------------

class TestNodeStyling:

    def test_get_node_names_basic(self):
        from dgraph_flex import DgraphFlex
        from fastcda import FastCDA
        dg = DgraphFlex()
        dg.add_edges(["A --> B", "B o-> C", "C o-o D"])
        names = FastCDA.get_node_names(dg)
        assert sorted(names) == ["A", "B", "C", "D"]

    def test_get_node_names_deduplication(self):
        from dgraph_flex import DgraphFlex
        from fastcda import FastCDA
        dg = DgraphFlex()
        dg.add_edges(["A --> B", "A --> C"])
        names = FastCDA.get_node_names(dg)
        assert names.count("A") == 1

    def test_resolve_prefix_match(self):
        from fastcda import FastCDA
        names = ["COG_1", "COG_2", "MRI_1", "PHQ9"]
        styles = [{"pattern": "COG*", "shape": "box", "fillcolor": "lightblue"}]
        resolved = FastCDA.resolve_node_styles(names, styles)
        assert "COG_1" in resolved
        assert "COG_2" in resolved
        assert resolved["COG_1"]["shape"] == "box"
        assert "MRI_1" not in resolved

    def test_resolve_exact_match(self):
        from fastcda import FastCDA
        names = ["PHQ9", "PHQ9_lag"]
        styles = [{"pattern": "PHQ9", "shape": "diamond"}]
        resolved = FastCDA.resolve_node_styles(names, styles)
        assert "PHQ9" in resolved
        assert "PHQ9_lag" not in resolved

    def test_resolve_suffix_match(self):
        from fastcda import FastCDA
        names = ["A_lag", "B_lag", "C"]
        styles = [{"pattern": "*_lag", "shape": "box"}]
        resolved = FastCDA.resolve_node_styles(names, styles)
        assert "A_lag" in resolved
        assert "B_lag" in resolved
        assert "C" not in resolved

    def test_resolve_order_override(self):
        from fastcda import FastCDA
        names = ["COG_1"]
        styles = [
            {"pattern": "*", "fillcolor": "white"},
            {"pattern": "COG*", "fillcolor": "lightblue"},
        ]
        resolved = FastCDA.resolve_node_styles(names, styles)
        assert resolved["COG_1"]["fillcolor"] == "lightblue"

    def test_resolve_partial_override(self):
        from fastcda import FastCDA
        names = ["COG_1"]
        styles = [
            {"pattern": "*", "shape": "oval", "fillcolor": "white"},
            {"pattern": "COG*", "fillcolor": "lightblue"},
        ]
        resolved = FastCDA.resolve_node_styles(names, styles)
        assert resolved["COG_1"]["shape"] == "oval"
        assert resolved["COG_1"]["fillcolor"] == "lightblue"

    def test_resolve_empty_styles(self):
        from fastcda import FastCDA
        resolved = FastCDA.resolve_node_styles(["A", "B"], [])
        assert resolved == {}

    def test_resolve_no_match(self):
        from fastcda import FastCDA
        styles = [{"pattern": "X*", "shape": "box"}]
        resolved = FastCDA.resolve_node_styles(["A", "B"], styles)
        assert resolved == {}

    def test_apply_node_styles(self):
        from dgraph_flex import DgraphFlex
        from fastcda import FastCDA
        dg = DgraphFlex()
        dg.add_edges(["COG_1 --> MRI_1", "MRI_1 --> PHQ9"])
        dg.load_graph()
        styles = [{"pattern": "COG*", "shape": "box"}]
        FastCDA.apply_node_styles(dg, styles)
        assert 'shape=box' in dg.dot.source

    def test_show_styled_graph_returns_dot(self):
        import graphviz
        from dgraph_flex import DgraphFlex
        from fastcda import FastCDA
        dg = DgraphFlex()
        dg.add_edges(["A --> B"])
        styles = [{"pattern": "*", "style": "filled", "fillcolor": "white"}]
        result = FastCDA.show_styled_graph(dg, styles)
        assert isinstance(result, graphviz.Digraph)
        assert 'fillcolor=white' in result.source

    def test_set_node_styles(self, fc_no_jvm):
        styles = [{"pattern": "COG*", "shape": "box"}]
        fc_no_jvm.set_node_styles(styles)
        assert fc_no_jvm.node_styles == styles

    def test_set_node_styles_defensive_copy(self, fc_no_jvm):
        styles = [{"pattern": "COG*", "shape": "box"}]
        fc_no_jvm.set_node_styles(styles)
        styles.append({"pattern": "X*", "shape": "oval"})
        assert len(fc_no_jvm.node_styles) == 1


# ---------------------------------------------------------------------------
# Paired / multi-graph comparison tests
# ---------------------------------------------------------------------------

class TestPairedGraphs:
    """Tests for the paired graph comparison feature.

    These tests use DgraphFlex directly (no JVM required).  Layout-dependent
    tests mock ``graphviz.Digraph.pipe`` to avoid needing Graphviz installed.
    """

    # --- _get_all_union_nodes ---

    def test_union_nodes_disjoint(self):
        from dgraph_flex import DgraphFlex
        from fastcda import FastCDA
        dg1 = DgraphFlex()
        dg1.add_edges(["A --> B"])
        dg2 = DgraphFlex()
        dg2.add_edges(["C --> D"])
        result = FastCDA._get_all_union_nodes([dg1, dg2])
        assert result == ["A", "B", "C", "D"]

    def test_union_nodes_overlapping(self):
        from dgraph_flex import DgraphFlex
        from fastcda import FastCDA
        dg1 = DgraphFlex()
        dg1.add_edges(["A --> B", "B --> C"])
        dg2 = DgraphFlex()
        dg2.add_edges(["B --> D", "C --> E"])
        result = FastCDA._get_all_union_nodes([dg1, dg2])
        assert result == ["A", "B", "C", "D", "E"]

    def test_union_nodes_identical(self):
        from dgraph_flex import DgraphFlex
        from fastcda import FastCDA
        dg1 = DgraphFlex()
        dg1.add_edges(["A --> B"])
        dg2 = DgraphFlex()
        dg2.add_edges(["A --> B"])
        result = FastCDA._get_all_union_nodes([dg1, dg2])
        assert result == ["A", "B"]

    def test_union_nodes_three_graphs(self):
        from dgraph_flex import DgraphFlex
        from fastcda import FastCDA
        dg1 = DgraphFlex()
        dg1.add_edges(["A --> B"])
        dg2 = DgraphFlex()
        dg2.add_edges(["C --> D"])
        dg3 = DgraphFlex()
        dg3.add_edges(["B --> E"])
        result = FastCDA._get_all_union_nodes([dg1, dg2, dg3])
        assert result == ["A", "B", "C", "D", "E"]

    # --- _get_connected_nodes ---

    def test_connected_nodes_basic(self):
        from dgraph_flex import DgraphFlex
        from fastcda import FastCDA
        dg = DgraphFlex()
        dg.add_edges(["A --> B", "C o-> D"])
        result = FastCDA._get_connected_nodes(dg)
        assert result == {"A", "B", "C", "D"}

    def test_connected_nodes_directed_only(self):
        from dgraph_flex import DgraphFlex
        from fastcda import FastCDA
        dg = DgraphFlex()
        dg.add_edges(["A --> B", "C o-o D"])
        result = FastCDA._get_connected_nodes(dg, directed_only=True)
        assert result == {"A", "B"}

    def test_connected_nodes_empty_graph(self):
        from dgraph_flex import DgraphFlex
        from fastcda import FastCDA
        dg = DgraphFlex()
        result = FastCDA._get_connected_nodes(dg)
        assert result == set()

    # --- _apply_disconnected_styling ---

    def test_disconnected_styling_applied(self):
        import graphviz
        from fastcda import FastCDA
        dot = graphviz.Digraph()
        dot.node('A')
        dot.node('B')
        dot.node('C')
        all_nodes = ['A', 'B', 'C']
        connected = {'A', 'B'}
        FastCDA._apply_disconnected_styling(dot, all_nodes, connected)
        source = dot.source
        assert '#BBBBBB' in source

    def test_disconnected_styling_none_disconnected(self):
        import graphviz
        from fastcda import FastCDA
        dot = graphviz.Digraph()
        dot.node('A')
        dot.node('B')
        all_nodes = ['A', 'B']
        connected = {'A', 'B'}
        source_before = dot.source
        FastCDA._apply_disconnected_styling(dot, all_nodes, connected)
        assert dot.source == source_before

    # --- _extract_positions ---

    def test_extract_positions_parses_plain(self):
        import graphviz
        from fastcda import FastCDA

        mock_plain = (
            "graph 1.0 3.5 2.0\n"
            "node A 0.5 1.0 0.75 0.5 A solid oval black lightgrey\n"
            "node B 2.0 1.0 0.75 0.5 B solid oval black lightgrey\n"
            "edge A B ...\n"
            "stop\n"
        )
        dot = graphviz.Digraph()
        dot.node('A')
        dot.node('B')
        with patch.object(dot, 'pipe',
                          return_value=mock_plain.encode('utf-8')):
            positions = FastCDA._extract_positions(dot)
        assert positions == {'A': ('0.5', '1.0'), 'B': ('2.0', '1.0')}

    # --- _apply_positions ---

    def test_apply_positions(self):
        import graphviz
        from fastcda import FastCDA
        dot = graphviz.Digraph()
        dot.node('A')
        dot.node('B')
        positions = {'A': ('1.5', '2.0'), 'B': ('3.0', '4.0')}
        FastCDA._apply_positions(dot, positions)
        source = dot.source
        assert 'pos="1.5,2.0!"' in source
        assert 'pos="3.0,4.0!"' in source

    # --- _build_union_dot ---

    def test_build_union_dot_contains_all_edges(self):
        from dgraph_flex import DgraphFlex
        from fastcda import FastCDA
        dg1 = DgraphFlex()
        dg1.add_edges(["A --> B"])
        dg2 = DgraphFlex()
        dg2.add_edges(["C --> D"])
        union_dot = FastCDA._build_union_dot([dg1, dg2])
        source = union_dot.source
        assert 'A -> B' in source
        assert 'C -> D' in source

    def test_build_union_dot_directed_only(self):
        from dgraph_flex import DgraphFlex
        from fastcda import FastCDA
        dg1 = DgraphFlex()
        dg1.add_edges(["A --> B"])
        dg2 = DgraphFlex()
        dg2.add_edges(["C o-o D"])
        union_dot = FastCDA._build_union_dot([dg1, dg2],
                                              directed_only=True)
        source = union_dot.source
        assert 'A -> B' in source
        # C o-o D excluded but C, D present as nodes
        assert 'C' in source
        assert 'D' in source

    def test_build_union_dot_deduplicates(self):
        from dgraph_flex import DgraphFlex
        from fastcda import FastCDA
        dg1 = DgraphFlex()
        dg1.add_edges(["A --> B"])
        dg2 = DgraphFlex()
        dg2.add_edges(["A --> B"])
        union_dot = FastCDA._build_union_dot([dg1, dg2])
        source = union_dot.source
        assert source.count('A -> B') == 1

    # --- _prepare_n_graphs (full pipeline, mocked layout) ---

    def _mock_plain_for_nodes(self, nodes):
        """Generate mock plain-format output for a set of nodes."""
        lines = ["graph 1.0 5.0 3.0"]
        for i, name in enumerate(sorted(nodes)):
            x = str(0.5 + i * 1.5)
            y = "1.0"
            lines.append(
                f"node {name} {x} {y} 0.75 0.5 {name} solid oval "
                f"black lightgrey")
        lines.append("stop")
        return "\n".join(lines)

    def test_prepare_n_graphs_gray_disconnected(self):
        from dgraph_flex import DgraphFlex
        from fastcda import FastCDA

        dg1 = DgraphFlex()
        dg1.add_edges(["A --> B", "B --> C"])
        dg2 = DgraphFlex()
        dg2.add_edges(["A --> B"])  # C is disconnected in dg2

        mock_plain = self._mock_plain_for_nodes(["A", "B", "C"])

        with patch('graphviz.Digraph.pipe',
                   return_value=mock_plain.encode('utf-8')):
            dots = FastCDA._prepare_n_graphs(
                [dg1, dg2], gray_disconnected=True)

        assert len(dots) == 2
        # In dot2 (dg2), C should have gray styling
        assert '#BBBBBB' in dots[1].source
        # Both should have pinned positions
        assert 'pos=' in dots[0].source
        assert 'pos=' in dots[1].source
        # Both should use neato engine
        assert dots[0].engine == 'neato'
        assert dots[1].engine == 'neato'

    def test_prepare_n_graphs_no_gray(self):
        from dgraph_flex import DgraphFlex
        from fastcda import FastCDA

        dg1 = DgraphFlex()
        dg1.add_edges(["A --> B", "B --> C"])
        dg2 = DgraphFlex()
        dg2.add_edges(["A --> B"])

        mock_plain = self._mock_plain_for_nodes(["A", "B", "C"])

        with patch('graphviz.Digraph.pipe',
                   return_value=mock_plain.encode('utf-8')):
            dots = FastCDA._prepare_n_graphs(
                [dg1, dg2], gray_disconnected=False)

        # No gray styling should be present for C
        assert '#BBBBBB' not in dots[1].source

    def test_prepare_n_graphs_with_node_styles(self):
        from dgraph_flex import DgraphFlex
        from fastcda import FastCDA

        dg1 = DgraphFlex()
        dg1.add_edges(["A_lag --> B"])
        dg2 = DgraphFlex()
        dg2.add_edges(["A_lag --> C"])

        mock_plain = self._mock_plain_for_nodes(["A_lag", "B", "C"])
        styles = [{"pattern": "*_lag", "shape": "box"}]

        with patch('graphviz.Digraph.pipe',
                   return_value=mock_plain.encode('utf-8')):
            dots = FastCDA._prepare_n_graphs(
                [dg1, dg2], node_styles=styles,
                gray_disconnected=False)

        assert 'shape=box' in dots[0].source
        assert 'shape=box' in dots[1].source

    def test_prepare_n_graphs_labels(self):
        from dgraph_flex import DgraphFlex
        from fastcda import FastCDA

        dg1 = DgraphFlex()
        dg1.add_edges(["A --> B"])
        dg2 = DgraphFlex()
        dg2.add_edges(["A --> B"])

        mock_plain = self._mock_plain_for_nodes(["A", "B"])

        with patch('graphviz.Digraph.pipe',
                   return_value=mock_plain.encode('utf-8')):
            dots = FastCDA._prepare_n_graphs(
                [dg1, dg2], labels=["Model A", "Model B"])

        assert 'Model A' in dots[0].source
        assert 'Model B' in dots[1].source

    def test_prepare_three_graphs(self):
        from dgraph_flex import DgraphFlex
        from fastcda import FastCDA

        dg1 = DgraphFlex()
        dg1.add_edges(["A --> B", "B --> C"])
        dg2 = DgraphFlex()
        dg2.add_edges(["A --> C"])
        dg3 = DgraphFlex()
        dg3.add_edges(["B --> C"])

        mock_plain = self._mock_plain_for_nodes(["A", "B", "C"])

        with patch('graphviz.Digraph.pipe',
                   return_value=mock_plain.encode('utf-8')):
            dots = FastCDA._prepare_n_graphs(
                [dg1, dg2, dg3], gray_disconnected=True,
                labels=["G1", "G2", "G3"])

        assert len(dots) == 3
        # dg2 is missing B as connected, dg3 is missing A as connected
        assert '#BBBBBB' in dots[1].source  # B disconnected in dg2
        assert '#BBBBBB' in dots[2].source  # A disconnected in dg3
        assert 'G1' in dots[0].source
        assert 'G2' in dots[1].source
        assert 'G3' in dots[2].source

    # --- save_n_graphs validation ---

    def test_save_n_graphs_mismatched_lengths(self):
        from dgraph_flex import DgraphFlex
        from fastcda import FastCDA

        dg1 = DgraphFlex()
        dg1.add_edges(["A --> B"])
        dg2 = DgraphFlex()
        dg2.add_edges(["A --> B"])

        with pytest.raises(ValueError, match="must match"):
            FastCDA.save_n_graphs([dg1, dg2], ["only_one"])
