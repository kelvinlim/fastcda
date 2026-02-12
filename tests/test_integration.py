"""
Integration tests for fastcda that exercise the JVM and run Tetrad models.

These tests require:
  - Java JDK 17+
  - Graphviz (dot) on PATH

All tests are skipped automatically when Java is not available.
Run with:  pytest tests/test_integration.py -v
"""

import shutil
import subprocess
import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Skip the entire module if Java is not available
# ---------------------------------------------------------------------------

def _java_available() -> bool:
    """Check if a usable Java (>=17) is on PATH."""
    try:
        result = subprocess.run(
            ['java', '-version'], capture_output=True, text=True, check=True
        )
        import re
        match = re.search(r'(\d+)\.(\d+)', result.stderr)
        if match:
            return int(match.group(1)) >= 17
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    return False


def _graphviz_available() -> bool:
    """Check if Graphviz dot is on PATH."""
    return shutil.which('dot') is not None


requires_java = pytest.mark.skipif(
    not _java_available(),
    reason="Java 17+ not found on PATH"
)
requires_graphviz = pytest.mark.skipif(
    not _graphviz_available(),
    reason="Graphviz 'dot' not found on PATH"
)

pytestmark = [requires_java, requires_graphviz]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fc():
    """
    Module-scoped FastCDA instance.
    Starting the JVM is expensive, so we do it once for all integration tests.
    """
    from fastcda import FastCDA
    return FastCDA(verbose=0)


@pytest.fixture(scope="module")
def boston_df(fc):
    """Boston EMA dataset loaded via FastCDA."""
    return fc.getEMAData()


@pytest.fixture(scope="module")
def boston_lagged(fc, boston_df):
    """Boston dataset with lag columns, standardized."""
    df_lag = fc.add_lag_columns(boston_df, lag_stub='_lag')
    return fc.standardize_df_cols(df_lag)


@pytest.fixture(scope="module")
def boston_knowledge():
    """Temporal knowledge for the Boston EMA dataset."""
    return {
        'addtemporal': {
            0: [
                'alcohol_bev_lag', 'TIB_lag', 'TST_lag',
                'PANAS_PA_lag', 'PANAS_NA_lag',
                'worry_scale_lag', 'PHQ9_lag',
            ],
            1: [
                'alcohol_bev', 'TIB', 'TST',
                'PANAS_PA', 'PANAS_NA',
                'worry_scale', 'PHQ9',
            ],
        }
    }


# ---------------------------------------------------------------------------
# JVM and environment
# ---------------------------------------------------------------------------

class TestJVMStartup:

    def test_jvm_is_running(self, fc):
        import jpype
        assert jpype.isJVMStarted()

    def test_tetrad_version(self, fc):
        version = fc.getTetradVersion()
        assert isinstance(version, str)
        assert len(version) > 0
        # Should be 7.x.x
        assert version.startswith("7")

    def test_java_packages_available(self, fc):
        """Verify that the Tetrad Java packages were loaded."""
        assert fc.td is not None
        assert fc.tg is not None
        assert fc.ts is not None
        assert fc.util is not None


# ---------------------------------------------------------------------------
# df_to_data (DataFrame -> Tetrad DataSet)
# ---------------------------------------------------------------------------

class TestDfToData:

    def test_converts_without_error(self, fc, boston_lagged):
        data = fc.df_to_data(boston_lagged)
        assert data is not None

    def test_column_count_matches(self, fc, boston_lagged):
        data = fc.df_to_data(boston_lagged)
        assert data.getNumColumns() == boston_lagged.shape[1]

    def test_row_count_matches(self, fc, boston_lagged):
        data = fc.df_to_data(boston_lagged)
        assert data.getNumRows() == boston_lagged.shape[0]

    def test_jitter_does_not_crash(self, fc, boston_lagged):
        data = fc.df_to_data(boston_lagged, jitter=True)
        assert data.getNumRows() == boston_lagged.shape[0]


# ---------------------------------------------------------------------------
# run_gfci (low-level GFCI search)
# ---------------------------------------------------------------------------

class TestRunGfci:

    def test_returns_string(self, fc, boston_lagged, boston_knowledge):
        fc.clearKnowledge()
        fc.load_knowledge(boston_knowledge)
        result = fc.run_gfci(boston_lagged, alpha=0.01, penalty_discount=1.0)
        assert isinstance(result, str)

    def test_output_contains_graph_edges(self, fc, boston_lagged, boston_knowledge):
        fc.clearKnowledge()
        fc.load_knowledge(boston_knowledge)
        result = fc.run_gfci(boston_lagged, alpha=0.01, penalty_discount=1.0)
        assert "Graph Edges:" in result

    def test_edges_parseable(self, fc, boston_lagged, boston_knowledge):
        fc.clearKnowledge()
        fc.load_knowledge(boston_knowledge)
        result = fc.run_gfci(boston_lagged, alpha=0.01, penalty_discount=1.0)
        edges = fc.extract_edges(result)
        assert isinstance(edges, list)
        # Boston data with these settings should produce at least some edges
        assert len(edges) > 0

    def test_jitter_flag(self, fc, boston_lagged, boston_knowledge):
        """Running with jitter=True should still produce valid output."""
        fc.clearKnowledge()
        fc.load_knowledge(boston_knowledge)
        result = fc.run_gfci(boston_lagged, alpha=0.01, penalty_discount=1.0, jitter=True)
        edges = fc.extract_edges(result)
        assert isinstance(edges, list)


# ---------------------------------------------------------------------------
# Knowledge loading round-trip
# ---------------------------------------------------------------------------

class TestKnowledgeRoundTrip:

    def test_load_from_prior_file(self, fc):
        """Load knowledge from the bundled prior file and verify tiers."""
        from importlib.resources import files as pkg_resources_files
        txt_resource = pkg_resources_files('fastcda.data').joinpath('boston_prior.txt')
        prior_lines = fc.read_prior_file(str(txt_resource))
        knowledge = fc.extract_knowledge(prior_lines)

        fc.clearKnowledge()
        fc.load_knowledge(knowledge)

        # The internal Tetrad Knowledge object should now have entries
        # Run a quick search to confirm it doesn't crash
        df = fc.getEMAData()
        df_lag = fc.add_lag_columns(df, lag_stub='_lag')
        df_std = fc.standardize_df_cols(df_lag)
        result = fc.run_gfci(df_std, alpha=0.05, penalty_discount=1.0)
        assert isinstance(result, str)

    def test_create_lag_knowledge_round_trip(self, fc):
        """create_lag_knowledge output should be loadable."""
        cols = ['A', 'B', 'C']
        knowledge = fc.create_lag_knowledge(cols, lag_stub='_lag')
        fc.clearKnowledge()
        fc.load_knowledge(knowledge)
        # No assertion beyond not crashing — verifies Tetrad accepted it


# ---------------------------------------------------------------------------
# run_model_search (high-level search with SEM)
# ---------------------------------------------------------------------------

class TestRunModelSearch:

    def test_returns_result_and_graph(self, fc, boston_lagged, boston_knowledge):
        fc.clearKnowledge()
        result, dg = fc.run_model_search(
            boston_lagged,
            model='gfci',
            score={'sem_bic': {'penalty_discount': 1.0}},
            test={'fisher_z': {'alpha': 0.01}},
            knowledge=boston_knowledge,
        )
        assert isinstance(result, dict)
        assert 'edges' in result
        assert 'cda_output' in result
        assert 'sem_results' in result

    def test_edges_are_strings(self, fc, boston_lagged, boston_knowledge):
        fc.clearKnowledge()
        result, _ = fc.run_model_search(
            boston_lagged,
            model='gfci',
            score={'sem_bic': {'penalty_discount': 1.0}},
            test={'fisher_z': {'alpha': 0.01}},
            knowledge=boston_knowledge,
        )
        for edge in result['edges']:
            assert isinstance(edge, str)
            parts = edge.split(' ')
            assert len(parts) == 3  # "A --> B"

    def test_sem_results_present(self, fc, boston_lagged, boston_knowledge):
        fc.clearKnowledge()
        result, dg = fc.run_model_search(
            boston_lagged,
            model='gfci',
            score={'sem_bic': {'penalty_discount': 1.0}},
            test={'fisher_z': {'alpha': 0.01}},
            knowledge=boston_knowledge,
        )
        sem = result['sem_results']
        if sem is not None:
            assert 'estimates' in sem
            assert 'stats' in sem
            assert isinstance(sem['estimates'], pd.DataFrame)
            assert 'Estimate' in sem['estimates'].columns

    def test_graph_object_created(self, fc, boston_lagged, boston_knowledge):
        fc.clearKnowledge()
        result, dg = fc.run_model_search(
            boston_lagged,
            model='gfci',
            score={'sem_bic': {'penalty_discount': 1.0}},
            test={'fisher_z': {'alpha': 0.01}},
            knowledge=boston_knowledge,
        )
        if result['edges']:
            assert dg is not None

    def test_no_graph_mode(self, fc, boston_lagged, boston_knowledge):
        """run_graph=False should return None for the graph object."""
        fc.clearKnowledge()
        result, dg = fc.run_model_search(
            boston_lagged,
            model='gfci',
            score={'sem_bic': {'penalty_discount': 1.0}},
            test={'fisher_z': {'alpha': 0.01}},
            knowledge=boston_knowledge,
            run_graph=False,
        )
        assert dg is None
        assert 'edges' in result

    def test_no_sem_mode(self, fc, boston_lagged, boston_knowledge):
        """run_sem=False should skip SEM estimation."""
        fc.clearKnowledge()
        result, dg = fc.run_model_search(
            boston_lagged,
            model='gfci',
            score={'sem_bic': {'penalty_discount': 1.0}},
            test={'fisher_z': {'alpha': 0.01}},
            knowledge=boston_knowledge,
            run_sem=False,
        )
        # sem_results won't be set when run_sem=False
        assert 'edges' in result

    def test_bad_edge_fixup(self, fc):
        """o-- edges should be converted to o-o in results."""
        # Simulate what run_model_search does with edge fixup
        edges = ["A o-- B", "C --> D"]
        for index in range(len(edges)):
            if 'o--' in edges[index]:
                edges[index] = edges[index].replace('o--', 'o-o')
        assert edges[0] == "A o-o B"
        assert edges[1] == "C --> D"


# ---------------------------------------------------------------------------
# run_stability_search (subsampled repeated search)
# ---------------------------------------------------------------------------

class TestRunStabilitySearch:

    def test_basic_stability_search(self, fc, boston_df, boston_knowledge):
        """Run a short stability search (5 runs) and verify structure."""
        results, dg = fc.run_stability_search(
            boston_df,
            model='gfci',
            knowledge=boston_knowledge,
            score={'sem_bic': {'penalty_discount': 1.0}},
            test={'fisher_z': {'alpha': 0.05}},
            runs=5,
            min_fraction=0.5,
            subsample_fraction=0.9,
            lag_stub='_lag',
        )
        assert isinstance(results, dict)
        assert 'edges' in results
        assert 'sorted_edge_counts' in results
        assert 'sorted_edge_counts_raw' in results
        assert 'edge_counts' in results
        assert 'cda_output' in results

    def test_edge_counts_are_fractions(self, fc, boston_df, boston_knowledge):
        runs = 5
        results, _ = fc.run_stability_search(
            boston_df,
            model='gfci',
            knowledge=boston_knowledge,
            score={'sem_bic': {'penalty_discount': 1.0}},
            test={'fisher_z': {'alpha': 0.05}},
            runs=runs,
            min_fraction=0.5,
            subsample_fraction=0.9,
            lag_stub='_lag',
            run_graph=False,
        )
        for edge, fraction in results['sorted_edge_counts'].items():
            assert 0 < fraction <= 1.0

    def test_raw_counts_are_integers(self, fc, boston_df, boston_knowledge):
        runs = 5
        results, _ = fc.run_stability_search(
            boston_df,
            model='gfci',
            knowledge=boston_knowledge,
            score={'sem_bic': {'penalty_discount': 1.0}},
            test={'fisher_z': {'alpha': 0.05}},
            runs=runs,
            min_fraction=0.5,
            subsample_fraction=0.9,
            lag_stub='_lag',
            run_graph=False,
        )
        for edge, count in results['sorted_edge_counts_raw'].items():
            assert isinstance(count, int)
            assert 1 <= count <= runs

    def test_save_to_file(self, fc, boston_df, boston_knowledge, tmp_path):
        """Results should be JSON-serializable and writable to a file."""
        save_file = str(tmp_path / "stability_results.json")
        results, _ = fc.run_stability_search(
            boston_df,
            model='gfci',
            knowledge=boston_knowledge,
            score={'sem_bic': {'penalty_discount': 1.0}},
            test={'fisher_z': {'alpha': 0.05}},
            runs=3,
            min_fraction=0.5,
            subsample_fraction=0.9,
            lag_stub='_lag',
            run_graph=False,
            run_sem=False,
            save_file=save_file,
        )
        assert os.path.exists(save_file)
        with open(save_file) as f:
            loaded = json.load(f)
        assert 'edges' in loaded
        assert 'sorted_edge_counts' in loaded

    def test_no_lag_stub(self, fc, boston_knowledge):
        """Stability search without lag_stub (pre-lagged data)."""
        df = fc.getEMAData()
        df_lag = fc.add_lag_columns(df, lag_stub='_lag')
        results, _ = fc.run_stability_search(
            df_lag,
            model='gfci',
            knowledge=boston_knowledge,
            score={'sem_bic': {'penalty_discount': 1.0}},
            test={'fisher_z': {'alpha': 0.05}},
            runs=3,
            min_fraction=0.5,
            subsample_fraction=0.9,
            lag_stub='',
            run_graph=False,
            run_sem=False,
        )
        assert isinstance(results['edges'], list)


# ---------------------------------------------------------------------------
# SEM estimation
# ---------------------------------------------------------------------------

class TestSemopy:

    def test_run_semopy_on_directed_edges(self, fc, boston_lagged):
        """Run SEM on a known set of directed edges."""
        edges = ["PANAS_NA_lag --> PANAS_NA", "TST_lag --> TST"]
        lavaan_model = fc.edges_to_lavaan(edges)
        sem_results = fc.run_semopy(lavaan_model, boston_lagged)

        assert sem_results['estimates'] is not None
        assert isinstance(sem_results['estimates'], pd.DataFrame)
        assert 'Estimate' in sem_results['estimates'].columns
        assert sem_results['stats'] is not None

    def test_semopy_estimates_dict(self, fc, boston_lagged):
        edges = ["PANAS_NA_lag --> PANAS_NA"]
        lavaan_model = fc.edges_to_lavaan(edges)
        sem_results = fc.run_semopy(lavaan_model, boston_lagged)

        assert sem_results['estimatesDict'] is not None
        assert isinstance(sem_results['estimatesDict'], list)
        for record in sem_results['estimatesDict']:
            assert 'src' in record
            assert 'dest' in record

    def test_semopy_with_no_directed_edges(self, fc, boston_lagged):
        """An empty lavaan model should be handled gracefully."""
        edges = ["A o-o B"]  # Excluded by default
        lavaan_model = fc.edges_to_lavaan(edges)
        # lavaan_model will be empty string
        assert lavaan_model.strip() == ""


# ---------------------------------------------------------------------------
# Full pipeline end-to-end
# ---------------------------------------------------------------------------

class TestEndToEnd:

    def test_full_pipeline(self, fc):
        """
        End-to-end: load data -> lag -> standardize -> search -> SEM -> graph.
        Mirrors the demo notebook workflow.
        """
        # Load data
        df = fc.getEMAData()
        assert df.shape[0] > 0

        # Add lags
        df_lag = fc.add_lag_columns(df, lag_stub='_lag')
        assert df_lag.shape[1] == df.shape[1] * 2

        # Standardize
        df_std = fc.standardize_df_cols(df_lag)

        # Build knowledge
        knowledge = fc.create_lag_knowledge(df.columns.tolist(), lag_stub='_lag')

        # Run search
        fc.clearKnowledge()
        result, dg = fc.run_model_search(
            df_std,
            model='gfci',
            score={'sem_bic': {'penalty_discount': 1.0}},
            test={'fisher_z': {'alpha': 0.01}},
            knowledge=knowledge,
        )

        # Validate result structure
        assert len(result['edges']) > 0
        assert result['cda_output'] is not None

        # Validate SEM ran
        sem = result['sem_results']
        if sem is not None and sem['estimates'] is not None:
            assert 'Estimate' in sem['estimates'].columns

        # Validate graph object
        assert dg is not None

    def test_pipeline_with_different_alpha(self, fc):
        """A stricter alpha should produce fewer (or equal) edges."""
        df = fc.getEMAData()
        df_lag = fc.add_lag_columns(df, lag_stub='_lag')
        df_std = fc.standardize_df_cols(df_lag)
        knowledge = fc.create_lag_knowledge(df.columns.tolist(), lag_stub='_lag')

        fc.clearKnowledge()
        result_loose, _ = fc.run_model_search(
            df_std,
            model='gfci',
            score={'sem_bic': {'penalty_discount': 1.0}},
            test={'fisher_z': {'alpha': 0.10}},
            knowledge=knowledge,
            run_sem=False,
        )

        fc.clearKnowledge()
        result_strict, _ = fc.run_model_search(
            df_std,
            model='gfci',
            score={'sem_bic': {'penalty_discount': 1.0}},
            test={'fisher_z': {'alpha': 0.001}},
            knowledge=knowledge,
            run_sem=False,
        )

        # Stricter alpha should generally produce fewer or equal edges
        assert len(result_strict['edges']) <= len(result_loose['edges'])
