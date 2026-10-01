import unittest
from pathlib import Path

from src.analysis.subgraphs import (
    build_fragment_interpretation_tables,
    build_fragment_selection_metric_rows,
    resolve_subgraph_analysis_options,
)
from src.training.pipeline import Config


class SubgraphFragmentMetricTests(unittest.TestCase):
    def test_subgraph_analysis_options_use_config_defaults(self):
        config = Config(
            {
                "save_model": "outputs/run/checkpoints/best_model.pt",
                "processed_dir": "data/processed/example/group_graph",
                "test_split": "ood_test",
                "subgraph_analysis_dir": "outputs/run/subgraph_analysis",
                "subgraph_num_visualize": 7,
                "subgraph_metric_min_available": 12,
                "subgraph_metric_top_k": 30,
            }
        )

        options = resolve_subgraph_analysis_options(config)

        self.assertEqual(options["model_path"], "outputs/run/checkpoints/best_model.pt")
        self.assertEqual(options["processed_dir"], "data/processed/example/group_graph")
        self.assertEqual(options["split"], "ood_test")
        self.assertEqual(str(options["output_dir"]), str(Path("outputs/run/subgraph_analysis")))
        self.assertIsNone(options["num_samples"])
        self.assertEqual(options["num_visualize"], 7)
        self.assertEqual(options["metric_min_available"], 12)
        self.assertEqual(options["metric_top_k"], 30)

    def test_subgraph_analysis_options_prefer_explicit_over_config(self):
        config = Config(
            {
                "save_model": "config_checkpoint.pt",
                "processed_dir": "config_processed",
                "subgraph_split": "iid_test",
                "subgraph_analysis_dir": "config_output",
                "subgraph_num_samples": 100,
                "subgraph_num_visualize": 3,
                "subgraph_metric_min_available": 10,
                "subgraph_metric_top_k": 25,
            }
        )

        options = resolve_subgraph_analysis_options(
            config,
            model_path="explicit_checkpoint.pt",
            processed_dir="explicit_processed",
            split="ood_test",
            num_samples=5,
            num_visualize=2,
            output_dir="explicit_output",
            metric_min_available=4,
            metric_top_k=6,
        )

        self.assertEqual(options["model_path"], "explicit_checkpoint.pt")
        self.assertEqual(options["processed_dir"], "explicit_processed")
        self.assertEqual(options["split"], "ood_test")
        self.assertEqual(options["num_samples"], 5)
        self.assertEqual(options["num_visualize"], 2)
        self.assertEqual(str(options["output_dir"]), "explicit_output")
        self.assertEqual(options["metric_min_available"], 4)
        self.assertEqual(options["metric_top_k"], 6)

    def test_fragment_metrics_are_label_and_source_aware(self):
        records = [
            {
                "num_views": 2,
                "row": {"label": "1"},
                "fragment_signatures": [
                    ("frag", "C=O"),
                    ("split_frag", "N"),
                    ("frag", "C=O"),
                ],
                "fragment_edges": [(0, 1), (1, 2)],
                "per_view_frag_ids": [[0, 1], [1, 2]],
                "agg_frag_ids": [0, 2],
                "per_view_edge_ids": [[0], [1]],
                "agg_edge_ids": [0],
            },
            {
                "num_views": 2,
                "row": {"label": "0"},
                "fragment_signatures": [
                    ("frag", "C=O"),
                    ("frag", "O"),
                ],
                "fragment_edges": [(0, 1)],
                "per_view_frag_ids": [[1], []],
                "agg_frag_ids": [1],
                "per_view_edge_ids": [[], []],
                "agg_edge_ids": [],
            },
        ]

        fragment_rows, edge_rows = build_fragment_selection_metric_rows(records)

        label_one_carbonyl_any = next(
            row
            for row in fragment_rows
            if row["label"] == "1"
            and row["selection_source"] == "any_view"
            and row["fragment_type"] == "frag"
            and row["fragment_smiles"] == "C=O"
        )
        self.assertEqual(label_one_carbonyl_any["available_count"], 2)
        self.assertEqual(label_one_carbonyl_any["selected_count"], 2)
        self.assertEqual(label_one_carbonyl_any["selection_rate"], 1.0)

        label_zero_carbonyl_any = next(
            row
            for row in fragment_rows
            if row["label"] == "0"
            and row["selection_source"] == "any_view"
            and row["fragment_type"] == "frag"
            and row["fragment_smiles"] == "C=O"
        )
        self.assertEqual(label_zero_carbonyl_any["available_count"], 1)
        self.assertEqual(label_zero_carbonyl_any["selected_count"], 0)
        self.assertEqual(label_zero_carbonyl_any["selection_rate"], 0.0)

        label_one_pair_any = next(
            row
            for row in edge_rows
            if row["label"] == "1"
            and row["selection_source"] == "any_view"
            and {row["fragment_a_smiles"], row["fragment_b_smiles"]} == {"C=O", "N"}
        )
        self.assertEqual(label_one_pair_any["available_count"], 2)
        self.assertEqual(label_one_pair_any["selected_count"], 2)
        self.assertEqual(label_one_pair_any["selection_rate"], 1.0)

    def test_fragment_pair_signature_is_undirected(self):
        records = [
            {
                "num_views": 1,
                "row": {"label": "1"},
                "fragment_signatures": [("frag", "N"), ("frag", "C=O")],
                "fragment_edges": [(0, 1)],
                "per_view_frag_ids": [[0, 1]],
                "agg_frag_ids": [0, 1],
                "per_view_edge_ids": [[0]],
                "agg_edge_ids": [0],
            },
            {
                "num_views": 1,
                "row": {"label": "1"},
                "fragment_signatures": [("frag", "C=O"), ("frag", "N")],
                "fragment_edges": [(0, 1)],
                "per_view_frag_ids": [[0, 1]],
                "agg_frag_ids": [0, 1],
                "per_view_edge_ids": [[0]],
                "agg_edge_ids": [0],
            },
        ]

        _, edge_rows = build_fragment_selection_metric_rows(records)
        view_zero_rows = [
            row for row in edge_rows if row["label"] == "1" and row["selection_source"] == "view_0"
        ]

        self.assertEqual(len(view_zero_rows), 1)
        self.assertEqual(view_zero_rows[0]["available_count"], 2)
        self.assertEqual(view_zero_rows[0]["selected_count"], 2)

    def test_fragment_metrics_keep_fragmentation_methods_separate(self):
        records = [
            {
                "num_views": 3,
                "view_methods": ["group_graph", "brics", "brics"],
                "row": {"label": "1"},
                "per_view_frag_ids": [[0], [1], []],
                "per_view_edge_ids": [[0], [0], []],
                "method_fragment_data": {
                    "group_graph": {
                        "num_fragments": 2,
                        "fragment_signatures": [("frag", "C=O"), ("frag", "N")],
                        "fragment_edges": [(0, 1)],
                        "atom_to_fragment": [0, 0, 1],
                        "agg_atoms": [0, 1],
                        "agg_frag_ids": [1],
                        "agg_edge_ids": [0],
                    },
                    "brics": {
                        "num_fragments": 2,
                        "fragment_signatures": [("brics", "C=O"), ("brics", "N")],
                        "fragment_edges": [(0, 1)],
                        "atom_to_fragment": [0, 1, 1],
                        "agg_atoms": [0],
                        "agg_frag_ids": [0],
                        "agg_edge_ids": [],
                    },
                },
            }
        ]

        fragment_rows, edge_rows = build_fragment_selection_metric_rows(records)

        group_graph_any = next(
            row
            for row in fragment_rows
            if row["fragmentation_method"] == "group_graph"
            and row["selection_source"] == "any_view"
            and row["fragment_smiles"] == "C=O"
        )
        self.assertEqual(group_graph_any["selected_count"], 1)

        brics_any_carbonyl = next(
            row
            for row in fragment_rows
            if row["fragmentation_method"] == "brics"
            and row["selection_source"] == "any_view"
            and row["fragment_smiles"] == "C=O"
        )
        self.assertEqual(brics_any_carbonyl["selected_count"], 0)

        brics_aggregated_carbonyl = next(
            row
            for row in fragment_rows
            if row["fragmentation_method"] == "brics"
            and row["selection_source"] == "aggregated"
            and row["fragment_smiles"] == "C=O"
        )
        self.assertEqual(brics_aggregated_carbonyl["selected_count"], 1)

        brics_view_sources = {
            row["selection_source"]
            for row in fragment_rows
            if row["fragmentation_method"] == "brics"
        }
        self.assertEqual(
            brics_view_sources,
            {"view_1", "view_2", "any_view", "aggregated"},
        )
        self.assertNotIn("view_0", brics_view_sources)

        group_graph_pair = next(
            row
            for row in edge_rows
            if row["fragmentation_method"] == "group_graph"
            and row["selection_source"] == "aggregated"
        )
        self.assertEqual(group_graph_pair["selected_count"], 1)

        brics_pair = next(
            row
            for row in edge_rows
            if row["fragmentation_method"] == "brics" and row["selection_source"] == "aggregated"
        )
        self.assertEqual(brics_pair["selected_count"], 0)

    def test_interpretation_tables_filter_and_rank_fragment_metrics(self):
        fragment_rows = [
            {
                "label": "1",
                "selection_source": "aggregated",
                "fragment_type": "frag",
                "fragment_smiles": "C=O",
                "available_count": 20,
                "selected_count": 10,
                "selection_rate": 0.5,
            },
            {
                "label": "0",
                "selection_source": "aggregated",
                "fragment_type": "frag",
                "fragment_smiles": "C=O",
                "available_count": 20,
                "selected_count": 2,
                "selection_rate": 0.1,
            },
            {
                "label": "1",
                "selection_source": "aggregated",
                "fragment_type": "frag",
                "fragment_smiles": "rare",
                "available_count": 1,
                "selected_count": 1,
                "selection_rate": 1.0,
            },
        ]

        tables = build_fragment_interpretation_tables(
            fragment_rows,
            [],
            min_available=5,
            top_k=10,
        )

        top_rates = tables["top_fragments_by_selection_rate.csv"]
        self.assertEqual(len(top_rates), 2)
        self.assertEqual(top_rates[0]["fragment_smiles"], "C=O")
        self.assertEqual(top_rates[0]["label"], "1")

        filter_summary = tables["interpretation_filter_summary.csv"][0]
        self.assertEqual(filter_summary["total_rows"], 3)
        self.assertEqual(filter_summary["kept_rows"], 2)
        self.assertEqual(filter_summary["dropped_low_support_rows"], 1)

    def test_interpretation_tables_keep_only_selected_report_set(self):
        fragment_rows = [
            {
                "label": "0",
                "selection_source": "aggregated",
                "fragment_type": "frag",
                "fragment_smiles": "C=O",
                "available_count": 10,
                "selected_count": 1,
                "selection_rate": 0.1,
            },
            {
                "label": "1",
                "selection_source": "aggregated",
                "fragment_type": "frag",
                "fragment_smiles": "C=O",
                "available_count": 10,
                "selected_count": 7,
                "selection_rate": 0.7,
            },
            {
                "label": "1",
                "selection_source": "any_view",
                "fragment_type": "frag",
                "fragment_smiles": "C=O",
                "available_count": 10,
                "selected_count": 9,
                "selection_rate": 0.9,
            },
            {
                "label": "1",
                "selection_source": "view_0",
                "fragment_type": "frag",
                "fragment_smiles": "C=O",
                "available_count": 10,
                "selected_count": 2,
                "selection_rate": 0.2,
            },
            {
                "label": "1",
                "selection_source": "view_1",
                "fragment_type": "frag",
                "fragment_smiles": "C=O",
                "available_count": 10,
                "selected_count": 8,
                "selection_rate": 0.8,
            },
        ]

        tables = build_fragment_interpretation_tables(
            fragment_rows,
            [],
            min_available=1,
            top_k=10,
        )

        self.assertEqual(
            set(tables),
            {
                "interpretation_filter_summary.csv",
                "top_fragments_by_selection_rate.csv",
                "top_fragments_by_selected_count.csv",
                "top_fragment_pairs_by_selection_rate.csv",
                "top_fragment_pairs_by_selected_count.csv",
                "top_functional_group_fragments.csv",
                "top_functional_group_pairs.csv",
            },
        )
        self.assertNotIn("fragment_label_selection_deltas.csv", tables)
        self.assertNotIn("fragment_enrichment.csv", tables)
        self.assertNotIn("fragment_view_selection_spread.csv", tables)
        self.assertNotIn("fragment_aggregation_deltas.csv", tables)

    def test_interpretation_tables_cover_fragment_pairs_and_functional_groups(self):
        edge_rows = [
            {
                "label": "1",
                "selection_source": "aggregated",
                "fragment_a_type": "frag",
                "fragment_a_smiles": "C=O",
                "fragment_b_type": "split_frag",
                "fragment_b_smiles": "N",
                "available_count": 12,
                "selected_count": 9,
                "selection_rate": 0.75,
            },
            {
                "label": "1",
                "selection_source": "aggregated",
                "fragment_a_type": "frag",
                "fragment_a_smiles": "O",
                "fragment_b_type": "split_frag",
                "fragment_b_smiles": "N",
                "available_count": 12,
                "selected_count": 3,
                "selection_rate": 0.25,
            },
        ]

        tables = build_fragment_interpretation_tables(
            [],
            edge_rows,
            min_available=5,
            top_k=10,
        )

        top_pairs = tables["top_fragment_pairs_by_selection_rate.csv"]
        self.assertEqual(top_pairs[0]["fragment_a_smiles"], "C=O")
        self.assertEqual(top_pairs[0]["fragment_b_smiles"], "N")

        group_pairs = tables["top_functional_group_pairs.csv"]
        self.assertEqual(len(group_pairs), 2)
        self.assertEqual(group_pairs[0]["rank_within_group"], 1)
        self.assertEqual(group_pairs[0]["fragment_a_smiles"], "C=O")
        self.assertNotIn("fragment_pair_type_selection_summary.csv", tables)
        self.assertNotIn("fragment_pair_enrichment.csv", tables)
        self.assertNotIn("fragment_pair_label_selection_deltas.csv", tables)
        self.assertNotIn("fragment_pair_view_selection_spread.csv", tables)
        self.assertNotIn("fragment_pair_aggregation_deltas.csv", tables)


if __name__ == "__main__":
    unittest.main()
