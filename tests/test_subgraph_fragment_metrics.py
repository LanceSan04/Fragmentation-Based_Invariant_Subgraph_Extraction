import unittest

from scripts.analyze_subgraphs import build_fragment_selection_metric_rows


class SubgraphFragmentMetricTests(unittest.TestCase):
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
            row for row in fragment_rows
            if row["label"] == "1"
            and row["selection_source"] == "any_view"
            and row["fragment_type"] == "frag"
            and row["fragment_smiles"] == "C=O"
        )
        self.assertEqual(label_one_carbonyl_any["available_count"], 2)
        self.assertEqual(label_one_carbonyl_any["selected_count"], 2)
        self.assertEqual(label_one_carbonyl_any["selection_rate"], 1.0)

        label_zero_carbonyl_any = next(
            row for row in fragment_rows
            if row["label"] == "0"
            and row["selection_source"] == "any_view"
            and row["fragment_type"] == "frag"
            and row["fragment_smiles"] == "C=O"
        )
        self.assertEqual(label_zero_carbonyl_any["available_count"], 1)
        self.assertEqual(label_zero_carbonyl_any["selected_count"], 0)
        self.assertEqual(label_zero_carbonyl_any["selection_rate"], 0.0)

        label_one_pair_any = next(
            row for row in edge_rows
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
            row for row in edge_rows
            if row["label"] == "1" and row["selection_source"] == "view_0"
        ]

        self.assertEqual(len(view_zero_rows), 1)
        self.assertEqual(view_zero_rows[0]["available_count"], 2)
        self.assertEqual(view_zero_rows[0]["selected_count"], 2)


if __name__ == "__main__":
    unittest.main()
