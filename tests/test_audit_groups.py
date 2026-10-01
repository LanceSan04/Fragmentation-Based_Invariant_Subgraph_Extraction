import unittest
from src.evaluation.assay_groups import group_report
from src.analysis.fragment_structure import topology


class AuditTests(unittest.TestCase):
    def test_auc_decomposition_including_ties(self):
        rows = [
            dict(assay="a", label=0, probability=0.1),
            dict(assay="a", label=1, probability=0.1),
            dict(assay="b", label=0, probability=0.9),
            dict(assay="b", label=1, probability=0.8),
        ]
        r = group_report(rows)
        fraction = r["within_pair_fraction"]
        self.assertAlmostEqual(
            r["pooled_auc"],
            fraction * r["within_pair_weighted_auc"] + (1 - fraction) * r["cross_assay_auc"],
        )
        self.assertAlmostEqual(r["macro_assay_auc"], 0.25)

    def test_single_class_assays_are_not_assigned_zero_auc(self):
        r = group_report(
            [dict(assay="a", label=0, probability=0.1), dict(assay="b", label=1, probability=0.9)]
        )
        self.assertIsNone(r["macro_assay_auc"])
        self.assertIsNone(r["within_pair_weighted_auc"])
        self.assertEqual(r["excluded_single_class_assays"], 2)
        self.assertEqual(r["cross_assay_auc"], 1.0)

    def test_components_include_isolated_fragments(self):
        r = topology(4, [(0, 1), (0, 1)])
        self.assertEqual(r["components"], 3)
        self.assertEqual(r["degrees"], [1, 1, 0, 0])
