import unittest
from rdkit import Chem
from src.experiments.motif_check import expanded_fixture, retention_geometry, fixture


class MotifExpansionTests(unittest.TestCase):
    def test_fixed_split_has_no_canonical_identity_overlap(self):
        train, held = expanded_fixture()
        canonical = lambda rows: [Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in rows]
        a, b = canonical(train), canonical(held)
        self.assertEqual(len(a), len(set(a)))
        self.assertEqual(len(b), len(set(b)))
        self.assertFalse(set(a) & set(b))
        self.assertGreaterEqual(len(held), 20)

    def test_class_balance_and_matched_atom_counts(self):
        for split in expanded_fixture():
            batch, targets = fixture(split, 0, 23)
            self.assertEqual(int(batch.y.sum()), len(split) // 2)
            self.assertEqual(sum(bool(t) for t in targets), len(split) // 2)
            for i in range(0, len(split), 2):
                self.assertEqual(Chem.MolFromSmiles(split[i]).GetNumAtoms(),
                                 Chem.MolFromSmiles(split[i + 1]).GetNumAtoms())

    def test_edgeless_motif_is_necessarily_retained(self):
        report = retention_geometry(["CC(=O)C"])
        self.assertEqual(report["mean_uniform_random_retention"], 1)


if __name__ == "__main__":
    unittest.main()
