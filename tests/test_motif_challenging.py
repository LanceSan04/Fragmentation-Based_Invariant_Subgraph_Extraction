import unittest
from rdkit import Chem
from src.experiments.motif_check import challenging_fixture, retention_geometry


class ChallengingMotifTests(unittest.TestCase):
    def test_disjoint_balanced_atom_and_connectivity_matched_pairs(self):
        train, held = challenging_fixture()
        canonical = [Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in train + held]
        self.assertEqual(len(set(canonical)), len(canonical))
        query = Chem.MolFromSmarts("[CX3]=[OX1]")
        for split in (train, held):
            for i in range(0, len(split), 2):
                positive, negative = [Chem.MolFromSmiles(s) for s in split[i:i + 2]]
                self.assertTrue(positive.HasSubstructMatch(query))
                self.assertFalse(negative.HasSubstructMatch(query))
                self.assertEqual(positive.GetNumAtoms(), negative.GetNumAtoms())
                edges = lambda m: {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))) for b in m.GetBonds()}
                self.assertEqual(edges(positive), edges(negative))

    def test_no_retention_ceiling_at_fixed_quarter_budget(self):
        _, held = challenging_fixture()
        geometry = retention_geometry(held, 0.25)
        self.assertEqual(geometry["always_retained_molecules"], 0)
        self.assertEqual(geometry["positive_molecules"], 16)
        self.assertLessEqual(geometry["mean_uniform_random_retention"], 0.5)
