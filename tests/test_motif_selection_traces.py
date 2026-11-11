import unittest
import torch
from src.experiments.motif_check import fixture, challenging_fixture, selection_trace, frozen_selection_draws
from tests import test_learnable_selection as fixtures


class SelectionTraceTests(unittest.TestCase):
    def test_motif_bookkeeping_uses_local_fragment_ids_in_all_molecules(self):
        smiles, _ = challenging_fixture()
        batch, targets = fixture(smiles[:4], 0, 51)
        details = dict(selected_atom_ids=torch.arange(batch.x.size(0)),
                       selected_fragment_mask=torch.zeros(0, dtype=torch.bool))
        trace = selection_trace(batch, targets, details)
        self.assertEqual([r["motif_retained"] for r in trace["per_molecule"]],
                         [True, None, True, None])
        details["selected_atom_ids"] = torch.empty(0, dtype=torch.long)
        trace = selection_trace(batch, targets, details)
        self.assertEqual(trace["motif_retention"], 0.)

    def test_frozen_draws_preserve_model_rng_and_deterministic_output(self):
        batch, model = fixtures.LearnableSelectionTests().fixture()
        _, targets = fixture(["CCOC(=O)C", "CCN(CC)CC"], 0, 23)
        model.eval()
        with torch.no_grad():
            before, details = model(batch, return_intermediates=True)
        buffers = {k: v.clone() for k, v in model.state_dict().items()}
        rng = torch.random.get_rng_state()
        first = frozen_selection_draws(model, batch, targets, 71, draws=3)
        second = frozen_selection_draws(model, batch, targets, 71, draws=3)
        self.assertEqual(first, second)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertFalse(model.training)
        self.assertEqual(model.ranking_mode, "learned")
        with torch.no_grad():
            torch.testing.assert_close(before, model(batch), rtol=0, atol=0)
        for key, tensor in model.state_dict().items():
            self.assertTrue(torch.equal(tensor, buffers[key]))
        trace = selection_trace(batch, targets, details)
        self.assertEqual(sum(len(r["atom_ids"]) for r in trace["per_molecule"]),
                         details["selected_atom_ids"].numel())
