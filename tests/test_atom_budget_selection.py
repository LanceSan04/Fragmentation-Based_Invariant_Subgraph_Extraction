"""Atom-budget selector checked against an exhaustive prefix oracle."""
import unittest
import torch
from src.models.atom_budget_selection import atom_budget_mask
from tests import test_random_selection_control as fixture_module
from src.models.sugar_ensemble import SuGAREnsemble


class AtomBudgetTests(unittest.TestCase):
    def test_prefix_oracle(self):
        generator = torch.Generator().manual_seed(102)
        for trial in range(100):
            sizes = torch.randint(1, 15, (12,), generator=generator)
            groups = torch.arange(3).repeat_interleave(4)
            edges = torch.tensor([[0,1,2,0,4,5,6,4], [1,2,3,3,5,6,7,6]])
            ranks = torch.randint(0, 4, (8,), generator=generator).float()
            eligible = torch.rand(8, generator=generator) > .3
            counts = sizes.reshape(3,4).sum(1)
            actual = atom_budget_mask(ranks, edges, groups, sizes, counts, eligible, .65)
            expected = torch.zeros(8, dtype=torch.bool)
            for graph in range(3):
                ids = [i for i in range(8) if eligible[i] and groups[edges[0,i]] == graph]
                ids.sort(key=lambda i: (-float(ranks[i]), i))
                candidates = []
                for k in range(1, len(ids)+1):
                    atoms = int(sizes[torch.unique(edges[:,ids[:k]])].sum())
                    candidates.append((abs(atoms-int(torch.ceil(counts[graph].double()*.65))), atoms, k))
                if candidates:
                    k = min(candidates)[2]
                    expected[ids[:k]] = True
            self.assertTrue(torch.equal(actual, expected), str(trial))

    def test_empty_edges_and_invalid_ratio(self):
        args = (torch.empty(0), torch.empty((2,0),dtype=torch.long), torch.tensor([0]),
                torch.tensor([6]), torch.tensor([6]), torch.empty(0,dtype=torch.bool))
        self.assertEqual(atom_budget_mask(*args, .65).numel(), 0)
        for ratio in (0, -1, 1.1, float('nan')):
            with self.assertRaises(ValueError): atom_budget_mask(*args, ratio)

    def test_default_selector_and_strict_state_compatibility(self):
        fixture = fixture_module.RandomSelectionControlTests()
        fixture.setUpClass()
        batch = fixture.batch([0,1,2])
        view = fixture.view(batch).eval()
        from src.models.fragment_graph import build_fragment_graph_batch
        from src.models.fragment_ciga import topk_mask_over_eligible
        fragment = build_fragment_graph_batch(batch)
        scores, eligible = view.score_fragment_edges(fragment)
        expected = topk_mask_over_eligible(scores, fragment.batch[fragment.edge_index[0]], .5, eligible)
        self.assertTrue(torch.equal(expected, view.select_fragment_edges(batch,fragment,scores,eligible)))
        other = fixture.view(batch)
        other.selection_budget = 'atoms'
        other.load_state_dict(view.state_dict(), strict=True)
        self.assertEqual(list(other.state_dict()), list(view.state_dict()))

    def test_model_paths_and_gradients(self):
        fixture = fixture_module.RandomSelectionControlTests()
        fixture.setUpClass()
        batch = fixture.batch([0,1,2])
        view = fixture.view(batch)
        view.selection_mode = 'learned'
        view.selection_budget = 'atoms'
        view.ranking_mode = 'random'
        view.eval()
        predictions, trace = view(batch, True)
        subgraph = view.get_invariant_subgraph(batch)
        self.assertEqual(subgraph.x.size(0), trace['selected_atom_ids'].numel())
        self.assertIsNotNone(trace['fragment_scores'])
        predictions[:,1].sum().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in view.fragment_masker.parameters()))
        with self.assertRaisesRegex(ValueError, 'independent'):
            SuGAREnsemble([view], inference_policy='shared')
        reversed_batch = fixture.batch([2,0,1])
        with torch.no_grad():
            reversed_prediction = view(reversed_batch)
        self.assertTrue(torch.allclose(predictions, reversed_prediction[[1,2,0]], atol=1e-6))
        # New control's random priorities must not consume dropout's global RNG.
        from src.models.fragment_graph import build_fragment_graph_batch
        fragment = build_fragment_graph_batch(batch)
        view.train()
        state = torch.random.get_rng_state()
        view.select_fragment_edges(batch, fragment, torch.ones(fragment.edge_index.size(1)), torch.ones(fragment.edge_index.size(1),dtype=torch.bool))
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))

if __name__ == '__main__': unittest.main()
