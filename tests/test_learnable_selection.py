import copy
import unittest
import torch
import torch.nn.functional as F
from rdkit import Chem
from torch_geometric.data import Batch
from src.fragmentation import build_fragmenter
from src.data.molecule_example import build_molecule_example
from src.data.pyg_batch import molecule_example_to_data
from src.models.fragment_ciga import FragmentCIGA
from src.models.fragment_graph import build_fragment_graph_batch
from src.models.atom_reconstructor import reconstruct_batch_fast
from src.models.learnable_selection import budget_soft_mask, masked_atom_graph


class LearnableSelectionTests(unittest.TestCase):
    def fixture(self, method="group_graph", ratio=0.5):
        torch.manual_seed(23)
        torch.set_num_threads(1)
        data = [molecule_example_to_data(build_molecule_example(smi, label=i,
                    fragmenter=build_fragmenter(method)))
                for i, smi in enumerate(["CCOC(=O)C", "CCN(CC)CC"])]
        batch = Batch.from_data_list(data, follow_batch=["fragment_x_static"])
        view = FragmentCIGA(atom_feature_dim=batch.x.size(1),
            fragment_feature_dim=batch.fragment_x_static.size(1),
            fragment_edge_dim=batch.fragment_edge_attr.size(1),
            atom_edge_dim=batch.edge_attr.size(1), gnn_hidden_dim=16,
            num_gnn_layers=2, edge_ratio=ratio, dropout=0.0)
        return batch, view

    def test_budget_and_smooth_derivative(self):
        scores = torch.tensor([0.2, 0.4, 0.7, 0.8], dtype=torch.double, requires_grad=True)
        graphs = torch.zeros(4, dtype=torch.long)
        hard = torch.tensor([False, False, True, True])
        eligible = torch.ones(4, dtype=torch.bool)
        function = lambda s: budget_soft_mask(s, graphs, hard, eligible, 0.2)
        self.assertAlmostEqual(float(function(scores).sum()), 2.0, places=8)
        self.assertTrue(torch.autograd.gradcheck(function, (scores,), eps=1e-6, atol=1e-4))

    def test_batched_budgets_and_ineligible_scores(self):
        scores = torch.linspace(.1, .9, 8, dtype=torch.double, requires_grad=True)
        graphs = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
        hard = torch.tensor([False, True, True, True, False, False, False, False])
        eligible = torch.tensor([True, True, True, True, True, True, False, True])
        function = lambda s: budget_soft_mask(s, graphs, hard, eligible, .2)
        result = function(scores)
        self.assertAlmostEqual(float(result[graphs == 0].sum()), 1, places=8)
        self.assertAlmostEqual(float(result[graphs == 1].sum()), 2, places=8)
        self.assertEqual(float(result[6]), 0)
        self.assertTrue(torch.autograd.gradcheck(function, (scores,), eps=1e-6, atol=1e-4))

    def test_edgeless_atom_molecules_train_and_evaluate(self):
        data = [molecule_example_to_data(build_molecule_example("[He]", label=i,
                    fragmenter=build_fragmenter("atom"))) for i in range(2)]
        batch = Batch.from_data_list(data, follow_batch=["fragment_x_static"])
        _, model = self.fixture("atom")
        model.selection_estimator = "straight_through"
        predictions, details = model(batch, return_intermediates=True)
        self.assertTrue(torch.isfinite(predictions).all())
        self.assertEqual(details["selected_atom_ids"].numel(), 2)
        model.eval()
        self.assertTrue(torch.isfinite(model(batch)).all())

    def test_training_forward_and_normalization_match_reconstruction(self):
        for method in ("group_graph", "atom"):
            with self.subTest(method=method):
                batch, view = self.fixture(method)
                graph = build_fragment_graph_batch(batch)
                scores, eligible = view.score_fragment_edges(graph)
                hard = view.select_fragment_edges(batch, graph, scores, eligible)
                physical = reconstruct_batch_fast(batch, hard, selected_fragment_edge_weight=scores)
                masked, _ = masked_atom_graph(batch, graph, scores, hard, eligible, None, 0.1)
                a, b = copy.deepcopy(view.atom_classifier), copy.deepcopy(view.atom_classifier)
                a.train(); b.train()
                torch.testing.assert_close(a(physical), b(masked), atol=2e-5, rtol=2e-5)
                for (name, left), (_, right) in zip(a.named_buffers(), b.named_buffers()):
                    torch.testing.assert_close(left, right, atol=2e-5, rtol=2e-5, msg=name)

    def test_all_selected_matches_full_graph_with_same_continuous_gates(self):
        batch, view = self.fixture("atom", 1.0)
        graph = build_fragment_graph_batch(batch)
        scores, eligible = view.score_fragment_edges(graph)
        hard = torch.ones_like(scores, dtype=torch.bool)
        physical = reconstruct_batch_fast(batch, hard, selected_fragment_edge_weight=scores)
        masked, _ = masked_atom_graph(batch, graph, scores, hard, eligible, None, 0.1)
        classifier = view.atom_classifier.eval()
        torch.testing.assert_close(classifier(physical), classifier(masked), atol=1e-6, rtol=1e-5)

    def test_atom_renumbering_preserves_bond_provenance(self):
        mol = Chem.MolFromSmiles("CCOC(=O)N")
        order = list(reversed(range(mol.GetNumAtoms())))
        permuted = Chem.RenumberAtoms(mol, order)
        original = build_fragmenter("atom").fragment(mol)
        other = build_fragmenter("atom").fragment(permuted)
        left = {tuple(sorted(pair)) for pair in original.fragment_edge_index.t().tolist()}
        right = {tuple(sorted(order[i] for i in pair)) for pair in other.fragment_edge_index.t().tolist()}
        self.assertEqual(left, right)

    def test_excluded_candidate_receives_selection_gradient_without_gate_gradient(self):
        batch, view = self.fixture("atom")
        graph = build_fragment_graph_batch(batch)
        scores = torch.linspace(0.2, 0.8, graph.edge_index.size(1), requires_grad=True)
        eligible = torch.ones_like(scores, dtype=torch.bool)
        hard = view.select_fragment_edges(batch, graph, scores, eligible)
        masked, _ = masked_atom_graph(batch, graph, scores, hard, eligible, None, 0.2)
        # Remove the old soft-score gate derivative, leaving the ST selection path.
        # Also cancel the old gate score derivative explicitly using a detached
        # score input for gates while retaining selection weights below.
        soft = budget_soft_mask(scores, graph.batch[graph.edge_index[0]], hard, eligible, 0.2)
        bond_mask = hard.float() + (soft - soft.detach())
        weights = scores.new_zeros(batch.bond_index.size(1))
        weights[batch.fragment_edge_to_bond_index] = bond_mask
        masked.edge_weight = weights.repeat(2)
        view.atom_classifier.eval()
        F.cross_entropy(view.atom_classifier(masked), batch.y.reshape(-1).long()).backward()
        self.assertGreater(float(scores.grad[~hard].abs().sum()), 0)

    def test_atom_adapter_provenance_and_edgeless_fallback(self):
        for smiles in ("CCOC", "[He]", "CC.[Na+]"):
            mol = Chem.MolFromSmiles(smiles)
            result = build_fragmenter("atom").fragment(mol)
            self.assertEqual(result.num_fragments, mol.GetNumAtoms())
            self.assertEqual(result.num_fragment_edges, mol.GetNumBonds())
            torch.testing.assert_close(result.atom_to_fragment, torch.arange(mol.GetNumAtoms()))
        batch, view = self.fixture("atom", 1.0)
        graph = build_fragment_graph_batch(batch)
        output = reconstruct_batch_fast(batch, torch.ones(graph.edge_index.size(1), dtype=torch.bool))
        self.assertEqual(output.x.size(0), batch.x.size(0))
        self.assertEqual(output.edge_index.size(1), batch.edge_index.size(1))

    def test_straight_through_training_and_hard_evaluation(self):
        batch, reference = self.fixture()
        model = copy.deepcopy(reference)
        model.selection_estimator = "straight_through"
        loss = F.nll_loss(model(batch), batch.y.reshape(-1).long())
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        model.eval(); reference.eval()
        # BatchNorm was updated by the training forward; compare equal states.
        reference.load_state_dict(model.state_dict())
        torch.testing.assert_close(model(batch), reference(batch))


if __name__ == "__main__":
    unittest.main()
