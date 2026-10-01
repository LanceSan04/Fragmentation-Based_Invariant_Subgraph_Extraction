"""FragmentCIGA: single-view trainable model for fragment-based invariant subgraph learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.explain.algorithm.utils import set_masks, clear_masks

from src.models.fragment_masker import FragmentMasker, segment_topk_mask
from src.models.gnn_encoders import SimpleGNN
from src.models.fragment_graph import build_fragment_graph_batch, get_fragment_view_value
from src.models.atom_budget_selection import atom_budget_mask
from src.models.fragment_edge_sampler import FragmentEdgeSampler
from src.models.random_fragment_selection import random_edge_ranks
from src.models.atom_reconstructor import reconstruct_batch, reconstruct_batch_fast
from src.data.atom_graph import ATOM_FDIM, BOND_FDIM


def topk_mask_over_eligible(scores, edge_batch, ratio, eligible_mask):
    """Select top-k edges per graph, forbidding sampler-dropped edges."""
    return segment_topk_mask(scores, edge_batch, ratio, eligible_mask=eligible_mask)


def restore_full_edge_outputs(
    num_full_edges,
    sampled_full_edge_ids,
    sampled_bool_mask,
    sampled_edge_scores,
):
    """Map sampled-graph scores/masks back to full fragment-edge order."""
    device = sampled_edge_scores.device
    full_mask = torch.zeros(num_full_edges, dtype=torch.bool, device=device)
    full_scores = torch.zeros(num_full_edges, dtype=sampled_edge_scores.dtype, device=device)
    if sampled_full_edge_ids.numel() == 0:
        return full_mask, full_scores

    sampled_full_edge_ids = sampled_full_edge_ids.to(device)
    full_mask[sampled_full_edge_ids] = sampled_bool_mask
    full_scores[sampled_full_edge_ids] = sampled_edge_scores
    return full_mask, full_scores


class AtomClassifier(nn.Module):
    """GIN(E) classifier for atom-level invariant subgraphs.

    Uses a GINE backbone (``aggr="add"``) to mirror SuGAr/CIGA's DrugOOD
    classifier. GIN's summed messages compose cleanly with the ``set_masks``
    edge gate below — ``mlp((1+eps)x + sum(gate_ij * msg_ij))`` — whereas an
    attention backbone computes its softmax coefficients independently of the
    gate, blunting the very signal that trains the fragment scorer.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        num_layers=3,
        dropout=0.0,
        num_classes=2,
        edge_dim=BOND_FDIM,
    ):
        super().__init__()
        self.gnn = SimpleGNN(input_dim, hidden_dim, num_layers, dropout, edge_dim=edge_dim)
        self.pred_head = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, num_classes),
        )

    def forward(self, data, return_embeddings=False):
        edge_weight = getattr(data, "edge_weight", None)
        # CIGA-style differentiable subgraph gating: use the fragment-edge
        # attention (carried on ``edge_weight``) as a multiplicative mask on every
        # message in every GNN layer, mirroring SuGAr's ``set_masks`` mechanism.
        # This is the path that lets the classification loss train the fragment
        # scorer; scaling ``edge_attr`` instead only weakly perturbs the conv and
        # cannot suppress an edge. ``apply_sigmoid=False`` since the scores are
        # already in [0, 1]. GINE adds no self-loops, so the mask aligns exactly
        # with ``edge_index``; the ``(1+eps)x`` self term is ungated (full
        # strength), matching SuGAr's GIN. Skip gating when the subgraph has no
        # edges.
        gate = edge_weight is not None and data.edge_index.numel() > 0
        if gate:
            set_masks(self.gnn, edge_weight, data.edge_index, apply_sigmoid=False)
        try:
            _, h_graph = self.gnn(data.x, data.edge_index, data.batch, edge_attr=data.edge_attr)
        finally:
            if gate:
                clear_masks(self.gnn)
        logits = self.pred_head(h_graph)
        if return_embeddings:
            return logits, h_graph
        return logits


class FragmentCIGA(nn.Module):
    """Single-view trainable model: fragment masking → reconstruction → classification.

    One FragmentCIGA instance is one "view" in the SuGAR ensemble. During training
    each view operates independently — its masker scores fragment edges, the top-k
    selection defines an invariant subgraph specific to this view, and its classifier
    is trained on that reconstructed atom-level subgraph. Multiple trained views are
    then wrapped in SuGAREnsemble for inference-time aggregation.
    """

    def __init__(
        self,
        atom_feature_dim=ATOM_FDIM,
        fragment_feature_dim=None,
        num_classes=2,
        num_gnn_layers=3,
        gnn_hidden_dim=128,
        edge_ratio=0.5,
        dropout=0.0,
        fragment_edge_dim=None,
        atom_edge_dim=BOND_FDIM,
        fragment_sample_ratio=1.0,
        sampler_seed=None,
        fragmentation_method=None,
        erm_baseline=False,
        selection_mode="learned",
        scoring_mode="ordered",
        selection_budget="edges",
        atom_retention_ratio=0.65,
        ranking_mode="learned",
    ):
        super().__init__()
        if fragment_feature_dim is None:
            raise ValueError("fragment_feature_dim must be provided")

        # ERM anchor: classify the whole atom graph, with no fragment scoring,
        # no selection and no edge gate. This is the reference the fragment
        # pipeline has to be priced against, since published ERM numbers for
        # DrugOOD EC50-Assay disagree by ~8 AUC points across papers.
        self.erm_baseline = erm_baseline
        if selection_mode not in ("learned", "random"):
            raise ValueError("selection_mode must be learned or random")
        if erm_baseline and selection_mode != "learned":
            raise ValueError("ERM has no fragment selection mode")
        if selection_budget not in ("edges", "atoms") or ranking_mode not in ("learned", "random"):
            raise ValueError("Invalid selection budget or ranking mode")
        if not 0 < atom_retention_ratio <= 1:
            raise ValueError("atom_retention_ratio must be in (0, 1]")
        if (erm_baseline or selection_mode == "random") and (
            selection_budget != "edges" or ranking_mode != "learned"
        ):
            raise ValueError("New budget/ranking settings require learned fragment selection mode")
        self.selection_budget = selection_budget
        self.atom_retention_ratio = float(atom_retention_ratio)
        self.ranking_mode = ranking_mode
        self._ranking_call_count = 0
        self.selection_mode = selection_mode
        self.random_selection_seed = 0 if sampler_seed is None else int(sampler_seed)

        self.fragment_edge_sampler = FragmentEdgeSampler(
            keep_ratio=fragment_sample_ratio,
            seed=sampler_seed,
        )
        self.fragmentation_method = fragmentation_method
        self.fragment_masker = FragmentMasker(
            input_dim=fragment_feature_dim,
            hidden_dim=gnn_hidden_dim,
            num_layers=num_gnn_layers,
            dropout=dropout,
            ratio=edge_ratio,
            edge_dim=fragment_edge_dim,
            scoring_mode=scoring_mode,
        )
        self.atom_classifier = AtomClassifier(
            input_dim=atom_feature_dim,
            hidden_dim=gnn_hidden_dim,
            num_layers=num_gnn_layers,
            dropout=dropout,
            num_classes=num_classes,
            edge_dim=atom_edge_dim,
        )

    def _forward_erm(self, batch_data, return_intermediates=False):
        """ERM anchor: classify the full atom graph, ungated.

        No fragment graph is built, the masker is never called, and no edge mask
        is set, so the classifier sees every bond at weight 1.0. Auxiliary losses
        that need fragment scores are reported as absent (``fragment_scores`` is
        None) and the trainer skips the diversity term for such views.
        """
        atom_batch = Data(
            x=batch_data.x,
            edge_index=batch_data.edge_index,
            edge_attr=batch_data.edge_attr,
            batch=batch_data.batch,
        )
        logits, h_graph = self.atom_classifier(atom_batch, return_embeddings=True)
        predictions = F.log_softmax(logits, dim=-1)

        if return_intermediates:
            return predictions, {
                "embeddings": h_graph,
                "fragment_scores": None,
                "fragment_edge_batch": None,
                "fragment_edge_eligible": None,
                "fragmentation_method": self.fragmentation_method,
                "logits": logits,
            }
        return predictions

    def score_fragment_edges(self, fragment_batch):
        """Return full-edge ranks and eligibility for learned or random selection."""
        sampled, eligible = self.fragment_edge_sampler(fragment_batch)
        if self.selection_mode == "random":
            sampled_scores = random_edge_ranks(sampled, self.random_selection_seed, self.training)
        else:
            sampled_scores, _ = self.fragment_masker(sampled)
        _, full_scores = restore_full_edge_outputs(
            fragment_batch.edge_index.size(1),
            sampled.full_fragment_edge_id,
            torch.zeros_like(sampled_scores, dtype=torch.bool),
            sampled_scores,
        )
        return full_scores, eligible

    def select_fragment_edges(self, batch_data, fragment_batch, scores, eligible):
        """One selection path shared by training, inference and extraction."""
        ranks = scores.detach()
        if self.ranking_mode == "random":
            if self.training:
                generator = torch.Generator(device=scores.device)
                generator.manual_seed(
                    self.random_selection_seed + 1000003 + self._ranking_call_count
                )
                self._ranking_call_count += 1
                ranks = torch.rand(scores.shape, device=scores.device, generator=generator)
            else:
                ranks = random_edge_ranks(fragment_batch, self.random_selection_seed, False)
        edge_batch = fragment_batch.batch[fragment_batch.edge_index[0]]
        if self.selection_budget == "edges":
            return topk_mask_over_eligible(ranks, edge_batch, self.fragment_masker.ratio, eligible)
        atom_to_fragment = get_fragment_view_value(
            batch_data, "atom_to_fragment", self.fragmentation_method
        )
        sizes = torch.bincount(atom_to_fragment, minlength=fragment_batch.x.size(0))
        atom_counts = torch.bincount(batch_data.batch, minlength=batch_data.num_graphs)
        return atom_budget_mask(
            ranks,
            fragment_batch.edge_index,
            fragment_batch.batch,
            sizes,
            atom_counts,
            eligible,
            self.atom_retention_ratio,
        )

    def forward(self, batch_data, return_intermediates=False, fragment_batch=None):
        """Per-view forward pass: mask → reconstruct → classify.

        Args:
            batch_data: Batched MoleculeData
            return_intermediates: If True, also return embeddings, fragment_scores,
                and logits (needed by SuGARTrainer's auxiliary losses).
            fragment_batch: Optional prebuilt batched fragment graph. The fragment
                graph is a pure function of ``batch_data`` and identical across all
                views, so the trainer builds it once per batch and passes it to
                every view.

        Returns:
            predictions: [batch_size, num_classes] log-probabilities
            intermediates: (only when return_intermediates=True) dict with
                'embeddings', 'fragment_scores', 'logits'
        """
        device = batch_data.x.device

        if self.erm_baseline:
            return self._forward_erm(batch_data, return_intermediates)

        if fragment_batch is None:
            fragment_batch = build_fragment_graph_batch(
                batch_data,
                method=self.fragmentation_method,
            ).to(device)

        edge_scores, eligible_mask = self.score_fragment_edges(fragment_batch)

        edge_batch = fragment_batch.batch[fragment_batch.edge_index[0]]
        bool_mask = self.select_fragment_edges(
            batch_data, fragment_batch, edge_scores, eligible_mask
        )

        reconstruction_weights = (
            bool_mask.to(edge_scores.dtype) if self.selection_mode == "random" else edge_scores
        )
        atom_batch = reconstruct_batch_fast(
            batch_data,
            bool_mask,
            selected_fragment_edge_weight=reconstruction_weights,
            fragmentation_method=self.fragmentation_method,
        )

        logits, h_graph = self.atom_classifier(atom_batch, return_embeddings=True)
        predictions = F.log_softmax(logits, dim=-1)

        if return_intermediates:
            return predictions, {
                "embeddings": h_graph,
                "selected_fragment_mask": bool_mask,
                "selected_atom_ids": atom_batch.selected_original_atom_indices,
                "selected_bond_ids": atom_batch.selected_original_bond_indices,
                "fragment_scores": edge_scores if self.selection_mode == "learned" else None,
                "fragment_edge_batch": edge_batch,
                "fragment_edge_eligible": eligible_mask.to(
                    device=edge_scores.device, dtype=torch.bool
                ),
                "fragmentation_method": self.fragmentation_method,
                "logits": logits,
            }
        return predictions

    def get_invariant_subgraph(self, batch_data):
        """Return the reconstructed atom subgraph for this view (for analysis).

        Args:
            batch_data: Batched MoleculeData

        Returns:
            Batched reconstructed atom-level PyG Data
        """
        if self.erm_baseline:
            raise ValueError(
                "ERM uses the full atom graph; fragment rationale extraction is not applicable."
            )
        device = batch_data.x.device
        fragment_batch = build_fragment_graph_batch(
            batch_data,
            method=self.fragmentation_method,
        ).to(device)
        edge_scores, eligible = self.score_fragment_edges(fragment_batch)
        edge_batch = fragment_batch.batch[fragment_batch.edge_index[0]]
        full_mask = self.select_fragment_edges(batch_data, fragment_batch, edge_scores, eligible)
        return reconstruct_batch(
            batch_data,
            full_mask,
            selected_fragment_edge_weight=(
                full_mask.to(edge_scores.dtype) if self.selection_mode == "random" else edge_scores
            ),
            fragmentation_method=self.fragmentation_method,
        )
