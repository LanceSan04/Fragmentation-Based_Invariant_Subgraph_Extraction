"""Training pipeline for SuGAR multi-view FragmentCIGA learning."""

import csv
from copy import deepcopy
from pathlib import Path

import torch
from torch.optim import Adam
from tqdm import tqdm

from src.losses.prediction import prediction_loss
from src.losses.supervised_contrastive import supervised_contrastive_loss
from src.losses.diversity import pairwise_diversity_loss
from src.losses.irm import invariant_risk_loss
from src.models.fragment_graph import build_fragment_graph_batch
from src.models.sugar_ensemble import SuGAREnsemble
from src.training.selection_metrics import trailing_mean

try:
    from sklearn.metrics import roc_auc_score as _sklearn_roc_auc
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


class SuGARTrainer:
    """Trainer for a list of FragmentCIGA views forming the SuGAR ensemble.

    Each FragmentCIGA view learns its own fragment-edge scorer and atom-level
    classifier. After training, the trained views are wrapped in SuGAREnsemble
    for shared-subgraph aggregation and soft-vote inference.
    """

    def __init__(
        self,
        views,
        device="cpu",
        learning_rate=0.001,
        weight_decay=0.0,
        lambda_contrast=1.0,
        lambda_diversity=1.0,
        lambda_irm=0.0,
        num_envs=1,
        env_split_key="domain_id",
        env_bucket_strategy="value",
        contrast_temperature=0.5,
        contrast_sampling="mul",
        track_roc_auc=True,
        loss_log_path=None,
        aggregation_mode="mean",
        selection_ma_window=1,
        selection_weight_avg=False,
        diversity_delay_epochs=0,
        diversity_ramp_epochs=0,
        early_stopping_min_epochs=0,
        inference_policy="shared",
        contrast_scope="pooled",
    ):
        """
        Args:
            views: List of FragmentCIGA models, one per SuGAr view.
            device: Device string ('cpu' or 'cuda').
            learning_rate: Adam learning rate, applied per-view.
            weight_decay: Adam weight decay.
            lambda_contrast: Weight for supervised contrastive loss.
            lambda_diversity: Weight for pairwise diversity loss.
            lambda_irm: Weight for IRM penalty. Requires num_envs > 1.
            num_envs: Number of environment partitions for IRM.
            env_split_key: Data key used to derive the IRM environment split
                (checked in each example's ``metadata`` dict, falling back to
                its ``environment_or_group`` value, e.g. DrugOOD's domain_id).
                Unique key values are grouped into num_envs buckets as evenly
                as possible; examples sharing a key value always land in the
                same environment. Set to None/"" to use a random split
                instead.
            env_bucket_strategy: How env_split_key domains are ordered before
                being chunked into num_envs buckets. ``"value"`` sorts by the
                raw key value (arbitrary for a nuisance axis); ``"size"`` sorts
                domains by their per-domain mean molecule size (atom count) so
                environments span a size gradient — a classic molecular OOD
                nuisance axis that, unlike prevalence bucketing, never collapses
                an environment to a single class.
            contrast_temperature: Temperature for contrastive loss.
            contrast_sampling: Contrastive sampling mode ('mul', 'var', 'cnc').
            track_roc_auc: If True, use ROC-AUC where available.
            loss_log_path: Optional CSV path for per-epoch diagnostics.
            aggregation_mode: Score-aggregation mode ('mean' or 'max') for the
                SuGAREnsemble used to score validation. Must match the mode used
                for final test-time inference so best-epoch/early-stop selection
                is made on the same shared-subgraph path.
            selection_ma_window: Window (in epochs) for the trailing moving
                average of the validation selection metric used to pick the
                best epoch. ``1`` reproduces raw single-best-epoch selection;
                a larger window (e.g. 5, matching the grid-search objective)
                uses only complete finite windows. Patience starts after the
                first eligible window; this reduces sensitivity to single peaks
                but does not remove selection bias.
            selection_weight_avg: If True, restore an SWA-style average of the
                top ``selection_ma_window`` snapshots (ranked by smoothed
                metric) instead of the single best snapshot. Off by default;
                rejected for BatchNorm models pending proper recalibration.
        """
        self.views = [v.to(device) for v in views]
        self.device = device
        self.optimizers = [
            Adam(v.parameters(), lr=learning_rate, weight_decay=weight_decay)
            for v in self.views
        ]
        self.lambda_contrast = lambda_contrast
        self.lambda_diversity = lambda_diversity
        self.lambda_irm = lambda_irm
        self.num_envs = num_envs
        self.env_split_key = env_split_key or None
        self.env_bucket_strategy = env_bucket_strategy
        self._env_split_key_warned = False
        self.contrast_temperature = contrast_temperature
        self.contrast_sampling = contrast_sampling
        if contrast_scope not in ('pooled','per_view'):
            raise ValueError('contrast_scope must be pooled or per_view')
        self.contrast_scope = contrast_scope
        self.track_roc_auc = track_roc_auc and _SKLEARN_AVAILABLE
        self.diversity_delay_epochs = int(diversity_delay_epochs)
        self.diversity_ramp_epochs = int(diversity_ramp_epochs)
        self.early_stopping_min_epochs = int(early_stopping_min_epochs)
        if min(self.diversity_delay_epochs, self.diversity_ramp_epochs, self.early_stopping_min_epochs) < 0:
            raise ValueError("Schedule durations must be nonnegative")
        self.effective_diversity_weight = self.lambda_diversity
        self.loss_log_path = Path(loss_log_path) if loss_log_path else None
        # Validation is scored through the same shared-subgraph aggregation +
        # soft-vote path used at final test time. The ensemble holds references
        # to the same view objects, so weight updates are reflected without any
        # re-wrapping.
        self.aggregation_mode = aggregation_mode
        self.selection_ma_window = int(selection_ma_window)
        if self.selection_ma_window < 1:
            raise ValueError("selection_ma_window must be positive")
        self.selection_weight_avg = bool(selection_weight_avg)
        if self.selection_weight_avg and any(
            isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
            for view in self.views for module in view.modules()
        ):
            raise ValueError(
                "selection_weight_avg is unsupported for BatchNorm models until "
                "normalization recalibration is implemented; set it to false."
            )
        self.ensemble = SuGAREnsemble(self.views, aggregation_mode=aggregation_mode, inference_policy=inference_policy)
        if track_roc_auc and not _SKLEARN_AVAILABLE:
            print("Warning: scikit-learn not found; ROC-AUC tracking disabled.")

    @staticmethod
    def loss_component_keys():
        """Return epoch-level loss component names tracked for diagnostics."""
        return (
            "prediction_loss",
            "contrastive_loss",
            "contrastive_loss_weighted",
            "diversity_loss",
            "diversity_loss_weighted",
            "irm_loss",
            "irm_loss_weighted",
        )

    def uses_irm(self):
        """Return whether the IRM penalty should participate in training."""
        return self.lambda_irm > 0 and self.num_envs > 1

    def _empty_components(self, device):
        zero = torch.zeros((), device=device)
        return {key: zero for key in self.loss_component_keys()}

    def _compute_components(self, per_view, labels, batch_env=None):
        """Compute total loss plus named components from per-view outputs."""
        components = self._empty_components(labels.device)

        # Reduction convention: per-view terms are SUMMED over views, matching
        # SuGAr, which backpropagates the sum of its base models' losses. Any
        # per-view term that is averaged instead has its lambda silently divided
        # by num_views relative to the prediction term.
        per_view_losses = [
            prediction_loss(pv["predictions"], labels)
            for pv in per_view
        ]
        components["prediction_loss"] = torch.stack(per_view_losses).sum()

        if self.lambda_contrast > 0:
            if self.contrast_scope == 'per_view':
                contrastive = torch.stack([
                    supervised_contrastive_loss(
                        pv['embeddings'], labels, temperature=self.contrast_temperature,
                        sampling=self.contrast_sampling,
                        y_pred=pv['logits'].detach().argmax(-1) if self.contrast_sampling.lower() == 'cnc' else None,
                    ) for pv in per_view
                ]).sum()
            else:
                contrastive = supervised_contrastive_loss(
                    torch.cat([pv['embeddings'] for pv in per_view]),
                    labels.repeat(len(per_view)),
                    temperature=self.contrast_temperature, sampling=self.contrast_sampling,
                    y_pred=torch.cat([pv['logits'].detach().argmax(-1) for pv in per_view])
                    if self.contrast_sampling.lower() == 'cnc' else None,
                )
            components['contrastive_loss'] = contrastive
            components["contrastive_loss_weighted"] = self.lambda_contrast * contrastive

        if self.lambda_diversity > 0 and len(self.views) > 1:
            diversity_terms = []
            grouped = {}
            for pv in per_view:
                # ERM views produce no fragment scores, so there is nothing to
                # diversify; they cannot participate in this term.
                if pv.get("fragment_scores") is None:
                    continue
                grouped.setdefault(pv.get("fragmentation_method"), []).append(pv)
            for method_views in grouped.values():
                if len(method_views) < 2:
                    continue
                eligible_masks = [pv.get("fragment_edge_eligible") for pv in method_views]
                if any(mask is None for mask in eligible_masks):
                    eligible_masks = None
                diversity_terms.append(
                    pairwise_diversity_loss(
                        [pv["fragment_scores"] for pv in method_views],
                        edge_batch=method_views[0].get("fragment_edge_batch"),
                        eligible_masks=eligible_masks,
                        num_graphs=labels.size(0),
                    )
                )
            if diversity_terms:
                # Averaged across fragmentation methods, not views: each method
                # is a separate fragment-edge space whose scores are not
                # comparable, and pairwise_diversity_loss already aggregates
                # over the view pairs within a method.
                diversity = torch.stack(diversity_terms).mean()
            else:
                diversity = labels.new_zeros(())
            components["diversity_loss"] = diversity
            components["diversity_loss_weighted"] = self.effective_diversity_weight * diversity

        if batch_env is not None:
            irm_terms = [
                invariant_risk_loss(pv["logits"], labels, batch_env)
                for pv in per_view
            ]
            if irm_terms:
                irm = torch.stack(irm_terms).sum()
                components["irm_loss"] = irm
                components["irm_loss_weighted"] = self.lambda_irm * irm

        total = (
            components["prediction_loss"]
            + components["contrastive_loss_weighted"]
            + components["diversity_loss_weighted"]
            + components["irm_loss_weighted"]
        )
        return total, components, per_view_losses

    def _average_components(self, component_totals, num_batches):
        return {
            key: value / num_batches if num_batches else 0.0
            for key, value in component_totals.items()
        }

    def _roc_auc_from_chunks(self, prob_chunks, label_chunks):
        if not self.track_roc_auc or not prob_chunks:
            return None
        all_probs = torch.cat(prob_chunks, dim=0).numpy()
        all_labels = torch.cat(label_chunks, dim=0).numpy()
        num_classes = all_probs.shape[1]
        try:
            if num_classes == 2:
                return _sklearn_roc_auc(all_labels, all_probs[:, 1])
            return _sklearn_roc_auc(
                all_labels,
                all_probs,
                multi_class="ovr",
                average="macro",
            )
        except ValueError:
            return None

    def _metric_for_selection(self, accuracy, roc_auc):
        if self.track_roc_auc and roc_auc is not None:
            return roc_auc, "auc"
        return accuracy, "acc"

    def _make_env_idx(self, loader):
        if not self.uses_irm():
            return None
        if self.env_split_key:
            env_idx = self._env_idx_from_key(loader.dataset)
            if env_idx is not None:
                return env_idx
        return torch.randint(0, self.num_envs, (len(loader.dataset),))

    def prepare_irm_environments(self, loaders):
        """Resolve and attach IRM env ids while metadata is still available.

        Call this before any batch-field stripping so an ``env_split_key`` that
        lives only in an example's ``metadata`` can still be read. Once the ids
        are written onto the ``Data`` objects, stripping ``metadata`` is safe.
        No-op unless IRM is active.
        """
        if not self.uses_irm():
            return
        for loader in loaders:
            if loader is not None:
                self._assign_env_ids(loader)

    def _assign_env_ids(self, loader):
        """Attach a per-example ``env_id`` tensor to each Data object in the loader.

        Environment ids are computed in dataset order but written onto the
        individual ``Data`` records, so they batch and shuffle together with
        their molecules (via ``batch.env_id``). This is robust to the train
        loader's ``shuffle=True``, unlike slicing a dataset-ordered ``env_idx``
        by ``step * batch_size``. If ids were already assigned (e.g. by
        ``prepare_irm_environments`` before metadata was stripped), they are
        reused rather than recomputed. Returns True if ids are present.
        """
        dataset = loader.dataset
        if len(dataset) and getattr(dataset[0], "env_id", None) is not None:
            return True
        env_idx = self._make_env_idx(loader)
        if env_idx is None:
            return False
        for i in range(len(dataset)):
            dataset[i].env_id = env_idx[i].view(1).long()
        return True

    def _env_idx_from_key(self, dataset):
        """Group examples into num_envs buckets keyed by env_split_key, or None if absent."""
        values = [self._extract_env_value(data, self.env_split_key) for data in dataset]
        if any(value is None for value in values):
            if not self._env_split_key_warned:
                print(
                    f"Warning: env_split_key {self.env_split_key!r} missing on some examples; "
                    "falling back to a random IRM environment split."
                )
                self._env_split_key_warned = True
            return None
        if self.env_bucket_strategy == "size":
            bucket_of_value = self._assign_env_buckets_by_size(
                dataset, values, self.num_envs
            )
        else:
            bucket_of_value = self._assign_env_buckets(values, self.num_envs)
        return torch.tensor([bucket_of_value[v] for v in values], dtype=torch.long)

    @staticmethod
    def _extract_env_value(data, key):
        """Look up ``key`` in a data example's metadata, or its environment_or_group value."""
        metadata = getattr(data, "metadata", None)
        if isinstance(metadata, dict) and key in metadata:
            return metadata[key]
        return getattr(data, "environment_or_group", None)

    @staticmethod
    def _chunk_into_buckets(ordered_unique_values, num_envs):
        """Chunk an ordered list of unique values into num_envs even buckets.

        Env 0 gets the first (smallest) values, the last env the largest. Bucket
        sizes differ by at most one (earlier envs absorb the remainder).
        """
        base_size, remainder = divmod(len(ordered_unique_values), num_envs)

        bucket_of_value = {}
        cursor = 0
        for env_id in range(num_envs):
            chunk_size = base_size + (1 if env_id < remainder else 0)
            for value in ordered_unique_values[cursor:cursor + chunk_size]:
                bucket_of_value[value] = env_id
            cursor += chunk_size
        return bucket_of_value

    @classmethod
    def _assign_env_buckets(cls, values, num_envs):
        """Map each unique value to one of num_envs buckets, sized as evenly as possible."""

        def sort_key(value):
            try:
                return (0, float(value))
            except (TypeError, ValueError):
                return (1, str(value))

        unique_values = sorted(set(values), key=sort_key)
        return cls._chunk_into_buckets(unique_values, num_envs)

    @classmethod
    def _assign_env_buckets_by_size(cls, dataset, values, num_envs):
        """Map each unique domain to a bucket ordered by mean molecule size.

        Accumulate the total and count of atom sizes per domain over the whole
        dataset, then order unique domains by mean size (size_sum / count),
        tie-breaking on the raw-value sort order for determinism, and chunk into
        num_envs buckets (env 0 = smallest-molecule domains, last env = largest).
        """
        size_sum = {}
        count = {}
        for data, value in zip(dataset, values):
            size_sum[value] = size_sum.get(value, 0.0) + cls._scalar_size(data)
            count[value] = count.get(value, 0) + 1

        def value_sort_key(value):
            try:
                return (0, float(value))
            except (TypeError, ValueError):
                return (1, str(value))

        value_rank = {
            value: rank
            for rank, value in enumerate(sorted(size_sum, key=value_sort_key))
        }

        def mean_size(value):
            n = count[value]
            return size_sum[value] / n if n else 0.0

        ordered = sorted(size_sum, key=lambda v: (mean_size(v), value_rank[v]))
        return cls._chunk_into_buckets(ordered, num_envs)

    @staticmethod
    def _scalar_size(data):
        """Atom count for a molecule Data object (0 if node features are absent)."""
        x = getattr(data, "x", None)
        if x is None:
            return 0
        return int(x.size(0))

    def _labels_from_batch(self, batch):
        labels = batch.y
        if labels.dim() == 2:
            labels = labels.squeeze(-1)
        return labels.long()

    def _forward_views(self, batch, use_aux):
        # The batched fragment graph is a pure function of the batch and method.
        # Build it once per fragmentation method and share it across views.
        fragment_batches = {}
        per_view = []
        for view in self.views:
            method = getattr(view, "fragmentation_method", None)
            if getattr(view, "erm_baseline", False):
                # ERM views never touch the fragment graph; don't build one.
                fragment_batch = None
            else:
                if method not in fragment_batches:
                    fragment_batches[method] = build_fragment_graph_batch(
                        batch,
                        method=method,
                    ).to(self.device)
                fragment_batch = fragment_batches[method]
            if use_aux:
                pred, interm = view(batch, return_intermediates=True, fragment_batch=fragment_batch)
                per_view.append({"predictions": pred, **interm})
            else:
                pred = view(batch, fragment_batch=fragment_batch)
                per_view.append({"predictions": pred})
        return per_view

    def train_epoch(self, train_loader, return_loss_components=False):
        """Train all FragmentCIGA views for one epoch."""
        for view in self.views:
            view.train()

        total_loss = 0.0
        component_totals = {key: 0.0 for key in self.loss_component_keys()}
        correct = 0
        total = 0
        num_batches = 0

        use_irm = self.uses_irm()
        use_aux = self.lambda_contrast > 0 or self.lambda_diversity > 0 or use_irm
        use_env = self._assign_env_ids(train_loader) if use_irm else False

        pbar = tqdm(train_loader, desc="Training")
        for step, batch in enumerate(pbar):
            batch = batch.to(self.device)
            labels = self._labels_from_batch(batch)
            per_view = self._forward_views(batch, use_aux)

            batch_env = None
            if use_env:
                batch_env = batch.env_id.to(self.device)

            loss, components, _ = self._compute_components(per_view, labels, batch_env)

            if step == 0 and getattr(self, 'log_gradient_influence', False):
                self.last_gradient_report = {}
                for group,attr in [('scorer','fragment_masker'),('classifier','atom_classifier')]:
                    params=[p for v in self.views for p in getattr(v,attr).parameters() if p.requires_grad]
                    for name in ['prediction_loss','contrastive_loss_weighted']:
                        term=components[name]
                        gradients=torch.autograd.grad(term,params,retain_graph=True,allow_unused=True) if term.requires_grad else []
                        self.last_gradient_report[group+'_'+name]=sum(float(g.detach().square().sum()) for g in gradients if g is not None)**.5

            for opt in self.optimizers:
                opt.zero_grad()
            loss.backward()
            for view in self.views:
                torch.nn.utils.clip_grad_norm_(view.parameters(), max_norm=1.0)
            for opt in self.optimizers:
                opt.step()

            total_loss += float(loss.detach().item())
            for key, value in components.items():
                component_totals[key] += float(value.detach().item())
            num_batches += 1

            avg_probs = torch.stack(
                [pv["predictions"].exp() for pv in per_view],
                dim=0,
            ).mean(dim=0)
            pred_labels = avg_probs.argmax(dim=-1)
            correct += (pred_labels == labels).sum().item()
            total += labels.size(0)

            current_acc = correct / total if total else 0.0
            pbar.set_postfix(
                {
                    "loss": f"{loss.detach().item():.4f}",
                    "pred": f"{components['prediction_loss'].detach().item():.4f}",
                    "acc": f"{current_acc:.4f}",
                }
            )

        avg_loss = total_loss / num_batches if num_batches else 0.0
        accuracy = correct / total if total else 0.0
        avg_components = self._average_components(component_totals, num_batches)
        if return_loss_components:
            return avg_loss, accuracy, avg_components
        return avg_loss, accuracy

    def evaluate_all(self, eval_loader, return_loss_components=False):
        """Single-pass evaluation returning per-view and ensemble metrics.

        The ensemble ``accuracy``/``roc_auc`` are scored through the configured
        inference policy (``self.ensemble``) — the
        same path used at final test time — so best-epoch and early-stop
        selection are made on the true inference path rather than on an average
        of independent per-view predictions. Per-view metrics still come from
        each view's own top-k subgraph, for diagnostics only.
        """
        for view in self.views:
            view.eval()

        use_irm = self.uses_irm()
        use_aux = self.lambda_contrast > 0 or self.lambda_diversity > 0 or use_irm
        num_views = len(self.views)
        use_env = self._assign_env_ids(eval_loader) if use_irm else False

        view_correct = [0] * num_views
        view_total = [0] * num_views
        view_loss_sum = [0.0] * num_views
        view_prob_chunks = [[] for _ in range(num_views)] if self.track_roc_auc else None
        view_label_chunks = [[] for _ in range(num_views)] if self.track_roc_auc else None

        total_loss = 0.0
        component_totals = {key: 0.0 for key in self.loss_component_keys()}
        correct = 0
        total = 0
        num_batches = 0
        prob_chunks = [] if self.track_roc_auc else None
        label_chunks = [] if self.track_roc_auc else None

        for step, batch in enumerate(eval_loader):
            batch = batch.to(self.device)
            labels = self._labels_from_batch(batch)

            # Eval never backprops: run the whole forward grad-free so the GAT
            # activations for all views aren't retained (the OOM source). IRM's
            # penalty is still computed as a reported value — it re-enables grad
            # internally only around its tiny dummy scalar, on detached logits.
            with torch.no_grad():
                per_view = self._forward_views(batch, use_aux)
                batch_env = None
                if use_env:
                    batch_env = batch.env_id.to(self.device)
                loss, components, per_view_losses = self._compute_components(per_view, labels, batch_env)
                # Shared-subgraph ensemble prediction: aggregate per-view edge
                # scores → one shared top-k subgraph → soft-vote. This is the
                # metric best-epoch/early-stop selection is made on.
                ens_log_probs = self.ensemble(batch, view_indices=None)

            for i, (pv, pv_loss) in enumerate(zip(per_view, per_view_losses)):
                preds = pv["predictions"]
                view_loss_sum[i] += float(pv_loss.item())
                view_correct[i] += (preds.argmax(dim=-1) == labels).sum().item()
                view_total[i] += labels.size(0)
                if self.track_roc_auc:
                    view_prob_chunks[i].append(preds.detach().exp().cpu())
                    view_label_chunks[i].append(labels.cpu())

            ens_probs = ens_log_probs.detach().exp()
            correct += (ens_probs.argmax(dim=-1) == labels).sum().item()
            total += labels.size(0)
            total_loss += float(loss.detach().item())
            for key, value in components.items():
                component_totals[key] += float(value.detach().item())
            num_batches += 1
            if self.track_roc_auc:
                prob_chunks.append(ens_probs.cpu())
                label_chunks.append(labels.detach().cpu())

        per_view_metrics = []
        for i in range(num_views):
            auc = (
                self._roc_auc_from_chunks(view_prob_chunks[i], view_label_chunks[i])
                if self.track_roc_auc
                else None
            )
            per_view_metrics.append({
                "loss": view_loss_sum[i] / num_batches if num_batches else 0.0,
                "acc": view_correct[i] / view_total[i] if view_total[i] else 0.0,
                "auc": auc,
            })

        avg_loss = total_loss / num_batches if num_batches else 0.0
        accuracy = correct / total if total else 0.0
        roc_auc = self._roc_auc_from_chunks(prob_chunks, label_chunks)
        avg_components = self._average_components(component_totals, num_batches)

        if return_loss_components:
            return avg_loss, accuracy, roc_auc, avg_components, per_view_metrics
        return avg_loss, accuracy, roc_auc, per_view_metrics

    def _loss_log_fieldnames(self):
        return (
            "epoch",
            "effective_diversity_weight",
            "train_loss",
            "train_prediction_loss",
            "train_contrastive_loss",
            "train_contrastive_loss_weighted",
            "train_diversity_loss",
            "train_diversity_loss_weighted",
            "train_irm_loss",
            "train_irm_loss_weighted",
            "train_acc",
            "val_loss",
            "val_prediction_loss",
            "val_contrastive_loss",
            "val_contrastive_loss_weighted",
            "val_diversity_loss",
            "val_diversity_loss_weighted",
            "val_irm_loss",
            "val_irm_loss_weighted",
            "val_acc",
            "val_auc",
            "val_selection_metric",
            "val_selection_metric_name",
            "val_selection_metric_smoothed",
            "val_acc_per_view",
            "val_auc_per_view",
            # Diagnostic-only per-epoch test-split metrics (populated only when a
            # diagnostic_loader is passed to train()); never used for selection.
            "test_loss",
            "test_acc",
            "test_auc",
        )

    def _initialize_loss_log(self):
        if self.loss_log_path is None:
            return
        self.loss_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.loss_log_path.open("w", newline="") as log_file:
            writer = csv.DictWriter(log_file, fieldnames=self._loss_log_fieldnames())
            writer.writeheader()
        print(f"Writing per-epoch loss components to {self.loss_log_path}")

    def _append_loss_log_row(self, row):
        if self.loss_log_path is None:
            return
        with self.loss_log_path.open("a", newline="") as log_file:
            writer = csv.DictWriter(log_file, fieldnames=self._loss_log_fieldnames())
            writer.writerow(row)

    def train(self, train_loader, val_loader=None, num_epochs=100, patience=20,
              restore_best_weights=True, diagnostic_loader=None, epoch_callback=None):
        """Full training loop with complete-window joint checkpoint selection.

        ``diagnostic_loader`` (typically the test split) is evaluated once per
        epoch purely to log its per-epoch AUC alongside validation. It is NEVER
        read for checkpoint selection or early stopping — it exists only to make
        the validation/test decoupling visible. Leaking it into selection would
        be test-set peeking, so keep it strictly out of the selection path.
        """
        if val_loader is not None and num_epochs < self.selection_ma_window:
            raise ValueError("epochs must be >= selection_ma_window; use window=1 for a one-epoch smoke test")
        num_views = len(self.views)
        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_auc": [],
            "val_acc_per_view": [],
            "val_auc_per_view": [],
            "val_selection_metric": [],
            "val_selection_metric_name": [],
            "val_selection_metric_smoothed": [],
            "test_loss": [],
            "test_acc": [],
            "test_auc": [],
        }
        for key in self.loss_component_keys():
            history[f"train_{key}"] = []
            history[f"val_{key}"] = []

        # Best-epoch selection is made on the configured ensemble metric
        # (the true inference path), and all views are snapshotted together at
        # that single epoch. Restoring each view from its own independently-best
        # epoch would mix maskers the aggregated score vote never validated.
        #
        # Selection is driven by a trailing moving average of the validation
        # metric (window = selection_ma_window) rather than the raw per-epoch
        # value. Complete windows reduce sensitivity to individual peaks but
        # do not guarantee better OOD performance. window=1 retains raw selection.
        raw_metric_history = []
        # Retained top snapshots ranked by smoothed metric. Capacity is 1 for
        # plain best-epoch restore, or the MA window when averaging weights.
        snapshot_capacity = (
            self.selection_ma_window if self.selection_weight_avg else 1
        )
        retained_snapshots = []  # list of dicts: {metric, epoch, weights}
        best_snapshot_epoch = None
        best_val_metric = -float("inf")
        best_metric_label = "acc"
        patience_counter = 0
        self._initialize_loss_log()

        from src.training.diversity_schedule import diversity_weight
        history["effective_diversity_weight"] = []
        stop_floor = max(self.early_stopping_min_epochs,
                         self.diversity_delay_epochs + self.diversity_ramp_epochs
                         if self.lambda_diversity > 0 else 0)
        for epoch in range(num_epochs):
            self.effective_diversity_weight = diversity_weight(
                self.lambda_diversity, epoch + 1, self.diversity_delay_epochs, self.diversity_ramp_epochs)
            history["effective_diversity_weight"].append(self.effective_diversity_weight)
            if epoch + 1 <= stop_floor:
                patience_counter = 0
            train_loss, train_acc, train_components = self.train_epoch(
                train_loader,
                return_loss_components=True,
            )
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            for key, value in train_components.items():
                history[f"train_{key}"].append(value)

            should_stop = False
            val_loss = None
            val_acc = None
            val_auc = None
            val_components = {key: None for key in self.loss_component_keys()}
            view_val_accs = []
            view_val_aucs = []
            val_selection_metric = None
            smoothed_metric = None
            metric_label = best_metric_label

            if val_loader is not None:
                val_loss, val_acc, val_auc, val_components, all_view_metrics = self.evaluate_all(
                    val_loader,
                    return_loss_components=True,
                )
                for vm in all_view_metrics:
                    view_val_accs.append(vm["acc"])
                    view_val_aucs.append(vm["auc"])
                val_selection_metric, metric_label = self._metric_for_selection(val_acc, val_auc)
                best_metric_label = metric_label

                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)
                history["val_auc"].append(val_auc)
                history["val_acc_per_view"].append(view_val_accs)
                history["val_auc_per_view"].append(view_val_aucs)
                history["val_selection_metric"].append(val_selection_metric)
                history["val_selection_metric_name"].append(metric_label)
                for key, value in val_components.items():
                    history[f"val_{key}"].append(value)

                view_acc_str = "  ".join(
                    f"v{i}={acc:.4f}" for i, acc in enumerate(view_val_accs)
                )
                auc_str = f" | Val AUC: {val_auc:.4f}" if val_auc is not None else ""
                print(
                    f"Epoch {epoch + 1}/{num_epochs} | "
                    f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
                    f"{auc_str} | Val per-view Acc [{view_acc_str}] | "
                    f"Selection val_{metric_label}: {val_selection_metric:.4f}"
                )

                # Never mix fallback accuracy with AUC in the same window.
                if (len(history["val_selection_metric_name"]) > 1 and
                        history["val_selection_metric_name"][-2] != metric_label):
                    raw_metric_history.clear()
                    retained_snapshots.clear()
                    best_val_metric = -float("inf")
                    best_snapshot_epoch = None
                    patience_counter = 0
                raw_metric_history.append(val_selection_metric)
                smoothed_metric = trailing_mean(raw_metric_history, self.selection_ma_window)
                history["val_selection_metric_smoothed"].append(smoothed_metric)

                if smoothed_metric is not None:
                    # Top-K includes runners-up, not only new all-time records.
                    if restore_best_weights and (
                        len(retained_snapshots) < snapshot_capacity or
                        smoothed_metric > retained_snapshots[-1]["metric"]
                    ):
                        self._retain_snapshot(
                            retained_snapshots, snapshot_capacity, smoothed_metric,
                            epoch + 1, [deepcopy(v.state_dict()) for v in self.views],
                        )
                        best_snapshot_epoch = retained_snapshots[0]["epoch"]
                    if smoothed_metric > best_val_metric:
                        best_val_metric = smoothed_metric
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    if epoch + 1 <= stop_floor:
                        patience_counter = 0
                    if patience is not None and epoch + 1 > stop_floor and patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch + 1}")
                        should_stop = True
            else:
                print(
                    f"Epoch {epoch + 1}/{num_epochs} | "
                    f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}"
                )

            # Diagnostic-only: log the test-split curve alongside validation so
            # the val/test decoupling is visible. Not used for any decision.
            test_loss = None
            test_acc = None
            test_auc = None
            if diagnostic_loader is not None:
                test_loss, test_acc, test_auc, _ = self.evaluate_all(diagnostic_loader)
                history["test_loss"].append(test_loss)
                history["test_acc"].append(test_acc)
                history["test_auc"].append(test_auc)
                test_auc_str = (
                    f" | Test AUC: {test_auc:.4f}" if test_auc is not None else ""
                )
                print(
                    f"           (diagnostic) Test Loss: {test_loss:.4f} | "
                    f"Test Acc: {test_acc:.4f}{test_auc_str}"
                )

            self._append_loss_log_row(
                {
                    "epoch": epoch + 1,
                    "effective_diversity_weight": self.effective_diversity_weight,
                    "train_loss": train_loss,
                    "train_prediction_loss": train_components["prediction_loss"],
                    "train_contrastive_loss": train_components["contrastive_loss"],
                    "train_contrastive_loss_weighted": train_components["contrastive_loss_weighted"],
                    "train_diversity_loss": train_components["diversity_loss"],
                    "train_diversity_loss_weighted": train_components["diversity_loss_weighted"],
                    "train_irm_loss": train_components["irm_loss"],
                    "train_irm_loss_weighted": train_components["irm_loss_weighted"],
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_prediction_loss": val_components["prediction_loss"],
                    "val_contrastive_loss": val_components["contrastive_loss"],
                    "val_contrastive_loss_weighted": val_components["contrastive_loss_weighted"],
                    "val_diversity_loss": val_components["diversity_loss"],
                    "val_diversity_loss_weighted": val_components["diversity_loss_weighted"],
                    "val_irm_loss": val_components["irm_loss"],
                    "val_irm_loss_weighted": val_components["irm_loss_weighted"],
                    "val_acc": val_acc,
                    "val_auc": val_auc,
                    "val_selection_metric": val_selection_metric,
                    "val_selection_metric_name": metric_label,
                    "val_selection_metric_smoothed": smoothed_metric,
                    "val_acc_per_view": "|".join(f"{v:.6f}" for v in view_val_accs),
                    "val_auc_per_view": "|".join(
                        "" if v is None else f"{v:.6f}" for v in view_val_aucs
                    ),
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                    "test_auc": test_auc,
                }
            )

            if epoch_callback is not None and val_loader is not None:
                epoch_callback(self, epoch + 1, val_selection_metric, smoothed_metric, metric_label)
            if should_stop:
                break

        if val_loader is not None and best_val_metric == -float("inf"):
            raise ValueError("No complete finite validation selection window was recorded")
        history["selected_checkpoint_epoch"] = best_snapshot_epoch
        history["best_selection_metric_smoothed"] = (
            best_val_metric if best_val_metric != -float("inf") else None
        )
        if restore_best_weights and retained_snapshots:
            window_note = (
                f" (MA window {self.selection_ma_window})"
                if self.selection_ma_window > 1
                else ""
            )
            if self.selection_weight_avg and len(retained_snapshots) > 1:
                epochs = [s["epoch"] for s in retained_snapshots]
                averaged = self._average_snapshots(
                    [s["weights"] for s in retained_snapshots]
                )
                for view, weights in zip(self.views, averaged):
                    view.load_state_dict(weights)
                print(
                    f"Restored SWA average of {len(retained_snapshots)} snapshots "
                    f"from epochs {sorted(epochs)} across all {num_views} views "
                    f"(best smoothed val_{best_metric_label} = {best_val_metric:.4f}"
                    f"{window_note})"
                )
            else:
                best = retained_snapshots[0]
                for view, weights in zip(self.views, best["weights"]):
                    view.load_state_dict(weights)
                print(
                    f"Restored best joint snapshot from epoch {best['epoch']} "
                    f"across all {num_views} views "
                    f"(best smoothed val_{best_metric_label} = {best_val_metric:.4f}"
                    f"{window_note})"
                )

        self._report_selection_diagnostics(history, best_snapshot_epoch, best_metric_label)
        return history

    @staticmethod
    def _retain_snapshot(retained, capacity, metric, epoch, weights):
        """Keep the top-``capacity`` snapshots ranked by smoothed metric (desc).

        ``retained`` is mutated in place and kept sorted best-first so callers
        can read ``retained[0]`` as the current best.
        """
        retained.append({"metric": metric, "epoch": epoch, "weights": weights})
        retained.sort(key=lambda s: s["metric"], reverse=True)
        del retained[capacity:]

    @staticmethod
    def _average_snapshots(snapshots):
        """Element-wise average a list of per-view state_dict lists (SWA-style).

        Each snapshot is a list of state_dicts (one per view). Only floating
        tensors are averaged; integer/bool buffers (e.g. sampler call counts)
        are taken from the first snapshot to avoid corrupting them.
        """
        num_views = len(snapshots[0])
        averaged = []
        for view_idx in range(num_views):
            base = snapshots[0][view_idx]
            merged = {}
            for key, ref in base.items():
                if torch.is_floating_point(ref):
                    stacked = torch.stack(
                        [snap[view_idx][key].float() for snap in snapshots], dim=0
                    )
                    merged[key] = stacked.mean(dim=0).to(ref.dtype)
                else:
                    merged[key] = ref.clone()
            averaged.append(merged)
        return averaged

    @staticmethod
    def _pearson(xs, ys):
        """Pearson correlation over paired, non-None samples; None if undefined."""
        pairs = [
            (x, y)
            for x, y in zip(xs, ys)
            if x is not None and y is not None
        ]
        n = len(pairs)
        if n < 2:
            return None
        mean_x = sum(x for x, _ in pairs) / n
        mean_y = sum(y for _, y in pairs) / n
        cov = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
        var_x = sum((x - mean_x) ** 2 for x, _ in pairs)
        var_y = sum((y - mean_y) ** 2 for _, y in pairs)
        if var_x <= 0 or var_y <= 0:
            return None
        return cov / (var_x ** 0.5 * var_y ** 0.5)

    def _report_selection_diagnostics(self, history, best_snapshot_epoch, metric_label):
        """Print the honest val→test gap and val↔test correlation.

        The "honest" gap compares validation and test AUC at the *same* restored
        epoch, instead of the biased running-max-val vs final-test comparison.
        A low correlation means validation is a poor selector for test — the
        core reason test AUC sits near the ERM baseline regardless of changes.
        """
        val_curve = history.get("val_auc", []) or []
        test_curve = history.get("test_auc", []) or []
        if not test_curve:
            return  # diagnostics only meaningful when a diagnostic_loader was used

        corr = self._pearson(val_curve, test_curve)
        corr_str = "n/a" if corr is None else f"{corr:.3f}"
        print(
            f"[selection diagnostics] per-epoch val↔test {metric_label} "
            f"correlation = {corr_str} over {len(test_curve)} epoch(s)"
        )

        if best_snapshot_epoch is not None:
            idx = best_snapshot_epoch - 1
            val_at = val_curve[idx] if 0 <= idx < len(val_curve) else None
            test_at = test_curve[idx] if 0 <= idx < len(test_curve) else None
            if val_at is not None and test_at is not None:
                print(
                    f"[selection diagnostics] at restored epoch {best_snapshot_epoch}: "
                    f"val {metric_label}={val_at:.4f}  test {metric_label}={test_at:.4f}  "
                    f"honest gap={val_at - test_at:+.4f}"
                )
