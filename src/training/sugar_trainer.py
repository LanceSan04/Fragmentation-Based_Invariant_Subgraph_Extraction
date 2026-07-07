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
from src.models.sugar_ensemble import SuGAREnsemble

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
        self.track_roc_auc = track_roc_auc and _SKLEARN_AVAILABLE
        self.loss_log_path = Path(loss_log_path) if loss_log_path else None
        # Validation is scored through the same shared-subgraph aggregation +
        # soft-vote path used at final test time. The ensemble holds references
        # to the same view objects, so weight updates are reflected without any
        # re-wrapping.
        self.aggregation_mode = aggregation_mode
        self.ensemble = SuGAREnsemble(self.views, aggregation_mode=aggregation_mode)
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

        per_view_losses = [
            prediction_loss(pv["predictions"], labels)
            for pv in per_view
        ]
        components["prediction_loss"] = torch.stack(per_view_losses).sum()

        if self.lambda_contrast > 0:
            all_emb = torch.cat([pv["embeddings"] for pv in per_view], dim=0)
            all_labels = labels.repeat(len(self.views))
            if self.contrast_sampling.lower() == "cnc":
                y_pred = torch.cat(
                    [pv["logits"].detach().argmax(dim=-1) for pv in per_view],
                    dim=0,
                )
            else:
                y_pred = None
            contrastive = supervised_contrastive_loss(
                all_emb,
                all_labels,
                temperature=self.contrast_temperature,
                sampling=self.contrast_sampling,
                y_pred=y_pred,
            )
            components["contrastive_loss"] = contrastive
            components["contrastive_loss_weighted"] = self.lambda_contrast * contrastive

        if self.lambda_diversity > 0 and len(self.views) > 1:
            # All views score the same fragment graph, so the edge->molecule map
            # is shared; take it from any view. edge_batch makes the cosine
            # per-molecule; eligible_masks restricts each view pair to edges both
            # actually scored, dropping the sampler's zero-fill artifact (a no-op
            # at eval, where the sampler is off and all edges are eligible).
            eligible_masks = [pv.get("fragment_edge_eligible") for pv in per_view]
            if any(m is None for m in eligible_masks):
                eligible_masks = None
            diversity = pairwise_diversity_loss(
                [pv["fragment_scores"] for pv in per_view],
                edge_batch=per_view[0].get("fragment_edge_batch"),
                eligible_masks=eligible_masks,
                num_graphs=labels.size(0),
            )
            components["diversity_loss"] = diversity
            components["diversity_loss_weighted"] = self.lambda_diversity * diversity

        if batch_env is not None:
            irm_terms = [
                invariant_risk_loss(pv["logits"], labels, batch_env)
                for pv in per_view
            ]
            if irm_terms:
                irm = torch.stack(irm_terms).mean()
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
        per_view = []
        for view in self.views:
            if use_aux:
                pred, interm = view(batch, return_intermediates=True)
                per_view.append({"predictions": pred, **interm})
            else:
                pred = view(batch)
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

        The ensemble ``accuracy``/``roc_auc`` are scored through the shared
        top-k subgraph aggregation + soft-vote path (``self.ensemble``) — the
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
            "val_acc_per_view",
            "val_auc_per_view",
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

    def train(self, train_loader, val_loader=None, num_epochs=100, patience=20, restore_best_weights=True):
        """Full training loop with AUC-aware per-view best-weight tracking."""
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
        }
        for key in self.loss_component_keys():
            history[f"train_{key}"] = []
            history[f"val_{key}"] = []

        # Best-epoch selection is made on the shared-subgraph ensemble metric
        # (the true inference path), and all views are snapshotted together at
        # that single epoch. Restoring each view from its own independently-best
        # epoch would mix maskers the aggregated score vote never validated.
        best_snapshot = None
        best_snapshot_epoch = None
        best_val_metric = -float("inf")
        best_metric_label = "acc"
        patience_counter = 0
        self._initialize_loss_log()

        for epoch in range(num_epochs):
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

                if val_selection_metric > best_val_metric:
                    best_val_metric = val_selection_metric
                    patience_counter = 0
                    if restore_best_weights:
                        best_snapshot = [deepcopy(v.state_dict()) for v in self.views]
                        best_snapshot_epoch = epoch + 1
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    should_stop = True
            else:
                print(
                    f"Epoch {epoch + 1}/{num_epochs} | "
                    f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}"
                )

            self._append_loss_log_row(
                {
                    "epoch": epoch + 1,
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
                    "val_acc_per_view": "|".join(f"{v:.6f}" for v in view_val_accs),
                    "val_auc_per_view": "|".join(
                        "" if v is None else f"{v:.6f}" for v in view_val_aucs
                    ),
                }
            )

            if should_stop:
                break

        if restore_best_weights and best_snapshot is not None:
            for view, weights in zip(self.views, best_snapshot):
                view.load_state_dict(weights)
            print(
                f"Restored best joint snapshot from epoch {best_snapshot_epoch} "
                f"across all {num_views} views "
                f"(best ensemble val_{best_metric_label} = {best_val_metric:.4f})"
            )

        return history
