"""Record two validation-only checkpoint policies on one fixed trajectory."""
import json
import math
from pathlib import Path

from src.training.checkpoint import save_training_checkpoint


class PolicyComparison:
    def __init__(self, config, feature_dims):
        self.config = config
        self.feature_dims = feature_dims
        self.directory = Path(config.run_dir) / "selection_comparison"
        self.records = {}

    def observe(self, trainer, epoch, raw, smoothed, metric_name):
        if metric_name != "auc" or not math.isfinite(raw):
            raise ValueError("Policy comparison requires finite validation AUC every epoch")
        for policy, score in (("best_raw", raw), ("best_ma_endpoint", smoothed)):
            if score is None:
                continue
            if policy in self.records and score <= self.records[policy]["selection_score"]:
                continue
            path = self.directory / f"{policy}.pt"
            save_training_checkpoint(
                path, trainer, self.config, self.feature_dims, epoch=epoch,
                metrics={self.config.validation_split: {"auc": raw}},
                include_optimizers=False,
            )
            self.records[policy] = dict(
                epoch=epoch, selection_score=score, validation_auc_at_selection=raw,
                checkpoint=str(path),
            )

    def finish(self, history):
        completed = len(history["train_loss"])
        if completed != self.config.epochs or len(self.records) != 2:
            raise RuntimeError("Policy comparison did not complete its fixed budget and both policies")
        report = dict(
            completed_epochs=completed, window=self.config.selection_ma_window,
            seed=self.config.seed, validation_split=self.config.validation_split,
            test_evaluated=False, checkpoint_purpose="evaluation, not exact training resume",
            policies=self.records,
        )
        path = self.directory / "comparison.json"
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Checkpoint policy comparison saved to {path}")
        return report
