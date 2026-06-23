"""Regression tests for SuGAr loss helpers.

Run this file from the repository root with:

    .\.venv\Scripts\python.exe -m unittest tests.test_losses -v
"""

import unittest

import torch

from src.losses.supervised_contrastive import supervised_contrastive_loss


class LossFunctionTests(unittest.TestCase):
    def test_supervised_contrastive_no_positive_pairs_returns_zero(self):
        embeddings = torch.randn(2, 8)
        labels = torch.tensor([0, 1])

        for sampling in ["mul", "var"]:
            loss = supervised_contrastive_loss(
                embeddings,
                labels,
                sampling=sampling,
            )
            self.assertTrue(torch.isfinite(loss))
            self.assertEqual(float(loss), 0.0)

    def test_cnc_contrastive_accepts_predictions_and_handles_no_positives(self):
        embeddings = torch.randn(2, 8)
        labels = torch.tensor([0, 1])
        y_pred = torch.tensor([0, 1])

        loss = supervised_contrastive_loss(
            embeddings,
            labels,
            sampling="cnc",
            y_pred=y_pred,
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(loss), 0.0)


if __name__ == "__main__":
    unittest.main()
