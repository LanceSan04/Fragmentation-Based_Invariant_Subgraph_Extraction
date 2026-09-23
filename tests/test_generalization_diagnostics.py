import unittest
import torch
from scripts.diagnose_generalization import gradient_vector, summarize_gradients


class GradientDiagnosticsTests(unittest.TestCase):
    def test_opposition_and_unused_parameters(self):
        a = torch.tensor([2., 3.], requires_grad=True)
        unused = torch.tensor([7.], requires_grad=True)
        loss = (a*a).sum()
        positive = gradient_vector(loss, [a, unused])
        negative = gradient_vector(-loss, [a, unused])
        self.assertTrue(torch.equal(positive, torch.tensor([4., 6., 0.])))
        stats = summarize_gradients({'positive': positive, 'negative': negative})
        self.assertAlmostEqual(stats['cosines']['positive__negative'], -1., places=6)
        self.assertIsNone(a.grad)

    def test_zero_gradient_is_not_alignment(self):
        p = torch.tensor([1.], requires_grad=True)
        vectors = {'inactive': gradient_vector(torch.tensor(0.), [p]),
                   'active': gradient_vector(p.sum(), [p])}
        self.assertIsNone(summarize_gradients(vectors)['cosines']['inactive__active'])


if __name__ == '__main__':
    unittest.main()
