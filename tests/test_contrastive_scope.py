import unittest
import torch
from src.training.sugar_trainer import SuGARTrainer
from src.losses.supervised_contrastive import supervised_contrastive_loss

class ContrastScopeTests(unittest.TestCase):
    def test_scope_matches_explicit_objective_and_gradients(self):
        torch.manual_seed(42)
        labels=torch.tensor([0,0,1,1])
        views=[]
        for _ in range(2):
            logits=torch.randn(4,2,requires_grad=True)
            views.append(dict(embeddings=torch.randn(4,8,requires_grad=True),logits=logits,predictions=logits.log_softmax(-1)))
        trainer=SuGARTrainer([],lambda_contrast=.1,lambda_diversity=0,lambda_irm=0)
        for scope in ['pooled','per_view']:
            trainer.contrast_scope=scope
            total,parts,_=trainer._compute_components(views,labels)
            if scope=='pooled':
                expected=supervised_contrastive_loss(torch.cat([v['embeddings'] for v in views]),labels.repeat(2),temperature=.5,sampling='mul')
            else:
                expected=sum(supervised_contrastive_loss(v['embeddings'],labels,temperature=.5,sampling='mul') for v in views)
            torch.testing.assert_close(parts['contrastive_loss'],expected)
            actual_grad=torch.autograd.grad(total,views[0]['embeddings'],retain_graph=True)[0]
            expected_grad=torch.autograd.grad(.1*expected,views[0]['embeddings'],retain_graph=True)[0]
            torch.testing.assert_close(actual_grad,expected_grad)
    def test_invalid_scope(self):
        with self.assertRaises(ValueError): SuGARTrainer([],contrast_scope='invalid')
