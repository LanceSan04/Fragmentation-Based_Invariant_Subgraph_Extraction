import copy
import unittest
from src.experiments.atom_budget_recovery import parity_report, check_completed_reference, require_replay_stability


class RecoveryTests(unittest.TestCase):
    def test_replay_stability_thresholds(self):
        require_replay_stability([dict(max_probability_difference=.0001953,mean_probability_difference=8.2e-8)])
        for maximum, mean in [(.002,1e-8),(.0005,.00002),(float('nan'),0)]:
            with self.assertRaises(ValueError):
                require_replay_stability([dict(max_probability_difference=maximum,mean_probability_difference=mean)])

    def fixture(self):
        expected = dict(epochs=3)
        config = dict(expected, seed=8401,sampler_seed=8401,early_stopping=False,
                      selection_ma_window=1,selection_weight_avg=False,greedy_model_selection=False,
                      loss_log_path=None,test_split=None,report_splits=[],loader_split_names=['train','ood_val'])
        history = [dict(epoch=i+1, validation={'independent':{'auc':v}}) for i,v in enumerate([.7,.8,.75])]
        result = dict(seed=8401,best={'independent':dict(auc=.8,epoch=2)},
                      results={'independent':{'independent':{'ood_val':{'auc':.8000031}}}})
        payload = dict(epoch=2,config=config.copy())
        return config,history,result,payload,expected,3,8401

    def test_accepts_observed_scale_and_rejects_meaningful_drift(self):
        result = self.fixture()[2]
        self.assertTrue(parity_report(result)['passed'])
        for bad in [.8002, float('nan'), float('inf')]:
            changed=copy.deepcopy(result)
            changed['results']['independent']['independent']['ood_val']['auc']=bad
            with self.assertRaises(ValueError): parity_report(changed)

    def test_recovery_requires_complete_and_matching_reference(self):
        check_completed_reference(*self.fixture())
        for field in ['config','history','checkpoint','result','test']:
            args=list(copy.deepcopy(self.fixture()))
            if field=='config':args[0]['epochs']=100
            if field=='history':args[1].pop()
            if field=='checkpoint':args[3]['epoch']=3
            if field=='result':args[2]['best']['independent']['epoch']=3
            if field=='test':args[2]['results']['independent']['independent']['ood_test']={'auc':.9}
            with self.assertRaises(ValueError, msg=field):check_completed_reference(*args)

if __name__=='__main__':unittest.main()
