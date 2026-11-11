from copy import deepcopy
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from src.experiments.checkpoints import digest
from src.experiments.fragment_confirmation import CANDIDATES, KI_CANDIDATES, evaluate_batch, main, require_validation_winner, validate


class FragmentConfirmationTests(unittest.TestCase):
    def test_selection_requires_complete_search_and_mean_validation_winner(self):
        search=dict(recipe_ids=[1,12],seeds=[10701,10702])
        rows=[dict(dataset='ki_assay',recipe=rid,seed=seed,status='completed',validation_auc=auc)
              for rid,values in [(1,[.7,.7]),(12,[.9,.4])]
              for seed,auc in zip(search['seeds'],values)]
        require_validation_winner(rows,'ki_assay',search,1)
        with self.assertRaisesRegex(ValueError,'not the validation-selected winner'):
            require_validation_winner(rows,'ki_assay',search,12)
        nonfinite=deepcopy(rows);nonfinite[0]['validation_auc']=float('nan')
        failed=deepcopy(rows);failed[0]['status']='failed'
        for bad in [rows[:-1],rows+[rows[0]],nonfinite,failed]:
            with self.assertRaisesRegex(ValueError,'Incomplete'):
                require_validation_winner(bad,'ki_assay',search,1)

    def test_ki_protocol_maps_sparse_recipe_ids_and_freezes_fifteen_runs(self):
        search=json.loads(Path('configs/ki_bounded_search_20261019.json').read_text())
        spec=json.loads(Path('configs/ki_confirmation_20261020.json').read_text())
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);ref=root/'reference.json';ref.write_text('{"configs":{"ec50_scaffold":{}}}')
            search['reference_plan_sha256']=digest(ref)
            search_path=root/'search.json';search_path.write_text(json.dumps(search))
            spec['search_config_sha256']=digest(search_path)
            validate(spec,search,ref,search_path)
            self.assertEqual(spec['candidates'][1]['recipe'],12)
            self.assertEqual(spec['candidates'][1]['parameters']['lambda_contrast'],0)
            for key,value in [('seeds',[10701,10702]),('epochs',100)]:
                bad=deepcopy(spec);bad[key]=value
                with self.assertRaises(ValueError):validate(bad,search,ref,search_path)
            bad=deepcopy(spec);bad['candidates'][1]['parameters']['lambda_contrast']=2
            with self.assertRaisesRegex(ValueError,'recipe changed'):validate(bad,search,ref,search_path)
            config=root/'confirm.json';config.write_text(json.dumps(spec));out=root/'not_created'
            args=['--config',str(config),'--search-config',str(search_path),'--reference-plan',str(ref),
                  '--project-root',str(root),'--output-dir',str(out)]
            with redirect_stdout(StringIO()):plan=main(args)
            self.assertEqual(plan['training_runs'],15)
            self.assertEqual(plan['ki_training_runs'],15)
            self.assertEqual(plan['ec50_training_runs'],0)
            self.assertEqual(plan['test_evaluated_for'],'KI only')
            self.assertFalse(out.exists())
            with redirect_stdout(StringIO()),redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit):main(args+['--then-ki-pilots'])

    def test_ki_validation_gate_and_summary_have_no_ec50_pairing(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);entries=self.entries(out)[:3];calls=[]
            for entry,(dataset,recipe) in zip(entries,KI_CANDIDATES):
                entry.update(dataset=dataset,recipe=recipe,seed=10801)
            def evaluate(checkpoint,split,*args,**kwargs):
                calls.append(split);return dict(auc=.7 if split=='ood_val' else .6)
            with patch('src.experiments.fragment_confirmation.evaluate_saved',side_effect=evaluate), \
                 patch('src.experiments.fragment_confirmation.prediction_report',return_value={}):
                report=evaluate_batch(entries,out,'cpu',diagnostics=False,
                    candidate_ids=KI_CANDIDATES,report_name='ki_confirmation.json')
            self.assertEqual(calls,['ood_val']*3+['ood_test']*3)
            self.assertEqual(len(report['candidates']),3)
            self.assertNotIn('scaffold_recipe04_minus_recipe07',report)
            self.assertTrue((out/'ki_confirmation.json').is_file())
            self.assertFalse((out/'ec50_confirmation.json').exists())

    def entries(self, root):
        entries=[]
        for i,(dataset,recipe) in enumerate(CANDIDATES):
            run=root/f'run{i}';run.mkdir();ckpt=run/'model.pt';ckpt.write_text('fixed weights')
            entries.append(dict(dataset=dataset,recipe=recipe,seed=10601,epoch=13,selected_auc=.7,
                parameters={},checkpoint=str(ckpt),checkpoint_sha256=digest(ckpt),
                run_dir=str(run),cache=str(root/'cache')))
        return entries

    def test_all_validation_precedes_all_test_and_paired_summary(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);entries=self.entries(out);calls=[]
            def evaluate(checkpoint,split,*args,**kwargs):
                calls.append(split)
                return dict(auc=np.float64(.7 if split=='ood_val' else .6),loss=.5,acc=.5)
            with patch('src.experiments.fragment_confirmation.evaluate_saved',side_effect=evaluate), \
                 patch('src.experiments.fragment_confirmation.prediction_report',return_value={}):
                report=evaluate_batch(entries,out,'cpu',diagnostics=False)
            self.assertEqual(calls,['ood_val']*4+['ood_test']*4)
            self.assertEqual(report['scaffold_recipe04_minus_recipe07']['ood_test']['values'],[0.])
            self.assertEqual(len(report['candidates']),4)

    def test_parity_failure_blocks_every_test_evaluation(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);entries=self.entries(out);calls=[]
            def evaluate(checkpoint,split,*args,**kwargs):
                calls.append(split);return dict(auc=.7 if len(calls)==1 else .8)
            with patch('src.experiments.fragment_confirmation.evaluate_saved',side_effect=evaluate), \
                 patch('src.experiments.fragment_confirmation.prediction_report',return_value={}):
                with self.assertRaises(ValueError):evaluate_batch(entries,out,'cpu',diagnostics=False)
            self.assertEqual(calls,['ood_val','ood_val'])

    def test_changed_checkpoint_blocks_test_after_successful_validation(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);entries=self.entries(out);calls=[]
            def evaluate(checkpoint,split,*args,**kwargs):
                calls.append(split)
                if len(calls)==4:Path(entries[0]['checkpoint']).write_text('changed')
                return dict(auc=.7)
            with patch('src.experiments.fragment_confirmation.evaluate_saved',side_effect=evaluate), \
                 patch('src.experiments.fragment_confirmation.prediction_report',return_value={}):
                with self.assertRaisesRegex(ValueError,'changed'):evaluate_batch(entries,out,'cpu',diagnostics=False)
            self.assertEqual(calls,['ood_val']*4)

    def test_frozen_twenty_run_scope_and_dry_run(self):
        search=json.loads(Path('configs/fragment_tuning_ec50_ki_20261018.json').read_text())
        spec=json.loads(Path('configs/fragment_confirmation_20261019.json').read_text())
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);ref=root/'reference.json';ref.write_text('{}')
            search['reference_plan_sha256']=digest(ref)
            search_path=root/'search.json';search_path.write_text(json.dumps(search))
            spec['search_config_sha256']=digest(search_path)
            validate(spec,search,ref,search_path)
            bad=deepcopy(spec);bad['seeds'][0]=10501
            with self.assertRaises(ValueError):validate(bad,search,ref,search_path)
            bad=deepcopy(spec);bad['candidates'][0]['recipe']=3
            with self.assertRaises(ValueError):validate(bad,search,ref,search_path)
            config=root/'confirmation.json';config.write_text(json.dumps(spec));out=root/'not_created'
            with redirect_stdout(StringIO()):
                plan=main(['--config',str(config),'--search-config',str(search_path),'--reference-plan',str(ref),
                    '--project-root',str(root),'--output-dir',str(out),'--then-ki-pilots'])
            self.assertEqual(plan['ec50_training_runs'],20)
            self.assertEqual(plan['ki_training_runs'],6)
            self.assertFalse(plan['ki_full_search'])
            self.assertFalse(out.exists())
