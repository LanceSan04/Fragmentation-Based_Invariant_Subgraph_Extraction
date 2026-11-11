"""Protocol, identity and test-gate checks for the bounded Ki comparison."""
from copy import deepcopy
from contextlib import redirect_stdout
from io import StringIO
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.experiments.checkpoints import digest
from src.experiments.ki_reference_controls import (
    ARMS, DATASETS, ROOT, arm_config, audit_source, base_config, evaluate_batch,
    main, reference_command, summarize, validate,
    reference_preflight,
)


class KiReferenceControlsTests(unittest.TestCase):
    def setUp(self):
        self.spec=json.loads((ROOT/'configs/ki_reference_controls_20261024.json').read_text())

    def test_frozen_scope_and_no_side_effect_dry_run(self):
        validate(self.spec)
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'never_created'
            with redirect_stdout(StringIO()):
                full=main(['--project-root',directory,'--output-dir',str(out),'--stage','full'])
                pilot=main(['--project-root',directory,'--output-dir',str(out),'--stage','pilot'])
            self.assertEqual(full['training_runs'],40)
            self.assertTrue(full['test_evaluation_planned'])
            self.assertEqual(pilot['training_runs'],8)
            self.assertFalse(pilot['test_evaluation_planned'])
            self.assertEqual(full['fragment_epochs'],150)
            self.assertEqual(full['reference_epochs'],40)
            self.assertFalse(out.exists())
        for key,value in [('seeds',[1,2]),('reference_epochs',150)]:
            altered=deepcopy(self.spec);altered[key]=value
            with self.assertRaises(ValueError):validate(altered)

    def test_random_ranking_preserves_gates_and_objectives(self):
        base=base_config(self.spec,ROOT,'ki_size','cpu',150)
        learned=arm_config(base,'fragment_learned');random=arm_config(base,'fragment_random')
        differences={key for key in learned if learned[key]!=random[key]}
        self.assertEqual(differences,{'ranking_mode'})
        self.assertEqual(random['message_gate_mode'],'learned')
        self.assertEqual(random['selection_mode'],'learned')
        erm=arm_config(base,'erm')
        self.assertEqual(erm['num_views'],3)
        self.assertEqual(erm['hidden_dim'],128)
        self.assertTrue(erm['erm_baseline'])
        self.assertEqual([erm[k] for k in ('lambda_contrast','lambda_diversity','lambda_irm')],[0,0,0])

    def test_source_audit_reports_imbalance_overlap_and_rejects_hash_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            raw=Path(directory)/'raw.json'
            rows=[dict(smiles='CCO',cls_label=0,assay_id=1),dict(smiles='CCC',cls_label=1,assay_id=1)]
            raw.write_text(json.dumps(dict(split={s:rows for s in ('train','ood_val','ood_test')},
                                          statistics={'thr_for_cls':5})))
            report=audit_source(raw,digest(raw))
            self.assertEqual(report['splits']['train']['negative'],1)
            self.assertEqual(report['canonical_overlap']['train__ood_val'],2)
            self.assertEqual(report['threshold_for_cls'],5)
            with self.assertRaisesRegex(ValueError,'source changed'):audit_source(raw,'0'*64)
            rows[0]['cls_label']=3
            raw.write_text(json.dumps(dict(split={s:rows for s in ('train','ood_val','ood_test')})))
            with self.assertRaisesRegex(ValueError,'binary'):audit_source(raw,digest(raw))

    def entries(self,root):
        rows=[]
        for dataset in DATASETS:
            for arm in ARMS:
                run=root/dataset/arm;run.mkdir(parents=True)
                checkpoint=run/'weights.pt';checkpoint.write_bytes(b'fixed weights')
                rows.append(dict(dataset=dataset,arm=arm,seed=17,run_dir=str(run),
                    checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),
                    selected_auc=.7,cache=str(root/'cache')))
        return rows

    def test_validation_failure_blocks_every_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);rows=self.entries(root);calls=[]
            def evaluate(*args,**kwargs):
                calls.append(args[1]);return {'auc':.8}
            with patch('src.experiments.ki_reference_controls.evaluate_saved',side_effect=evaluate):
                with self.assertRaisesRegex(ValueError,'parity'):
                    evaluate_batch(rows,root,self.spec,root,'cpu',smoke=True,test=True)
            self.assertEqual(calls,['ood_val'])
            self.assertFalse((root/'test_progress.json').exists())

    def test_changed_late_checkpoint_blocks_first_test_after_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);rows=self.entries(root);calls=[]
            def evaluate(*args,**kwargs):
                calls.append(args[1]);return {'auc':.7}
            def reference(row,*args):
                row['metrics']={'ood_val':{'auc':.7}}
                if row is rows[-1]:Path(rows[0]['checkpoint']).write_bytes(b'changed')
            with patch('src.experiments.ki_reference_controls.evaluate_saved',side_effect=evaluate), \
                 patch('src.experiments.ki_reference_controls.reference_validation',side_effect=reference), \
                 patch('src.experiments.ki_reference_controls.reference_predictions'), \
                 patch('src.experiments.ki_reference_controls.project_retention',return_value={}), \
                 patch('src.experiments.ki_reference_controls.prediction_report',return_value={}):
                with self.assertRaisesRegex(ValueError,'checkpoint changed'):
                    evaluate_batch(rows,root,self.spec,root,'cpu',smoke=True,test=True)
            self.assertTrue(calls)
            self.assertEqual(set(calls),{'ood_val'})

    def test_summary_is_paired_and_retains_negative_results(self):
        rows=[dict(dataset=d,arm=arm,seed=17,metrics={'ood_val':{'auc':auc}})
              for d in DATASETS for arm,auc in zip(ARMS,[.6,.7,.8,.75])]
        result=summarize(rows,False)
        self.assertAlmostEqual(result['ki_size']['ood_val']['learned_minus_random']['mean'],-.1)
        self.assertEqual(result['ki_size']['ood_val']['learned_minus_random']['positive_seeds'],0)
        self.assertIsNone(result['ki_size']['ood_val']['learned_minus_random']['sample_sd'])
        import numpy as np
        for row in rows:row['metrics']['ood_val']['auc']=np.float64(row['metrics']['ood_val']['auc'])
        json.dumps(summarize(rows,False),allow_nan=False)

    def test_incomplete_batch_is_rejected_before_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);rows=self.entries(root)
            with patch('src.experiments.ki_reference_controls.evaluate_saved') as evaluate:
                with self.assertRaisesRegex(ValueError,'Incomplete'):
                    evaluate_batch(rows[:-1],root,self.spec,root,'cpu',smoke=True,test=True)
                evaluate.assert_not_called()

    def test_adapter_dataset_identity_and_reference_command(self):
        # Parse just the pure identity function; avoid importing the upstream namespace.
        import ast
        path=ROOT/'baselines/sugar/adapters/run.py'
        tree=ast.parse(path.read_text())
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='dataset_identity')
        datasets=tuple('drugood_lbap_core_'+e+'_'+s for e in ('ec50','ki') for s in ('assay','scaffold','size'))
        scope=dict(Path=Path,DATASETS=datasets)
        exec(compile(ast.Module(body=[function],type_ignores=[]),str(path),'exec'),scope)
        identify=scope['dataset_identity']
        self.assertEqual(identify('lbap_core_ki_size.json'),'drugood_lbap_core_ki_size')
        self.assertEqual(identify(None),'drugood_lbap_core_ec50_assay')
        with self.assertRaisesRegex(ValueError,'disagrees'):
            identify('lbap_core_ki_size.json','drugood_lbap_core_ec50_assay')
        command=reference_command(Path('lbap_core_ki_size.json'),Path('out'),'cpu',10900,3,self.spec,False)
        self.assertEqual(command[command.index('--dataset')+1],'drugood_lbap_core_ki_size')
        self.assertEqual(command[command.index('--views')+1],'10')
        self.assertNotIn('--test-protocol',command)
        explicit=reference_command(Path('lbap_core_ki_size.json'),Path('out'),'cpu',10900,3,self.spec,False,
            interpreter='/project/.venv-sugar/bin/python')
        self.assertEqual(explicit[0],'/project/.venv-sugar/bin/python')

    def test_missing_reference_dependencies_block_all_training(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'pilot'
            with patch('src.experiments.ki_reference_controls.reference_preflight',side_effect=RuntimeError('missing ogb')), \
                 patch('src.experiments.ki_reference_controls.run_seed') as train, \
                 patch('src.experiments.ki_reference_controls.execute') as upstream, \
                 patch('src.experiments.ki_reference_controls.verify_inputs') as audit:
                with self.assertRaisesRegex(RuntimeError,'missing ogb'):
                    main(['--project-root',directory,'--output-dir',str(out),'--stage','pilot',
                          '--reference-python','/project/.venv-reference/bin/python','--execute'])
                train.assert_not_called();upstream.assert_not_called();audit.assert_not_called()
            status=json.loads((out/'status.json').read_text())
            self.assertEqual(status['trained'],0)
            self.assertEqual(status['stage'],'failed')

    def test_preflight_uses_the_selected_interpreter_without_training(self):
        import subprocess
        result=subprocess.CompletedProcess([],0,stdout=json.dumps({'python':'chosen','training':False}),stderr='')
        with patch('src.experiments.ki_reference_controls.subprocess.run',return_value=result) as run:
            report=reference_preflight('/project/.venv-sugar/bin/python','cpu')
        self.assertFalse(report['training'])
        self.assertEqual(run.call_args.args[0][0],'/project/.venv-sugar/bin/python')

    def test_reference_validation_uses_the_same_environment_as_training(self):
        from src.experiments.ki_reference_controls import reference_validation
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run=root/'reference';(run/'validation').mkdir(parents=True)
            validation=dict(test_evaluated=False,checkpoint_sha256='fixed',
                provenance={'source':{'raw_sha256':'raw'}},
                variants={'all_mean':{'views':[0,1],'auc':.7}},validation_retention={})
            (run/'validation/completed.json').write_text(json.dumps(validation))
            entry=dict(run_dir=str(run),checkpoint='weights.pt',checkpoint_sha256='fixed',
                       raw_path='raw.json',reference_python='/project/.venv-sugar/bin/python')
            with patch('src.experiments.ki_reference_controls.execute') as execute:
                reference_validation(entry,'cpu',3)
            self.assertEqual(execute.call_args.args[0][0],entry['reference_python'])


if __name__=='__main__':unittest.main()
