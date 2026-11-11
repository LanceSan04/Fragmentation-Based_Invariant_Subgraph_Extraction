from copy import deepcopy
from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest

from src.experiments.checkpoints import digest
from src.experiments.fragment_tuning import DATASETS, FIXED, build_config, contained, main, validate


class FragmentTuningTests(unittest.TestCase):
    def setUp(self):
        self.spec=json.loads(Path('configs/fragment_tuning_ec50_ki_20261018.json').read_text())
        self.reference=json.loads(Path('outputs/cross_domain/mentor_transfer_20261016_v1/results/plan.json').read_text()) if Path('outputs/cross_domain/mentor_transfer_20261016_v1/results/plan.json').is_file() else {'configs':{'ec50_scaffold':{}}}

    def test_all_six_configs_use_pinned_sources_and_same_recipe(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for dataset in DATASETS:
                base=build_config(self.spec,self.reference,dataset,root,'cpu')
                self.assertEqual(base['dataset'],'drugood_lbap_core_'+dataset)
                self.assertEqual(base['epochs'],150)
                self.assertEqual({k:base[k] for k in FIXED},FIXED)
                self.assertEqual(Path(base['raw_path']),root/'data/raw/drugood'/f'lbap_core_{dataset}.json')
                self.assertEqual(base['loader_split_names'],['train','ood_val'])
                self.assertIsNone(base['test_split'])
                self.assertFalse(base['report_splits'])
                self.assertFalse(base['early_stopping'])
                self.assertFalse(base['compare_checkpoint_rules'])

    def test_bounded_ki_scope_coverage_and_parent_integrity(self):
        spec=json.loads(Path('configs/ki_bounded_search_20261019.json').read_text())
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'reference.json';p.write_text('{}');spec['reference_plan_sha256']=digest(p)
            validate(spec,p)
            self.assertEqual(len(spec['recipes'])*len(spec['seeds'])*len(spec['datasets']),36)
            self.assertEqual(spec['recipe_ids'],[1,2,3,4,5,12])
            for col in [0,2,3]:self.assertEqual(set(Counter(r[col] for r in spec['recipes']).values()),{2})
            self.assertEqual({r[1] for r in spec['recipes']},{.3,.5,.7,.9})
            bad=deepcopy(spec);bad['recipes'][0][1]=.3
            with self.assertRaises(ValueError):validate(bad,p)
            bad=deepcopy(spec);bad['recipe_ids'][-1]=6
            with self.assertRaises(ValueError):validate(bad,p)
            bad=deepcopy(spec);bad['parent_config_sha256']='0'*64
            with self.assertRaises(ValueError):validate(bad,p)

    def test_bounded_ki_dry_run_preserves_recipe_ids_and_no_test(self):
        spec=json.loads(Path('configs/ki_bounded_search_20261019.json').read_text())
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);ref=root/'reference.json';ref.write_text('{"configs":{"ec50_scaffold":{}}}')
            spec['reference_plan_sha256']=digest(ref)
            config=root/'ki.json';config.write_text(json.dumps(spec));out=root/'not_created'
            from contextlib import redirect_stdout
            from io import StringIO
            with redirect_stdout(StringIO()):
                plan=main(['--config',str(config),'--reference-plan',str(ref),'--project-root',str(root),
                           '--datasets','ki_assay','ki_scaffold','ki_size','--output-dir',str(out)])
            self.assertEqual(plan['training_runs'],36)
            self.assertEqual(plan['recipe_ids'],[1,2,3,4,5,12])
            self.assertFalse(plan['test_evaluated'])
            self.assertFalse(out.exists())

    def test_validation_rejects_budget_policy_and_source_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            reference=Path(directory)/'reference.json';reference.write_text('{}')
            spec=deepcopy(self.spec);spec['reference_plan_sha256']=digest(reference)
            validate(spec,reference)
            self.assertEqual(len(spec['recipes'])*len(spec['seeds']),24)
            bad=deepcopy(spec);bad['epochs']=100
            with self.assertRaises(ValueError):validate(bad,reference)
            bad=deepcopy(spec);bad['fixed']['inference_policy']='shared'
            with self.assertRaises(ValueError):validate(bad,reference)
            reference.write_text('{"changed":true}')
            with self.assertRaises(ValueError):validate(spec,reference)

    def test_data_path_cannot_escape_project(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):contained(Path(directory),'../escape')

    def test_dry_run_does_not_create_output_or_require_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);reference=root/'reference.json'
            reference.write_text(json.dumps({'configs':{'ec50_scaffold':{}}}))
            spec=deepcopy(self.spec);spec['reference_plan_sha256']=digest(reference)
            config=root/'config.json';config.write_text(json.dumps(spec));out=root/'not_created'
            args=['--config',str(config),'--reference-plan',str(reference),'--project-root',str(root),
                  '--datasets','ec50_assay','ec50_scaffold','ec50_size','--output-dir',str(out)]
            from contextlib import redirect_stdout
            from io import StringIO
            with redirect_stdout(StringIO()):plan=main(args)
            self.assertEqual(plan['training_runs'],72)
            self.assertEqual(plan['epochs'],150)
            self.assertFalse(out.exists())
            with redirect_stdout(StringIO()):pilot=main(args+['--stage','pilot'])
            self.assertEqual(pilot['training_runs'],6)
            self.assertEqual(pilot['epochs'],10)
            with redirect_stdout(StringIO()):prep=main(args+['--stage','prepare'])
            self.assertEqual(prep['training_runs'],0)

    def test_raw_mismatch_stops_before_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);reference=root/'reference.json'
            reference.write_text(json.dumps({'configs':{'ec50_scaffold':{}}}))
            spec=deepcopy(self.spec);spec['reference_plan_sha256']=digest(reference)
            raw=contained(root,spec['datasets']['ki_assay']['raw_path'])
            raw.parent.mkdir(parents=True);raw.write_text('{}')
            config=root/'config.json';config.write_text(json.dumps(spec));out=root/'not_created'
            with self.assertRaisesRegex(ValueError,'Pinned raw source changed'):
                main(['--config',str(config),'--reference-plan',str(reference),'--project-root',str(root),
                      '--datasets','ki_assay','--stage','prepare','--output-dir',str(out),'--execute'])
            self.assertFalse(out.exists())
