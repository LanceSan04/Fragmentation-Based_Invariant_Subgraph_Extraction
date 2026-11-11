"""Frozen validation-only contextual-feature pilot or full comparison; no continuation."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import torch

from src.experiments.checkpoints import digest, evaluate_saved, require_parity
from src.experiments.cross_domain import verify_cache, prepare_cache
from src.experiments.fragment_tuning import contained
from src.experiments.plans import ROOT, make_plan
from src.experiments.selection import run_seed
from src.experiments.sugar_reference_search import write


def validate(spec):
    family = spec['datasets'][0]['dataset'].split('_')[0]
    scope = (family, spec['epochs'], tuple(spec['seeds']))
    allowed = {('ec50', 5, (11001,)),
               ('ec50', 150, (11011,11012,11013,11014,11015)),
               ('ec50', 5, (11201,)),
               ('ec50', 150, (11211,11212,11213,11214,11215)),
               ('ic50', 5, (11101,)),
               ('ic50', 150, (11111,11112,11113,11114,11115))}
    if (spec['schema'] != 1 or scope not in allowed
            or spec['arms'] != ['scorer_only', 'scorer_and_classifier']
            or [r['dataset'] for r in spec['datasets']] != [family+'_'+s for s in ('assay','scaffold','size')]
            or spec['base']['fragment_feature_mode'] != 'molecular_gnn'
            or spec['base']['selection_estimator'] != 'hard'
            or spec['base']['inference_policy'] != 'independent'):
        raise ValueError('Comparison scope changed; prepare a new named protocol')
    if spec['seeds'][0] in (11201,11211):
        if (spec['base'].get('dataset_version')!='drugood_official_chembl30_20261028'
                or any(not r['raw_path'].startswith('data/raw/drugood_official_chembl30_20261028/')
                       for r in spec['datasets'])):
            raise ValueError('Official EC50 protocol requires separately versioned ChEMBL30 inputs')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default=str(ROOT/'configs/context_features_pilot_20261028.json'))
    p.add_argument('--project-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--evaluate-test', action='store_true',
                   help='After every trajectory finishes, replay all frozen checkpoints then evaluate test')
    p.add_argument('--stage', choices=['prepare', 'train'], default='train',
                   help='Prepare new versioned caches only, or train using verified existing caches')
    p.add_argument('--smoke', action='store_true', help='Synthetic CPU fixture, two epochs; not real-data evidence')
    a = p.parse_args(argv)
    spec = json.loads(Path(a.config).read_text()); validate(spec)
    if a.evaluate_test and (a.stage!='train' or (spec['epochs']!=150 and not a.smoke)):
        p.error('Test evaluation requires full training protocol, or an explicit synthetic smoke')
    if a.smoke and a.stage == 'prepare':
        p.error('Cache preparation stage is for real pinned datasets; use ordinary --smoke for synthetic checks')
    if a.smoke and spec['epochs'] != 5:
        p.error('Synthetic smoke uses the pilot protocol, not the full 30-run specification')
    project = a.project_root.resolve()
    epochs = 2 if a.smoke else spec['epochs']
    plan = dict(protocol=spec, config_sha256=digest(a.config),
        stage=a.stage,
        training_runs=0 if a.stage=='prepare' else len(spec['datasets'])*len(spec['arms'])*len(spec['seeds']),
        epochs=epochs, synthetic=a.smoke, device=a.device, test_evaluated=False,
        primary_rule='best_raw_pooled_validation_auc', inference_policy='independent',
        automatic_continuation=False, checkpoint_replay_tolerance=1e-4)
    plan['test_planned']=a.evaluate_test
    if not a.execute:
        print(json.dumps(plan, indent=2)); return plan
    # Training never silently preprocesses; cache preparation is an explicit separate stage.
    configs = []
    for entry in spec['datasets']:
        cfg = deepcopy(spec['base']); cfg.update(entry['parameters'])
        cfg.update(dataset='drugood_lbap_core_'+entry['dataset'], device=a.device,
            epochs=epochs, raw_path=str(contained(project, entry['raw_path'])),
            processed_dir=str(contained(project, entry['cache_path'])),
            test_split=None, report_splits=[], loader_split_names=['train','ood_val'],
            log_test_curve=False, compare_checkpoint_rules=False, num_workers=0 if a.smoke else 4)
        if not a.smoke:
            if digest(cfg['raw_path']) != entry['raw_sha256']:
                raise ValueError('Raw source changed: '+entry['dataset'])
            if a.stage == 'train' or Path(cfg['processed_dir']).exists():
                verify_cache(cfg)
        configs.append((entry['dataset'], cfg))
    out = a.output_dir.resolve(); out.mkdir(parents=True, exist_ok=False)
    write(out/'plan.json', plan)
    rows = []
    try:
        if a.stage == 'prepare':
            for dataset, cfg in configs:
                cache = Path(cfg['processed_dir'])
                manifest = verify_cache(cfg) if cache.exists() else prepare_cache(cfg)
                rows.append(dict(dataset=dataset, status='prepared', cache=str(cache),
                                 manifest=manifest))
                write(out/'status.json', rows)
            report = dict(plan=plan, datasets=rows, training_runs=0,
                          test_evaluated=False, automatic_continuation=False)
            write(out/'completed.json', report)
            return report
        for dataset, base in configs:
            if a.smoke:
                from src.data.preprocess import run_preprocessing
                from src.training.pipeline import Config
                torch.set_num_threads(1)
                _, fixture = make_plan(out/dataset/'fixture', a.device, True)
                base.update(fixture)
                base.update(fragment_feature_mode='molecular_gnn', fragment_encoder_layers=2,
                    num_views=2, view_fragmentations={'group_graph':2}, hidden_dim=16,
                    gnn_layers=1, epochs=epochs, num_workers=0, report_within_assay=False,
                    split_names=['train','ood_val'], loader_split_names=['train','ood_val'],
                    compare_checkpoint_rules=False, test_split=None, report_splits=[])
                if a.evaluate_test:
                    base['split_names']=['train','ood_val','ood_test']
                else:
                    base['splits'].pop('ood_test', None)
                run_preprocessing(Config(base))
            for seed in spec['seeds']:
                for arm in spec['arms']:
                    values = deepcopy(base)
                    values['classifier_feature_mode'] = 'raw' if arm=='scorer_only' else 'encoder'
                    run = out/dataset/arm/f'seed{seed}'; run.parent.mkdir(parents=True, exist_ok=True)
                    row = dict(dataset=dataset, arm=arm, seed=seed, status='running')
                    rows.append(row); write(out/'status.json', rows)
                    result = run_seed(values, run, seed, False, ('independent',))
                    checkpoint = run/'best_independent.pt'
                    before = digest(checkpoint)
                    replay = evaluate_saved(checkpoint, 'ood_val', Path(values['processed_dir']),
                        a.device, 'independent', seed, prediction_path=run/'validation_predictions.npz')
                    require_parity(replay['auc'], result['best']['independent']['auc'])
                    if digest(checkpoint) != before: raise ValueError('Checkpoint changed during replay')
                    row.update(status='completed', validation=replay,
                        checkpoint=str(checkpoint), checkpoint_sha256=before,
                        selected_epoch=result['best']['independent']['epoch'])
                    write(out/'status.json', rows)
        report = dict(plan=plan, runs=rows, test_evaluated=False,
            interpretation=('Feasibility pilot only. No performance claim or automatic larger study.'
                if spec['epochs']==5 else
                'Fixed paired comparison. Report every seed; validation only, no automatic test evaluation or follow-up.'))
        if a.evaluate_test:
            from src.experiments.context_test import evaluate_frozen
            test=evaluate_frozen(rows,{dataset:cfg['processed_dir'] for dataset,cfg in configs},
                                 out/'test_evaluation',a.device)
            report.update(test_evaluated=True,test_summary=test['summary'])
            report['interpretation']='Frozen validation-selected checkpoints; every paired validation/test result reported without test-based model selection.'
        write(out/'completed.json', report); return report
    except Exception as error:
        if rows and rows[-1]['status'] == 'running':
            rows[-1].update(status='failed', error=str(error))
            write(out/'status.json', rows)
        write(out/'failed.json', dict(error=str(error), runs=rows, test_planned=a.evaluate_test))
        raise


if __name__ == '__main__':
    main()
