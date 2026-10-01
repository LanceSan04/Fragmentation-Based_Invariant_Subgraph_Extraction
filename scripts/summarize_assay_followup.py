"""Create a readable report from the completed, frozen assay follow-up."""
import argparse
import json
from pathlib import Path
import statistics as st


def read(path):
    return json.loads(path.read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results',type=Path)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    root=args.results.resolve()
    if not read(root/'complete.json')['complete']: raise ValueError('Incomplete study')
    synthetic=read(root/'frozen_plan.json')['synthetic']
    report=read(root/'summary.json')
    val=read(root/'validation_summary.json')
    rows=report['per_seed']
    seeds=sorted({r['seed'] for r in rows})
    arms=['pooled','off','per_view']
    def values(arm,policy):
        return [next(r['auc'] for r in rows if r['arm']==arm and r['selected_by']==policy and r['seed']==s) for s in seeds]
    lines=['# Assay follow-up results — 2026-09-26','',
           'All four recommendations were executed as a frozen-checkpoint study. No new training or hyperparameter search was performed. The ordered pooled-contrastive model remains the declared reference. Agent/skill files were not changed.','',
           '## Test ROC-AUC','',
           'Primary: independent checkpoint selection and independent inference. Secondary: shared checkpoint selection and independent inference. These are different saved epochs from the same training trajectories. All predeclared seeds were included.','',
           '| Seed | Pooled 0.1 | Off | Per-view 0.1 |','|---|---:|---:|---:|']
    for i,seed in enumerate(seeds):lines.append('| '+str(seed)+' | '+' | '.join(f'{values(a,"independent")[i]:.4f}' for a in arms)+' |')
    lines.extend(['','| Arm | Primary test mean ± sample SD | Secondary test mean ± sample SD |','|---|---:|---:|'])
    for a in arms:
        cells=[]
        for policy in ['independent','shared']:
            v=values(a,policy);cells.append(f'{st.mean(v):.4f} ± {st.stdev(v):.4f}' if len(v)>1 else f'{v[0]:.4f}')
        lines.append('| '+a+' | '+' | '.join(cells)+' |')
    lines.extend(['','Paired differences compare the same seed. Positive means the alternative is higher than pooled:',''])
    for a in ['off','per_view']:
        delta=[x-y for x,y in zip(values(a,'independent'),values('pooled','independent'))]
        lines.append(f'- {a}: mean difference {st.mean(delta):+.4f}; higher on {sum(x>0 for x in delta)}/{len(delta)} seeds. Individual differences: '+', '.join(f'{d:+.4f}' for d in delta)+'.')
    lines.extend(['','Secondary per-seed test AUC (shared selection, independent inference):','',
                  '| Seed | Pooled 0.1 | Off | Per-view 0.1 |','|---|---:|---:|---:|'])
    for i,seed in enumerate(seeds):
        lines.append('| '+str(seed)+' | '+' | '.join(f'{values(a,"shared")[i]:.4f}' for a in arms)+' |')
    lines.extend(['','## Validation and within-assay diagnostics','',
                  'An assay is one experimental measurement context. Within-assay AUC compares positive and negative molecules with the same assay ID. Overall AUC also includes comparisons across assays. Macro gives each usable assay equal weight; pair-weighted gives more weight to assays with more positive-negative pairs. Single-class assays cannot supply within-assay AUC.','',
                  '| Arm | Overall validation | Macro within-assay | Pair-weighted within-assay |','|---|---:|---:|---:|'])
    for a in arms:
        rs=[r for r in val if r['arm']==a and r['selected_by']=='independent']
        lines.append('| '+a+' | '+' | '.join(f'{st.mean(r[k] for r in rs):.4f}' for k in ['auc','macro','within'])+' |')
    lines.extend(['','## What each view retains','',
                  'Descriptive sample: 100 validation molecules per checkpoint, evenly spaced by original atom-count rank, using the same molecules across arms and seeds. This is not a random dataset sample. Jaccard is intersection divided by union; 1 means identical selections.','',
                  '| Arm | Atoms retained | Bonds retained | Components | Atom Jaccard | Bond Jaccard |','|---|---:|---:|---:|---:|---:|'])
    for a in arms:
        ds=[read(f) for f in (root/'retention').glob(a+'_*/eval_edge0.5.json')]
        means=[st.mean(d['summary'][k]['mean'] for d in ds) for k in ['atom_fraction','bond_fraction','components']]
        overlaps=[st.mean(d['overlap'][k]['mean'] for d in ds) for k in ['atom_jaccard','bond_jaccard']]
        lines.append(f'| {a} | {means[0]:.1%} | {means[1]:.1%} | {means[2]:.2f} | {overlaps[0]:.3f} | {overlaps[1]:.3f} |')
    lines.extend(['','## Pooled-reference ratio sensitivity','',
                  'Frozen weights only. Normal evaluation uses all fragment edges as candidates; training sample ratio does not filter them. Changing an inference ratio does not establish how a model trained with that ratio would perform.','',
                  '| Edge ratio | Atoms retained | Bonds retained | Components | Atom overlap |','|---|---:|---:|---:|---:|'])
    for edge in [.3,.5,.7]:
        ds=[read(f) for f in (root/'retention').glob(f'pooled_*/eval_edge{edge}.json')]
        m=[st.mean(d['summary'][k]['mean'] for d in ds) for k in ['atom_fraction','bond_fraction','components']]
        j=st.mean(d['overlap']['atom_jaccard']['mean'] for d in ds)
        lines.append(f'| {edge} | {m[0]:.1%} | {m[1]:.1%} | {m[2]:.2f} | {j:.3f} |')
    lines.extend(['','At the original edge ratio 0.5:','',
                  '| Original molecule size | Atoms retained | Bonds retained | Components |','|---|---:|---:|---:|'])
    ds=[read(f) for f in (root/'retention').glob('pooled_*/eval_edge0.5.json')]
    for group in ['small_1_25','medium_26_50','large_51_plus']:
        present=[d['by_size'][group] for d in ds if group in d['by_size']]
        if not present:continue
        m=[st.mean(d[k]['mean'] for d in present) for k in ['atom_fraction','bond_fraction','components']]
        lines.append(f'| {group} | {m[0]:.1%} | {m[1]:.1%} | {m[2]:.2f} |')
    original=root/'original_connectivity_check.json'
    if original.exists():
        lines.extend(['','Original connectivity was checked locally against the identical SHA256-verified validation cache:',''])
        for group,g in read(original)['groups'].items():
            lines.append(f"- {group}: {g['unique_molecules']} distinct molecules, {g['original_atoms_min']}–{g['original_atoms_max']} atoms; original components mean {g['original_components_mean']:.2f}, selected mean {g['selected_components_mean']:.2f}, selected median {g['selected_components_median']:.1f}. Counts are repeated over five seeds and five views, not independent molecule samples.")
    lines.extend(['','Sampling-only probes use 40 size-stratified validation molecules and two fixed draws per checkpoint. Only the sampler is in training mode; dropout stays off. They are not training runs or standard evaluation predictions.','',
                  '| Edge ratio | Sample ratio | Atoms retained | Bonds retained | Selected/full connections |','|---|---|---:|---:|---:|'])
    for edge in [.3,.5,.7]:
        for sample in [.5,.7,.9]:
            ds=[read(f) for f in (root/'retention').glob(f'pooled_*/sample_edge{edge}_keep{sample}_draw*.json')]
            m=[st.mean(d['summary'][k]['mean'] for d in ds) for k in ['atom_fraction','bond_fraction','selected_connection_fraction']]
            lines.append(f'| {edge} | {sample} | {m[0]:.1%} | {m[1]:.1%} | {m[2]:.1%} |')
    lines.extend(['','## Interpretation limits and next decision','',
                  '- No test results were used to pick checkpoints, seeds or ratios for this batch. All validation parity checks passed before test predictions. The pre-test decision is stored in decision_before_test.json.',
                  '- The assay test split was inspected during earlier project work. This is an exploratory comparison, not a new untouched holdout.',
                  '- Five seeds give a limited view of training variability. They do not quantify uncertainty across new assays/datasets.',
                  '- Per-view contrastive changes both which embeddings are compared and the scale of the loss. A performance difference cannot isolate those mechanisms.',
                  '- Retention and overlap describe geometry, not chemical usefulness or causal explanations. Sampler-only probes do not simulate retraining.',
                  '- The 40-molecule sampling probes and 100-molecule evaluation probes use different rank grids. Do not interpret their aggregate difference as a paired training/evaluation gap.',
                  '- Preserve these results. Any later retention-controlled training comparison needs a separately frozen validation protocol; do not chase the best observed test setting.','',
                  '## Files','',f'- Full local results: `{root}`',
                  '- `frozen_plan.json`: every checkpoint/config/hash and fixed comparison.',
                  '- `validation_summary.json` and `validation_*.json`: validation AUC and per-assay/prediction data.',
                  '- `retention/<arm>_<seed>/*.json`: original atom/bond IDs, counts, size bins, overlap and ratio probes.',
                  '- `decision_before_test.json`: timestamped validation-only decision.',
                  '- `summary.json` and `test_*.json`: final test AUCs and individual predictions.',
                  '- `../launch.json`, `../console.log`, `../source-manifest.json`: server execution and source provenance.',''])
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text('\n'.join(lines),encoding='utf-8')
    print(args.report)


if __name__=='__main__':main()
