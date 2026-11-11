"""Frozen validation-only scorer-context intervention; no training or AUC."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch
import statistics
import torch
from torch_geometric.data import Batch
from src.experiments.checkpoints import restore, digest
from src.data.dataloader import load_processed_split, _expected_metadata
from src.analysis.retention import stratified_indices
from src.models.fragment_graph import build_fragment_graph_batch
from src.models.fragment_edge_sampler import random_fragment_edge_mask, filter_fragment_graph_edges


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--samples', type=int, default=25)
    args = parser.parse_args(argv)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    records = json.loads(Path(args.inputs).read_text())
    rows = []
    for record in records:
        assert digest(record['checkpoint']) == record['expected_sha256']
        config, views = restore(record['checkpoint'], record['cache'], 'cpu')
        if any(v.fragment_feature_mode != "static_pool" for v in views):
            raise ValueError("This legacy context intervention supports static fragment features only")
        data = load_processed_split(Path(record['cache'])/'ood_val.pt', _expected_metadata(config))['molecule_data']
        for index in stratified_indices(data, args.samples):
            molecule = data[index].clone()
            for key in ('metadata','smiles','molecule_id','fragment_to_atoms'):
                if key in molecule: del molecule[key]
            batch = Batch.from_data_list([molecule], follow_batch=config.follow_batch)
            fragments = build_fragment_graph_batch(batch)
            edges = fragments.edge_index.size(1)
            for number, view in enumerate(views):
                eligible = random_fragment_edge_mask(fragments.batch[fragments.edge_index[0]],
                    .7, True, seed=record['seed']*100000 + index*10 + number)
                sampled = filter_fragment_graph_edges(fragments, eligible)
                with torch.no_grad():
                    partial, _ = view.fragment_masker(sampled)
                    full, _ = view.fragment_masker(fragments)
                restored = torch.zeros_like(full)
                restored[sampled.full_fragment_edge_id] = partial
                scores = {'sampled': restored, 'full_masked': full*eligible, 'full': full}
                for ranking in ('learned','random'):
                    view.ranking_mode = ranking
                    traces, probabilities = {}, {}
                    for context, values in scores.items():
                        with torch.no_grad(), patch.object(view,'score_fragment_edges', return_value=(values,eligible)):
                            prediction, trace = view(batch, return_intermediates=True, fragment_batch=fragments)
                        traces[context] = trace
                        probabilities[context] = float(prediction.exp()[0,1])
                    for context in ('full_masked','full'):
                        first = traces['sampled']['selected_fragment_mask']
                        second = traces[context]['selected_fragment_mask']
                        if ranking == 'random': assert torch.equal(first,second)
                        union = (first|second).sum().item()
                        rows.append(dict(dataset=record['dataset'],seed=record['seed'],index=index,view=number,
                            ranking=ranking,context=context,full_edges=edges,eligible_edges=int(eligible.sum()),
                            score_mean_abs_delta=float((full[eligible]-restored[eligible]).abs().mean()) if eligible.any() else None,
                            mask_jaccard=float((first&second).sum())/union if union else 1.,
                            positive_probability_delta=probabilities[context]-probabilities['sampled'],
                            sampled_atom_fraction=traces['sampled']['selected_atom_ids'].numel()/batch.x.size(0),
                            alternative_atom_fraction=traces[context]['selected_atom_ids'].numel()/batch.x.size(0)))
        print(record['dataset'], record['seed'], 'complete', flush=True)
    summary = {}
    for dataset in sorted({r['dataset'] for r in rows}):
        summary[dataset] = {}
        for ranking in ('learned','random'):
            summary[dataset][ranking] = {}
            for context in ('full_masked','full'):
                subset = [r for r in rows if r['dataset']==dataset and r['ranking']==ranking and r['context']==context]
                summary[dataset][ranking][context] = dict(n=len(subset),
                    mean_mask_jaccard=statistics.mean(r['mask_jaccard'] for r in subset),
                    mean_absolute_probability_change=statistics.mean(abs(r['positive_probability_delta']) for r in subset),
                    mean_signed_probability_change=statistics.mean(r['positive_probability_delta'] for r in subset),
                    changed_masks=sum(r['mask_jaccard']<1 for r in subset))
    (out/'results.json').write_text(json.dumps(dict(summary=summary,rows=rows,
        note='Frozen older hard checkpoints, validation only, fixed 70% eligibility; dropout/BN eval. No AUC, training, or proof of improved performance. Full_masked isolates encoder context; full also retains ineligible scores for internal gates.'),indent=2))
    print(json.dumps(summary),flush=True)


if __name__=='__main__': main()
