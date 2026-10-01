"""Replay frozen checkpoints on validation and decompose ROC-AUC by assay."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from train_sugar import Config,build_views_from_feature_dims,strip_non_tensor_batch_fields
from src.data.dataloader import build_split_dataloaders
from src.training.checkpoint import load_checkpoint,load_view_state_dicts
from src.models.sugar_ensemble import SuGAREnsemble


def group_report(rows):
    groups={}
    for r in rows:groups.setdefault(r['assay'],[]).append(r)
    def auc(g):return float(roc_auc_score([r['label'] for r in g],[r['probability'] for r in g]))
    report={};within_pairs=0;within_wins=0.;valid=[]
    for k,g in groups.items():
        pos=sum(r['label'] for r in g);neg=len(g)-pos
        score=auc(g) if pos and neg else None
        report[k]=dict(n=len(g),positive=pos,negative=neg,auc=score)
        if score is not None:
            valid.append(score);within_pairs+=pos*neg;within_wins+=score*pos*neg
    pos=sum(r['label'] for r in rows);total_pairs=pos*(len(rows)-pos)
    pooled=auc(rows) if total_pairs else None
    cross_pairs=total_pairs-within_pairs
    return dict(pooled_auc=pooled,macro_assay_auc=float(np.mean(valid)) if valid else None,
                within_pair_weighted_auc=within_wins/within_pairs if within_pairs else None,
                within_pair_fraction=within_pairs/total_pairs if total_pairs else None,
                cross_assay_auc=(pooled*total_pairs-within_wins)/cross_pairs if cross_pairs else None,
                usable_assays=len(valid),excluded_single_class_assays=len(groups)-len(valid),assays=report)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoints',nargs='+');p.add_argument('--processed-dir',required=True);p.add_argument('--output-dir',required=True)
    args=p.parse_args();out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(17)
    for checkpoint in args.checkpoints:
        path=Path(checkpoint);payload=load_checkpoint(path)
        config=Config(dict(payload['config'],device='cpu',processed_dir=str(Path(args.processed_dir).resolve()),num_workers=0,loader_split_names=['ood_val']))
        loaders=build_split_dataloaders(config.processed_dir,config)
        assays=[str(d.metadata['assay_id']) for d in loaders['ood_val'].dataset]
        strip_non_tensor_batch_fields(loaders)
        views=build_views_from_feature_dims(config,payload['feature_dims']);load_view_state_dicts(views,payload)
        model=SuGAREnsemble(views,aggregation_mode=config.aggregation_mode,inference_policy=config.inference_policy);model.eval()
        rows=[]
        with torch.no_grad():
            for batch in loaders['ood_val']:
                prob=model(batch).exp()[:,1].tolist()
                for label,value in zip(batch.y.reshape(-1).tolist(),prob):
                    rows.append(dict(index=len(rows),assay=assays[len(rows)],label=int(label),probability=value))
        report=group_report(rows)
        report.update(checkpoint=str(path),checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),effective_config=config.to_dict(),molecules=rows)
        name=path.parent.parent.name+'_'+path.parent.name+'_'+path.stem
        (out/(name+'.json')).write_text(json.dumps(report,indent=2))
        print(name,{k:report[k] for k in ['pooled_auc','macro_assay_auc','within_pair_weighted_auc','within_pair_fraction']},flush=True)


if __name__=='__main__':main()
