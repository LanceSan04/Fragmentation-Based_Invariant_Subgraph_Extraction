"""Compare saved views on validation only; no training or checkpoint writes."""
import argparse
from datetime import datetime
from itertools import combinations
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from train_sugar import Config, build_views_from_feature_dims, strip_non_tensor_batch_fields
from src.training.checkpoint import load_checkpoint, load_view_state_dicts
from src.data.dataloader import build_split_dataloaders
from src.models.sugar_ensemble import SuGAREnsemble
from src.models.fragment_ciga import topk_mask_over_eligible


from src.experiments.metrics import correlation, metrics


def compare(checkpoint, processed_dir=None, device="cpu", max_batches=None):
    payload = load_checkpoint(checkpoint)
    values = dict(payload["config"])
    split = values["validation_split"]
    if not split or "test" in split.lower():
        raise ValueError("Checkpoint must identify a validation split, not test")
    values.update(loader_split_names=[split], test_split=None, report_splits=[], num_workers=0)
    if processed_dir: values["processed_dir"] = str(processed_dir)
    cfg = Config(values)
    views = build_views_from_feature_dims(cfg, payload["feature_dims"])
    load_view_state_dicts(views,payload)
    if len(views)<2 or any(v.erm_baseline for v in views) or len({v.fragmentation_method for v in views})!=1:
        raise ValueError("This report requires >=2 non-ERM views of one fragmentation method")
    for v in views: v.to(device).eval()
    loaders=build_split_dataloaders(cfg.processed_dir,cfg)
    strip_non_tensor_batch_fields(loaders)
    ensemble=SuGAREnsemble(views,aggregation_mode=cfg.get("aggregation_mode","mean"))
    chunks={}; labels=[]; overlaps={pair:[] for pair in combinations(range(len(views)),2)}
    score_correlations={pair:[] for pair in overlaps}
    def add(key,value): chunks.setdefault(key,[]).append(value.detach().cpu().numpy())
    with torch.no_grad():
        for index,batch in enumerate(loaders[split]):
            if max_batches is not None and index>=max_batches: break
            batch=batch.to(device); labels.extend(batch.y.reshape(-1).cpu().tolist())
            own=[]; scores=[]; masks=[]; edge_batch=None
            for i,v in enumerate(views):
                pred,pv=v(batch,return_intermediates=True)
                own.append(pred.exp()); add(f"own_view_{i}",pred.exp())
                scores.append(pv["fragment_scores"])
                edge_batch=pv["fragment_edge_batch"]
                masks.append(topk_mask_over_eligible(scores[-1],edge_batch,v.fragment_masker.ratio,pv["fragment_edge_eligible"]))
            for pair in overlaps:
                a,b=pair
                for graph in range(batch.num_graphs):
                    loc=edge_batch==graph
                    union=(masks[a][loc]|masks[b][loc]).sum().item()
                    if union: overlaps[pair].append(float((masks[a][loc]&masks[b][loc]).sum())/union)
                    corr=correlation(scores[a][loc].cpu().numpy(),scores[b][loc].cpu().numpy())
                    if corr is not None: score_correlations[pair].append(corr)
            shared=[]
            handles=[v.atom_classifier.register_forward_hook(lambda m,a,result: shared.append(result.softmax(-1))) for v in views]
            try: prediction=ensemble(batch).exp()
            finally:
                for h in handles:h.remove()
            assert torch.allclose(torch.stack(shared).mean(0),prediction,atol=1e-6)
            add("full_ensemble",prediction); add("independent_graph_vote",torch.stack(own).mean(0))
            for i in range(len(views)):
                add(f"shared_view_{i}",shared[i])
                add(f"drop_vote_{i}",torch.stack([p for j,p in enumerate(shared) if i!=j]).mean(0))
                add(f"drop_entire_view_{i}",ensemble(batch,view_indices=[j for j in range(len(views)) if j!=i]).exp())
            print(f"Compared validation batch {index+1}",flush=True)
    if not labels: raise ValueError("No validation examples")
    y=np.asarray(labels); arrays={k:np.concatenate(v) for k,v in chunks.items()}
    pairs=[]
    for (a,b),overlap in overlaps.items():
        entry=dict(a=a,b=b,edge_jaccard_mean=float(np.mean(overlap)) if overlap else None,
                   edge_jaccard_n=len(overlap),score_correlation_mean=float(np.mean(score_correlations[a,b])) if score_correlations[a,b] else None,
                   score_correlation_n=len(score_correlations[a,b]))
        for mode in ["own","shared"]:
            pa,pb=arrays[f"{mode}_view_{a}"],arrays[f"{mode}_view_{b}"]
            entry[mode]=dict(probability_correlation=correlation(pa[:,1],pb[:,1]),
                disagreement=float(np.mean(pa.argmax(1)!=pb.argmax(1))),
                both_wrong=float(np.mean((pa.argmax(1)!=y)&(pb.argmax(1)!=y))))
        pairs.append(entry)
    return dict(checkpoint=str(checkpoint),split=split,n=len(y),partial=max_batches is not None,
                selected_epoch=(payload.get("history") or {}).get("selected_checkpoint_epoch"),
                metrics={k:metrics(y,p) for k,p in arrays.items()},pairs=pairs,
                note="No retraining. Jaccard excludes molecules with empty union; score correlation excludes constant/short vectors. Drop-entire-view changes both rationale and vote. Drop-vote keeps the full shared rationale fixed.")


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--processed-dir")
    parser.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    parser.add_argument("--max-batches",type=int)
    parser.add_argument("--output-dir")
    args=parser.parse_args(argv)
    if args.max_batches is not None and args.max_batches<1: parser.error("max-batches must be positive")
    torch.set_num_threads(4)
    out=Path(args.output_dir or f"outputs/analysis/views/{datetime.now():%Y%m%d_%H%M%S}")
    out.mkdir(parents=True,exist_ok=False)
    result=compare(args.checkpoint,args.processed_dir,args.device,args.max_batches)
    (out/"comparison.json").write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(result["metrics"],indent=2))
    print(f"Saved {out/'comparison.json'}")


if __name__=="__main__": main()
