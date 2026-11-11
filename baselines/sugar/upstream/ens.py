import argparse
import os
import os.path as osp
from copy import deepcopy
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

# from DrugOOD.drugood.datasets import build_dataset
# from mmcv import Config
# from ogb.graphproppred import Evaluator, PygGraphPropPredDataset
from ogb.graphproppred import Evaluator
from sklearn.metrics import matthews_corrcoef
from torch_geometric.data import DataLoader
# from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import global_mean_pool
from tqdm import tqdm

from datasets.drugood_dataset import DrugOOD
from datasets.graphss2_dataset import get_dataloader_per, get_dataset
from datasets.mnistsp_dataset import CMNIST75sp
from datasets.spmotif_dataset import SPMotif

from models.ciga import GNNERM, CIGA

from utils.util import args_print, set_seed

from datasets.mnistsp_dataset import MNIST75sp, CMNIST75sp


@torch.no_grad()
def eval_model(model, device, loader, evaluator, eval_metric='acc', save_pred=False, visual=False, index=1):
    model.eval()
    y_true = []
    y_pred = []

    for batch in loader:
        batch = batch.to(device)
        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)
            is_labeled = batch.y == batch.y


            if eval_metric == 'acc':
                if len(batch.y.size()) == len(batch.y.size()):
                    y_true.append(batch.y.view(-1, 1).detach().cpu())
                    y_pred.append(torch.argmax(pred.detach(), dim=1).view(-1, 1).cpu())
                else:
                    y_true.append(batch.y.unsqueeze(-1).detach().cpu())
                    y_pred.append(pred.argmax(-1).unsqueeze(-1).detach().cpu())
            elif eval_metric == 'rocauc':
                # print(pred)
                pred = F.softmax(pred, dim=-1)[:, 1].unsqueeze(-1)
                # print('----------',pred)
                if len(batch.y.size()) == len(batch.y.size()):
                    y_true.append(batch.y.view(-1, 1).detach().cpu())
                    y_pred.append(pred.detach().view(-1, 1).cpu())
                else:
                    y_true.append(batch.y.unsqueeze(-1).detach().cpu())
                    y_pred.append(pred.unsqueeze(-1).detach().cpu())
            elif eval_metric == 'mat':
                y_true.append(batch.y.unsqueeze(-1).detach().cpu())
                y_pred.append(pred.argmax(-1).unsqueeze(-1).detach().cpu())
            elif eval_metric == 'ap':
                y_true.append(batch.y.view(pred.shape).detach().cpu())
                y_pred.append(pred.detach().cpu())
            else:
                if is_labeled.size() != pred.size():
                    with torch.no_grad():
                        pred, rep = model(batch, return_data="rep", debug=True)
                        print(rep.size())
                    print(batch)
                    print(global_mean_pool(batch.x, batch.batch).size())
                    print(pred.shape)
                    print(batch.y.size())
                    print(sum(is_labeled))
                    print(batch.y)
                batch.y = batch.y[is_labeled]
                pred = pred[is_labeled]
                y_true.append(batch.y.view(pred.shape).unsqueeze(-1).detach().cpu())
                y_pred.append(pred.detach().unsqueeze(-1).cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    if eval_metric == 'mat':
        res_metric = matthews_corrcoef(y_true, y_pred)
    else:
        input_dict = {"y_true": y_true, "y_pred": y_pred}
        res_metric = evaluator.eval(input_dict)[eval_metric]

    if save_pred:
        return res_metric, y_pred
    else:
        return res_metric


@torch.no_grad()
def sub_softavg(models, device, val_loader, test_loader, evaluator, eval_metric='acc'):

    acc = []


    attentions = []
    for model in models:
        model.eval()
        test_acc = eval_model(model,
                              device,
                              val_loader,
                              evaluator,
                              eval_metric=eval_metric)
        # print('test',test_acc)
        acc.append(test_acc)
        y_true = []
        y_pred = []
        model_attention = []

        for batch in test_loader:
            batch = batch.to(device)
            if batch.x.shape[0] == 1:
                pass
            else:
                with torch.no_grad():
                    attention = model(batch, return_data="attention")

                if len(batch.y.size()) == len(batch.y.size()):
                    y_true.append(batch.y.view(-1, 1).detach().cpu())
                    model_attention.append(attention)


                else:
                    print('else')


        y_true = torch.cat(y_true, dim=0).numpy()

        attentions.append(model_attention)

    confidence = torch.softmax(torch.tensor(acc).float(), dim=0)

    avgs = []
    for i in range(len(attentions[0])):
        avg = confidence[0] * attentions[0][int(i)]

        for j in range(len(models)):
            if j == 0:
                pass
            else:
                avg = avg + confidence[j] * attentions[j][i]


        avgs.append(avg)



    res_metric1 = soft_vote(models, device, test_loader, evaluator, eval_metric=eval_metric, subg_avg=True, avgs=avgs)
    #res_metric2 = hard_vote(models, device, test_loader, evaluator, eval_metric=eval_metric, subg_avg=True, avgs=avgs)
    return res_metric1


@torch.no_grad()
def sub_max(models, device, loader, evaluator, eval_metric='acc'):


    attentions = []
    for model in models:
        model.eval()

        y_true = []

        model_attention = []

        for batch in loader:
            batch = batch.to(device)
            if batch.x.shape[0] == 1:
                pass
            else:
                with torch.no_grad():
                    attention = model(batch, return_data="attention")

                if len(batch.y.size()) == len(batch.y.size()):
                    y_true.append(batch.y.view(-1, 1).detach().cpu())
                    model_attention.append(attention)


                else:
                    print('else')

        y_true = torch.cat(y_true, dim=0).numpy()

        attentions.append(model_attention)


    maxs = []
    for i in range(len(attentions[0])):
        max = attentions[0][i].cpu().numpy()

        for j in range(len(models)):
            if j == 0:
                pass
            else:
                max = np.maximum.reduce([max, attentions[j][i].cpu().numpy()])


        maxs.append(torch.tensor(max))



    res_metric1 = soft_vote(models, device, loader, evaluator, eval_metric=eval_metric, subg_avg=True, avgs=maxs)
    #res_metric2 = hard_vote(models, device, loader, evaluator, eval_metric=eval_metric, subg_avg=True, avgs=maxs)
    return res_metric1


@torch.no_grad()
def sub_avg(models, device, loader, evaluator, eval_metric='acc'):

    do_show = True
    attentions = []
    for model in models:
        model.eval()
        y_true = []

        model_attention = []

        for batch in loader:
            batch = batch.to(device)
            if batch.x.shape[0] == 1:
                pass
            else:
                with torch.no_grad():
                    attention = model(batch, return_data="attention")

                if len(batch.y.size()) == len(batch.y.size()):
                    y_true.append(batch.y.view(-1, 1).detach().cpu())
                    model_attention.append(attention)
                else:
                    print('else')


        y_true = torch.cat(y_true, dim=0).numpy()
        attentions.append(model_attention)

    avgs = []
    for i in range(len(attentions[0])):
        avg = attentions[0][i]
        for j in range(len(models)):
            if j == 0:
                pass
            else:
                avg = (avg + attentions[j][i])

        avg = avg / len(models)
        avgs.append(avg)

    res_metric1 = soft_vote(models, device, loader, evaluator, eval_metric=eval_metric, subg_avg=True, avgs=avgs)
    #res_metric2 = hard_vote(models, device, loader, evaluator, eval_metric=eval_metric, subg_avg=True, avgs=avgs)
    return res_metric1


@torch.no_grad()
def soft_vote(models, device, loader, evaluator, eval_metric='acc', subg_avg=False, avgs=None):
    # print('soft_vote')
    preds = []
    acc = []
    confidences = []

    first = True
    for model in models:
        model.eval()
        test_acc = eval_model(model,
                              device,
                              loader,
                              evaluator,
                              eval_metric=eval_metric)
        # print('test',test_acc)
        acc.append(test_acc)
        y_true = []
        y_pred = []
        confidence = []
        do_show = True
        if subg_avg:
            for a, batch in zip(avgs, loader):
                batch = batch.to(device)
                if batch.x.shape[0] == 1:
                    pass
                else:
                    with torch.no_grad():
                        pred = model(batch, subgraph_avg=True, pred_edge_weight=a.to(device))
                    is_labeled = batch.y == batch.y


                    if eval_metric == 'acc':
                        if len(batch.y.size()) == len(batch.y.size()):
                            y_true.append(batch.y.view(-1, 1).detach().cpu())

                            pred = torch.softmax(pred, dim=1)
                            y_pred.append(pred.detach().cpu())
                        else:
                            print('else')
                            y_true.append(batch.y.unsqueeze(-1).detach().cpu())
                            y_pred.append(pred.argmax(-1).unsqueeze(-1).detach().cpu())
                    elif eval_metric == 'rocauc':
                        # print(pred)
                        pred = F.softmax(pred, dim=-1)[:, 1].unsqueeze(-1)
                        # print('----------',pred)
                        if len(batch.y.size()) == len(batch.y.size()):
                            y_true.append(batch.y.view(-1, 1).detach().cpu())
                            y_pred.append(pred.detach().view(-1, 1).cpu())
                        else:
                            y_true.append(batch.y.unsqueeze(-1).detach().cpu())
                            y_pred.append(pred.unsqueeze(-1).detach().cpu())
                    elif eval_metric == 'mat':
                        y_true.append(batch.y.unsqueeze(-1).detach().cpu())
                        y_pred.append(pred.argmax(-1).unsqueeze(-1).detach().cpu())
                    elif eval_metric == 'ap':
                        y_true.append(batch.y.view(pred.shape).detach().cpu())
                        y_pred.append(pred.detach().cpu())

                    else:
                        if is_labeled.size() != pred.size():
                            with torch.no_grad():
                                pred, rep = model(batch, return_data="rep", debug=True)
                                print(rep.size())
                            print(batch)
                            print(global_mean_pool(batch.x, batch.batch).size())
                            print(pred.shape)
                            print(batch.y.size())
                            print(sum(is_labeled))
                            print(batch.y)
                        batch.y = batch.y[is_labeled]
                        pred = pred[is_labeled]
                        y_true.append(batch.y.view(pred.shape).unsqueeze(-1).detach().cpu())
                        y_pred.append(pred.detach().unsqueeze(-1).cpu())
        else:
            for batch in loader:
                batch = batch.to(device)
                if batch.x.shape[0] == 1:
                    pass
                else:
                    with torch.no_grad():
                        pred = model(batch)
                        # print('pred',pred)
                    is_labeled = batch.y == batch.y
                    if eval_metric == 'acc':
                        if len(batch.y.size()) == len(batch.y.size()):
                            y_true.append(batch.y.view(-1, 1).detach().cpu())

                            pred = torch.softmax(pred, dim=1)
                            y_pred.append(pred.detach().cpu())
                        else:
                            print('else')
                            y_true.append(batch.y.unsqueeze(-1).detach().cpu())
                            y_pred.append(pred.argmax(-1).unsqueeze(-1).detach().cpu())
                    elif eval_metric == 'rocauc':
                        # print(pred)
                        pred = F.softmax(pred, dim=-1)[:, 1].unsqueeze(-1)
                        # print('----------',pred)
                        if len(batch.y.size()) == len(batch.y.size()):
                            y_true.append(batch.y.view(-1, 1).detach().cpu())
                            y_pred.append(pred.detach().view(-1, 1).cpu())
                        else:
                            y_true.append(batch.y.unsqueeze(-1).detach().cpu())
                            y_pred.append(pred.unsqueeze(-1).detach().cpu())
                    elif eval_metric == 'mat':
                        y_true.append(batch.y.unsqueeze(-1).detach().cpu())
                        y_pred.append(pred.argmax(-1).unsqueeze(-1).detach().cpu())
                    elif eval_metric == 'ap':
                        y_true.append(batch.y.view(pred.shape).detach().cpu())
                        y_pred.append(pred.detach().cpu())

                    else:
                        if is_labeled.size() != pred.size():
                            with torch.no_grad():
                                pred, rep = model(batch, return_data="rep", debug=True)
                                print(rep.size())
                            print(batch)
                            print(global_mean_pool(batch.x, batch.batch).size())
                            print(pred.shape)
                            print(batch.y.size())
                            print(sum(is_labeled))
                            print(batch.y)
                        batch.y = batch.y[is_labeled]
                        pred = pred[is_labeled]
                        y_true.append(batch.y.view(pred.shape).unsqueeze(-1).detach().cpu())
                        y_pred.append(pred.detach().unsqueeze(-1).cpu())

        y_true = torch.cat(y_true, dim=0).numpy()
        y_pred = torch.cat(y_pred, dim=0).numpy()

        if eval_metric == 'acc':
            if first:
                # print(first)
                p = np.zeros((y_pred.shape[0], y_pred.shape[1]))
                first = False
            p = np.add(p, y_pred)

        if eval_metric == 'rocauc':
            preds.append(y_pred)


    if eval_metric == 'rocauc':
        confidence = torch.softmax(torch.tensor(acc).float(), dim=0)


        for j in range(len(models)):
            if j == 0:
                avg = confidence[j] * preds[j]
            else:
                avg = avg + confidence[j] * preds[j]

        p = avg
    if eval_metric == 'acc':
        p = torch.argmax(torch.tensor(p).detach(), dim=1).view(-1, 1).cpu().numpy()

    input_dict = {"y_true": y_true, "y_pred": p}

    res_metric = evaluator.eval(input_dict)[eval_metric]

    return res_metric


parser = argparse.ArgumentParser(description='sugar')
parser.add_argument('--device', default=0, type=int, help='cuda device')
parser.add_argument('--root', default='./data', type=str, help='directory for datasets.')
parser.add_argument('--dataset', default='drugood_lbap_core_ec50_assay', type=str)
parser.add_argument('--bias', default='0.33', type=str, help='select bias extend')
parser.add_argument('--feature', type=str, default="full", help='full feature or simple feature')
parser.add_argument('--drop_mode', default='drop_edge', type=str)

# training config
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--epoch', default=400, type=int, help='training iterations')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate for the predictor')
parser.add_argument('--seed', nargs='?', default='[1,2,3,4,5]', help='model id')
parser.add_argument('--pretrain', default=20, type=int, help='pretrain epoch before early stopping')

# model config
parser.add_argument('--emb_dim', default=32, type=int)
parser.add_argument('--r', default=0.25, type=float, help='selected ratio')
parser.add_argument('-c_dim', '--classifier_emb_dim', default=32, type=int)
parser.add_argument('-c_in', '--classifier_input_feat', default='raw', type=str)
parser.add_argument('--model', default='gin', type=str)
parser.add_argument('--pooling', default='mean', type=str)
parser.add_argument('--num_layers', default=2, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--early_stopping', default=5, type=int)
parser.add_argument('--dropout', default=0.5, type=float)
parser.add_argument('--virtual_node', action='store_true')
parser.add_argument('--eval_metric',
                    default='',
                    type=str,
                    help='specify a particular eval metric, e.g., mat for MatthewsCoef')

# Invariant Learning baselines config
parser.add_argument('--num_envs', default=1, type=int, help='num of envs need to be partitioned')
parser.add_argument('--irm_p', default=1, type=float, help='penalty weight')
parser.add_argument('--irm_opt', default='irm', type=str, help='algorithms to use')

# Invariant Graph Learning config
parser.add_argument('--erm', action='store_true')  # whether to use normal GNN arch
parser.add_argument('--ginv_opt', default='ginv', type=str)  # which interpretable GNN archs to use
parser.add_argument('--dir', default=0, type=float)
parser.add_argument('--contrast_t', default=1.0, type=float, help='temperature prameter in contrast loss')
# strength of the contrastive reg, \alpha in the paper
parser.add_argument('--contrast', default=4, type=float)
parser.add_argument('--not_norm', action='store_true')  # whether not using normalization for the constrast loss
parser.add_argument('-c_sam', '--contrast_sampling', default='mul', type=str)
parser.add_argument('-c_rep', '--contrast_rep', default='rep', type=str)
# pooling method for the last two options in c_rep
parser.add_argument('-c_pool', '--contrast_pooling', default='add', type=str)

parser.add_argument('-s_rep', '--spurious_rep', default='rep', type=str)
parser.add_argument('--spu_coe', default=0, type=float)

# misc
parser.add_argument('--no_tqdm', action='store_true')
parser.add_argument('--commit', default='', type=str, help='experiment name')
parser.add_argument('--save_model', action='store_true')  # save pred to ./pred if not empty
parser.add_argument('--sample_rate', default=100, type=int, help='sample rate')
parser.add_argument('--wd', type=float, default=1e-4, help='weight decay (default: 1e-4)')
parser.add_argument('--init', type=int, default=1)
parser.add_argument('--metrictest', action='store_true')
parser.add_argument('--divp', action='store_true')
parser.add_argument('--penalty', type=float,default=1,help='penalty weight for diversity regularizer')

parser.add_argument('--model_num', type=int, default=10)

args = parser.parse_args()

erm_model = None  # used to obtain pesudo labels for CNC sampling in contrastive loss

args.seed = eval(args.seed)

device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
eval_metric = 'acc' if len(args.eval_metric) == 0 else args.eval_metric
edge_dim = -1.
if args.dataset.lower().startswith('drugood'):
    # drugood_lbap_core_ic50_assay.json
    config_path = os.path.join("configs", args.dataset + ".py")
    # print(config_path)
    cfg = Config.fromfile(config_path)
    # print(cfg.data.ood_val)
    # print(cfg.data.ood_test)
    root = os.path.join(args.root, "DrugOOD")


    train_dataset = DrugOOD(root=root, dataset=build_dataset(cfg.data.train), name=args.dataset, mode="train")
    val_dataset = DrugOOD(root=root, dataset=build_dataset(cfg.data.ood_val), name=args.dataset, mode="ood_val")
    test_dataset = DrugOOD(root=root, dataset=build_dataset(cfg.data.ood_test), name=args.dataset,
                            mode="ood_test")

    if args.eval_metric == 'auc' or args.eval_metric == 'rocauc':
        evaluator = Evaluator('ogbg-molhiv')
        eval_metric = args.eval_metric = 'rocauc'
    else:
        evaluator = Evaluator('ogbg-ppa')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    input_dim = 39
    edge_dim = 10
    num_classes = 2
elif args.dataset.lower() in ['spmotif','sumotif']:

    train_dataset = SPMotif(os.path.join(args.root, f'{args.dataset}-{args.bias}/'), mode='train')
    val_dataset = SPMotif(os.path.join(args.root, f'{args.dataset}-{args.bias}/'), mode='val')
    test_dataset = SPMotif(os.path.join(args.root, f'{args.dataset}-{args.bias}/'), mode='test')

    input_dim = 4
    num_classes = 3
    evaluator = Evaluator('ogbg-ppa')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

elif args.dataset.lower() in ['graph-sst5']:
    dataset = get_dataset(dataset_dir=args.root, dataset_name=args.dataset, task=None)
    dataloader = get_dataloader_per(dataset, batch_size=args.batch_size, small_to_large=True, seed=1)
    train_loader = dataloader['train']
    valid_loader = dataloader['eval']
    test_loader = dataloader['test']
    input_dim = 768
    num_classes = int(args.dataset[-1].lower()) if args.dataset[-1].lower() in ['2', '5'] else 3
    evaluator = Evaluator('ogbg-ppa')
elif args.dataset.lower() in ['graph-twitter']:
    dataset = get_dataset(dataset_dir=args.root, dataset_name=args.dataset, task=None)
    dataloader = get_dataloader_per(dataset, batch_size=args.batch_size, small_to_large=False, seed=1)
    train_loader = dataloader['train']
    valid_loader = dataloader['eval']
    test_loader = dataloader['test']
    
    input_dim = 768
    num_classes = int(args.dataset[-1].lower()) if args.dataset[-1].lower() in ['2', '5'] else 3
    evaluator = Evaluator('ogbg-ppa')
    
elif args.dataset.lower() in ['cmnist']:
    n_val_data = 5000
    train_dataset = CMNIST75sp(os.path.join(args.root, 'CMNISTSP/'), mode='train')
    test_dataset = CMNIST75sp(os.path.join(args.root, 'CMNISTSP/'), mode='test')
    perm_idx = torch.randperm(len(test_dataset), generator=torch.Generator().manual_seed(0))
    test_val = test_dataset[perm_idx]
    val_dataset, test_dataset = test_val[:n_val_data], test_val[n_val_data:]
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    input_dim = 7
    num_classes = 2
    evaluator = Evaluator('ogbg-ppa')
else:
    raise Exception("Invalid dataset name")

if args.erm:
    model = GNNERM(input_dim=input_dim,
                   edge_dim=edge_dim,
                   out_dim=num_classes,
                   gnn_type=args.model,
                   num_layers=args.num_layers,
                   emb_dim=args.emb_dim,
                   drop_ratio=args.dropout,
                   graph_pooling=args.pooling,
                   virtual_node=args.virtual_node).to(device)

    model_optimizer = torch.optim.Adam(list(model.parameters()), lr=args.lr)

else:
    model = CIGA(ratio=args.r,
                 input_dim=input_dim,
                 edge_dim=edge_dim,
                 out_dim=num_classes,
                 gnn_type=args.model,
                 num_layers=args.num_layers,
                 emb_dim=args.emb_dim,
                 drop_ratio=args.dropout,
                 graph_pooling=args.pooling,
                 virtual_node=args.virtual_node,
                 c_dim=args.classifier_emb_dim,
                 c_in=args.classifier_input_feat,
                 c_rep=args.contrast_rep,
                 c_pool=args.contrast_pooling,
                 s_rep=args.spurious_rep).to(device)
    params = list(model.parameters())
    mean = sum([torch.mean(p) for p in params]) / len(params)
    print('mean', mean)
    model_optimizer = torch.optim.Adam(list(model.parameters()), lr=args.lr)
print(model)


if args.erm:
    file_prefix=f'erm_{args.dataset}_{args.model}_{args.classifier_input_feat}_dout{args.dropout}_penalty'
elif args.divp:
    file_prefix=f'{args.dataset}_{args.model}_{args.classifier_input_feat}_dout{args.dropout}_r{args.r}_penalty'
else:
    file_prefix=f'{args.dataset}_{args.model}_{args.classifier_input_feat}_dout{args.dropout}_r{args.r}_sub_allcontrast'

 
def path_construct(i):
    if args.dataset.lower() in ['spmotif', 'mspmotif','sumotif']:
        file = glob.glob(os.path.join(f'{file_prefix}/{args.sample_rate}%subgraph({args.drop_mode})100%feature_{args.dataset}_{args.contrast}_random_{args.init}_shareinit_{i}_{args.bias}.pt'))
        if args.divp:
            file = glob.glob(os.path.join(f'{file_prefix}/{args.sample_rate}%subgraph({args.drop_mode})100%feature_{args.dataset}_{args.contrast}_shared_init_{args.init}_{i}_{args.bias}_{args.penalty}.pt'))
    else:

        file = glob.glob(os.path.join(f'{file_prefix}/{args.sample_rate}%subgraph({args.drop_mode})100%feature_{args.dataset}_{args.contrast}_random_{args.init}_shareinit_{i}.pt'))
        if args.divp:
            file = glob.glob(os.path.join(f'{file_prefix}/{args.sample_rate}%subgraph({args.drop_mode})100%feature_{args.dataset}_{args.contrast}_random_init_{i}_{args.penalty}.pt'))

    return file


gnns = []


avg = []
# SUGAR(ENS)
for i in range(1,args.model_num+1):
    set_seed(args.init)
    if args.erm:
        model = GNNERM(input_dim=input_dim,
                   edge_dim=edge_dim,
                   out_dim=num_classes,
                   gnn_type=args.model,
                   num_layers=args.num_layers,
                   emb_dim=args.emb_dim,
                   drop_ratio=args.dropout,
                   graph_pooling=args.pooling,
                   virtual_node=args.virtual_node).to(device)
    else:
        model = CIGA(ratio=args.r,
                 input_dim=input_dim,
                 edge_dim=edge_dim,
                 out_dim=num_classes,
                 gnn_type=args.model,
                 num_layers=args.num_layers,
                 emb_dim=args.emb_dim,
                 drop_ratio=args.dropout,
                 graph_pooling=args.pooling,
                 virtual_node=args.virtual_node,
                 c_dim=args.classifier_emb_dim,
                 c_in=args.classifier_input_feat,
                 c_rep=args.contrast_rep,
                 c_pool=args.contrast_pooling,
                 s_rep=args.spurious_rep).to(device)
    file=path_construct(i)

    print(file[0])
    model.load_state_dict(torch.load(file[0]))


    gnns.append(model)
    layer_outputs=[]
    def hook(module, input, output):
        layer_outputs.append(output)

    model.eval()
    mo_acc = eval_model(model, device, test_loader, evaluator, eval_metric=eval_metric)

    avg.append(mo_acc)
    if i % args.model_num == 0:
        for ac in avg:
            print(ac)
        print('Result')
        if not args.erm:
            result1 = sub_softavg(gnns, device, valid_loader, test_loader, evaluator, eval_metric=eval_metric)
            print('AVG_soft:',result1)

            result1 = sub_max(gnns, device, test_loader, evaluator, eval_metric=eval_metric)
            print('MAX:',result1)

            result1 = sub_avg(gnns, device, test_loader, evaluator, eval_metric=eval_metric)

            print('AVG_hard',result1)


        print(float(torch.tensor(avg).mean()))
        print(float(torch.tensor(avg).std()))



dict_folder_to_score = {}
for i in range(1, args.model_num+1):
    set_seed(i)
    file=path_construct(i)
    print(file[0])
    model.load_state_dict(torch.load(file[0]))
    model.eval()
    mo_acc = eval_model(model, device, valid_loader, evaluator, eval_metric=eval_metric)
    print('acc', mo_acc)
    dict_folder_to_score[file[0]] = mo_acc
sorted_checkpoints = sorted(dict_folder_to_score.keys(), key=lambda x: dict_folder_to_score[x], reverse=True)
print(sorted_checkpoints)
for method in range(3):
    if args.erm:
        if method<3:
            continue

    selected_indexes = []
    best_result = -float("inf")
    dict_best_results = {}
    for i in range(0, len(sorted_checkpoints)):

        selected_indexes.append(i)
        selected_checkpoints = [sorted_checkpoints[index] for index in selected_indexes]

        gnns = []
        for folder in selected_checkpoints:
            # seed=int(folder.split('_')[1])
            if args.erm:
                model = GNNERM(input_dim=input_dim,
                        edge_dim=edge_dim,
                        out_dim=num_classes,
                        gnn_type=args.model,
                        num_layers=args.num_layers,
                        emb_dim=args.emb_dim,
                        drop_ratio=args.dropout,
                        graph_pooling=args.pooling,
                        virtual_node=args.virtual_node).to(device)
            else:
                model = CIGA(ratio=args.r,
                         input_dim=input_dim,
                         edge_dim=edge_dim,
                         out_dim=num_classes,
                         gnn_type=args.model,
                         num_layers=args.num_layers,
                         emb_dim=args.emb_dim,
                         drop_ratio=args.dropout,
                         graph_pooling=args.pooling,
                         virtual_node=args.virtual_node,
                         c_dim=args.classifier_emb_dim,
                         c_in=args.classifier_input_feat,
                         c_rep=args.contrast_rep,
                         c_pool=args.contrast_pooling,
                         s_rep=args.spurious_rep).to(device)

            model.load_state_dict(torch.load(folder))
            gnns.append(model)
        if method == 0:
            result1 = sub_softavg(gnns, device, valid_loader, valid_loader, evaluator,
                                     eval_metric=eval_metric)
        elif method == 1:
            result1= sub_max(gnns, device, valid_loader, evaluator, eval_metric=eval_metric)
        elif method == 2:
            result1= sub_avg(gnns, device, valid_loader, evaluator, eval_metric=eval_metric)


        if result1 > best_result:
            best_result = result1

        else:
            selected_indexes.pop(-1)
        # print('ind',selected_indexes)
    selected_checkpoints = [sorted_checkpoints[index] for index in selected_indexes]
    # print(selected_checkpoints)
    gnns = []
    for folder in selected_checkpoints:
        # seed=int(folder.split('_')[1])
        if args.erm:
            model = GNNERM(input_dim=input_dim,
                    edge_dim=edge_dim,
                    out_dim=num_classes,
                    gnn_type=args.model,
                    num_layers=args.num_layers,
                    emb_dim=args.emb_dim,
                    drop_ratio=args.dropout,
                    graph_pooling=args.pooling,
                    virtual_node=args.virtual_node).to(device)
        else:
            model = CIGA(ratio=args.r,
                     input_dim=input_dim,
                     edge_dim=edge_dim,
                     out_dim=num_classes,
                     gnn_type=args.model,
                     num_layers=args.num_layers,
                     emb_dim=args.emb_dim,
                     drop_ratio=args.dropout,
                     graph_pooling=args.pooling,
                     virtual_node=args.virtual_node,
                     c_dim=args.classifier_emb_dim,
                     c_in=args.classifier_input_feat,
                     c_rep=args.contrast_rep,
                     c_pool=args.contrast_pooling,
                     s_rep=args.spurious_rep).to(device)

        model.load_state_dict(torch.load(folder))

        gnns.append(model)

    if method == 0:
        result1 = sub_softavg(gnns, device, valid_loader, test_loader, evaluator,
                                 eval_metric=eval_metric)
        print('AVG_soft:',result1)
    elif method == 1:
        result1= sub_max(gnns, device, test_loader, evaluator, eval_metric=eval_metric)
        print('MAX:',result1)
    elif method == 2:
        result1 = sub_avg(gnns, device, test_loader, evaluator, eval_metric=eval_metric)
        print('AVG_hard',result1)
#
def merge(models):
    if args.erm:
        md = GNNERM(input_dim=input_dim,
                    edge_dim=edge_dim,
                    out_dim=num_classes,
                    gnn_type=args.model,
                    num_layers=args.num_layers,
                    emb_dim=args.emb_dim,
                    drop_ratio=args.dropout,
                    graph_pooling=args.pooling,
                    virtual_node=args.virtual_node).to(device)
    else:
        md = CIGA(ratio=args.r,
                    input_dim=input_dim,
                    edge_dim=edge_dim,
                    out_dim=num_classes,
                    gnn_type=args.model,
                    num_layers=args.num_layers,
                    emb_dim=args.emb_dim,
                    drop_ratio=args.dropout,
                    graph_pooling=args.pooling,
                    virtual_node=args.virtual_node,
                    c_dim=args.classifier_emb_dim,
                    c_in=args.classifier_input_feat,
                    c_rep=args.contrast_rep,
                    c_pool=args.contrast_pooling,
                    s_rep=args.spurious_rep).to(device)
    alphal = [1/len(models) for i in range(0, len(models))]

    for j in range(len(models)):
        if j==0:
            for p1,p2 in zip(md.parameters(),models[j].parameters()):
                p1.data=p2.data*alphal[j]
        else:
            for p1,p2 in zip(md.parameters(),models[j].parameters()):
                p1.data=p1.data+p2.data*alphal[j]
    return md

#SUGAR(WA)

seed_n=[]
origin_acc=[]
diwa_acc=[]
valid_acc=[]
avg_val_acc=[]
gnns = []
if args.erm:
    model = GNNERM(input_dim=input_dim,
                edge_dim=edge_dim,
                out_dim=num_classes,
                gnn_type=args.model,
                num_layers=args.num_layers,
                emb_dim=args.emb_dim,
                drop_ratio=args.dropout,
                graph_pooling=args.pooling,
                virtual_node=args.virtual_node).to(device)
else:
    model = CIGA(ratio=args.r,
                input_dim=input_dim,
                edge_dim=edge_dim,
                out_dim=num_classes,
                gnn_type=args.model,
                num_layers=args.num_layers,
                emb_dim=args.emb_dim,
                drop_ratio=args.dropout,
                graph_pooling=args.pooling,
                virtual_node=args.virtual_node,
                c_dim=args.classifier_emb_dim,
                c_in=args.classifier_input_feat,
                c_rep=args.contrast_rep,
                c_pool=args.contrast_pooling,
                s_rep=args.spurious_rep).to(device)
swa_model1 = torch.optim.swa_utils.AveragedModel(model,device=device)
for i in range(1,args.model_num+1):
    seed_n.append(i)
    if args.erm:
        model = GNNERM(input_dim=input_dim,
                   edge_dim=edge_dim,
                   out_dim=num_classes,
                   gnn_type=args.model,
                   num_layers=args.num_layers,
                   emb_dim=args.emb_dim,
                   drop_ratio=args.dropout,
                   graph_pooling=args.pooling,
                   virtual_node=args.virtual_node).to(device)
    else:
        model = CIGA(ratio=args.r,
                    input_dim=input_dim,
                    edge_dim=edge_dim,
                    out_dim=num_classes,
                    gnn_type=args.model,
                    num_layers=args.num_layers,
                    emb_dim=args.emb_dim,
                    drop_ratio=args.dropout,
                    graph_pooling=args.pooling,
                    virtual_node=args.virtual_node,
                    c_dim=args.classifier_emb_dim,
                    c_in=args.classifier_input_feat,
                    c_rep=args.contrast_rep,
                    c_pool=args.contrast_pooling,
                    s_rep=args.spurious_rep).to(device)

    file=path_construct(i)
    model.load_state_dict(torch.load(file[0]))
    gnns.append(model)
    model.eval()

    mo_acc=eval_model(model, device, test_loader, evaluator, eval_metric=eval_metric)
    print('acc')
    print( mo_acc)
    origin_acc.append(mo_acc)

set_seed(args.init)
swa_model1=merge(gnns)
torch.optim.swa_utils.update_bn(train_loader, swa_model1,device=device)
swa_model1.eval()
acc = eval_model(swa_model1, device, test_loader, evaluator, eval_metric=eval_metric)
print(acc)






import glob
if args.erm:
    model = GNNERM(input_dim=input_dim,
                edge_dim=edge_dim,
                out_dim=num_classes,
                gnn_type=args.model,
                num_layers=args.num_layers,
                emb_dim=args.emb_dim,
                drop_ratio=args.dropout,
                graph_pooling=args.pooling,
                virtual_node=args.virtual_node).to(device)
else:    
    model = CIGA(ratio=args.r,
                input_dim=input_dim,
                edge_dim=edge_dim,
                out_dim=num_classes,
                gnn_type=args.model,
                num_layers=args.num_layers,
                emb_dim=args.emb_dim,
                drop_ratio=args.dropout,
                graph_pooling=args.pooling,
                virtual_node=args.virtual_node,
                c_dim=args.classifier_emb_dim,
                c_in=args.classifier_input_feat,
                c_rep=args.contrast_rep,
                c_pool=args.contrast_pooling,
                s_rep=args.spurious_rep).to(device)
swa_model1 = torch.optim.swa_utils.AveragedModel(model,device=device)
dict_folder_to_score = {}
for i in range(1, args.model_num+1):
    set_seed(args.init)
    
    file=path_construct(i)
    print(file)
    model.load_state_dict(torch.load(file[0]))
    model.eval()
    mo_acc = eval_model(model, device, valid_loader, evaluator, eval_metric=eval_metric)

    dict_folder_to_score[file[0]] = mo_acc
sorted_checkpoints = sorted(dict_folder_to_score.keys(), key=lambda x: dict_folder_to_score[x], reverse=True)

selected_indexes = []
best_result = -float("inf")
dict_best_results = {}
for i in range(0, len(sorted_checkpoints)):

    selected_indexes.append(i)
    selected_checkpoints = [sorted_checkpoints[index] for index in selected_indexes]

    gnns = []


    for j in range(len(selected_checkpoints)):

        if args.erm:
            model = GNNERM(input_dim=input_dim,
                    edge_dim=edge_dim,
                    out_dim=num_classes,
                    gnn_type=args.model,
                    num_layers=args.num_layers,
                    emb_dim=args.emb_dim,
                    drop_ratio=args.dropout,
                    graph_pooling=args.pooling,
                    virtual_node=args.virtual_node).to(device)
        else:
            model = CIGA(ratio=args.r,
                        input_dim=input_dim,
                        edge_dim=edge_dim,
                        out_dim=num_classes,
                        gnn_type=args.model,
                        num_layers=args.num_layers,
                        emb_dim=args.emb_dim,
                        drop_ratio=args.dropout,
                        graph_pooling=args.pooling,
                        virtual_node=args.virtual_node,
                        c_dim=args.classifier_emb_dim,
                        c_in=args.classifier_input_feat,
                        c_rep=args.contrast_rep,
                        c_pool=args.contrast_pooling,
                        s_rep=args.spurious_rep).to(device)



        model.load_state_dict(torch.load(selected_checkpoints[j]))
        gnns.append(model)
    set_seed(args.init)
    if len(gnns)>1:
        swa_model1=merge(gnns)
        torch.optim.swa_utils.update_bn(train_loader, swa_model1,device=device)
    else:
        swa_model1=gnns[0]
    swa_model1.eval()
    result1 = eval_model(swa_model1, device, valid_loader, evaluator, eval_metric=eval_metric)
    if result1 > best_result:
        best_result = result1
    else:
        selected_indexes.pop(-1)

selected_checkpoints = [sorted_checkpoints[index] for index in selected_indexes]

gnns = []
for j in range(len(selected_checkpoints)):

    if args.erm:
        model = GNNERM(input_dim=input_dim,
                edge_dim=edge_dim,
                out_dim=num_classes,
                gnn_type=args.model,
                num_layers=args.num_layers,
                emb_dim=args.emb_dim,
                drop_ratio=args.dropout,
                graph_pooling=args.pooling,
                virtual_node=args.virtual_node).to(device)
    else:
        model = CIGA(ratio=args.r,
                    input_dim=input_dim,
                    edge_dim=edge_dim,
                    out_dim=num_classes,
                    gnn_type=args.model,
                    num_layers=args.num_layers,
                    emb_dim=args.emb_dim,
                    drop_ratio=args.dropout,
                    graph_pooling=args.pooling,
                    virtual_node=args.virtual_node,
                    c_dim=args.classifier_emb_dim,
                    c_in=args.classifier_input_feat,
                    c_rep=args.contrast_rep,
                    c_pool=args.contrast_pooling,
                    s_rep=args.spurious_rep).to(device)
    set_seed(args.init)
    model.load_state_dict(torch.load(selected_checkpoints[j]))


    gnns.append(model)
if len(gnns)>1:
    swa_model1=merge(gnns)
    torch.optim.swa_utils.update_bn(train_loader, swa_model1, device=device)
else:
    swa_model1=gnns[0]
swa_model1.eval()
result1 = eval_model(swa_model1, device, test_loader, evaluator, eval_metric=eval_metric)
print('WA',result1)

