import argparse
import os
import os.path as osp
from copy import deepcopy
from datetime import datetime
import random
import numpy as np
import torch
import torch.nn.functional as F

from DrugOOD.drugood.datasets import build_dataset
from mmcv import Config
from ogb.graphproppred import Evaluator, PygGraphPropPredDataset
from ogb.graphproppred import Evaluator
from sklearn.metrics import matthews_corrcoef
from torch_geometric.data import DataLoader
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import global_mean_pool
from tqdm import tqdm

from datasets.drugood_dataset import DrugOOD
from datasets.graphss2_dataset import get_dataloader_per, get_dataset
from datasets.mnistsp_dataset import CMNIST75sp
from datasets.spmotif_dataset import SPMotif
from models.gnn_ib import GIB
from models.ciga import GNNERM, CIGA, GNNPooling
from models.losses import get_contrast_loss, get_irm_loss
from utils.logger import Logger
from utils.util import args_print, set_seed
from torch_geometric.data import InMemoryDataset, Data, Batch

@torch.no_grad()
def eval_model(model, device, loader, evaluator, eval_metric='acc', save_pred=False,visual=False,index=1):
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
                pred = F.softmax(pred, dim=-1)[:, 1].unsqueeze(-1)
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

def delete_edge(data,args,seed):

    import torch_geometric.utils as utils
    
    
    row, col = data.edge_index
    sample_rate=args.sample_rate/100
    #print(sample_rate)
    true_count = int(row.size(0) * sample_rate)
    false_count = row.size(0) - true_count
    random.seed(seed)
    edge_mask = [True] * true_count + [False] * false_count
    random.shuffle(edge_mask)

    edge_index = data.edge_index[:, edge_mask]
    edge_attr = data.edge_attr[edge_mask]
    #print(edge_index.shape)
    if args.dataset.lower() in ['spmotif','sumotif']:
        data = Data(edge_index=edge_index, edge_attr=edge_attr, x=data.x,idx=data.idx,y=data.y)
    else:
        data = Data(edge_index=edge_index, edge_attr=edge_attr, x=data.x, node_label=data.node_label, idx=data.idx,
                    y=data.y)
    
    #print('after',data)
    return data

def main(args):

    erm_model = None  # used to obtain pesudo labels for CNC sampling in contrastive loss
    
    args.seed = eval(args.seed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    #if args.metrictest:
     #   device = torch.device("cpu")
    def ce_loss(a, b, reduction='mean'):
        return F.cross_entropy(a, b, reduction=reduction)

    criterion = ce_loss
    eval_metric = 'acc' if len(args.eval_metric) == 0 else args.eval_metric
    edge_dim = -1.


    best_weights = None

    seed=args.init
        
        

    if args.dataset.lower().startswith('drugood'):

        config_path = os.path.join("configs", args.dataset + ".py")
        cfg = Config.fromfile(config_path)
        root = os.path.join(args.root, "DrugOOD")
        train_dataset = DrugOOD(root=root, dataset=build_dataset(cfg.data.train), name=args.dataset, mode="train",seed=seed)
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
        dataloader = get_dataloader_per(dataset, batch_size=args.batch_size, small_to_large=True, seed=args.seed)
        train_loader = dataloader['train']
        valid_loader = dataloader['eval']
        test_loader = dataloader['test']
        input_dim = 768
        num_classes = int(args.dataset[-1].lower()) if args.dataset[-1].lower() in ['2', '5'] else 3
        evaluator = Evaluator('ogbg-ppa')
    elif args.dataset.lower() in ['graph-twitter']:
        dataset = get_dataset(dataset_dir=args.root, dataset_name=args.dataset, task=None)
        dataloader = get_dataloader_per(dataset, batch_size=args.batch_size, small_to_large=False, seed=args.seed)
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

    # log
    datetime_now = datetime.now().strftime("%Y%m%d-%H%M%S")
    all_info = {
        'test_acc': [],
        'train_acc': [],
        'val_acc': [],
    }


    # models and optimizers
    if args.erm:
        def model_init():
            set_seed(args.init)
            model = GNNERM(input_dim=input_dim,
                        edge_dim=edge_dim,
                        out_dim=num_classes,
                        gnn_type=args.model,
                        num_layers=args.num_layers,
                        emb_dim=args.emb_dim,
                        drop_ratio=args.dropout,
                        graph_pooling=args.pooling,
                        virtual_node=args.virtual_node).to(device)
            return model

        models = [model_init() for seed in args.seed]
        for model in models:
            params = list(model.parameters())
            mean = sum([torch.mean(p) for p in params]) / len(params)
            print('mean', mean)

        def opt(model):
            import random
            lr = lambda r: r.choice([1e-3, 3e-3, 5e-3,7e-3,9e-3])
            args.lr = lr(random)
            print(args.lr)
            wd = lambda r: r.choice([1e-4, 1e-5,1e-6])
            args.wd = wd(random)
            print('wd', args.wd)

            return torch.optim.Adam(list(model.parameters()), lr=args.lr, weight_decay=args.wd)

        model_optimizers = [opt(model) for model in models]

    else:
        def model_init():
            set_seed(args.init)
            
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
            return model
        
        models=[model_init() for seed in args.seed]
        for model in models:
            params = list(model.parameters())
            mean = sum([torch.mean(p) for p in params]) / len(params)
            # print('mean', mean)

        model_optimizers = [torch.optim.Adam(list(model.parameters()), lr=args.lr) for model in models]

        
    #print(model)
    last_train_acc, last_test_acc, last_val_acc = [0 for _ in range(len(models))], [0 for _ in range(len(models))], [0 for _ in range(len(models))]
    cnt = [0 for _ in range(len(models))]
    # generate environment partitions
    if args.num_envs > 1:
        env_idx = (torch.sigmoid(torch.randn(len(train_loader.dataset))) > 0.5).long()
        print(f"num env 0: {sum(env_idx == 0)} num env 1: {sum(env_idx == 1)}")
    best_weights = [None for _ in range(len(models))]


    
    for epoch in range(args.epoch):
        # for epoch in tqdm(range(args.epoch)):


        all_loss, n_bw = 0, 0
        all_div_p=0
        all_losses = [{} for _ in range(len(models))]
        contrast_loss, all_contrast_loss = torch.zeros(1).to(device), 0.
        spu_pred_loss = torch.zeros(1).to(device)

        num_batch = (len(train_loader.dataset) // args.batch_size) + int(
            (len(train_loader.dataset) % args.batch_size) > 0)
        for step, batch in tqdm(enumerate(train_loader), total=num_batch, desc=f"Epoch [{epoch}] >>  ", disable=args.no_tqdm, ncols=60):
            
            n_bw += 1

            batch.to(device)
            is_labeled = batch.y == batch.y
            edge_atts=[]
            batch_losses=[]
            

            for i,model in enumerate(models):
                graph=batch
                data_list=[]

                for j in range(len(graph.y)) :
                    data=delete_edge(graph[j],args,i+1)
                    data_list.append(data)
                from torch_geometric.data import Batch

                graph = Batch.from_data_list(data_list)

                graph.to(device)
                is_labeled = graph.y == graph.y
                model.train()
            
                torch.autograd.set_detect_anomaly(True)
                if not args.erm:
                    layer_outputs=[]
                    def hook(module, input, output):
                        layer_outputs.append(output)
                    # handle = model.module.edge_att.register_forward_hook(hook)
                    handle = model.edge_att.register_forward_hook(hook)
                
                if args.dir > 0:
                    # obtain dir losses
                    dir_loss, causal_pred, spu_pred, causal_rep = model.get_dir_loss(graph,
                                                                                    graph.y,
                                                                                    criterion,
                                                                                    is_labeled=is_labeled,
                                                                                    return_data='rep')

                    spu_loss = criterion(spu_pred[is_labeled], graph.y[is_labeled])
                    pred_loss = criterion(causal_pred[is_labeled], graph.y[is_labeled])
                    pred_loss = pred_loss + spu_loss + args.dir * (epoch ** 1.6) * dir_loss
                    all_losses['cls'] = (all_losses.get('cls', 0) * (n_bw - 1) + pred_loss.item()) / n_bw
                    all_losses['dir'] = (all_losses.get('dir', 0) * (n_bw - 1) + dir_loss.item()) / n_bw
                    all_losses['spu'] = (all_losses.get('spu', 0) * (n_bw - 1) + spu_loss.item()) / n_bw
                elif args.ginv_opt.lower() == 'gib':
                    # obtain gib loss
                    pred_loss, causal_rep = model.get_ib_loss(graph, return_data="rep")
                    all_losses['cls'] = (all_losses.get('cls', 0) * (n_bw - 1) + pred_loss.item()) / n_bw
                else:

                    # obtain ciga I(G_S;Y) losses
                    if args.spu_coe > 0 and not args.erm:

                        if args.contrast_rep.lower() == "feat":
                            (causal_pred, spu_pred), causal_rep = model(graph, return_data="feat", return_spu=True)
                        else:
                            (causal_pred, spu_pred), causal_rep = model(graph, return_data="rep", return_spu=True)

                        spu_pred_loss = criterion(spu_pred[is_labeled], graph.y[is_labeled], reduction='none')
                        pred_loss = criterion(causal_pred[is_labeled], graph.y[is_labeled], reduction='none')

                        assert spu_pred_loss.size() == pred_loss.size()
                        # hinge loss
                        spu_loss_weight = torch.zeros(spu_pred_loss.size()).to(device)
                        spu_loss_weight[spu_pred_loss > pred_loss] = 1.0

                        spu_pred_loss = spu_pred_loss.dot(spu_loss_weight) / (sum(spu_pred_loss > pred_loss) + 1e-6)
                        pred_loss = pred_loss.mean()
                        all_losses['spu'] = (all_losses.get('spu', 0) * (n_bw - 1) + spu_pred_loss.item()) / n_bw
                        all_losses['cls'] = (all_losses.get('cls', 0) * (n_bw - 1) + pred_loss.item()) / n_bw
                    else:

                        if args.contrast_rep.lower() == "feat":
                            causal_pred, causal_rep = model(graph, return_data="feat")
                        else:
                            causal_pred, causal_rep = model(graph, return_data="rep")

                        pred_loss = criterion(causal_pred[is_labeled], graph.y[is_labeled])
                        
                        all_losses[i]['cls'] = (all_losses[i].get('cls', 0) * (n_bw - 1) + pred_loss.item()) / n_bw
                contrast_loss = 0
                contrast_coe = args.contrast

                if args.contrast > 0:
                    # obtain contrast loss
                    if args.contrast_sampling.lower() in ['cnc', 'cncp']:
                        # cncp referes to only contrastig the positive examples in cnc
                        if erm_model == None:
                            model_path = os.path.join('erm_model', args.dataset) + ".pt"
                            erm_model = GNNERM(input_dim=input_dim,
                                            edge_dim=edge_dim,
                                            out_dim=num_classes,
                                            gnn_type=args.model,
                                            num_layers=args.num_layers,
                                            emb_dim=args.emb_dim,
                                            drop_ratio=args.dropout,
                                            graph_pooling=args.pooling,
                                            virtual_node=args.virtual_node).to(device)
                            erm_model.load_state_dict(torch.load(model_path, map_location=device))
                            print("Loaded model from ", model_path)
                        # obtain the erm predictions to sampling pos/neg pairs in cnc
                        erm_model.eval()
                        with torch.no_grad():
                            erm_y_pred = erm_model(graph)
                        erm_y_pred = erm_y_pred.argmax(-1)
                    else:
                        erm_y_pred = None
                    contrast_loss = get_contrast_loss(causal_rep,
                                                    graph.y.view(-1),
                                                    norm=F.normalize if not args.not_norm else None,
                                                    contrast_t=args.contrast_t,
                                                    sampling=args.contrast_sampling,
                                                    y_pred=erm_y_pred)
                    all_losses[i]['contrast'] = (all_losses[i].get('contrast', 0) * (n_bw - 1) + contrast_loss.item()) / n_bw
                    all_contrast_loss += contrast_loss.item()

                if args.num_envs > 1:
                    print('indicate invariant learning')
                    # indicate invariant learning
                    batch_env_idx = env_idx[step * args.batch_size:step * args.batch_size + graph.y.size(0)]
                    if 'molhiv' in args.dataset.lower():
                        batch_env_idx = batch_env_idx.view(graph.y.shape)
                    causal_pred, labels, batch_env_idx = causal_pred[is_labeled], graph.y[is_labeled], batch_env_idx[
                        is_labeled]
                    if args.irm_opt.lower() == 'eiil':
                        dummy_w = torch.tensor(1.).to(device).requires_grad_()
                        loss = F.nll_loss(causal_pred * dummy_w, labels, reduction='none')
                        env_w = torch.randn(batch_env_idx.size(0)).cuda().requires_grad_()
                        optimizer = torch.optim.Adam([env_w], lr=1e-3)
                        for i in range(20):
                            # penalty for env a
                            lossa = (loss.squeeze() * env_w.sigmoid()).mean()
                            grada = torch.autograd.grad(lossa, [dummy_w], create_graph=True)[0]
                            penaltya = torch.sum(grada ** 2)
                            # penalty for env b
                            lossb = (loss.squeeze() * (1 - env_w.sigmoid())).mean()
                            gradb = torch.autograd.grad(lossb, [dummy_w], create_graph=True)[0]
                            penaltyb = torch.sum(gradb ** 2)
                            # negate
                            npenalty = -torch.stack([penaltya, penaltyb]).mean()
                            # step
                            optimizer.zero_grad()
                            npenalty.backward(retain_graph=True)
                            optimizer.step()
                        new_batch_env_idx = (env_w.sigmoid() > 0.5).long()
                        env_idx[step * args.batch_size:step * args.batch_size +
                                                    graph.y.size(0)][labels] = new_batch_env_idx.to(env_idx.device)
                        irm_loss = get_irm_loss(causal_pred, labels, new_batch_env_idx, criterion=criterion)
                    elif args.irm_opt.lower() == 'ib-irm':
                        ib_penalty = causal_rep.var(dim=0).mean()
                        irm_loss = get_irm_loss(causal_pred, labels, batch_env_idx,
                                                criterion=criterion) + ib_penalty / args.irm_p
                        all_losses['ib'] = (all_losses.get('ib', 0) * (n_bw - 1) + ib_penalty.item()) / n_bw
                    elif args.irm_opt.lower() == 'vrex':
                        loss_0 = criterion(causal_pred[batch_env_idx == 0], labels[batch_env_idx == 0])
                        loss_1 = criterion(causal_pred[batch_env_idx == 1], labels[batch_env_idx == 1])
                        irm_loss = torch.var(torch.FloatTensor([loss_0, loss_1]).to(device))
                    else:
                        irm_loss = get_irm_loss(causal_pred, labels, batch_env_idx, criterion=criterion)
                    all_losses['irm'] = (all_losses.get('irm', 0) * (n_bw - 1) + irm_loss.item()) / n_bw
                    pred_loss += irm_loss * args.irm_p

                batch_loss =pred_loss + contrast_coe * contrast_loss + args.spu_coe * spu_pred_loss 
                batch_losses.append(batch_loss)
                
                if not args.erm:
                    edge_atts.append(layer_outputs[0])
                    handle.remove()
            if not args.erm:
                for i in range(len(models)):
                    f_min, f_max = edge_atts[i].min(), edge_atts[i].max()
                    edge_atts[i] = (edge_atts[i] - f_min) / (f_max - f_min+ 1e-6 )

                for i in range(len(models)):
                    reg=1e-2
                    diversity_p=args.penalty*reg*sum([torch.mm(edge_atts[i].T ,edge_atts[j]).squeeze() for j in range(len(edge_atts)) if j != i])
                    batch_losses[i]+=diversity_p

                    all_div_p+=diversity_p

            
            total_loss=sum(batch_losses)

            for opt in model_optimizers:
                opt.zero_grad()
            total_loss.backward()

            for opt in model_optimizers:
                opt.step()

            

        all_contrast_loss /= n_bw
        all_loss /= n_bw
        all_div_p /=n_bw

        for i,model in enumerate(models):
            model.eval()
            train_acc = eval_model(model, device, train_loader, evaluator, eval_metric=eval_metric)
            val_acc = eval_model(model, device, valid_loader, evaluator, eval_metric=eval_metric)
            test_acc = eval_model(model,
                                device,
                                test_loader,
                                evaluator,
                                eval_metric=eval_metric)
            print("      [{:3d}/{:d}]".format(epoch, args.epoch) +
                    "\n       train_ACC: {:.4f} "
                    "\n       valid_ACC: {:.4f} "
                    "\n       tests_ACC: {:.4f} \n".format(
                        train_acc,
                        val_acc,
                        test_acc))
            if val_acc <= last_val_acc[i]:
                pass
            else:
                last_train_acc[i] = train_acc
                last_val_acc[i] = val_acc
                last_test_acc[i] = test_acc

                if args.save_model:

                    best_weights[i] = deepcopy(model.state_dict())


    #
    if args.save_model:
        print("Saving best weights..")
        for seed in range(len(models)):
            if args.erm:
                model_path = os.path.join(f'erm_{args.dataset}_{args.model}_{args.classifier_input_feat}_dout{args.dropout}_penalty/{args.sample_rate}%subgraph({args.drop_mode})100%feature_{args.dataset}_{args.contrast}_shared_init_{args.init}_{seed+1}_{args.bias}_{args.penalty}') + ".pt"
                isExists=os.path.exists(f'erm_{args.dataset}_{args.model}_{args.classifier_input_feat}_dout{args.dropout}_penalty')
                if not isExists:
                    os.makedirs(f'erm_{args.dataset}_{args.model}_{args.classifier_input_feat}_dout{args.dropout}_penalty') 
            else:
                if args.dataset.lower() in ['spmotif','sumotif']:
                    model_path = os.path.join(f'{args.dataset}_{args.model}_{args.classifier_input_feat}_dout{args.dropout}_r{args.r}_penalty/{args.sample_rate}%subgraph({args.drop_mode})100%feature_{args.dataset}_{args.contrast}_shared_init_{args.init}_{seed+1}_{args.bias}_{args.penalty}') + ".pt"
                    
                else:
                    model_path = os.path.join(f'{args.dataset}_{args.model}_{args.classifier_input_feat}_dout{args.dropout}_r{args.r}_penalty/{args.sample_rate}%subgraph({args.drop_mode})100%feature_{args.dataset}_{args.contrast}_shared_init_{args.init}_{seed+1}_{args.penalty}') + ".pt"
                isExists=os.path.exists(f'{args.dataset}_{args.model}_{args.classifier_input_feat}_dout{args.dropout}_r{args.r}_penalty')
                if not isExists:
                    os.makedirs(f'{args.dataset}_{args.model}_{args.classifier_input_feat}_dout{args.dropout}_r{args.r}_penalty') 

            for k, v in best_weights[seed].items():
                best_weights[seed][k] = v.cpu()
            torch.save(best_weights[seed], model_path)
        print("Done..")



    print("\n\n\n")
    torch.cuda.empty_cache()


if __name__ == "__main__":
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
    # emb dim of classifier
    parser.add_argument('-c_dim', '--classifier_emb_dim', default=32, type=int)
    # inputs of classifier
    # raw:  raw feat
    # feat: hidden feat from featurizer
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
    parser.add_argument('-c_pool', '--contrast_pooling', default='add', type=str)
    parser.add_argument('-s_rep', '--spurious_rep', default='rep', type=str)

    parser.add_argument('--spu_coe', default=0, type=float)

    # misc
    parser.add_argument('--no_tqdm', action='store_true')
    parser.add_argument('--commit', default='', type=str, help='experiment name')
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--sample_rate',default=100,type=int,help='sample rate')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay (default: 1e-4)')
    parser.add_argument('--init', type=int,default=1)
    # strength of the diversity reg, \beta in the paper
    parser.add_argument('--penalty', type=float,default=1)
    args = parser.parse_args()
    main(args)
