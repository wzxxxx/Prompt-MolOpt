import pandas as pd
from rdkit import Chem
import torch as th
from maskgnn import collate_molgraphs, EarlyStopping, run_a_train_epoch, \
    run_an_eval_epoch, set_random_seed, RGCN, Meter
import pickle as pkl
from build_data import build_mol_graph_for_one_mol, MolGraphDataset, collate_molgraphs
from torch.utils.data import DataLoader
from rdkit import Chem
import itertools
from typing import List


# fix parameters of model
def SME_pred_for_mols(smis, model_name, rgcn_hidden_feats=[64, 64, 64], ffn_hidden_feats=128, batch_size=128,
                      lr=0.0003, classification=False):
    args = {}
    args['device'] = "cuda"
    args['node_data_field'] = 'node'
    args['edge_data_field'] = 'edge'
    args['substructure_mask'] = 'smask'
    # model parameter
    args['num_epochs'] = 500
    args['patience'] = 30
    args['batch_size'] = batch_size
    args['mode'] = 'higher'
    args['in_feats'] = 40
    args['classification'] = classification
    args['rgcn_hidden_feats'] = rgcn_hidden_feats
    args['ffn_hidden_feats'] = ffn_hidden_feats
    args['rgcn_drop_out'] = 0
    args['ffn_drop_out'] = 0
    args['lr'] = lr
    args['loop'] = True
    # task name (model name)
    args['task_name'] = model_name  # change
    y_pred_sum = None
    dataset = MolGraphDataset(smis)
    data_loader = DataLoader(dataset, batch_size=args['batch_size'], collate_fn=collate_molgraphs)
    for seed in range(10):
        model = RGCN(ffn_hidden_feats=args['ffn_hidden_feats'],
                     ffn_dropout=args['ffn_drop_out'],
                     rgcn_node_feats=args['in_feats'], rgcn_hidden_feats=args['rgcn_hidden_feats'],
                     rgcn_drop_out=args['rgcn_drop_out'],
                     classification=args['classification'])
        stopper = EarlyStopping(patience=args['patience'], task_name=args['task_name'] + '_' + str(seed + 1),
                                mode=args['mode'])
        model.to(args['device'])
        stopper.load_checkpoint(model)
        model.eval()
        eval_meter = Meter()
        for batch_idx, batch_mol_bg in enumerate(data_loader):
            print('{} {}/{}'.format(seed, batch_idx+1, len(data_loader)))
            batch_mol_bg = batch_mol_bg.to(args['device'])
            with th.no_grad():
                rgcn_node_feats = batch_mol_bg.ndata.pop(args['node_data_field']).float().to(args['device'])
                rgcn_edge_feats = batch_mol_bg.edata.pop(args['edge_data_field']).long().to(args['device'])
                smask_feats = batch_mol_bg.ndata.pop(args['substructure_mask']).unsqueeze(dim=1).float().to(
                    args['device'])

                preds, weight = model(batch_mol_bg, rgcn_node_feats, rgcn_edge_feats, smask_feats)
                eval_meter.update(preds, preds)
                th.cuda.empty_cache()
        _, y_pred = eval_meter.compute_metric('return_pred_true')
        if args['classification']:
            y_pred = th.sigmoid(y_pred)
            y_pred = y_pred.squeeze().numpy()
        else:
            y_pred = y_pred.squeeze().numpy()
        if y_pred_sum is None:
            y_pred_sum = y_pred
        else:
            y_pred_sum += y_pred
    y_pred_mean = y_pred_sum / 10
    return y_pred_mean.tolist()




generate_data = pd.read_csv('../data/generate_data/single_opt_data_train.csv')
for model_name in ['ESOL', 'Mutagenicity', 'hERG', 'BBBP']:
    with open('../result/hyperparameter_{}.pkl'.format(model_name), 'rb') as f:
        hyperparameter = pkl.load(f)
    origin_smi_list = generate_data['src_smi'].tolist()
    opt_smi_list = generate_data['tgt_smi'].tolist()
    origin_pred_list = SME_pred_for_mols(smis=origin_smi_list, model_name=model_name, batch_size=256,
                                         rgcn_hidden_feats=hyperparameter['rgcn_hidden_feats'],
                                         ffn_hidden_feats=hyperparameter['ffn_hidden_feats'],
                                         lr=hyperparameter['lr'], classification=hyperparameter['classification'])
    opt_pred_list = SME_pred_for_mols(smis=opt_smi_list, model_name=model_name, batch_size=256,
                                      rgcn_hidden_feats=hyperparameter['rgcn_hidden_feats'],
                                      ffn_hidden_feats=hyperparameter['ffn_hidden_feats'],
                                      lr=hyperparameter['lr'], classification=hyperparameter['classification'])
    generate_data['src_pred_{}'.format(model_name)] = origin_pred_list
    generate_data['tgt_pred_{}'.format(model_name)] = opt_pred_list
generate_data.to_csv('../data/generate_data/single_opt_data_train_all_pred.csv'.format(model_name), index=False)