import random
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

# opt
for generate_data_name in ['Mutagenicity_ESOL', 'Mutagenicity_lipop', 'BBBP_lipop']:
    for mode in ['higher', 'lower']:
        generate_data = pd.read_csv('../data/generate_data/{}_{}_data.csv'.format(generate_data_name, mode))
        for model_name in ['ESOL', 'Mutagenicity', 'hERG', 'BBBP', 'lipop']:
            with open('../result/hyperparameter_{}.pkl'.format(model_name), 'rb') as f:
                hyperparameter = pkl.load(f)
            if model_name in generate_data_name:
                print('{} {} {}'.format(generate_data_name, model_name, mode))


                origin_smi_list = generate_data['origin_smi'].tolist()
                opt_smi_list = generate_data['opt_smi'].tolist()

                origin_pred_list = SME_pred_for_mols(smis=origin_smi_list, model_name=model_name, batch_size=32,
                                                  rgcn_hidden_feats=hyperparameter['rgcn_hidden_feats'],
                                                  ffn_hidden_feats=hyperparameter['ffn_hidden_feats'],
                                                  lr=hyperparameter['lr'], classification=hyperparameter['classification'])
                opt_pred_list = SME_pred_for_mols(smis=opt_smi_list, model_name=model_name, batch_size=32,
                                                  rgcn_hidden_feats=hyperparameter['rgcn_hidden_feats'],
                                                  ffn_hidden_feats=hyperparameter['ffn_hidden_feats'],
                                                  lr=hyperparameter['lr'], classification=hyperparameter['classification'])
                generate_data[f'origin_pred_{model_name}'] = origin_pred_list
                generate_data[f'opt_pred_{model_name}'] = opt_pred_list
        # 对数据做简单处理，删除不成功的分子，并添加标签(类似数据扩充)
        # 对数据做简单处理，删除不成功的分子，并添加标签(类似数据扩充)
        if (generate_data_name == 'Mutagenicity_ESOL') & (mode == 'higher'):
            generate_data = generate_data[
                (generate_data['origin_pred_Mutagenicity'] < 0.5) & (generate_data['opt_pred_Mutagenicity'] >= 0.5)]
            generate_data = generate_data[(generate_data['origin_pred_ESOL'] - generate_data['opt_pred_ESOL']) >= 0.5]
            generate_data['task_name'] = ['Mutagenicity ESOL' for x in range(len(generate_data))]
            generate_data_2 = generate_data.copy()
            generate_data_2['task_name'] = ['ESOL Mutagenicity' for x in range(len(generate_data))]
            generate_data = pd.concat([generate_data, generate_data_2], axis=0)
        if (generate_data_name == 'Mutagenicity_ESOL') & (mode == 'lower'):
            generate_data = generate_data[
                (generate_data['origin_pred_Mutagenicity'] >= 0.5) & (generate_data['opt_pred_Mutagenicity'] < 0.5)]
            generate_data = generate_data[(generate_data['origin_pred_ESOL'] - generate_data['opt_pred_ESOL']) <= -0.5]
            generate_data['task_name'] = ['Mutagenicity ESOL' for x in range(len(generate_data))]
            generate_data_2 = generate_data.copy()
            generate_data_2['task_name'] = ['ESOL Mutagenicity' for x in range(len(generate_data))]
            generate_data = pd.concat([generate_data, generate_data_2], axis=0)

        if (generate_data_name == 'Mutagenicity_lipop') & (mode == 'higher'):
            generate_data = generate_data[
                (generate_data['origin_pred_Mutagenicity'] < 0.5) & (generate_data['opt_pred_Mutagenicity'] >= 0.5)]
            generate_data = generate_data[(generate_data['origin_pred_lipop'] - generate_data['opt_pred_lipop']) >= 0.5]
            generate_data['task_name'] = ['Mutagenicity lipop' for x in range(len(generate_data))]
            generate_data_2 = generate_data.copy()
            generate_data_2['task_name'] = ['lipop Mutagenicity' for x in range(len(generate_data))]
            generate_data = pd.concat([generate_data, generate_data_2], axis=0)
        if (generate_data_name == 'Mutagenicity_lipop') & (mode == 'lower'):
            generate_data = generate_data[
                (generate_data['origin_pred_Mutagenicity'] >= 0.5) & (generate_data['opt_pred_Mutagenicity'] < 0.5)]
            generate_data = generate_data[
                (generate_data['origin_pred_lipop'] - generate_data['opt_pred_lipop']) <= -0.5]
            generate_data['task_name'] = ['Mutagenicity lipop' for x in range(len(generate_data))]
            generate_data_2 = generate_data.copy()
            generate_data_2['task_name'] = ['lipop Mutagenicity' for x in range(len(generate_data))]
            generate_data = pd.concat([generate_data, generate_data_2], axis=0)

        if (generate_data_name == 'BBBP_lipop') & (mode == 'higher'):
            generate_data = generate_data[
                (generate_data['origin_pred_BBBP'] < 0.5) & (generate_data['opt_pred_BBBP'] >= 0.5)]
            generate_data = generate_data[
                (generate_data['origin_pred_lipop'] - generate_data['opt_pred_lipop']) <= -0.5]
            generate_data['task_name'] = ['BBBP lipop' for x in range(len(generate_data))]
            generate_data_2 = generate_data.copy()
            generate_data_2['task_name'] = ['lipop BBBP' for x in range(len(generate_data))]
            generate_data = pd.concat([generate_data, generate_data_2], axis=0)
        if (generate_data_name == 'BBBP_lipop') & (mode == 'lower'):
            generate_data = generate_data[
                (generate_data['origin_pred_BBBP'] >= 0.5) & (generate_data['opt_pred_BBBP'] < 0.5)]
            generate_data = generate_data[
                (generate_data['origin_pred_lipop'] - generate_data['opt_pred_lipop']) >= -0.5]
            generate_data['task_name'] = ['BBBP lipop' for x in range(len(generate_data))]
            generate_data_2 = generate_data.copy()
            generate_data_2['task_name'] = ['lipop BBBP' for x in range(len(generate_data))]
            generate_data = pd.concat([generate_data, generate_data_2], axis=0)
        # if (generate_data_name == 'hERG_ESOL') & (mode=='higher'):
        #     generate_data = generate_data[(generate_data['origin_pred_hERG']<0.5)&(generate_data['opt_pred_hERG']>=0.5)]
        #     generate_data = generate_data[(generate_data['origin_pred_ESOL']-generate_data['opt_pred_ESOL'])>=0.5]
        #     generate_data['task_name'] = ['hERG ESOL' for x in range(len(generate_data))]
        #     generate_data_2 = generate_data.copy()
        #     generate_data_2['task_name'] = ['ESOL hERG' for x in range(len(generate_data))]
        #     generate_data = pd.concat([generate_data, generate_data_2], axis=0)
        # if (generate_data_name == 'hERG_ESOL') & (mode=='lower'):
        #     generate_data = generate_data[(generate_data['origin_pred_hERG']>=0.5)&(generate_data['opt_pred_hERG']<0.5)]
        #     generate_data = generate_data[(generate_data['origin_pred_ESOL']-generate_data['opt_pred_ESOL'])<=-0.5]
        #     generate_data['task_name'] = ['hERG ESOL' for x in range(len(generate_data))]
        #     generate_data_2 = generate_data.copy()
        #     generate_data_2['task_name'] = ['ESOL hERG' for x in range(len(generate_data))]
        #     generate_data = pd.concat([generate_data, generate_data_2], axis=0)
        # if (generate_data_name == 'Mutagenicity_BBBP') & (mode=='higher'):
        #     generate_data = generate_data[(generate_data['origin_pred_Mutagenicity']<0.5)&(generate_data['opt_pred_Mutagenicity']>=0.5)]
        #     generate_data = generate_data[(generate_data['origin_pred_BBBP']>=0.5)&(generate_data['opt_pred_BBBP']<0.5)]
        #     generate_data['task_name'] = ['Mutagenicity BBBP' for x in range(len(generate_data))]
        #     generate_data_2 = generate_data.copy()
        #     generate_data_2['task_name'] = ['BBBP Mutagenicity' for x in range(len(generate_data))]
        #     generate_data = pd.concat([generate_data, generate_data_2], axis=0)
        # if (generate_data_name == 'Mutagenicity_BBBP') & (mode=='lower'):
        #     generate_data = generate_data[(generate_data['origin_pred_Mutagenicity']>=0.5)&(generate_data['opt_pred_Mutagenicity']<0.5)]
        #     generate_data = generate_data[(generate_data['origin_pred_BBBP']<0.5)&(generate_data['opt_pred_BBBP']>=0.5)]
        #     generate_data['task_name'] = ['Mutagenicity BBBP' for x in range(len(generate_data))]
        #     generate_data_2 = generate_data.copy()
        #     generate_data_2['task_name'] = ['BBBP Mutagenicity' for x in range(len(generate_data))]
        #     generate_data = pd.concat([generate_data, generate_data_2], axis=0)
        # if (generate_data_name == 'hERG_Mutagenicity') & (mode=='higher'):
        #     generate_data = generate_data[(generate_data['origin_pred_hERG']<0.5)&(generate_data['opt_pred_hERG']>=0.5)]
        #     generate_data = generate_data[(generate_data['origin_pred_Mutagenicity']<0.5)&(generate_data['opt_pred_Mutagenicity']>=0.5)]
        #     generate_data['task_name'] = ['hERG Mutagenicity' for x in range(len(generate_data))]
        #     generate_data_2 = generate_data.copy()
        #     generate_data_2['task_name'] = ['Mutagenicity hERG' for x in range(len(generate_data))]
        #     generate_data = pd.concat([generate_data, generate_data_2], axis=0)
        # if (generate_data_name == 'hERG_Mutagenicity') & (mode=='lower'):
        #     generate_data = generate_data[(generate_data['origin_pred_hERG']>=0.5)&(generate_data['opt_pred_hERG']<0.5)]
        #     generate_data = generate_data[(generate_data['origin_pred_Mutagenicity']>=0.5)&(generate_data['opt_pred_Mutagenicity']<0.5)]
        #     generate_data['task_name'] = ['hERG Mutagenicity' for x in range(len(generate_data))]
        #     generate_data_2 = generate_data.copy()
        #     generate_data_2['task_name'] = ['Mutagenicity hERG' for x in range(len(generate_data))]
        #     generate_data = pd.concat([generate_data, generate_data_2], axis=0)
        generate_data.to_csv('../data/generate_data/{}_{}_data_pred.csv'.format(generate_data_name, mode))

# re_opt
for generate_data_name in ['Mutagenicity_ESOL', 'Mutagenicity_lipop', 'BBBP_lipop']:
    for mode in ['higher', 'lower']:
        generate_data = pd.read_csv('../data/generate_data/re_opt_{}_{}_data.csv'.format(generate_data_name, mode))
        for model_name in ['ESOL', 'Mutagenicity', 'hERG', 'BBBP', 'lipop']:
            with open('../result/hyperparameter_{}.pkl'.format(model_name), 'rb') as f:
                hyperparameter = pkl.load(f)
            if model_name in generate_data_name:
                print('{} {} {}'.format(generate_data_name, model_name, mode))


                origin_smi_list = generate_data['origin_smi'].tolist()
                opt_smi_list = generate_data['opt_smi'].tolist()

                origin_pred_list = SME_pred_for_mols(smis=origin_smi_list, model_name=model_name, batch_size=32,
                                                  rgcn_hidden_feats=hyperparameter['rgcn_hidden_feats'],
                                                  ffn_hidden_feats=hyperparameter['ffn_hidden_feats'],
                                                  lr=hyperparameter['lr'], classification=hyperparameter['classification'])
                opt_pred_list = SME_pred_for_mols(smis=opt_smi_list, model_name=model_name, batch_size=32,
                                                  rgcn_hidden_feats=hyperparameter['rgcn_hidden_feats'],
                                                  ffn_hidden_feats=hyperparameter['ffn_hidden_feats'],
                                                  lr=hyperparameter['lr'], classification=hyperparameter['classification'])
                generate_data[f'origin_pred_{model_name}'] = origin_pred_list
                generate_data[f'opt_pred_{model_name}'] = opt_pred_list
        # 对数据做简单处理，删除不成功的分子，并添加标签(类似数据扩充)
        if (generate_data_name == 'Mutagenicity_ESOL') & (mode=='higher'):
            generate_data = generate_data[(generate_data['origin_pred_Mutagenicity']<0.5)&(generate_data['opt_pred_Mutagenicity']>=0.5)]
            generate_data = generate_data[(generate_data['origin_pred_ESOL']-generate_data['opt_pred_ESOL'])>=0.5]
            generate_data['task_name'] = ['Mutagenicity ESOL' for x in range(len(generate_data))]
            generate_data_2 = generate_data.copy()
            generate_data_2['task_name'] = ['ESOL Mutagenicity' for x in range(len(generate_data))]
            generate_data = pd.concat([generate_data, generate_data_2], axis=0)
        if (generate_data_name == 'Mutagenicity_ESOL') & (mode=='lower'):
            generate_data = generate_data[(generate_data['origin_pred_Mutagenicity']>=0.5)&(generate_data['opt_pred_Mutagenicity']<0.5)]
            generate_data = generate_data[(generate_data['origin_pred_ESOL']-generate_data['opt_pred_ESOL'])<=-0.5]
            generate_data['task_name'] = ['Mutagenicity ESOL' for x in range(len(generate_data))]
            generate_data_2 = generate_data.copy()
            generate_data_2['task_name'] = ['ESOL Mutagenicity' for x in range(len(generate_data))]
            generate_data = pd.concat([generate_data, generate_data_2], axis=0)
        
        if (generate_data_name == 'Mutagenicity_lipop') & (mode=='higher'):
            generate_data = generate_data[(generate_data['origin_pred_Mutagenicity']<0.5)&(generate_data['opt_pred_Mutagenicity']>=0.5)]
            generate_data = generate_data[(generate_data['origin_pred_lipop']-generate_data['opt_pred_lipop'])>=0.5]
            generate_data['task_name'] = ['Mutagenicity lipop' for x in range(len(generate_data))]
            generate_data_2 = generate_data.copy()
            generate_data_2['task_name'] = ['lipop Mutagenicity' for x in range(len(generate_data))]
            generate_data = pd.concat([generate_data, generate_data_2], axis=0)
        if (generate_data_name == 'Mutagenicity_lipop') & (mode=='lower'):
            generate_data = generate_data[(generate_data['origin_pred_Mutagenicity']>=0.5)&(generate_data['opt_pred_Mutagenicity']<0.5)]
            generate_data = generate_data[(generate_data['origin_pred_lipop']-generate_data['opt_pred_lipop'])<=-0.5]
            generate_data['task_name'] = ['Mutagenicity lipop' for x in range(len(generate_data))]
            generate_data_2 = generate_data.copy()
            generate_data_2['task_name'] = ['lipop Mutagenicity' for x in range(len(generate_data))]
            generate_data = pd.concat([generate_data, generate_data_2], axis=0)
            
        if (generate_data_name == 'BBBP_lipop') & (mode=='higher'):
            generate_data = generate_data[(generate_data['origin_pred_BBBP']<0.5)&(generate_data['opt_pred_BBBP']>=0.5)]
            generate_data = generate_data[(generate_data['origin_pred_lipop']-generate_data['opt_pred_lipop'])<=-0.5]
            generate_data['task_name'] = ['BBBP lipop' for x in range(len(generate_data))]
            generate_data_2 = generate_data.copy()
            generate_data_2['task_name'] = ['lipop BBBP' for x in range(len(generate_data))]
            generate_data = pd.concat([generate_data, generate_data_2], axis=0)
        if (generate_data_name == 'BBBP_lipop') & (mode=='lower'):
            generate_data = generate_data[(generate_data['origin_pred_BBBP']>=0.5)&(generate_data['opt_pred_BBBP']<0.5)]
            generate_data = generate_data[(generate_data['origin_pred_lipop']-generate_data['opt_pred_lipop'])>=-0.5]
            generate_data['task_name'] = ['BBBP lipop' for x in range(len(generate_data))]
            generate_data_2 = generate_data.copy()
            generate_data_2['task_name'] = ['lipop BBBP' for x in range(len(generate_data))]
            generate_data = pd.concat([generate_data, generate_data_2], axis=0)
        
        # if (generate_data_name == 'hERG_ESOL') & (mode=='higher'):
        #     generate_data = generate_data[(generate_data['origin_pred_hERG']<0.5)&(generate_data['opt_pred_hERG']>=0.5)]
        #     generate_data = generate_data[(generate_data['origin_pred_ESOL']-generate_data['opt_pred_ESOL'])>=0.5]
        #     generate_data['task_name'] = ['hERG ESOL' for x in range(len(generate_data))]
        #     generate_data_2 = generate_data.copy()
        #     generate_data_2['task_name'] = ['ESOL hERG' for x in range(len(generate_data))]
        #     generate_data = pd.concat([generate_data, generate_data_2], axis=0)
        # if (generate_data_name == 'hERG_ESOL') & (mode=='lower'):
        #     generate_data = generate_data[(generate_data['origin_pred_hERG']>=0.5)&(generate_data['opt_pred_hERG']<0.5)]
        #     generate_data = generate_data[(generate_data['origin_pred_ESOL']-generate_data['opt_pred_ESOL'])<=-0.5]
        #     generate_data['task_name'] = ['hERG ESOL' for x in range(len(generate_data))]
        #     generate_data_2 = generate_data.copy()
        #     generate_data_2['task_name'] = ['ESOL hERG' for x in range(len(generate_data))]
        #     generate_data = pd.concat([generate_data, generate_data_2], axis=0)
        # if (generate_data_name == 'Mutagenicity_BBBP') & (mode=='higher'):
        #     generate_data = generate_data[(generate_data['origin_pred_Mutagenicity']<0.5)&(generate_data['opt_pred_Mutagenicity']>=0.5)]
        #     generate_data = generate_data[(generate_data['origin_pred_BBBP']>=0.5)&(generate_data['opt_pred_BBBP']<0.5)]
        #     generate_data['task_name'] = ['Mutagenicity BBBP' for x in range(len(generate_data))]
        #     generate_data_2 = generate_data.copy()
        #     generate_data_2['task_name'] = ['BBBP Mutagenicity' for x in range(len(generate_data))]
        #     generate_data = pd.concat([generate_data, generate_data_2], axis=0)
        # if (generate_data_name == 'Mutagenicity_BBBP') & (mode=='lower'):
        #     generate_data = generate_data[(generate_data['origin_pred_Mutagenicity']>=0.5)&(generate_data['opt_pred_Mutagenicity']<0.5)]
        #     generate_data = generate_data[(generate_data['origin_pred_BBBP']<0.5)&(generate_data['opt_pred_BBBP']>=0.5)]
        #     generate_data['task_name'] = ['Mutagenicity BBBP' for x in range(len(generate_data))]
        #     generate_data_2 = generate_data.copy()
        #     generate_data_2['task_name'] = ['BBBP Mutagenicity' for x in range(len(generate_data))]
        #     generate_data = pd.concat([generate_data, generate_data_2], axis=0)
        # if (generate_data_name == 'hERG_Mutagenicity') & (mode=='higher'):
        #     generate_data = generate_data[(generate_data['origin_pred_hERG']<0.5)&(generate_data['opt_pred_hERG']>=0.5)]
        #     generate_data = generate_data[(generate_data['origin_pred_Mutagenicity']<0.5)&(generate_data['opt_pred_Mutagenicity']>=0.5)]
        #     generate_data['task_name'] = ['hERG Mutagenicity' for x in range(len(generate_data))]
        #     generate_data_2 = generate_data.copy()
        #     generate_data_2['task_name'] = ['Mutagenicity hERG' for x in range(len(generate_data))]
        #     generate_data = pd.concat([generate_data, generate_data_2], axis=0)
        # if (generate_data_name == 'hERG_Mutagenicity') & (mode=='lower'):
        #     generate_data = generate_data[(generate_data['origin_pred_hERG']>=0.5)&(generate_data['opt_pred_hERG']<0.5)]
        #     generate_data = generate_data[(generate_data['origin_pred_Mutagenicity']>=0.5)&(generate_data['opt_pred_Mutagenicity']<0.5)]
        #     generate_data['task_name'] = ['hERG Mutagenicity' for x in range(len(generate_data))]
        #     generate_data_2 = generate_data.copy()
        #     generate_data_2['task_name'] = ['Mutagenicity hERG' for x in range(len(generate_data))]
        #     generate_data = pd.concat([generate_data, generate_data_2], axis=0)
        generate_data.to_csv('../data/generate_data/re_opt_{}_{}_data_pred.csv'.format(generate_data_name, mode))
