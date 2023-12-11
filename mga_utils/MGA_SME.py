import numpy as np
import build_data
import pandas as pd
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from maskgnn import collate_molgraphs, EarlyStopping, run_a_train_epoch, \
    run_an_eval_epoch, set_random_seed, MGA, pos_weight
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import pickle as pkl


# fix parameters of model
def MGA_SME_hyperopt(times, data_name, max_evals=30,):
    args = {}
    args['device'] = "cuda"
    args['node_data_field'] = 'node'
    args['edge_data_field'] = 'edge'
    args['classification_metric_name'] = 'roc_auc'
    args['regression_metric_name'] = 'r2'
    args['substructure_mask'] = 'smask'
    # model parameter
    args['num_epochs'] = 500
    args['patience'] = 30
    args['batch_size'] = 256
    args['in_feats'] = 40
    args['max_evals'] = max_evals
    args['loop'] = True
    args['mode'] = 'higher'
    # task name (model name)
    args['task_name'] = data_name  # change
    args['data_name'] = data_name  # change
    args['times'] = times

    args['task_name'] = data_name  # change
    args['data_name'] = data_name  # change
    args['bin_path'] = '../data/graph_data/' + args['data_name'] + '.bin'
    args['group_path'] = '../data/graph_data/' + args['data_name'] + '_group.csv'
    args['origin_path'] = '../data/origin_data/' + args['data_name'] + '.csv'
    args['classification_num'] = 0
    args['regression_num'] = 0
    columns_list = pd.read_csv(args['origin_path']).columns.tolist()
    args['task_name_list'] = [x for x in columns_list if x not in ['smiles', 'group', 'origin_smi']]
    print(args['task_name_list'])
    # generate classification_num
    for task in args['task_name_list']:
        if task in ['Pgp-inh', 'Pgp-sub', 'HIA', 'F(20%)', 'F(30%)', 'BBB', 'CYP1A2-inh', 'CYP1A2-sub', 'CYP2C19-inh', 'CYP2C19-sub', 'CYP2C9-inh', 'CYP2C9-sub', 'CYP2D6-inh',
                    'CYP2D6-sub', 'CYP3A4-inh', 'CYP3A4-sub', 'T12', 'hERG', 'H-HT', 'DILI', 'Ames', 'ROA', 'FDAMDD', 'SkinSen', 'Carcinogenicity', 'EC', 'EI',
                    'Respiratory', 'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5',
                    'SR-HSE', 'SR-MMP', 'SR-p53', 'Mutagenicity', 'BBBP', 'drd2']:
            args['classification_num'] = args['classification_num'] + 1
        if task in ['LogS', 'LogD', 'LogP', 'PPB', 'VDss', 'Fu', 'CL', 'BCF', 'IGC50', 'LC50', 'LC50DM', 'Caco-2', 'ESOL', 'lipop', 'plogp', 'qed']:
            args['regression_num'] = args['regression_num'] + 1

    # generate classification_num
    if args['classification_num'] != 0 and args['regression_num'] != 0:
        args['task_class'] = 'classification_regression'
    if args['classification_num'] != 0 and args['regression_num'] == 0:
        args['task_class'] = 'classification'
    if args['classification_num'] == 0 and args['regression_num'] != 0:
        args['task_class'] = 'regression'
    print('Classification task:{}, Regression Task:{}'.format(args['classification_num'], args['regression_num']))

    result_pd = pd.DataFrame(columns=args['task_name_list'] + ['group'] + args['task_name_list'] + ['group']
                                     + args['task_name_list'] + ['group'])

    space = {'rgcn_hidden_feats': hp.choice('rgcn_hidden_feats',
                                            [[64, 64], [128, 128], [256, 256], [64, 64, 64], [128, 128, 128],
                                             [256, 256, 256]]),
             'ffn_hidden_feats': hp.choice('ffn_hidden_feats', [64, 128, 256]),
             'ffn_drop_out': hp.choice('ffn_drop_out', [0, 0.1, 0.2, 0.3, 0.4, 0.5]),
             'rgcn_drop_out': hp.choice('rgcn_drop_out', [0, 0.1, 0.2, 0.3, 0.4, 0.5]),
             'lr': hp.choice('lr', [0.003, 0.001, 0.0003, 0.0001]),
             }

    args['task_number'] = args['classification_num'] + args['regression_num']

    train_set, val_set, test_set = build_data.load_graph_from_csv_bin_for_splited(
        bin_path=args['bin_path'], group_path=args['group_path'], random_shuffle=False, seed=2022)

    print("Molecule graph is loaded!")
    train_loader = DataLoader(dataset=train_set,
                              batch_size=args['batch_size'],
                              shuffle=True,
                              collate_fn=collate_molgraphs)

    val_loader = DataLoader(dataset=val_set,
                            batch_size=args['batch_size'],
                            collate_fn=collate_molgraphs)

    test_loader = DataLoader(dataset=test_set,
                             batch_size=args['batch_size'],
                             collate_fn=collate_molgraphs)
    pos_weight_np = pos_weight(train_set, classification_num=args['classification_num'])
    loss_criterion_c = torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight_np.to(args['device']))
    loss_criterion_r = torch.nn.MSELoss(reduction='none')

    if max_evals !=0:
    
        def hyperopt_my_mga(parameter):
            model = MGA(ffn_hidden_feats=parameter['ffn_hidden_feats'], n_tasks=args['task_number'],
                        ffn_dropout=parameter['ffn_drop_out'],
                        in_feats=args['in_feats'], rgcn_hidden_feats=parameter['rgcn_hidden_feats'],
                        rgcn_drop_out=parameter['rgcn_drop_out'])
            stopper = EarlyStopping(patience=args['patience'], task_name=args['task_name'], mode=args['mode'])
            model.to(args['device'])
            for epoch in range(args['num_epochs']):
                # Train
                lr = parameter['lr']
                optimizer = Adam(model.parameters(), lr=lr)

                _ = run_a_train_epoch(args, model, train_loader, loss_criterion_c, loss_criterion_r, optimizer)
                # Validation and early stop
                train_score = np.mean(run_an_eval_epoch(args, model, train_loader, out_path=None))
                val_score = np.mean(run_an_eval_epoch(args, model, val_loader, out_path=None))
                early_stop = stopper.step(val_score, model)
                print('epoch {:d}/{:d}, lr: {:.6f}, train: {:.4f}, valid: {:.4f}, best valid score {:.4f}'.format(
                      epoch + 1, args['num_epochs'], lr,  train_score, val_score,
                      stopper.best_score))
                if early_stop:
                    break
            stopper.load_checkpoint(model)
            val_score = -np.mean(run_an_eval_epoch(args, model, val_loader, out_path=None))
            return {'loss': val_score, 'status': STATUS_OK, 'model': model}

        # hyper-parameter optimization
        trials = Trials()
        best = fmin(hyperopt_my_mga, space, algo=tpe.suggest, trials=trials, max_evals=args['max_evals'])
        print(best)

        # load the best model parameters
        args['rgcn_hidden_feats'] = [[64, 64], [128, 128], [256, 256], [64, 64, 64], [128, 128, 128], [256, 256, 256]][
            best['rgcn_hidden_feats']]
        args['ffn_hidden_feats'] = [64, 128, 256][best['ffn_hidden_feats']]
        args['rgcn_drop_out'] = [0, 0.1, 0.2, 0.3, 0.4, 0.5][best['rgcn_drop_out']]
        args['ffn_drop_out'] = [0, 0.1, 0.2, 0.3, 0.4, 0.5][best['ffn_drop_out']]
        args['lr'] = [0.003, 0.001, 0.0003, 0.0001][best['lr']]
    else:
        args['rgcn_hidden_feats'] = [256, 256]
        args['ffn_hidden_feats'] = 256
        args['rgcn_drop_out'] = 0.2
        args['ffn_drop_out'] = 0.2
        args['lr'] = 0.001

    for time_id in range(args['times']):
        set_random_seed(2022+time_id*10)
        print('***************************************************************************************************')
        print('{}, {}/{} time, task_number: {}, task: {}'.format(args['task_name'], time_id + 1,
                                         args['times'], args['task_number'], args['task_name_list']))
        print('***************************************************************************************************')
        model = MGA(ffn_hidden_feats=args['ffn_hidden_feats'], n_tasks=args['task_number'],
                    ffn_dropout=args['ffn_drop_out'],
                    in_feats=args['in_feats'], rgcn_hidden_feats=args['rgcn_hidden_feats'],
                    rgcn_drop_out=args['rgcn_drop_out'])
        stopper = EarlyStopping(patience=args['patience'], task_name=args['task_name']+'_'+str(time_id+1), mode=args['mode'])
        model.to(args['device'])
        for epoch in range(args['num_epochs']):
            # Train
            lr = args['lr']
            optimizer = Adam(model.parameters(), lr=lr)
            _ = run_a_train_epoch(args, model, train_loader, loss_criterion_c, loss_criterion_r, optimizer)
            # Validation and early stop
            train_score = np.mean(run_an_eval_epoch(args, model, train_loader, out_path=None))
            val_score = np.mean(run_an_eval_epoch(args,  model, val_loader, out_path=None))
            test_score = np.mean(run_an_eval_epoch(args, model, test_loader, out_path=None))
            early_stop = stopper.step(val_score, model)
            print('epoch {:d}/{:d}, lr: {:.6f}, train: {:.4f}, valid: {:.4f}, best valid score {:.4f}, '
                  'test: {:.4f}'.format(
                  epoch + 1, args['num_epochs'], lr,  train_score, val_score,
                  stopper.best_score, test_score))
            if early_stop:
                break
        stopper.load_checkpoint(model)

        pred_name = 'mol_{}'.format(time_id + 1)
        test_score = run_an_eval_epoch(args, model, test_loader,
                                               out_path='../prediction/mol/' + args['task_name'] + '_' + pred_name + '_test')
        train_score = run_an_eval_epoch(args, model, train_loader,
                                                out_path='../prediction/mol/' + args['task_name'] + '_' + pred_name + '_train')
        val_score = run_an_eval_epoch(args, model, val_loader,
                                              out_path='../prediction/mol/' + args['task_name'] + '_' + pred_name + '_val')
        # deal result
        result = train_score + ['training'] + val_score + ['valid'] + test_score + ['test']
        result_pd.loc[time_id] = result

        print('********************************{}, {}_times_result*******************************'.format(
            args['task_name'], time_id + 1))
        print("training_result:", train_score)
        print("val_result:", val_score)
        print("test_result:", test_score)

    with open('../result/hyperparameter_{}.pkl'.format(data_name), 'wb') as f:
        pkl.dump(args, f, pkl.HIGHEST_PROTOCOL)
    result_pd.to_csv('../result/SMEG_' + args['task_name'] + '_all_result.csv', index=False)



















