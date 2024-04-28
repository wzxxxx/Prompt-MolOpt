from sklearn.metrics import roc_auc_score, mean_squared_error, precision_recall_curve, auc,  r2_score
import torch.nn.functional as F
import dgl
import numpy as np
import pandas as pd
import random
from dgl.nn.pytorch.conv import RelGraphConv
from torch import nn
import torch as th
from dgl.readout import sum_nodes
import os


class WeightAndSum(nn.Module):
    def __init__(self, in_feats, task_num=1, return_embedding=False):
        super(WeightAndSum, self).__init__()
        self.in_feats = in_feats
        self.task_num = task_num
        self.return_embedding=return_embedding
        self.atom_weighting_specific = nn.ModuleList([self.atom_weight(self.in_feats) for _ in range(self.task_num)])
        self.shared_weighting = self.atom_weight(self.in_feats)
    def forward(self, bg, feats, smask):
        feat_list = []
        atom_list = []
        # cal specific feats
        if self.return_embedding:
            for i in range(self.task_num):
                with bg.local_scope():
                    bg.ndata['h'] = feats
                    weight = self.atom_weighting_specific[i](feats) * smask
                    bg.ndata['w'] = weight
                feat_list.append((weight*feats).unsqueeze(1))
        else:
            for i in range(self.task_num):
                with bg.local_scope():
                    bg.ndata['h'] = feats
                    weight = self.atom_weighting_specific[i](feats)*smask
                    bg.ndata['w'] = weight
                    specific_feats_sum = sum_nodes(bg, 'h', 'w')
                    atom_list.append(bg.ndata['w'])
                feat_list.append(specific_feats_sum)
        return feat_list

    def atom_weight(self, in_feats):
        return nn.Sequential(
            nn.Linear(in_feats, 1),
            nn.Sigmoid()
            )


class RGCNLayer(nn.Module):
    """Single layer RGCN for updating node features
    Parameters
    ----------
    in_feats : int
        Number of input atom features
    out_feats : int
        Number of output atom features
    num_rels: int
        Number of bond type
    activation : activation function
        Default to be ReLU
    loop: bool:
        Whether to use self loop
        Default to be False
    residual : bool
        Whether to use residual connection, default to be True
    batchnorm : bool
        Whether to use batch normalization on the output,
        default to be True
    rgcn_drop_out : float
        The probability for dropout. Default to be 0., i.e. no
        dropout is performed.
    hyperbolic: str
        Riemannian Manifolds. Defalt: 'Poincare'
    """

    def __init__(self, in_feats, out_feats, num_rels=65, activation=F.relu, loop=False,
                 residual=True, batchnorm=True, rgcn_drop_out=0.5):
        super(RGCNLayer, self).__init__()

        self.activation = activation
        self.graph_conv_layer = RelGraphConv(in_feats, out_feats, num_rels=num_rels, regularizer='basis',
                                             num_bases=num_rels, bias=True, activation=activation,
                                             self_loop=loop, dropout=rgcn_drop_out)
        self.residual = residual
        if residual:
            self.res_connection = nn.Linear(in_feats, out_feats)

        self.bn = batchnorm
        if batchnorm:
            self.bn_layer = nn.BatchNorm1d(out_feats)

    def forward(self, bg, node_feats, etype, norm=None):
        """Update atom representations
        Parameters
        ----------
        bg : BatchedDGLGraph
            Batched DGLGraphs for processing multiple molecules in parallel
        node_feats : FloatTensor of shape (N, M1)
            * N is the total number of atoms in the batched graph
            * M1 is the input atom feature size, must match in_feats in initialization
        etype: int
            bond type
        norm: th.Tensor
            Optional edge normalizer tensor. Shape: :math:`(|E|, 1)`
        Returns
        -------
        new_feats : FloatTensor of shape (N, M2)
            * M2 is the output atom feature size, must match out_feats in initialization
        """
        new_feats = self.graph_conv_layer(bg, node_feats, etype, norm)
        if self.residual:
            res_feats = self.activation(self.res_connection(node_feats))
            new_feats = new_feats + res_feats
        if self.bn:
            new_feats = self.bn_layer(new_feats)
        del res_feats
        th.cuda.empty_cache()
        return new_feats


class BaseGNN(nn.Module):
    def __init__(self, gnn_out_feats, n_tasks, ffn_hidden_feats, ffn_dropout=0.25, return_embedding=False):
        super(BaseGNN, self).__init__()
        self.task_num = n_tasks
        self.gnn_layers = nn.ModuleList()
        self.fc_in_feats = gnn_out_feats
        self.return_embedding = return_embedding
        self.readout = WeightAndSum(gnn_out_feats, self.task_num, self.return_embedding)

        self.fc_layers1 = nn.ModuleList([self.fc_layer(ffn_dropout, self.fc_in_feats, ffn_hidden_feats) for _ in range(self.task_num)])
        self.fc_layers2 = nn.ModuleList(
            [self.fc_layer(ffn_dropout, ffn_hidden_feats, ffn_hidden_feats) for _ in range(self.task_num)])
        self.fc_layers3 = nn.ModuleList(
            [self.fc_layer(ffn_dropout, ffn_hidden_feats, ffn_hidden_feats) for _ in range(self.task_num)])

        self.output_layer1 = nn.ModuleList(
            [self.output_layer(ffn_hidden_feats, 1) for _ in range(self.task_num)])

    def forward(self, bg, node_feats, edge_feats, smask_feats):
        # Update atom features with GNNs
        for gnn in self.gnn_layers:
            node_feats = gnn(bg, node_feats, edge_feats)

        # Compute molecule features from atom features
        feats_list = self.readout(bg, node_feats, smask_feats)
        if self.return_embedding:
            prediction_all = th.cat(feats_list, dim=1)

        else:
            for i in range(self.task_num):
                # mol_feats = th.cat([feats_list[-1], feats_list[i]], dim=1)
                mol_feats = feats_list[i]
                h1 = self.fc_layers1[i](mol_feats)
                h2 = self.fc_layers2[i](h1)
                h3 = self.fc_layers3[i](h2)
                predict = self.output_layer1[i](h3)
                if i == 0:
                    prediction_all = predict
                else:
                    prediction_all = th.cat([prediction_all, predict], dim=1)
        return prediction_all

    def fc_layer(self, dropout, in_feats, hidden_feats):
        return nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_feats, hidden_feats),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_feats)
                )

    def output_layer(self, hidden_feats, out_feats):
        return nn.Sequential(
                nn.Linear(hidden_feats, out_feats)
                )


class MGA(BaseGNN):
    def __init__(self, in_feats, ffn_hidden_feats, rgcn_hidden_feats, rgcn_drop_out=0.25, ffn_dropout=0.25, n_tasks=1, return_embedding=False):
        super(MGA, self).__init__(gnn_out_feats=rgcn_hidden_feats[-1],
                                  ffn_hidden_feats=ffn_hidden_feats,
                                  n_tasks=n_tasks,
                                  ffn_dropout=ffn_dropout,
                                  return_embedding=return_embedding
                                  )

        for i in range(len(rgcn_hidden_feats)):
            out_feats = rgcn_hidden_feats[i]
            self.gnn_layers.append(RGCNLayer(in_feats, out_feats, loop=True, rgcn_drop_out=rgcn_drop_out))
            in_feats = out_feats


def pro2label(x):
    if x < 0.5:
        return 0
    else:
        return 1


class Meter(object):
    """Track and summarize model performance on a dataset for
    (multi-label) binary classification."""
    def __init__(self):
        self.mask = []
        self.y_pred = []
        self.y_true = []

    def update(self, y_pred, y_true, mask):
        """Update for the result of an iteration
        Parameters
        ----------
        y_pred : float32 tensor
            Predicted molecule labels with shape (B, T),
            B for batch size and T for the number of tasks
        y_true : float32 tensor
            Ground truth molecule labels with shape (B, T)
        mask : float32 tensor
            Mask for indicating the existence of ground
            truth labels with shape (B, T)
        """
        self.y_pred.append(y_pred.detach().cpu())
        self.y_true.append(y_true.detach().cpu())
        self.mask.append(mask.detach().cpu())

    def roc_auc_score(self):
        """Compute roc-auc score for each task.
        Returns
        -------
        list of float
            roc-auc score for all tasks
        """
        mask = th.cat(self.mask, dim=0)
        y_pred = th.cat(self.y_pred, dim=0)
        y_true = th.cat(self.y_true, dim=0)
        # Todo: support categorical classes
        # This assumes binary case only
        y_pred = th.sigmoid(y_pred)
        n_tasks = y_true.shape[1]
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            scores.append(round(roc_auc_score(task_y_true, task_y_pred), 4))
        return scores

    def return_pred_true(self):
        """Compute roc-auc score for each task.
        Returns
        -------
        list of float
            roc-auc score for all tasks
        """
        y_pred = th.cat(self.y_pred, dim=0)
        y_true = th.cat(self.y_true, dim=0)
        y_mask = th.cat(self.mask, dim=0)
        return y_true, y_pred, y_mask

    def l1_loss(self, reduction):
        """Compute l1 loss for each task.
        Returns
        -------
        list of float
            l1 loss for all tasks
        reduction : str
            * 'mean': average the metric over all labeled data points for each task
            * 'sum': sum the metric over all labeled data points for each task
        """
        mask = th.cat(self.mask, dim=0)
        y_pred = th.cat(self.y_pred, dim=0)
        y_true = th.cat(self.y_true, dim=0)
        n_tasks = y_true.shape[1]
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            scores.append(F.l1_loss(task_y_true, task_y_pred, reduction=reduction).item())
        return scores

    def rmse(self):
        """Compute RMSE for each task.
        Returns
        -------
        list of float
            rmse for all tasks
        """
        mask = th.cat(self.mask, dim=0)
        y_pred = th.cat(self.y_pred, dim=0)
        y_true = th.cat(self.y_true, dim=0)
        n_data, n_tasks = y_true.shape
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            scores.append(np.sqrt(F.mse_loss(task_y_pred, task_y_true).cpu().item()))
        return scores

    def mae(self):
        """Compute MAE for each task.
        Returns
        -------
        list of float
            mae for all tasks
        """
        mask = th.cat(self.mask, dim=0)
        y_pred = th.cat(self.y_pred, dim=0)
        y_true = th.cat(self.y_true, dim=0)
        n_data, n_tasks = y_true.shape
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            scores.append(mean_squared_error(task_y_true, task_y_pred))
        return scores

    def r2(self):
        """Compute R2 for each task.
        Returns
        -------
        list of float
            r2 for all tasks
        """
        mask = th.cat(self.mask, dim=0)
        y_pred = th.cat(self.y_pred, dim=0)
        y_true = th.cat(self.y_true, dim=0)
        n_data, n_tasks = y_true.shape
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            # print(task_y_true.shape, task_y_pred.shape)
            scores.append(round(r2_score(task_y_true, task_y_pred), 4))
        return scores

    def roc_precision_recall_score(self):
        """Compute AUC_PRC for each task.
        Returns
        -------
        list of float
            AUC_PRC for all tasks
        """
        mask = th.cat(self.mask, dim=0)
        y_pred = th.cat(self.y_pred, dim=0)
        y_true = th.cat(self.y_true, dim=0)
        # Todo: support categorical classes
        # This assumes binary case only
        y_pred = th.sigmoid(y_pred)
        n_tasks = y_true.shape[1]
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            precision, recall, _thresholds = precision_recall_curve(task_y_true, task_y_pred)
            scores.append(auc(recall, precision))
        return scores

    def compute_metric(self, metric_name, reduction='mean'):
        """Compute metric for each task.
        Parameters
        ----------
        metric_name : str
            Name for the metric to compute.
        reduction : str
            Only comes into effect when the metric_name is l1_loss.
            * 'mean': average the metric over all labeled data points for each task
            * 'sum': sum the metric over all labeled data points for each task
        Returns
        -------
        list of float
            Metric value for each task
        """
        assert metric_name in ['roc_auc', 'l1', 'rmse', 'mae', 'roc_prc', 'r2', 'return_pred_true'], \
            'Expect metric name to be "roc_auc", "l1" or "rmse", "mae", "roc_prc", "r2", "return_pred_true", got {}'.format(metric_name)
        assert reduction in ['mean', 'sum']
        if metric_name == 'roc_auc':
            return self.roc_auc_score()
        if metric_name == 'l1':
            return self.l1_loss(reduction)
        if metric_name == 'rmse':
            return self.rmse()
        if metric_name == 'mae':
            return self.mae()
        if metric_name == 'roc_prc':
            return self.roc_precision_recall_score()
        if metric_name == 'r2':
            return self.r2()
        if metric_name == 'return_pred_true':
            return self.return_pred_true()


def set_random_seed(seed=10):
    """Set random seed.
    Parameters
    ----------
    seed : int
        Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    dgl.random.seed(seed)
    # dgl.seed(seed)
    th.manual_seed(seed)
    th.backends.cudnn.benchmark = False
    th.backends.cudnn.deterministic = True
    if th.cuda.is_available():
        th.cuda.manual_seed(seed)


def collate_molgraphs(data):
    smiles, g_rgcn, labels, smask, sub_name, mask = map(list, zip(*data))
    rgcn_bg = dgl.batch(g_rgcn)
    labels = th.tensor(labels)
    mask = th.tensor(mask)
    return smiles, rgcn_bg, labels, smask, sub_name, mask


def collate_molgraphs_pred(mol_gs):
    # Batch the graphs
    mol_bg = dgl.batch(mol_gs)
    return mol_bg


def pos_weight(train_set, classification_num):
    smiles, g_rgcn, labels, smask, sub_name, mask = map(list, zip(*train_set))
    labels = np.array(labels)
    task_pos_weight_list = []
    for task in range(classification_num):
        num_pos = 0
        num_impos = 0
        for i in labels[:, task]:
            if i == 1:
                num_pos = num_pos + 1
            if i == 0:
                num_impos = num_impos + 1
        weight = num_impos / (num_pos+0.00000001)
        task_pos_weight_list.append(weight)
    task_pos_weight = th.tensor(task_pos_weight_list)
    return task_pos_weight


def sesp_score(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i] == 1:
            tp = tp + 1
        if y_true[i] == y_pred[i] == 0:
            tn = tn + 1
        if y_true[i] == 0 and y_pred[i] == 1:
            fp = fp + 1
        if y_true[i] == 1 and y_pred[i] == 0:
            fn = fn + 1
    sensitivity = round(tp / (tp + fn + 0.0000001), 4)
    specificity = round(tn / (tn + fp + 0.0000001), 4)
    return sensitivity, specificity


def run_a_train_epoch(args, model, data_loader, loss_criterion_c, loss_criterion_r,  optimizer):
    model.train()
    train_meter_c = Meter()
    train_meter_r = Meter()
    n_mol = 0
    for batch_id, batch_data in enumerate(data_loader):
        smiles, rgcn_bg, labels, smask_idx, sub_name, mask = batch_data
        mask = mask.float().to(args['device'])
        rgcn_bg = rgcn_bg.to(args['device'])
        labels = labels.float().to(args['device'])
        rgcn_node_feats = rgcn_bg.ndata.pop(args['node_data_field']).float().to(args['device'])
        rgcn_edge_feats = rgcn_bg.edata.pop(args['edge_data_field']).long().to(args['device'])
        smask_feats = rgcn_bg.ndata.pop(args['substructure_mask']).unsqueeze(dim=1).float().to(args['device'])
        logits = model(rgcn_bg, rgcn_node_feats, rgcn_edge_feats, smask_feats)
        # calculate loss according to different task class
        if args['task_class'] == 'classification_regression':
            # split classification and regression
            logits_c = logits[:,:args['classification_num']]
            labels_c = labels[:,:args['classification_num']]
            mask_c = mask[:,:args['classification_num']]

            logits_r = logits[:,args['classification_num']:]
            labels_r = labels[:,args['classification_num']:]
            mask_r = mask[:,args['classification_num']:]
            # chose loss function according to task_weight
            loss = (loss_criterion_c(logits_c, labels_c)*(mask_c != 0).float()).mean() \
                   + (loss_criterion_r(logits_r, labels_r)*(mask_r != 0).float()).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print('epoch {:d}/{:d}, batch {:d}/{:d}, loss {:.4f}'.format(
            #     epoch + 1, args['num_epochs'], batch_id + 1, len(data_loader), loss.item()))
            train_meter_c.update(logits_c, labels_c, mask_c)
            train_meter_r.update(logits_r, labels_r, mask_r)
            del rgcn_bg, mask, labels, loss, logits_c, logits_r, labels_c, labels_r, mask_c, mask_r
            th.cuda.empty_cache()
        elif args['task_class'] == 'classification':
            # chose loss function according to task_weight
            loss = (loss_criterion_c(logits, labels)*(mask != 0).float()).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print('epoch {:d}/{:d}, batch {:d}/{:d}, loss {:.4f}'.format(
            #     epoch + 1, args['num_epochs'], batch_id + 1, len(data_loader), loss.item()))
            train_meter_c.update(logits, labels, mask)
            del rgcn_bg, mask, labels, loss,  logits
            th.cuda.empty_cache()
        else:
            loss = (loss_criterion_r(logits, labels)*(mask != 0).float()).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print('epoch {:d}/{:d}, batch {:d}/{:d}, loss {:.4f}'.format(
            #     epoch + 1, args['num_epochs'], batch_id + 1, len(data_loader), loss.item()))
            train_meter_r.update(logits, labels, mask)
            del rgcn_bg, mask, labels, loss,  logits

        n_mol = n_mol + len(smiles)
        optimizer.step()
        th.cuda.empty_cache()
    if args['task_class'] == 'classification_regression':
        train_score = np.mean(train_meter_c.compute_metric(args['classification_metric_name']) +
                              train_meter_r.compute_metric(args['regression_metric_name']))
    elif args['task_class'] == 'classification':
        train_score = np.mean(train_meter_c.compute_metric(args['classification_metric_name']))
    else:
        train_score = np.mean(train_meter_r.compute_metric(args['regression_metric_name']))
    return train_score


def reshape_array(arr):
    if len(arr.shape) == 1:  # 检查数组是一维的
        return arr.reshape(-1, 1)
    else:
        return arr  # 保持不变


def run_an_eval_epoch(args, model, data_loader, out_path):
    model.eval()
    eval_meter_c = Meter()
    eval_meter_r = Meter()
    smiles_list = []
    n_mol = 0
    smask_idx_list = []
    sub_name_list = []
    with th.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            # print('{}/{}'.format(batch_id, len(data_loader)))
            smiles, rgcn_bg, labels, smask_idx, sub_name, mask = batch_data
            rgcn_bg = rgcn_bg.to(args['device'])
            mask = mask.float().to(args['device'])
            labels = labels.float().to(args['device'])

            rgcn_node_feats = rgcn_bg.ndata.pop(args['node_data_field']).float().to(args['device'])
            rgcn_edge_feats = rgcn_bg.edata.pop(args['edge_data_field']).long().to(args['device'])
            smask_feats = rgcn_bg.ndata.pop(args['substructure_mask']).unsqueeze(dim=1).float().to(args['device'])
            logits = model(rgcn_bg, rgcn_node_feats, rgcn_edge_feats, smask_feats)
            if args['task_class'] == 'classification_regression':
                # split classification and regression
                logits_c = logits[:, :args['classification_num']]
                labels_c = labels[:, :args['classification_num']]
                mask_c = mask[:, :args['classification_num']]
                logits_r = logits[:, args['classification_num']:]
                labels_r = labels[:, args['classification_num']:]
                mask_r = mask[:, args['classification_num']:]
                # Mask non-existing labels
                eval_meter_c.update(logits_c, labels_c, mask_c)
                eval_meter_r.update(logits_r, labels_r, mask_r)
                del rgcn_bg,  mask, labels, logits_c, logits_r, labels_c, labels_r, mask_c, mask_r
                th.cuda.empty_cache()
            elif args['task_class'] == 'classification':
                # Mask non-existing labels
                eval_meter_c.update(logits, labels, mask)
                del rgcn_bg,  mask, labels, logits
                th.cuda.empty_cache()
            else:
                # Mask non-existing labels
                eval_meter_r.update(logits, labels, mask)
                del rgcn_bg,  mask, labels,  logits
                th.cuda.empty_cache()
            smask_idx_list = smask_idx_list + smask_idx
            sub_name_list = sub_name_list + sub_name
            n_mol = n_mol + len(smiles)
            smiles_list = smiles_list + smiles
            th.cuda.empty_cache()
    # return prediction
    if args['task_class'] == 'classification_regression':
        # return classification result
        y_true_c, y_pred_c, y_mask_c = eval_meter_c.compute_metric('return_pred_true')
        y_true_c = y_true_c.squeeze().numpy()
        y_mask_c = y_mask_c.squeeze().numpy()
        y_pred_c = th.sigmoid(y_pred_c).squeeze().numpy()

        # return regression result
        y_true_r, y_pred_r, y_mask_r = eval_meter_r.compute_metric('return_pred_true')
        y_true_r = y_true_r.squeeze().numpy()
        y_pred_r = y_pred_r.squeeze().numpy()
        y_mask_r = y_mask_r.squeeze().numpy()
        y_result = np.concatenate((reshape_array(y_true_c), reshape_array(y_true_r), reshape_array(y_pred_c), reshape_array(y_pred_r), reshape_array(y_mask_c), reshape_array(y_mask_r)), axis=1)
    if args['task_class'] == 'classification':
        # return classification result
        y_true_c, y_pred_c, y_mask_c = eval_meter_c.compute_metric('return_pred_true')
        if args['task_number']==1:
            y_true_c = y_true_c.numpy()
            y_mask_c = y_mask_c.numpy()
            y_pred_c = th.sigmoid(y_pred_c).numpy()
        else:
            y_true_c = y_true_c.squeeze().numpy()
            y_mask_c = y_mask_c.squeeze().numpy()
            y_pred_c = th.sigmoid(y_pred_c).squeeze().numpy()
        y_result = np.concatenate((reshape_array(y_true_c), reshape_array(y_pred_c), reshape_array(y_mask_c)), axis=1)
    if args['task_class'] == 'regression':
        # return regression result
        y_true_r, y_pred_r, y_mask_r = eval_meter_r.compute_metric('return_pred_true')
        if args['task_number']==1:
            y_true_r = y_true_r.numpy()
            y_pred_r = y_pred_r.numpy()
            y_mask_r = y_mask_r.numpy()
        else:
            y_true_r = y_true_r.squeeze().numpy()
            y_pred_r = y_pred_r.squeeze().numpy()
            y_mask_r = y_mask_r.squeeze().numpy()
        y_result = np.concatenate((reshape_array(y_true_r), reshape_array(y_pred_r), reshape_array(y_mask_r)), axis=1)
    label_columns = ['{}_label'.format(task_name) for task_name in args['task_name_list']]
    mask_columns = ['{}_mask'.format(task_name) for task_name in args['task_name_list']]
    columns = label_columns + args['task_name_list'] + mask_columns
    prediction_pd = pd.DataFrame(y_result, columns=columns)
    prediction_pd['smiles'] = smiles_list
    prediction_pd['sub_name'] = sub_name_list
    # save prediction
    if out_path is not None:
        np.save(out_path + '_smask_index.npy', smask_idx_list)
        prediction_pd.to_csv(out_path + '_prediction.csv', index=False)
    if args['task_class'] == 'classification_regression':
        return eval_meter_c.compute_metric(args['classification_metric_name']) + \
               eval_meter_r.compute_metric(args['regression_metric_name'])
    elif args['task_class'] == 'classification':
        return eval_meter_c.compute_metric(args['classification_metric_name'])
    else:
        return eval_meter_r.compute_metric(args['regression_metric_name'])


class EarlyStopping(object):
    """Early stop performing
    Parameters
    ----------
    mode : str
        * 'higher': Higher metric suggests a better model
        * 'lower': Lower metric suggests a better model
    patience : int
        Number of epochs to wait before early stop
        if the metric stops getting improved
    taskname : str or None
        Filename for storing the model checkpoints

    """
    
    def __init__(self, pretrained_model='Null_early_stop.pth', mode='higher', patience=10, filename=None,
                 task_name="None",
                 former_task_name="None"):
        if filename is None:
            task_name = task_name
            filename = '../checkpoints/mga/{}_early_stop.pth'.format(task_name)
        former_filename = '../checkpoints/mga/{}_early_stop.pth'.format(former_task_name)
        
        assert mode in ['higher', 'lower']
        self.mode = mode
        if self.mode == 'higher':
            self._check = self._check_higher
        else:
            self._check = self._check_lower
        
        self.patience = patience
        self.counter = 0
        self.filename = filename
        self.former_filename = former_filename
        self.best_score = None
        self.early_stop = False
        self.pretrained_model = '../checkpoints/mga/' + pretrained_model
    
    def _check_higher(self, score, prev_best_score):
        return (score > prev_best_score)
    
    def _check_lower(self, score, prev_best_score):
        return (score < prev_best_score)
    
    def step(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self._check(score, self.best_score):
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            print(
                'EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop
    
    def nosave_step(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self._check(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            print(
                'EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop
    
    def save_checkpoint(self, model):
        '''Saves model when the metric on the validation set gets improved.'''
        th.save({'model_state_dict': model.state_dict()}, self.filename)
        # print(self.filename)
    
    def load_checkpoint(self, model):
        '''Load model saved with early stopping.'''
        # model.load_state_dict(th.load(self.filename)['model_state_dict'])
        model.load_state_dict(th.load(self.filename, map_location=th.device('cpu'))['model_state_dict'])
    
    def load_former_model(self, model):
        '''Load model saved with early stopping.'''
        model.load_state_dict(th.load(self.former_filename)['model_state_dict'])
        # model.load_state_dict(th.load(self.former_filename, map_location=th.device('cpu'))['model_state_dict'])






