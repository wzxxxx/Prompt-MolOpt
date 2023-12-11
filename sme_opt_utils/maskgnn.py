import datetime
from sklearn.metrics import accuracy_score, r2_score
from sklearn import metrics
from dgl.data.graph_serialize import save_graphs
import torch.nn.functional as F
import dgl
import numpy as np
import pandas as pd
import random
from dgl.nn.pytorch.conv import RelGraphConv
from torch import nn
import torch as th
from dgl.readout import sum_nodes



class WeightAndSum(nn.Module):
    """Compute importance weights for atoms and perform a weighted sum.

    Parameters
    ----------
    in_feats : int
        Input atom feature size
    """
    
    def __init__(self, in_feats):
        super(WeightAndSum, self).__init__()
        self.in_feats = in_feats
        self.atom_weighting = nn.Sequential(
            nn.Linear(in_feats, 1),
            nn.Sigmoid()
        )
    
    def forward(self, g, feats, smask):
        """Compute molecule representations out of atom representations

        Parameters
        ----------
        g : DGLGraph
            DGLGraph with batch size B for processing multiple molecules in parallel
        feats : FloatTensor of shape (N, self.in_feats)
            Representations for all atoms in the molecules
            * N is the total number of atoms in all molecules
        smask: substructure mask, atom node for 0, substructure node for 1.

        Returns
        -------
        FloatTensor of shape (B, self.in_feats)
            Representations for B molecules
        """
        with g.local_scope():
            g.ndata['h'] = feats
            weight = self.atom_weighting(g.ndata['h']) * smask
            g.ndata['w'] = weight
            h_g_sum = sum_nodes(g, 'h', 'w')
        return h_g_sum, weight


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
    """HRGCN based predictor for multitask prediction on molecular graphs
    We assume each task requires to perform a binary classification.
    Parameters
    ----------
    gnn_out_feats : int
        Number of atom representation features after using GNN
    len_descriptors : int
        length of descriptors
    hyperbolic: str
        Riemannian Manifolds. Defalt: 'Poincare'
    rgcn_drop_out: float
        dropout rate for HRGCN layer
    n_tasks : int
        Number of prediction tasks
    classifier_hidden_feats : int
        Number of molecular graph features in hidden layers of the MLP Classifier
    dropout : float
        The probability for dropout. Default to be 0., i.e. no
        dropout is performed.
    return_weight: bool
        Wether to return atom weight defalt=False
    """
    
    def __init__(self, gnn_rgcn_out_feats, ffn_hidden_feats, ffn_dropout=0.25, classification=True):
        super(BaseGNN, self).__init__()
        self.classification = classification
        self.rgcn_gnn_layers = nn.ModuleList()
        self.readout = WeightAndSum(gnn_rgcn_out_feats)
        self.fc_layers1 = self.fc_layer(ffn_dropout, gnn_rgcn_out_feats, ffn_hidden_feats)
        self.fc_layers2 = self.fc_layer(ffn_dropout, ffn_hidden_feats, ffn_hidden_feats)
        self.fc_layers3 = self.fc_layer(ffn_dropout, ffn_hidden_feats, ffn_hidden_feats)
        self.predict = self.output_layer(ffn_hidden_feats, 1)
    
    def forward(self, rgcn_bg, rgcn_node_feats, rgcn_edge_feats, smask_feats):
        """Multi-task prediction for a batch of molecules
        """
        # Update atom features with GNNs
        for rgcn_gnn in self.rgcn_gnn_layers:
            rgcn_node_feats = rgcn_gnn(rgcn_bg, rgcn_node_feats, rgcn_edge_feats)
        # Compute molecule features from atom features and bond features
        graph_feats, weight = self.readout(rgcn_bg, rgcn_node_feats, smask_feats)
        h1 = self.fc_layers1(graph_feats)
        h2 = self.fc_layers2(h1)
        h3 = self.fc_layers3(h2)
        out = self.predict(h3)
        return out, weight
    
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


class RGCN(BaseGNN):
    """HRGCN based predictor for multitask prediction on molecular graphs
    We assume each task requires to perform a binary classification.
    Parameters
    ----------
    in_feats : int
        Number of input atom features
    Rgcn_hidden_feats : list of int
        rgcn_hidden_feats[i] gives the number of output atom features
        in the i+1-th HRGCN layer
    n_tasks : int
        Number of prediction tasks
    len_descriptors : int
        length of descriptors
    return_weight : bool
        Wether to return weight
    classifier_hidden_feats : int
        Number of molecular graph features in hidden layers of the MLP Classifier
    is_descriptor: bool
        Wether to use descriptor
    loop : bool
        Wether to use self loop
    gnn_drop_rate : float
        The probability for dropout of HRGCN layer. Default to be 0.5
    dropout : float
        The probability for dropout of MLP layer. Default to be 0.
    """
    
    def __init__(self, ffn_hidden_feats, rgcn_node_feats, rgcn_hidden_feats, rgcn_drop_out=0.25, ffn_dropout=0.25,
                 classification=True):
        super(RGCN, self).__init__(gnn_rgcn_out_feats=rgcn_hidden_feats[-1],
                                       ffn_hidden_feats=ffn_hidden_feats,
                                       ffn_dropout=ffn_dropout,
                                       classification=classification,
                                       )
        for i in range(len(rgcn_hidden_feats)):
            rgcn_out_feats = rgcn_hidden_feats[i]
            self.rgcn_gnn_layers.append(RGCNLayer(rgcn_node_feats, rgcn_out_feats, loop=True,
                                                  rgcn_drop_out=rgcn_drop_out))
            rgcn_node_feats = rgcn_out_feats


def pro2label(x):
    if x < 0.5:
        return 0
    else:
        return 1


class Meter(object):
    """Track and summarize model performance on a dataset for
    (multi-label) binary classification."""
    
    def __init__(self):
        self.y_pred = []
        self.y_true = []
    
    def update(self, y_pred, y_true):
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
    
    def accuracy_score(self):
        """Compute accuracy score for each task.
        Returns
        -------
        list of float
            roc-auc score for all tasks
        """
        
        y_pred = th.cat(self.y_pred, dim=0)
        y_pred = th.sigmoid(y_pred)
        y_pred = y_pred.numpy()
        y_pred_label = np.array([pro2label(x) for x in y_pred])
        y_true = th.cat(self.y_true, dim=0).numpy()
        scores = round(accuracy_score(y_true, y_pred_label), 4)
        return scores
    
    def r2(self):
        """Compute r2 score for each task.
        Returns
        -------
        list of float
            roc-auc score for all tasks
        """
        y_pred = th.cat(self.y_pred, dim=0).numpy()
        y_true = th.cat(self.y_true, dim=0).numpy()
        
        scores = round(r2_score(y_true, y_pred), 4)
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
        # Todo: support categorical classes
        return y_true, y_pred
    
    def compute_metric(self, metric_name):
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
        assert metric_name in ['accuracy', 'r2', 'return_pred_true'], \
            'Expect metric name to be "roc_auc", "accuracy", "return_pred_true", got {}'.format(metric_name)
        if metric_name == 'accuracy':
            return self.accuracy_score()
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
    smiles, g_rgcn, labels, smask, sub_smi = map(list, zip(*data))
    rgcn_bg = dgl.batch(g_rgcn)
    labels = th.tensor(labels)
    return smiles, rgcn_bg, labels, smask, sub_smi


def collate_molgraphs_pred(mol_gs):
    # Batch the graphs
    mol_bg = dgl.batch(mol_gs)
    return mol_bg



def pos_weight(train_set):
    smiles, g_rgcn, labels, smask, sub_smi = map(list, zip(*train_set))
    labels = np.array(labels)
    task_pos_weight_list = []
    num_pos = 0
    num_neg = 0
    for i in labels:
        if i == 1:
            num_pos = num_pos + 1
        if i == 0:
            num_neg = num_neg + 1
    weight = num_neg / (num_pos+0.00000001)
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


def run_a_train_epoch(args, model, data_loader, loss_criterion, optimizer):
    model.train()
    train_meter = Meter()
    total_loss = 0
    n_mol = 0
    for batch_id, batch_data in enumerate(data_loader):
        smiles, rgcn_bg, labels, smask_idx, sub_smi = batch_data
        rgcn_bg = rgcn_bg.to(args['device'])
        labels = labels.unsqueeze(dim=1).float().to(args['device'])
        
        rgcn_node_feats = rgcn_bg.ndata.pop(args['node_data_field']).float().to(args['device'])
        rgcn_edge_feats = rgcn_bg.edata.pop(args['edge_data_field']).long().to(args['device'])
        smask_feats = rgcn_bg.ndata.pop(args['substructure_mask']).unsqueeze(dim=1).float().to(args['device'])
        
        preds, weight = model(rgcn_bg, rgcn_node_feats, rgcn_edge_feats, smask_feats)
        loss = (loss_criterion(preds, labels)).mean()
        optimizer.zero_grad()
        loss.backward()
        total_loss = total_loss + loss * len(smiles)
        n_mol = n_mol + len(smiles)
        optimizer.step()
        train_meter.update(preds, labels)
        del labels, rgcn_bg, rgcn_edge_feats, rgcn_node_feats, loss
        th.cuda.empty_cache()
    train_score = round(train_meter.compute_metric(args['metric_name']), 4)
    average_loss = total_loss / n_mol
    return train_score, average_loss


def run_an_eval_epoch(args, model, data_loader, loss_criterion, out_path):
    model.eval()
    smiles_list = []
    eval_meter = Meter()
    total_loss = 0
    n_mol = 0
    smask_idx_list = []
    sub_smi_list = []
    with th.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            # print('{}/{}'.format(batch_id, len(data_loader)))
            smiles, rgcn_bg, labels, smask_idx, sub_smi = batch_data
            rgcn_bg = rgcn_bg.to(args['device'])
            labels = labels.unsqueeze(dim=1).float().to(args['device'])
            
            rgcn_node_feats = rgcn_bg.ndata.pop(args['node_data_field']).float().to(args['device'])
            rgcn_edge_feats = rgcn_bg.edata.pop(args['edge_data_field']).long().to(args['device'])
            smask_feats = rgcn_bg.ndata.pop(args['substructure_mask']).unsqueeze(dim=1).float().to(args['device'])
            
            preds, weight = model(rgcn_bg, rgcn_node_feats, rgcn_edge_feats, smask_feats)
            loss = (loss_criterion(preds, labels)).mean()
            smask_idx_list = smask_idx_list + smask_idx
            sub_smi_list = sub_smi_list + sub_smi
            total_loss = total_loss + loss * len(smiles)
            n_mol = n_mol + len(smiles)
            eval_meter.update(preds, labels)
            del labels, rgcn_bg, rgcn_edge_feats, rgcn_node_feats
            smiles_list = smiles_list + smiles
            th.cuda.empty_cache()
        average_loss = total_loss / n_mol
    prediction_pd = pd.DataFrame()
    y_true, y_pred = eval_meter.compute_metric('return_pred_true')
    y_true = y_true.squeeze().numpy()
    if args['classification']:
        y_pred = th.sigmoid(y_pred)
        y_pred = y_pred.squeeze().numpy()
    else:
        y_pred = y_pred.squeeze().numpy()
    y_true_list = y_true.tolist()
    y_pred_list = y_pred.tolist()
    # save prediction
    prediction_pd['smiles'] = smiles_list
    prediction_pd['label'] = y_true_list
    prediction_pd['pred'] = y_pred_list
    prediction_pd['sub_smi'] = sub_smi_list
    if out_path is not None:
        prediction_pd.to_csv(out_path + '_prediction.csv', index=False)
    if args['classification']:
        y_pred_label = [1 if x >= 0.5 else 0 for x in y_pred_list]
        accuracy = round(metrics.accuracy_score(y_true_list, y_pred_label), 4)
        mcc = round(metrics.matthews_corrcoef(y_true_list, y_pred_label), 4)
        se, sp = sesp_score(y_true_list, y_pred_label)
        pre, rec, f1, sup = metrics.precision_recall_fscore_support(y_true_list, y_pred_label, zero_division=0)
        f1 = round(f1[1], 4)
        rec = round(rec[1], 4)
        pre = round(pre[1], 4)
        err = round(1 - accuracy, 4)
        result = [accuracy, se, sp, f1, pre, rec, err, mcc]
        return result, average_loss
    else:
        r2 = round(metrics.r2_score(y_true_list, y_pred_list), 4)
        mae = round(metrics.mean_absolute_error(y_true_list, y_pred_list), 4)
        rmse = round(metrics.mean_squared_error(y_true_list, y_pred_list) ** 0.5, 4)
        result = [r2, mae, rmse]
        return result, average_loss



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
        Filename for storing the model checkpoint

    """
    
    def __init__(self, pretrained_model='Null_early_stop.pth', mode='higher', patience=10, filename=None,
                 task_name="None",
                 former_task_name="None"):
        if filename is None:
            task_name = task_name
            filename = '../checkpoints/sme/{}_early_stop.pth'.format(task_name)
        former_filename = '../checkpoints/sme/{}_early_stop.pth'.format(former_task_name)
        
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
        self.pretrained_model = '../checkpoints/sme/' + pretrained_model
    
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





