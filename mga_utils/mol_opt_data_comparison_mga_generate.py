import pandas as pd
import numpy as np
import re
import os
from rdkit import Chem
from maskgnn import EarlyStopping, run_a_train_epoch, \
    run_an_eval_epoch, set_random_seed, collate_molgraphs_pred, Meter, MGA
import torch as th
from build_data import MolGraphDataset
from torch.utils.data import DataLoader
from MGA_SME import MGA_SME_hyperopt
from maskgnn import set_random_seed
import argparse
import warnings

warnings.filterwarnings('ignore')


def tokenize_smiles(smi):
    pattern = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    return tokens


def make_vocab(data_file, vocab_file):
    vocab = {}
    data = pd.read_csv(data_file)
    smi_list = data.src_smi.tolist()+data.tgt_smi.tolist()
    tag_list = data.task_name.tolist()
    for tag in tag_list:
        tag_tokens = tag.split()
        for token in tag_tokens:
            if token in vocab:
                vocab[token] += 1
            else:
                vocab[token] = 1
    for smi in smi_list:
        smi_tokens = tokenize_smiles(smi)
        for token in smi_tokens:
            if token in vocab:
                vocab[token] += 1
            else:
                vocab[token] = 1
    with open(vocab_file, "w") as of:
        of.write("_PAD\n_UNK\n_SOS\n_EOS\n[other_atom]\nMolStart\n")
        for token, count in vocab.items():
            of.write(f"{token}\t{count}\n")


def load_vocab(vocab_file):
    vocab = {}
    with open(vocab_file, "r") as f:
        for i, line in enumerate(f):
            token = line.strip().split("\t")[0]
            vocab[token] = i
    return vocab


def token_gmol_match(smiles, vocab):
    initial_atom_list = ['c', 'N', 'C', 'O', 'Cl', 'o', 'S', 'P', 'n', 'Br', 'F', 's', 'I', 'B']
    all_token_list = list(vocab.keys())
    all_atom_list = initial_atom_list + [element for element in all_token_list if element.startswith('[')]
    # token2dic
    token2ids = {}
    for i, w in enumerate(all_token_list):
        token2ids[w] = i
    tokens = tokenize_smiles(smiles)
    tokens_atom_n = 0
    for i, token in enumerate(tokens):
        if token not in all_token_list:
            if token.startswith('['):
                tokens[i] = '[other_atom]'
                tokens_atom_n = tokens_atom_n + 1
            else:
                tokens[i] = '_UNK'
        if token in all_atom_list:
            tokens_atom_n = tokens_atom_n + 1
    mol_atom_n = Chem.MolFromSmiles(smiles).GetNumAtoms()
    if mol_atom_n== tokens_atom_n:
        return mol_atom_n
    else:
        return -1


def remove_mismatch(data_file, vocab):
    data = pd.read_csv(data_file)
    src_smis = data.src_smi.tolist()
    data['src_atom_n'] = [token_gmol_match(smi, vocab) for smi in src_smis]
    print(f'Tokens and GMol atoms not MisMatch: {len(data[data["src_atom_n"]==-1])} {data[data["src_atom_n"]==-1].src_smi.tolist()}')
    data=data[data["src_atom_n"]!=-1]
    data.to_csv(data_file, index=False)


def get_src_token_ids(tokens, vocab, embed, tag_indicator, max_len):
    initial_atom_list = ['c', 'N', 'C', 'O', 'Cl', 'o', 'S', 'P', 'n', 'Br', 'F', 's', 'I', 'B']
    all_token_list = list(vocab.keys())
    all_atom_list = initial_atom_list + [element for element in all_token_list if element.startswith('[')]
    atom_index = 0
    token_ids = []
    token_feats = []
    atom_feats = (embed*tag_indicator).sum(dim=1)
    non_atom_feats = th.zeros(atom_feats[0:1].size())
    for i, token in enumerate(tokens):
        # ADD NOT IDENTIFIED TOKEN
        if token not in all_token_list:
            if token.startswith('['):
                tokens[i] = '[other_atom]'
            else:
                tokens[i] = '_UNK'
        token = tokens[i]
        token_ids.append(vocab[token])

        # 为非原子token添加 0 feats
        if token in all_atom_list:
            token_feats.append(atom_feats[atom_index:atom_index+1])
            atom_index=atom_index+1
        else:
            token_feats.append(non_atom_feats)
    token_ids = token_ids[:max_len - 1]
    token_ids.append(vocab["_EOS"])
    token_feats = token_feats[:max_len - 1]
    token_feats.append(non_atom_feats)
    lens = len(token_ids)
    while len(token_ids) < max_len:
        token_ids.append(vocab["_PAD"])
        token_feats.append(non_atom_feats)
    token_feats = th.cat(token_feats, dim=0).unsqueeze(dim=0).to(th.float16)
    return token_ids, lens, token_feats


def get_tgt_token_ids(tokens, vocab, max_len):
    all_token_list = list(vocab.keys())
    token_ids = []
    for i, token in enumerate(tokens):
        # ADD NOT IDENTIFIED TOKEN
        if token not in all_token_list:
            if token.startswith('['):
                tokens[i] = '[other_atom]'
            else:
                tokens[i] = '_UNK'
        token = tokens[i]
        token_ids.append(vocab[token])

    token_ids = token_ids[:max_len - 1]
    token_ids.append(vocab["_EOS"])
    lens = len(token_ids)
    while len(token_ids) < max_len:
        token_ids.append(vocab["_PAD"])
    return token_ids, lens


# fix parameters of model
def SME_pred_for_mols(smis, model_name='None', task_number=5, batch_size=128):
    args = {}
    args['device'] = 'cpu'
    args['node_data_field'] = 'node'
    args['edge_data_field'] = 'edge'
    args['substructure_mask'] = 'smask'
    # model parameter
    args['num_epochs'] = 500
    args['patience'] = 30
    args['batch_size'] = batch_size
    args['mode'] = 'higher'
    args['in_feats'] = 40
    args['rgcn_hidden_feats'] = [256, 256]
    args['ffn_hidden_feats'] = 256
    args['rgcn_drop_out'] = 0.2
    args['ffn_drop_out'] = 0.2
    args['lr'] = 0.001
    args['loop'] = True
    # task name (model name)
    args['task_name'] = model_name  # change
    dataset = MolGraphDataset(smis)
    data_loader = DataLoader(dataset, batch_size=args['batch_size'], collate_fn=collate_molgraphs_pred)
    model = MGA(ffn_hidden_feats=args['ffn_hidden_feats'], n_tasks=task_number,
                ffn_dropout=args['ffn_drop_out'],
                in_feats=args['in_feats'], rgcn_hidden_feats=args['rgcn_hidden_feats'],
                rgcn_drop_out=args['rgcn_drop_out'], return_embedding=True)
    stopper = EarlyStopping(patience=args['patience'], task_name=args['task_name']+'_'+str(1),
                            mode=args['mode'])
    model.to(args['device'])
    stopper.load_checkpoint(model)
    model.eval()
    embed_all = []
    for batch_idx, batch_mol_bg in enumerate(data_loader):
        if batch_idx%100==0:
            print('{}/{}'.format(batch_idx+1, len(data_loader)))
        batch_mol_bg = batch_mol_bg.to(args['device'])
        with th.no_grad():
            rgcn_node_feats = batch_mol_bg.ndata.pop(args['node_data_field']).float().to(args['device'])
            rgcn_edge_feats = batch_mol_bg.edata.pop(args['edge_data_field']).long().to(args['device'])
            smask_feats = batch_mol_bg.ndata.pop(args['substructure_mask']).unsqueeze(dim=1).float().to(
                args['device'])
            embedding = model(batch_mol_bg, rgcn_node_feats, rgcn_edge_feats, smask_feats)
            embed_all.append(embedding.detach().cpu().to(th.float16))
            th.cuda.empty_cache()
    embed_all = th.cat(embed_all, dim=0)
    return embed_all


def build_s2s_data(data_file, model_name, tag_list, vocab, group='train', max_len=150, s2s_file='a.csv', batch_size=128):
    data = pd.read_csv(data_file)
    data = data[data['group']==group]
    src_smis = data.src_smi.tolist()
    tgt_smis = data.tgt_smi.tolist()
    tag_name_list = data.task_name.tolist()
    src_atom_n = data.src_atom_n.tolist()

    src_token_ids = []
    src_lens = []
    tgt_token_ids = []
    tgt_lens = []
    token_mga_feats = []

    # 计算各性质src_embed
    src_embed = SME_pred_for_mols(src_smis, task_number=len(tag_list), model_name=model_name, batch_size=batch_size)
    atom_idx_start = 0
    for i in range(len(src_smis)):
        atom_idx_end = atom_idx_start + src_atom_n[i]
        src_embed_i = src_embed[atom_idx_start:atom_idx_end]
        tag_indicator = th.tensor([[1 if tag in tag_name_list[i].split() else 0 for tag in tag_list]]).unsqueeze(2)
        if i%10000==0:
            print(f'{i}/{len(src_smis)} smiles is tokenize.')
        src_token = tag_name_list[i].split() + ['MolStart'] + tokenize_smiles(src_smis[i])
        src_token_ids.append(get_src_token_ids(src_token, vocab, src_embed_i, tag_indicator, max_len)[0])
        src_lens.append(get_src_token_ids(src_token, vocab, src_embed_i, tag_indicator, max_len)[1])
        token_mga_feats.append(get_src_token_ids(src_token, vocab, src_embed_i, tag_indicator, max_len)[2])


        tgt_token_ids.append(get_tgt_token_ids(tokenize_smiles(tgt_smis[i]), vocab, max_len)[0])
        tgt_lens.append(get_tgt_token_ids(tokenize_smiles(tgt_smis[i]), vocab, max_len)[1])
        atom_idx_start = atom_idx_end
    del src_embed
    token_mga_feats = th.cat(token_mga_feats, dim=0).numpy()
    np.savez(s2s_file,
             src_token_ids=src_token_ids,
             src_lens=src_lens,
             tgt_token_ids=tgt_token_ids,
             tgt_lens=tgt_lens,
             src_mga_feats=token_mga_feats
             )


if __name__ == '__main__':
    set_random_seed(10)

    # parameter
    model_name = 'QED_DRD2_data_for_MGA_1_early_stop'
    data_file = '../data/origin_data/Prompt_MolOpt_for_QED_DRD2.csv'
    vocab_file = '../preprocessed/comparison_mol_opt_data/comparison_vocab_smiles.txt'
    train_s2s_file = '../preprocessed/comparison_mol_opt_data/train_0.npz'
    val_s2s_file = '../preprocessed/comparison_mol_opt_data/val_0.npz'
    test_s2s_file = '../preprocessed/comparison_mol_opt_data/test_0.npz'
    max_len=150
    batch_size=128

    # generate MGA model
    if os.path.exists(f'../checkpoints/mga/{model_name}_1_early_stop.pth'):
        print(f'Load {model_name} MGA model...')
    else:
        MGA_SME_hyperopt(1, model_name, 0)
    # generate vacab
    if os.path.exists(vocab_file):
        print(f"Vocab file exists in {vocab_file}")
    else:
        make_vocab(data_file, vocab_file)

    # test match
    vocab = load_vocab(vocab_file)
    remove_mismatch(data_file, vocab)
    print('Remove mismatch is over.!')

    tag_list = ['drd2', 'plogp', 'qed']
    build_s2s_data(data_file=data_file, model_name=model_name, tag_list=tag_list, vocab=vocab, group='training', max_len=max_len, s2s_file=train_s2s_file, batch_size=batch_size)
    build_s2s_data(data_file=data_file, model_name=model_name, tag_list=tag_list, vocab=vocab, group='val', max_len=max_len, s2s_file=val_s2s_file, batch_size=batch_size)
    build_s2s_data(data_file=data_file, model_name=model_name, tag_list=tag_list, vocab=vocab, group='test', max_len=max_len, s2s_file=test_s2s_file, batch_size=batch_size)




