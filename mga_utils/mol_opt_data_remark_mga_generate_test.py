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


from rdkit import Chem
from rdkit.Chem import rdmolfiles
from rdkit import Chem
from rdkit.Chem import rdmolops
import re
import re

def split_by_opt_frag(string):
    fragments = []
    current_fragment = ""

    for token in string.split("[Opt Frag]"):
        if token:
            current_fragment += token
            fragments.append(current_fragment)
            current_fragment = ""
        else:
            current_fragment += "[Opt Frag]"

    return fragments


def find_number(s):
    # 使用正则表达式匹配[*:n]模式,对切点根据虚拟atom进行排序，使得先出现的是保留的子结构
    matches = re.findall(r'\[\*\:(-?\d+)\]', s)
    # 如果找到匹配，则返回第一个匹配的数字，否则返回None
    return int(matches[0]) if matches else None


def return_mol_opt_str(cano_str, type='src'):
    ''' input: cano_str = "Cc1ccc([*:1])cc1[*:2].[*:-1]C(=O)O.[*:-2][N+](=O)[O-]"
        output: Cc1ccc([*:1])cc1[*:2][Opt Frag][*:1]C(=O)O.[*:2][N+](=O)[O-]
    '''

    # 根据点号划分字符串
    parts = cano_str.split('.')
    # 找出每个片段中的数字，并根据数字对片段进行排序
    parts = sorted(parts, key=find_number, reverse=True)
    # 遍历每个部分并查找首次出现的 [*:-n] 模式
    new_parts = []
    add_opt_frag = 0
    if type == 'src':
        for part in parts:
            match = re.search(r'\[\*\:-\d+\]', part)
            if match and add_opt_frag == 0:
                new_parts.append('[Opt Frag]')
                new_parts.append(part)
                add_opt_frag += 1
            else:
                new_parts.append(part)
        # 重新组合字符串
        mol_opt_str = '.'.join(new_parts)
        mol_opt_str = mol_opt_str.replace('.[Opt Frag].', '[Opt Frag]')
        mol_opt_str = mol_opt_str.replace(':-', ':')
    if type == 'tgt':
        for part in parts:
            match = re.search(r'\[\*\:-\d+\]', part)
            if add_opt_frag > 0:
                new_parts.append(part)
            if match and add_opt_frag == 0:
                new_parts.append(part)
                add_opt_frag += 1
        # 重新组合字符串
        mol_opt_str = '.'.join(new_parts)
        mol_opt_str = mol_opt_str.replace('[Opt Frag].', '')
        mol_opt_str = mol_opt_str.replace(':-', ':')
    return mol_opt_str


def remove_atom_mapping_and_get_canonical_smiles(mol):
    for atom in mol.GetAtoms():
        atom.ClearProp('molAtomMapNumber')
    mol.UpdatePropertyCache(strict=False)
    rdmolfiles.CanonicalRankAtoms(mol)
    return Chem.MolToSmiles(mol, canonical=True)


def return_remark_src_smi(src_smi, tgt_smi):
    '''src smi 中优化基团未确定，重新确定优化基团，构建数据集'''
    # 将SMARTS转换为分子
    tgt_mol = Chem.MolFromSmiles(tgt_smi)
    src_mol = Chem.MolFromSmiles(src_smi)
    for atom in src_mol.GetAtoms():
        atom.ClearProp('molAtomMapNumber')
    # 创建一个副本并移除映射编号为0的原子
    editable_tgt_mol = Chem.RWMol(tgt_mol)
    atom_indices_to_remove = [atom.GetIdx() for atom in editable_tgt_mol.GetAtoms() if atom.GetAtomMapNum() == 0]

    # 从最高索引开始删除原子，以避免索引问题
    for idx in sorted(atom_indices_to_remove, reverse=True):
        editable_tgt_mol.RemoveAtom(idx)

    # 创建保留部分的mol对象
    kept_part_mol = Chem.Mol(editable_tgt_mol)

    # 使用更灵活的子结构匹配
    kept_part_smarts = Chem.MolToSmarts(kept_part_mol)
    loose_kept_part_mol = Chem.MolFromSmarts(kept_part_smarts, mergeHs=True)
    match = src_mol.GetSubstructMatch(loose_kept_part_mol)

    # 为匹配的原子添加原子映射
    editable_mol = Chem.RWMol(src_mol)
    # 设置映射编号为1，表示保留的部分
    for atom_idx in range(editable_mol.GetNumAtoms()):
        if atom_idx in match:
            editable_mol.GetAtomWithIdx(atom_idx).SetAtomMapNum(1)
        else:
            editable_mol.GetAtomWithIdx(atom_idx).SetAtomMapNum(0)

    # 将修改后的分子转换为smiles
    remark_src_smi = Chem.MolToSmiles(editable_mol)
    return remark_src_smi


def return_nomark_smi(mark_smi):
    return re.sub(r'\[\*:\-?\d+\]', '*', mark_smi)


def return_mol_opt_smi(smi, type='src'):
    # 将SMARTS字符串转换为分子对象
    mol = Chem.MolFromSmarts(smi)

    # 返回所有键的index
    def get_broken_bond_indices(mol):
        bond_indices = []
        for bond in mol.GetBonds():
            begin_atommapnum = mol.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetAtomMapNum()
            end_atommapnum = mol.GetAtomWithIdx(bond.GetEndAtomIdx()).GetAtomMapNum()
            if begin_atommapnum != end_atommapnum:
                bond_indices.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        return bond_indices

    broken_bond_indices = get_broken_bond_indices(mol)

    mol = Chem.RWMol(mol)
    virtual_atom_num = 1
    for broken_bond_indice in broken_bond_indices:
        mol.RemoveBond(broken_bond_indice[0], broken_bond_indice[1])
        for atom_idx in broken_bond_indice:
            virtual_atom = Chem.Atom(0)
            mol.AddAtom(virtual_atom)
            # 为虚拟节点添加表示是否改动
            if mol.GetAtomWithIdx(atom_idx).GetAtomMapNum() == 1:
                mol.GetAtomWithIdx(mol.GetNumAtoms() - 1).SetAtomMapNum(-virtual_atom_num)
            else:
                mol.GetAtomWithIdx(mol.GetNumAtoms() - 1).SetAtomMapNum(virtual_atom_num)
            mol.AddBond(atom_idx, mol.GetNumAtoms() - 1, Chem.BondType.SINGLE)
        virtual_atom_num += 1

    for idx, atom in enumerate(mol.GetAtoms()):
        # 为原子添加映射编号(改动的为负值，不改动的为正值)
        if atom.GetSymbol() != '*':
            atom.SetAtomMapNum(idx)
    # 更新分子，使其正确表示原子和键的连接
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES)
        mol.UpdatePropertyCache(strict=False)
    # 更新分子，使其正确表示原子和键的连接
    except:
        # Handle the exception if needed
        pass
    # 输出分子的SMILES字符串，展示原子映射编号
    clear_mol = mol
    for atom in clear_mol.GetAtoms():
        if atom.GetSymbol() != '*':
            atom.ClearProp('molAtomMapNumber')
    cano_smi = Chem.MolToSmiles(clear_mol, kekuleSmiles=False)
    mol_opt_smi = return_mol_opt_str(cano_smi, type)
    return mol_opt_smi


def return_mol_opt_smi_atom_index(smi):
    # 将SMARTS字符串转换为分子对象
    mol = Chem.MolFromSmarts(smi)
    atom_number = mol.GetNumAtoms()

    # 有些分子没有移除立体构型会识别不出来子结构匹配部分，因此续提前移除子结构匹配部分
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
            bond.SetStereo(Chem.rdchem.BondStereo.STEREONONE)
    Chem.RemoveStereochemistry(mol)

    # 返回所有键的index
    def get_broken_bond_indices(mol):
        bond_indices = []
        for bond in mol.GetBonds():
            begin_atommapnum = mol.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetAtomMapNum()
            end_atommapnum = mol.GetAtomWithIdx(bond.GetEndAtomIdx()).GetAtomMapNum()
            if begin_atommapnum != end_atommapnum:
                bond_indices.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        return bond_indices

    broken_bond_indices = get_broken_bond_indices(mol)

    mol = Chem.RWMol(mol)
    virtual_atom_num = 1
    for broken_bond_indice in broken_bond_indices:
        mol.RemoveBond(broken_bond_indice[0], broken_bond_indice[1])
        for atom_idx in broken_bond_indice:
            virtual_atom = Chem.Atom(0)
            mol.AddAtom(virtual_atom)
            # 为虚拟节点添加表示是否改动
            if mol.GetAtomWithIdx(atom_idx).GetAtomMapNum() == 1:
                mol.GetAtomWithIdx(mol.GetNumAtoms() - 1).SetAtomMapNum(-virtual_atom_num)
            else:
                mol.GetAtomWithIdx(mol.GetNumAtoms() - 1).SetAtomMapNum(virtual_atom_num)
            mol.AddBond(atom_idx, mol.GetNumAtoms() - 1, Chem.BondType.SINGLE)
        virtual_atom_num += 1

    for idx, atom in enumerate(mol.GetAtoms()):
        # 为原子添加映射编号(改动的为负值，不改动的为正值)
        if atom.GetSymbol() != '*':
            atom.SetAtomMapNum(idx)
    # 更新分子，使其正确表示原子和键的连接
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES)
        mol.UpdatePropertyCache(strict=False)
    # 更新分子，使其正确表示原子和键的连接
    except:
        # Handle the exception if needed
        pass
    # 输出分子的SMILES字符串，展示原子映射编号
    clear_mol = mol
    for atom in clear_mol.GetAtoms():
        if atom.GetSymbol() != '*':
            atom.ClearProp('molAtomMapNumber')
    cano_smi = Chem.MolToSmiles(clear_mol, kekuleSmiles=False)
    cano_smi_sort = return_mol_opt_str(cano_smi, type='src').replace('[Opt Frag]', '.')
    clear_mol = Chem.MolFromSmarts(return_nomark_smi(cano_smi_sort))
    mol_index = list(mol.GetSubstructMatches(clear_mol)[0])
    #将虚拟节点的index设置为-1
    new_atom_index = [x if x < atom_number else -1 for x in mol_index]
    return new_atom_index


def preprocess_opt_data(src_smi):
    # 根据输入的分子处理成我们需要的格式
    try:
        new_src_smi = return_mol_opt_smi(src_smi, type='src')
        src_atom_index = return_mol_opt_smi_atom_index(src_smi)
        return new_src_smi, src_atom_index
    except:
        print(f'SRC: {src_smi}  is preprocessed failed.')
        return 'CC', 'CC', []


def tokenize_smiles(smi):
    pattern = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    return tokens


def make_vocab(data_file, vocab_file):
    vocab = {}
    data = pd.read_csv(data_file)
    src_smi_list = data.src_smi.tolist()
    tgt_smi_list = data.tgt_smi.tolist()
    smi_list = []
    for i in range(len(src_smi_list)):
        if i%10000==0:
            print(f"{i}/{len(src_smi_list)} is preprocessed.")
        src_smi, tgt_smi, _ = preprocess_opt_data(src_smi_list[i], tgt_smi_list[i])
        smi_list.append(src_smi)
        smi_list.append(tgt_smi)
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
        of.write("_PAD\n_UNK\n_SOS\n_EOS\n[other_atom]\n[Pharmacophores]\n")
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
    all_atom_list = initial_atom_list + [element for element in all_token_list if element.startswith('[')
                                         and element not in ['[Pharmacophores]', '[Opt Frag]']]
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


def get_src_token_ids(tokens, vocab, embed, tag_indicator, max_len, src_atom_index):
    initial_atom_list = ['c', 'N', 'C', 'O', 'Cl', 'o', 'S', 'P', 'n', 'Br', 'F', 's', 'I', 'B']
    all_token_list = list(vocab.keys())
    all_atom_list = initial_atom_list + [element for element in all_token_list if element.startswith('[')
                                         and element not in ['[Pharmacophores]', '[Opt Frag]']]
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
        if token in all_atom_list or token.startswith('[*'):
            if src_atom_index[atom_index]!=-1:
                token_feats.append(atom_feats[src_atom_index[atom_index]:src_atom_index[atom_index]+1])
                atom_index=atom_index+1
            else:
                token_feats.append(non_atom_feats)
                atom_index = atom_index + 1
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
    args['device'] = "cpu"
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


def build_fs_data(data_file, model_name, tag_list, vocab, group='train', max_len=150, fs_file='a.csv', batch_size=128,
                  test_file='a_test.csv'):
    data = pd.read_csv(data_file)
    data = data[data['group'] == group]
    src_smis = data.src_smi.tolist()
    tgt_smis = data.tgt_smi.tolist()

    cano_src_smis = data.cano_src_smi.tolist()
    tag_name_list = data.task_name.tolist()
    src_atom_n = data.src_atom_n.tolist()

    src_token_ids = []
    src_lens = []
    tgt_token_ids = []
    tgt_lens = []
    token_mga_feats = []
    # 计算各性质src_embed
    src_embed = SME_pred_for_mols(cano_src_smis, model_name=model_name, batch_size=batch_size)
    atom_idx_start = 0
    retain_subs = []
    opt_frags = []
    for i in range(len(src_smis)):
        atom_idx_end = atom_idx_start + src_atom_n[i]
        src_embed_i = src_embed[atom_idx_start:atom_idx_end]
        tag_indicator = th.tensor([[1 if tag in tag_name_list[i].split() else 0 for tag in tag_list]]).unsqueeze(2)
        if i % 10000 == 0:
            print(f'{i}/{len(src_smis)} smiles is tokenize.')
        src_smi, src_atom_index = preprocess_opt_data(src_smis[i])
        retain_sub = src_smi.split("[Opt Frag]")[0]
        retain_subs.append(retain_sub)
        tgt_smi = tgt_smis[i]
        if src_smi == 'CC':
            atom_idx_start = atom_idx_end
        else:
            src_token = tag_name_list[i].split() + ['[Pharmacophores]'] + tokenize_smiles(src_smi)
            src_token_ids_i, src_lens_i, token_mga_feats_i = get_src_token_ids(src_token, vocab, src_embed_i,
                                                                               tag_indicator, max_len, src_atom_index)
            src_token_ids.append(src_token_ids_i)
            src_lens.append(src_lens_i)
            token_mga_feats.append(token_mga_feats_i)
            tgt_token_ids.append(get_tgt_token_ids(tokenize_smiles(tgt_smi), vocab, max_len)[0])
            tgt_lens.append(get_tgt_token_ids(tokenize_smiles(tgt_smi), vocab, max_len)[1])
            atom_idx_start = atom_idx_end
    if group == 'test':
        data['Pharmacophores'] = retain_subs
        data.to_csv(test_file, index=False)
    del data, src_smis, tgt_smis, cano_src_smis, src_embed, retain_subs, opt_frags
    token_mga_feats = th.cat(token_mga_feats, dim=0).numpy()
    np.savez(fs_file,
             src_token_ids=src_token_ids,
             src_lens=src_lens,
             tgt_token_ids=tgt_token_ids,
             tgt_lens=tgt_lens,
             src_mga_feats=token_mga_feats
             )

if __name__ == '__main__':
    # 此处代码有部分修改，这里标记的不是保留部分（原本保留部分标记为1），替换部分少一些，这里修改的是待优化基团（因为待优化基团少一些），待替换基团的标记为1
    set_random_seed(10)
    # parameter
    case_task_name='hERG_BBBP'
    model_name = 'ADMET_data_for_MGA'
    data_file = f'../data/origin_data/{case_task_name}_case_study_test_0.csv'
    vocab_file = '../preprocessed/fs_mga_remark_data/remark_vocab_smiles.txt'
    train_fs_file = f'../preprocessed/{case_task_name}_case_study/train_0.npz'
    val_fs_file = f'../dpreprocessed/{case_task_name}_case_study/val_0.npz'
    test_fs_file = f'../preprocessed/{case_task_name}_case_study/test_0.npz'
    test_file = f'../preprocessed/{case_task_name}_case_study/test_0.csv'
    max_len=150
    batch_size=128
    task_name = 'ADMET_data_for_MGA'

    # generate MGA model
    if os.path.exists(f'../checkpoints/mga/{task_name}_1_early_stop.pth'):
        print(f'Load {task_name} MGA model...')
    else:
        MGA_SME_hyperopt(1, task_name, 0)
    if os.path.exists(vocab_file):
        print(f"Vocab file exists in {vocab_file}")
    else:
        make_vocab(data_file, vocab_file)

    # test match
    vocab = load_vocab(vocab_file)
    remove_mismatch(data_file, vocab)
    print('Remove mismatch is over.!')

    tag_list = ['Mutagenicity', 'hERG', 'BBBP', 'ESOL', 'lipop']
    build_fs_data(data_file=data_file, model_name=model_name, tag_list=tag_list, vocab=vocab, group='test', max_len=max_len, fs_file=test_fs_file, batch_size=batch_size, test_file=test_file)



