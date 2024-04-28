import argparse
import logging
import os
import sys

import pandas as pd
import torch
from models.seq2seq import Seq2Seq
from models.seq2seq_sme import Seq2Seq_sme
from torch.utils.data import DataLoader
from utils import parsing
from utils.data_utils import canonicalize_smiles, load_vocab, S2SDataset
from utils.train_utils import log_tensor, param_count, set_seed, setup_logger
from types import SimpleNamespace
from rdkit.Chem import rdmolfiles
from rdkit.Chem import rdmolops
from rdkit import Chem

from rdkit.Chem.rdchem import ChiralType
import numpy as np
from rdkit.Chem import rdmolops
from rdkit import Chem


def get_neighbors_by_cip_rank(mol, chiral_atom_idx):
    chiral_atom = mol.GetAtomWithIdx(chiral_atom_idx)
    neighbors = list(chiral_atom.GetNeighbors())
    neighbors_rank = [int(x.GetProp('_CIPRank')) for x in neighbors]
    return [i[0] for i in sorted(enumerate(neighbors_rank), key=lambda x:x[1])]

def get_bond_type(mol, atom1_idx, atom2_idx):
    bond = mol.GetBondBetweenAtoms(atom1_idx, atom2_idx)

    if bond is not None:
        bond_type = bond.GetBondType()
        return bond_type
    else:
        return None


def is_molecule_valid(mol):
    try:
        rdmolops.SanitizeMol(mol)
        return True
    except:
        return False


def return_mark_opt_smi(src_smi, retain_sub, opt_frag):
    try:
        """输入保留结构的smi和优化基团的smi，输出优化后分子的smi"""
        retain_sub = Chem.MolFromSmarts(retain_sub)
        opt_frag = Chem.MolFromSmiles(opt_frag)
        new_mol = Chem.RWMol(retain_sub)
        new_mol.UpdatePropertyCache()
        Chem.AssignAtomChiralTagsFromStructure(new_mol)

        # 记录初始分子的手性顺序，以及手性中心index
        src_mol = Chem.MolFromSmiles(src_smi)
        src_chiral_atom_idx = []
        src_chiral_atom_neighbor_idx = []

        # 记录原连接原子
        connnect_atom_dic1 = {}
        connect_bond_dic = {}
        virtual_atom_list = []
        for atom in new_mol.GetAtoms():
            if atom.GetSymbol()=='*':
                idx_v_atom = atom.GetIdx()
                virtual_atom_list.append(idx_v_atom)
                idx_v_atom_neighbor = [neighbor.GetIdx() for neighbor in atom.GetNeighbors()][0]
                connnect_atom_dic1[f'*{str(atom.GetAtomMapNum())}'] = idx_v_atom_neighbor
                connect_bond_dic[f'*{str(atom.GetAtomMapNum())}'] = get_bond_type(new_mol, idx_v_atom, idx_v_atom_neighbor)
                # 记录手性中心原子
                change_atom_index = atom.GetNeighbors()[0].GetIdx()
                chiral_atom = src_mol.GetAtomWithIdx(change_atom_index)
                if chiral_atom.GetChiralTag() in [ChiralType.CHI_TETRAHEDRAL_CW, ChiralType.CHI_TETRAHEDRAL_CCW]:
                    # 获取与手性中心相连的邻居原子
                    # neighbors = [atom.GetIdx() for atom in chiral_atom.GetNeighbors()]
                    src_chiral_atom_idx.append(change_atom_index)
                    src_chiral_atom_neighbor_idx.append(get_neighbors_by_cip_rank(src_mol, change_atom_index))
        # 记录优化结构中的连接原子
        connnect_atom_dic2 = {}
        # 添加官能团
        for atom in opt_frag.GetAtoms():
            if atom.GetSymbol() == '*':
                idx_v_atom = atom.GetIdx() + len(retain_sub.GetAtoms())
                virtual_atom_list.append(idx_v_atom)
                idx_v_atom_neighbor = [neighbor.GetIdx() for neighbor in atom.GetNeighbors()][0] + len(retain_sub.GetAtoms())
                connnect_atom_dic2[f'*{str(atom.GetAtomMapNum())}'] = idx_v_atom_neighbor
            new_mol.AddAtom(atom)
        for atom in new_mol.GetAtoms():
            if atom.GetIdx() < len(retain_sub.GetAtoms()):
                atom.SetAtomMapNum(0)
            else:
                atom.SetAtomMapNum(1)

        for bond in opt_frag.GetBonds():
            begin_idx = bond.GetBeginAtomIdx() + len(retain_sub.GetAtoms())
            end_idx = bond.GetEndAtomIdx() + len(retain_sub.GetAtoms())
            new_mol.AddBond(begin_idx, end_idx, bond.GetBondType())
        virtual_atom_list = sorted(virtual_atom_list, reverse=True)
        connect_bond_list = [[connnect_atom_dic1[key], connnect_atom_dic2[key], connect_bond_dic[key]] for key in connnect_atom_dic1.keys()]
        # 为官能团与原分子添加连接键
        for bond in connect_bond_list:
            # 添加连接化学键
            begin_idx = bond[0]
            end_idx = bond[1]
            bond_type = bond[2]
            new_mol.AddBond(begin_idx, end_idx, bond_type)
        # 删除虚拟节点*
        for atom_idx in virtual_atom_list:
            new_mol.RemoveAtom(atom_idx)
        # 检查手性是否跟原有一致
        if is_molecule_valid(new_mol.GetMol()):
            smi = Chem.MolToSmiles(new_mol)
            if '.' not in smi:
                Chem.MolToSmiles(new_mol)
                new_mol.UpdatePropertyCache()
                new_mol = Chem.MolFromSmiles(Chem.MolToSmiles(new_mol))
                Chem.AssignAtomChiralTagsFromStructure(new_mol)
                for j, chiral_atom_idx in enumerate(src_chiral_atom_idx):
                    new_chiral_atom = new_mol.GetAtomWithIdx(chiral_atom_idx)
                    if new_chiral_atom.GetChiralTag() in [ChiralType.CHI_TETRAHEDRAL_CW, ChiralType.CHI_TETRAHEDRAL_CCW]:
                        # 获取与手性中心相连的邻居原子
                        new_neighbors = (get_neighbors_by_cip_rank(new_mol, chiral_atom_idx))
                        #如果手性顺序发生了变化，进行修改，保持不变，使其仍然保留原有手性。
                        if new_neighbors != src_chiral_atom_neighbor_idx[j]:
                            new_chiral_tag = [x for x in [ChiralType.CHI_TETRAHEDRAL_CW, ChiralType.CHI_TETRAHEDRAL_CCW] if x != new_chiral_atom.GetChiralTag()]
                            new_chiral_atom.SetChiralTag(new_chiral_tag[0])
                return Chem.MolToSmiles(new_mol)
            else:
                return False
        else:
            return False

    except:
        return False


def remove_atom_mapping_and_get_canonical_smiles(smi):
    mol = Chem.MolFromSmiles(smi)
    for atom in mol.GetAtoms():
        atom.ClearProp('molAtomMapNumber')
    mol.UpdatePropertyCache(strict=False)
    rdmolfiles.CanonicalRankAtoms(mol)
    cano_smi = Chem.MolToSmiles(mol, canonical=True)
    cano_smi = Chem.MolToSmiles(Chem.MolFromSmiles(cano_smi), canonical=True)
    return cano_smi


def get_predict_parser():
    parser = argparse.ArgumentParser("predict")
    parsing.add_common_args(parser)
    parsing.add_preprocess_args(parser)
    parsing.add_train_args(parser)
    parsing.add_predict_args(parser)

    return parser


def main(args):
    parsing.log_args(args)
    test_csv = pd.read_csv(args.test_csv)
    cano_src_smis = test_csv.cano_src_smi.tolist()
    src_smis = test_csv.src_smi.tolist()
    src_group = test_csv.group.tolist()
    task_name = test_csv.task_name.tolist()
    retain_subs = test_csv.Pharmacophores.tolist()

    if args.do_predict and os.path.exists(args.result_file):
        logging.info(f"Result file found at {args.result_file}, skipping prediction")

    elif args.do_predict and not os.path.exists(args.result_file):
        # os.makedirs(os.path.join("./results", args.data_name), exist_ok=True)

        # initialization ----------------- model
        assert os.path.exists(args.load_from), f"{args.load_from} does not exist!"
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        state = torch.load(args.load_from)
        pretrain_args = state["args"]
        pretrain_state_dict = state["state_dict"]


        assert args.model == pretrain_args.model, f"Pretrained model is {pretrain_args.model}!"
        if args.model == "s2s":
            model_class = Seq2Seq
            dataset_class = S2SDataset
        elif args.model == "s2s_sme":
            model_class = Seq2Seq_sme
            dataset_class = S2SDataset
        else:
            raise ValueError(f"Model {args.model} not supported!")

        # initialization ----------------- vocab
        vocab = load_vocab(args.vocab_file)
        vocab_tokens = [k for k, v in sorted(vocab.items(), key=lambda tup: tup[1])]

        model = model_class(pretrain_args, vocab)
        model.load_state_dict(pretrain_state_dict)
        logging.info(f"Loaded pretrained state_dict from {args.load_from}")

        model.to(device)
        model.eval()

        logging.info(model)
        logging.info(f"Number of parameters = {param_count(model)}")

        # initialization ----------------- data
        test_dataset = dataset_class(pretrain_args, file=args.test_bin)
        test_dataset.batch(
            batch_type=args.batch_type,
            batch_size=args.predict_batch_size
        )
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=lambda _batch: _batch[0],
            pin_memory=True
        )

        all_predictions = []

        with torch.no_grad():
            src_smi_idx = 0
            for test_idx, test_batch in enumerate(test_loader):
                if test_idx % args.log_iter == 0:
                    logging.info(f"Doing inference on test step {test_idx}")
                    sys.stdout.flush()

                test_batch.to(device)
                results = model.predict_step(
                    mol_opt_batch=test_batch,
                    batch_size=test_batch.size,
                    beam_size=args.beam_size,
                    n_best=args.beam_size,
                    min_length=args.predict_min_len,
                    max_length=args.predict_max_len
                )

                for i, predictions in enumerate(results["predictions"]):
                    smis = [src_smis[src_smi_idx+i], cano_src_smis[src_smi_idx+i]]
                    for prediction in predictions:
                        predicted_idx = prediction.detach().cpu().numpy()
                        predicted_tokens = [vocab_tokens[idx] for idx in predicted_idx[:-1]]
                        opt_frag_smi = "".join(predicted_tokens)
                        smi = return_mark_opt_smi(src_smis[src_smi_idx+i], retain_subs[src_smi_idx+i], opt_frag_smi)
                        if smi:
                            smis.append(smi)
                        else:
                            pass
                    cano_smis = []
                    for smi in smis:
                        if len(cano_smis) == (args.n_best+2):
                            break
                        try:
                            cano_smi = remove_atom_mapping_and_get_canonical_smiles(smi)
                            if cano_smi not in cano_smis and cano_smi!='':
                                cano_smis.append(smi)
                        except:
                            pass
                    NaN_cano_smi = ['CC' for i in range(args.n_best+2-len(cano_smis))]
                    cano_smis = cano_smis + NaN_cano_smi
                    # cano_smis = ",".join(cano_smis)
                    all_predictions.append(cano_smis)
                src_smi_idx = src_smi_idx + len(results["predictions"])
        columns = ['src smi', 'cano src smi'] + [f'opt smi {i+1}' for i in range(args.n_best)]
        data = pd.DataFrame(all_predictions, columns=columns)
        data['group'] = src_group
        data['task name'] = task_name
        data['retain_subs'] = retain_subs
        data.to_csv(args.result_file, index=False)


if __name__ == "__main__":
    predict_parser = get_predict_parser()
    args = predict_parser.parse_args()

    # set random seed (just in case)
    set_seed(args.seed)

    # logger setup
    logger = setup_logger(args, warning_off=True)
    args = vars(args)
    args_s2s = {
    "do_predict": True,
    "model": "s2s_sme",
    "data_name": "s2s_mga_data",
    "test_bin": "./preprocessed/hERG_BBBP_case_study/test_0.npz",
    "test_csv": "./preprocessed/hERG_BBBP_case_study/test_0.csv",
    "result_file": "./result/PromptP_molopt_hERG_BBBP_case_study_result.csv",
    "log_file": "PromptP_molopt_hERG_BBBP_case_study_predict.log",
    "load_from": "./checkpoints/fs_mga_remark_data/model.500000_99.pt",
    "vocab_file" : "./preprocessed/fs_mga_remark_data/remark_vocab_smiles.txt",
    "seed": 1216,
    "batch_type": "tokens",
    "predict_batch_size": 4096,
    "beam_size": 50,
    "n_best": 20,
    "predict_min_len": 1,
    "predict_max_len": 512,
    "log_iter": 100
    }
    args = {**args, **args_s2s}
    args = SimpleNamespace(**args)

    torch.set_printoptions(profile="full")
    main(args)


