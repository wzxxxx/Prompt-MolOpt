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
from rdkit import Chem
from mga_utils.mol_opt_data_mga_generate_for_iteration import build_s2s_data, load_mga_vocab
from properties import similarity, get_prop


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


def predict(args):
    # parsing.log_args(args)
    test_csv = pd.read_csv(args.test_csv)
    # 初始数据选择测试集
    if 'group' in test_csv.columns:
        test_csv = test_csv[test_csv['group']=='test']

    # src smi最初来自初始分子，后面来自opt mol
    if 'opt smi' not in test_csv.columns:
        src_smis = test_csv.src_smi.tolist()
    else:
        src_smis = test_csv['opt smi'].tolist()

    # origin 最初来自src smi，后来来自origin smi
    if 'origin smi' not in test_csv.columns:
        origin_smis = src_smis
    else:
        origin_smis = test_csv['origin smi'].tolist()
    task_name = test_csv.task_name.tolist()
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
        vocab = load_vocab(pretrain_args.vocab_file)
        vocab_tokens = [k for k, v in sorted(vocab.items(), key=lambda tup: tup[1])]

        model = model_class(pretrain_args, vocab)
        model.load_state_dict(pretrain_state_dict)
        logging.info(f"Loaded pretrained state_dict from {args.load_from}")

        model.to(device)
        model.eval()

        # logging.info(model)
        # logging.info(f"Number of parameters = {param_count(model)}")

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
                    smis = [origin_smis[src_smi_idx+i], src_smis[src_smi_idx+i]]
                    for prediction in predictions:
                        predicted_idx = prediction.detach().cpu().numpy()
                        predicted_tokens = [vocab_tokens[idx] for idx in predicted_idx[:-1]]
                        smi = "".join(predicted_tokens)
                        smis.append(smi)
                    cano_smis = []
                    for smi in smis:
                        if len(cano_smis) == (args.n_best+2):
                            break
                        try:
                            cano_smi = remove_atom_mapping_and_get_canonical_smiles(smi)
                            if cano_smi not in cano_smis and cano_smi!='':
                                cano_smis.append(cano_smi)
                        except:
                            pass
                    NaN_cano_smi = ['CC' for i in range(args.n_best+2-len(cano_smis))]
                    cano_smis = cano_smis + NaN_cano_smi
                    # cano_smis = ",".join(cano_smis)
                    all_predictions.append(cano_smis)
                src_smi_idx = src_smi_idx + len(results["predictions"])
        columns = ['origin smi', 'src smi'] + [f'opt smi {i+1}' for i in range(args.n_best)]
        data = pd.DataFrame(all_predictions, columns=columns)
        data['task name'] = task_name
        return data



def return_mol_score(origin_smi, src_smi, opt_smi, iteration):
    sim_score = similarity(origin_smi, opt_smi, sim_type='binary')
    drd2_score = get_prop(opt_smi, prop='drd2')
    qed_score = get_prop(opt_smi, prop='qed')
    drd2_qed_socre = drd2_score + qed_score
    return origin_smi, src_smi, opt_smi, sim_score, drd2_score, qed_score, drd2_qed_socre, iteration


def main(args):
    test_data = pd.read_csv(args.test_csv)
    test_data = test_data[test_data['group']=='test']
    test_data = test_data[test_data['task_name']=='drd2 qed']
    src_smis = test_data['src_smi'].tolist()
    origin_smis = [remove_atom_mapping_and_get_canonical_smiles(smi) for smi in src_smis]
    all_prediction = pd.DataFrame()
    task_name='drd2 qed'

    src_drd2_scores = {smi:get_prop(smi, prop='drd2') for smi in src_smis}
    src_qed_scores = {smi:get_prop(smi, prop='qed') for smi in src_smis}
    # parameter
    model_name = 'fs_comparison_mol_opt_relabel'
    s2s_file = './preprocessed/fs_comparison_mol_opt_data/'
    vocab_file = './preprocessed/fs_comparison_mol_opt_data/fs_comparison_vocab_smiles.txt'
    tag_list = ['drd2', 'plogp', 'qed']
    vocab = load_mga_vocab(vocab_file)
    all_result = pd.DataFrame()
    for iteration in range(5):
        if iteration!=0:
            test_data = result_next_iter
            src_smis = test_data['opt smi'].tolist()
        print(f'{iteration+1} iteration')
        args.test_csv = f"{s2s_file}{task_name}_{iteration+1}.csv"
        test_data.to_csv(args.test_csv, index=False)
        args.test_bin = f"{s2s_file}{task_name}_{iteration+1}.npz"
        build_s2s_data(src_smis, iteration+1, task_name, model_name, tag_list, vocab, s2s_file=s2s_file, batch_size=128)
        print('Data is built.')
        result_iter = predict(args)
        # 选择过的result
        result_next_iter = pd.DataFrame()
        for i, smi in enumerate(origin_smis):
            result_iter_i = result_iter[result_iter['origin smi']==smi]
            src_smis_i = result_iter_i['src smi'].tolist()
            src_smis_i = [item for item in src_smis_i for _ in range(args.n_best)]
            opt_mol_columns = [f'opt smi {i+1}' for i in range(args.n_best)]
            opt_smis_i = result_iter_i[opt_mol_columns].stack().tolist()
            scores_i = [return_mol_score(smi, src_smis_i[i], opt_smis_i[i],  iteration+1) for i, opt_smi in enumerate(opt_smis_i) if opt_smi!='CC']
            result_i = pd.DataFrame(scores_i, columns=['origin smi', 'src smi', 'opt smi', 'sim_score', 'opt_drd2_score', 'opt_qed_score', 'opt_drd2_qed_score', 'iteration'])

            result_i['src_drd2_score'] = [src_drd2_scores[x] for x in result_i['origin smi'].tolist()]
            result_i['src_qed_score'] = [src_qed_scores[x] for x in result_i['origin smi'].tolist()]
            result_i['task_name'] = [task_name for x in range(len(result_i))]
            result_i.drop_duplicates(subset=['origin smi', 'src smi', 'opt smi', 'iteration'], inplace=True, keep='first')
            all_result = pd.concat([all_result, result_i], axis=0)
            # 根据阈值划分取前五
            result_sel_i = result_i[result_i['sim_score']>=args.cutoff_sim].copy()
            result_sel_i = result_sel_i.sort_values(by='opt_drd2_qed_score', ascending=False)
            result_next_iter = pd.concat([result_next_iter, result_sel_i[:5]])
    all_result.to_csv(args.result_file, index=False)




if __name__ == "__main__":
    for cutoff_sim in [0.4, 0.5, 0.6]:
        predict_parser = get_predict_parser()
        args = predict_parser.parse_args()

        # set random seed (just in case)
        set_seed(args.seed)

        # logger setup
        logger = setup_logger(args, warning_off=True)
        args = vars(args)
        args_s2s = {
        "cutoff_sim":cutoff_sim,
        "do_predict": True,
        "model": "s2s_sme",
        "test_csv": './preprocessed/comparison_mol_opt.csv',
        "result_file": f"./result/5_{cutoff_sim}_comparison_result.csv",
        "log_file": "5_comparison_result.log",
        "load_from": "./checkpoints/comparison_mol_opt_data/model.500000_99.pt",
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


