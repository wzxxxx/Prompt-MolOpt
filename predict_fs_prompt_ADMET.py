import argparse
import logging
import numpy as np
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
    test_csv = test_csv[test_csv['group']=='test']
    src_smis = test_csv.src_smi.tolist()
    src_group = test_csv.group.tolist()
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
                    smis = [src_smis[src_smi_idx+i]]
                    for prediction in predictions:
                        predicted_idx = prediction.detach().cpu().numpy()
                        predicted_tokens = [vocab_tokens[idx] for idx in predicted_idx[:-1]]
                        smi = "".join(predicted_tokens)
                        smis.append(smi)
                    cano_smis = []
                    for smi in smis:
                        if len(cano_smis) == (args.n_best+1):
                            break
                        try:
                            cano_smi = remove_atom_mapping_and_get_canonical_smiles(smi)
                            if cano_smi not in cano_smis and cano_smi!='':
                                cano_smis.append(cano_smi)
                        except:
                            pass
                    NaN_cano_smi = ['CC' for i in range(args.n_best+1-len(cano_smis))]
                    cano_smis = cano_smis + NaN_cano_smi
                    # cano_smis = ",".join(cano_smis)
                    all_predictions.append(cano_smis)
                src_smi_idx = src_smi_idx + len(results["predictions"])
        columns = ['src smi'] + [f'opt smi {i+1}' for i in range(args.n_best)]
        data = pd.DataFrame(all_predictions, columns=columns)
        data['group'] = src_group
        data['task name'] = task_name
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
    "data_name": "fs_mga_data",
    "test_bin": "./preprocessed/fs_mga_data/test_0.npz",
    "test_csv": "./data/origin_data/fs_mol_opt_data.csv",
    "result_file": "./result/Prompt_MolOpt_fs_ADMET_result.csv",
    "log_file": "Prompt_MolOpt_fs_ADMET_result.log",
    "load_from": "./checkpoints/fs_mga_data/model.500000_99.pt",
    "vocab_file" : "./preprocessed/s2s_mga_data/vocab_smiles.txt",
    "seed": 1216,
    "batch_type": "tokens",
    "predict_batch_size": 4096,
    "beam_size": 50,
    "n_best": 5,
    "predict_min_len": 1,
    "predict_max_len": 512,
    "log_iter": 100
    }
    args = {**args, **args_s2s}
    args = SimpleNamespace(**args)

    torch.set_printoptions(profile="full")
    main(args)


