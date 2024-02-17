import logging
import os.path

import networkx as nx
import numpy as np
import re
import selfies as sf
import sys
import time
import torch
from rdkit import Chem
from torch.utils.data import Dataset
from typing import Dict, List, Tuple
from utils.chem_utils import ATOM_FDIM, BOND_FDIM, get_atom_features_sparse, get_bond_features


def tokenize_selfies_from_smiles(smi: str) -> str:
    encoded_selfies = sf.encoder(smi)
    tokens = list(sf.split_selfies(encoded_selfies))
    assert encoded_selfies == "".join(tokens)

    return " ".join(tokens)


def tokenize_smiles(smi: str) -> str:
    pattern = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == "".join(tokens), f"Tokenization mismatch. smi: {smi}, tokens: {tokens}"

    return " ".join(tokens)


def canonicalize_smiles(smiles, remove_atom_number=False, trim=True, suppress_warning=False):
    cano_smiles = ""

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        cano_smiles = ""

    else:
        if trim and mol.GetNumHeavyAtoms() < 2:
            if not suppress_warning:
                logging.info(f"Problematic smiles: {smiles}, setting it to 'CC'")
            cano_smiles = "CC"          # TODO: hardcode to ignore
        else:
            if remove_atom_number:
                [a.ClearProp('molAtomMapNumber') for a in mol.GetAtoms()]
            cano_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)

    return cano_smiles


def len2idx(lens) -> np.ndarray:
    # end_indices = np.cumsum(np.concatenate(lens, axis=0))
    end_indices = np.cumsum(lens)
    start_indices = np.concatenate([[0], end_indices[:-1]], axis=0)
    indices = np.stack([start_indices, end_indices], axis=1)

    return indices


class S2SBatch:
    def __init__(self,
                 src_token_ids: torch.Tensor,
                 src_lengths: torch.Tensor,
                 tgt_token_ids: torch.Tensor,
                 tgt_lengths: torch.Tensor,
                 src_mga_feats: torch.Tensor):
        self.src_mga_feats = None
        self.src_token_ids = src_token_ids
        self.src_lengths = src_lengths
        self.tgt_token_ids = tgt_token_ids
        self.tgt_lengths = tgt_lengths
        self.src_mga_feats = src_mga_feats

        self.size = len(src_lengths)

    def to(self, device):
        self.src_token_ids = self.src_token_ids.to(device)
        self.src_lengths = self.src_lengths.to(device)
        self.tgt_token_ids = self.tgt_token_ids.to(device)
        self.tgt_lengths = self.tgt_lengths.to(device)
        self.src_mga_feats = self.src_mga_feats.to(device)

    def pin_memory(self):
        self.src_token_ids = self.src_token_ids.pin_memory()
        self.src_lengths = self.src_lengths.pin_memory()
        self.tgt_token_ids = self.tgt_token_ids.pin_memory()
        self.tgt_lengths = self.tgt_lengths.pin_memory()
        self.src_mga_feats = self.src_mga_feats.pin_memory()

        return self

    def log_tensor_shape(self):
        logging.info(f"src_token_ids: {self.src_token_ids.shape}, "
                     f"src_lengths: {self.src_lengths}, "
                     f"tgt_token_ids: {self.tgt_token_ids.shape}, "
                     f"tgt_lengths: {self.tgt_lengths}, "
                     f"src_mga_feats: {self.src_mga_feats}, "
                     )


class S2SDataset(Dataset):
    def __init__(self, args, file: str):
        self.args = args

        self.src_token_ids = []
        self.src_lens = []
        self.tgt_token_ids = []
        self.tgt_lens = []
        self.src_mga_feats = []

        self.data_indices = []
        self.batch_sizes = []
        self.batch_starts = []
        self.batch_ends = []

        logging.info(f"Loading preprocessed features from {file}")
        feat = np.load(file)
        for attr in ["src_token_ids", "src_lens", "tgt_token_ids", "tgt_lens", "src_mga_feats"]:
            setattr(self, attr, feat[attr])

        assert len(self.src_token_ids) == len(self.src_lens) == len(self.tgt_token_ids) == len(self.tgt_lens)==len(self.src_mga_feats), \
            f"Lengths of source and target mismatch!"

        self.data_size = len(self.src_token_ids)
        self.data_indices = np.arange(self.data_size)

        logging.info(f"Loaded and initialized S2SDataset, size: {self.data_size}")

    def sort(self):
        start = time.time()

        logging.info(f"Calling S2SDataset.sort()")
        sys.stdout.flush()
        self.data_indices = np.argsort(self.src_lens + self.tgt_lens)

        logging.info(f"Done, time: {time.time() - start: .2f} s")
        sys.stdout.flush()

    def shuffle_in_bucket(self, bucket_size: int):
        start = time.time()

        logging.info(f"Calling S2SDataset.shuffle_in_bucket()")
        sys.stdout.flush()

        for i in range(0, self.data_size, bucket_size):
            np.random.shuffle(self.data_indices[i:i+bucket_size])

        logging.info(f"Done, time: {time.time() - start: .2f} s")
        sys.stdout.flush()

    def batch(self, batch_type: str, batch_size: int):
        start = time.time()

        logging.info(f"Calling S2SDataset.batch()")
        sys.stdout.flush()

        self.batch_sizes = []

        if batch_type == "samples":
            raise NotImplementedError

        elif batch_type == "atoms":
            raise NotImplementedError

        elif batch_type == "tokens":
            sample_size = 0
            max_batch_src_len = 0
            max_batch_tgt_len = 0

            for data_idx in self.data_indices:
                src_len = self.src_lens[data_idx]
                tgt_len = self.tgt_lens[data_idx]

                max_batch_src_len = max(src_len, max_batch_src_len)
                max_batch_tgt_len = max(tgt_len, max_batch_tgt_len)
                while self.args.enable_amp and not max_batch_src_len % 8 == 0:          # for amp
                    max_batch_src_len += 1
                while self.args.enable_amp and not max_batch_tgt_len % 8 == 0:          # for amp
                    max_batch_tgt_len += 1

                if (max_batch_src_len + max_batch_tgt_len) * (sample_size + 1) <= batch_size:
                    sample_size += 1
                elif self.args.enable_amp and not sample_size % 8 == 0:
                    sample_size += 1
                else:
                    self.batch_sizes.append(sample_size)

                    sample_size = 1
                    max_batch_src_len = src_len
                    max_batch_tgt_len = tgt_len
                    while self.args.enable_amp and not max_batch_src_len % 8 == 0:      # for amp
                        max_batch_src_len += 1
                    while self.args.enable_amp and not max_batch_tgt_len % 8 == 0:      # for amp
                        max_batch_tgt_len += 1

            # lastly
            self.batch_sizes.append(sample_size)
            self.batch_sizes = np.array(self.batch_sizes)
            assert np.sum(self.batch_sizes) == self.data_size, \
                f"Size mismatch! Data size: {self.data_size}, sum batch sizes: {np.sum(self.batch_sizes)}"

            self.batch_ends = np.cumsum(self.batch_sizes)
            self.batch_starts = np.concatenate([[0], self.batch_ends[:-1]])

        else:
            raise ValueError(f"batch_type {batch_type} not supported!")

        logging.info(f"Done, time: {time.time() - start: .2f} s, total batches: {self.__len__()}")
        sys.stdout.flush()

    def __getitem__(self, index: int) -> S2SBatch:
        batch_start = self.batch_starts[index]
        batch_end = self.batch_ends[index]

        data_indices = self.data_indices[batch_start:batch_end]

        # collating, essentially
        src_token_ids = self.src_token_ids[data_indices]
        src_lengths = self.src_lens[data_indices]
        tgt_token_ids = self.tgt_token_ids[data_indices]
        tgt_lengths = self.tgt_lens[data_indices]
        src_mga_feats = self.src_mga_feats[data_indices]

        src_token_ids = src_token_ids[:, :max(src_lengths)]
        tgt_token_ids = tgt_token_ids[:, :max(tgt_lengths)]
        src_mga_feats = src_mga_feats[:, :max(src_lengths), :]

        src_token_ids = torch.as_tensor(src_token_ids, dtype=torch.long)
        tgt_token_ids = torch.as_tensor(tgt_token_ids, dtype=torch.long)
        src_lengths = torch.tensor(src_lengths, dtype=torch.long)
        tgt_lengths = torch.tensor(tgt_lengths, dtype=torch.long)
        src_mga_feats = torch.tensor(src_mga_feats, dtype=torch.float32)
        s2s_batch = S2SBatch(
            src_token_ids=src_token_ids,
            src_lengths=src_lengths,
            tgt_token_ids=tgt_token_ids,
            tgt_lengths=tgt_lengths,
            src_mga_feats=src_mga_feats
        )
        # s2s_batch.log_tensor_shape()
        return s2s_batch

    def __len__(self):
        return len(self.batch_sizes)


class G2SBatch:
    def __init__(self,
                 fnode: torch.Tensor,
                 fmess: torch.Tensor,
                 agraph: torch.Tensor,
                 bgraph: torch.Tensor,
                 atom_scope: List,
                 bond_scope: List,
                 tgt_token_ids: torch.Tensor,
                 tgt_lengths: torch.Tensor,
                 distances: torch.Tensor = None):
        self.fnode = fnode
        self.fmess = fmess
        self.agraph = agraph
        self.bgraph = bgraph
        self.atom_scope = atom_scope
        self.bond_scope = bond_scope
        self.tgt_token_ids = tgt_token_ids
        self.tgt_lengths = tgt_lengths
        self.distances = distances

        self.size = len(tgt_lengths)

    def to(self, device):
        self.fnode = self.fnode.to(device)
        self.fmess = self.fmess.to(device)
        self.agraph = self.agraph.to(device)
        self.bgraph = self.bgraph.to(device)
        self.tgt_token_ids = self.tgt_token_ids.to(device)
        self.tgt_lengths = self.tgt_lengths.to(device)

        if self.distances is not None:
            self.distances = self.distances.to(device)

    def pin_memory(self):
        self.fnode = self.fnode.pin_memory()
        self.fmess = self.fmess.pin_memory()
        self.agraph = self.agraph.pin_memory()
        self.bgraph = self.bgraph.pin_memory()
        self.tgt_token_ids = self.tgt_token_ids.pin_memory()
        self.tgt_lengths = self.tgt_lengths.pin_memory()

        if self.distances is not None:
            self.distances = self.distances.pin_memory()

        return self

    def log_tensor_shape(self):
        logging.info(f"fnode: {self.fnode.shape}, "
                     f"fmess: {self.fmess.shape}, "
                     f"tgt_token_ids: {self.tgt_token_ids.shape}, "
                     f"tgt_lengths: {self.tgt_lengths}")


def make_vocab(fns: Dict[str, List[Tuple[str, str]]], vocab_file: str, tokenized=True):
    assert tokenized, f"Vocab can only be made from tokenized files"

    logging.info(f"Making vocab from {fns}")
    vocab = {}

    for phase, file_list in fns.items():
        for src_file, tgt_file in file_list:
            for fn in [src_file, tgt_file]:
                with open(fn, "r") as f:
                    for line in f:
                        tokens = line.strip().split()
                        for token in tokens:
                            if token in vocab:
                                vocab[token] += 1
                            else:
                                vocab[token] = 1

    logging.info(f"Saving vocab into {vocab_file}")
    with open(vocab_file, "w") as of:
        of.write("_PAD\n_UNK\n_SOS\n_EOS\n")
        for token, count in vocab.items():
            of.write(f"{token}\t{count}\n")


def load_vocab(vocab_file: str) -> Dict[str, int]:
    if os.path.exists(vocab_file):
        logging.info(f"Loading vocab from {vocab_file}")
    else:
        vocab_file = "../preprocessed/default_vocab_smiles.txt"
        logging.info(f"Vocab file invalid, loading default vocab from {vocab_file}")

    vocab = {}
    with open(vocab_file, "r") as f:
        for i, line in enumerate(f):
            token = line.strip().split("\t")[0]
            vocab[token] = i

    return vocab


def data_util_test():
    pass


if __name__ == "__main__":
    data_util_test()
