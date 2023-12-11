from MGA_SME import MGA_SME_hyperopt
from maskgnn import set_random_seed
import argparse
import warnings

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    set_random_seed(10)
    # for task_name in ['basic_physicochemical', 'absorption', 'metabolism', 'toxicity', 'tox21', 'Caco-2', 'T12', 'CL']:
    # for task_name in ['mol_opt_relabel']:
    for task_name in ['fs_d_comparison_mol_opt_relabel']:
        MGA_SME_hyperopt(1, task_name, 0)