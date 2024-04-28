from MGA_SME import MGA_SME_hyperopt
from maskgnn import set_random_seed
import argparse
import warnings

warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser(description='Develop MGA models')
parser.add_argument('--task_name', type=str, help='the task name')
args = parser.parse_args()

if __name__ == '__main__':
    set_random_seed(10)
    task_name = args.task_name
    MGA_SME_hyperopt(1, task_name, 0)