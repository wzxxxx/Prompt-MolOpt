# first used the code s2s_opt_mol_all_pred.py in sme_opt_utils to generate the prediction of the opt mols
import pandas as pd

model_list = ['Prompt_MolOpt_ADMET', 'Prompt_MolOpt_no_embed_ADMET']
task_list = ['hERG', 'lipop', 'Mutagenicity', 'ESOL', 'BBBP']


for task in task_list:
    for model in model_list:
        data_t_all_num = 0
        data_t_succeed = 0
        for i in range(1, 6):
            data = pd.read_csv(f'{model}_result_all_pred.csv')
            data = data[data['task name']==f'{task}']
            data = data[data['re_group'] == 'test_1']
            data_i = data[(data[f'opt smi {i}']!='CC')]
            len_data = len(data)
            if task in ['ESOL', 'lipop']:
                data_i_succeed = data_i[(data_i[f'src pred {task}'] - data_i[f'opt smi {i} pred {task}'])<=-0.5]
            if task in ['Mutagenicity', 'hERG']:
                data_i = data_i[data_i[f'src pred {task}']>=0.5]
                data_i_succeed = data_i[data_i[f'opt smi {i} pred {task}']<0.5]
            if task in ['BBBP']:
                data_i = data_i[data_i[f'src pred {task}']<0.5]
                data_i_succeed = data_i[data_i[f'opt smi {i} pred {task}']>=0.5]
            data_t_all_num = data_t_all_num + len(data)
            data_t_succeed = data_t_succeed + len(data_i_succeed)
        print(f'{task} {model} {round(data_t_succeed/data_t_all_num, 3)}, {data_t_all_num/5}')

