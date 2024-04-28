# first used the code s2s_opt_mol_all_pred.py in sme_opt_utils to generate the prediction of the opt mols
import pandas as pd

task_names = ['Prompt_MolOpt_ADMET', 'Prompt_MolOpt_fs_ADMET',]

test_set = 'test_2'
origin_task = 'Mutagenicity'

# zero shot
for add_tag_task in ['ESOL', 'lipop']:
    task_name = 'Prompt_MolOpt_ADMET'
    data_t_all_num = 0
    data_t_succeed = 0
    for i in range(1, 6):
        data_d = pd.read_csv(f'{task_name}_result_all_pred.csv')
        data_d = data_d[data_d['re_group'] == test_set]
        data_d.drop_duplicates(subset=['src smi', 'task name'], inplace=True)

        if add_tag_task in ['ESOL', 'lipop'] and task_name=='Prompt_MolOpt_ADMET':
            data_zero_shot = data_d.copy()
            data_zero_shot = data_zero_shot[data_zero_shot['task name']==origin_task]
            # print(f'*************************Top {task_name} {i} opt result********************')
            data_d = data_d[data_d['task name'] == f'{origin_task} {add_tag_task}']
            data_d = data_d[data_d['re_group']==test_set]
            data_d_smiles = data_d['src smi'].tolist()
            data_d_i = data_zero_shot[data_zero_shot['src smi'].isin(data_d_smiles)]
            data_d_i = data_d_i[(data_d_i[f'opt smi {i}'] != 'CC')]
            len_data_d = len(data_d)
            if add_tag_task == '':
                len_data_d = 10 ** 10
            if origin_task in ['Mutagenicity', 'hERG']:
                data_d_i_succeed = data_d_i[data_d_i[f'opt smi {i} pred {origin_task}'] < 0.5]
            else:
                data_d_i_succeed = data_d_i[data_d_i[f'opt smi {i} pred {origin_task}'] >= 0.5]
            add_data_d_i_succeed = data_d_i_succeed[data_d_i_succeed[f'src pred {add_tag_task}']-data_d_i_succeed[f'opt smi {i} pred {add_tag_task}']<=-0.5]
        data_t_all_num = data_t_all_num + len(data_d)
        data_t_succeed = data_t_succeed + len(add_data_d_i_succeed)
    print(f'Single Prompt token, {task_name} {origin_task} {add_tag_task} {round(data_t_succeed/data_t_all_num, 3)} {data_t_all_num/5}')


for add_tag_task in ['ESOL', 'lipop']:
    for task_name in task_names:
        data_t_all_num = 0
        data_t_succeed = 0
        for i in range(1, 6):
            data_d = pd.read_csv(f'{task_name}_result_all_pred.csv')
            data_d = data_d[data_d['re_group'] == test_set]
            data_d.drop_duplicates(subset=['src smi', 'task name'], inplace=True)

            if add_tag_task in ['ESOL', 'lipop']:
                # print(f'*************************Top {task_name} {i} opt result********************')
                data_d = data_d[data_d['task name'] == f'{origin_task} {add_tag_task}']
                data_d = data_d[data_d['re_group']==test_set]
                data_d_i = data_d[(data_d[f'opt smi {i}'] != 'CC')]
                len_data_d = len(data_d)
                if add_tag_task == '':
                    len_data_d = 10 ** 10
                if origin_task in ['Mutagenicity', 'hERG']:
                    data_d_i_succeed = data_d_i[data_d_i[f'opt smi {i} pred {origin_task}'] < 0.5]
                else:
                    data_d_i_succeed = data_d_i[data_d_i[f'opt smi {i} pred {origin_task}'] >= 0.5]
                add_data_d_i_succeed = data_d_i[data_d_i[f'src pred {add_tag_task}']-data_d_i[f'opt smi {i} pred {add_tag_task}']<=-0.5]
                add_data_d_i_succeed = data_d_i_succeed[data_d_i_succeed[f'src pred {add_tag_task}']-data_d_i_succeed[f'opt smi {i} pred {add_tag_task}']<=-0.5]
            data_t_all_num = data_t_all_num + len(data_d)
            data_t_succeed = data_t_succeed + len(add_data_d_i_succeed)
        print(f'{task_name} {origin_task} {add_tag_task} {round(data_t_succeed/data_t_all_num, 3)} {data_t_all_num/5}')