# first used the code s2s_opt_mol_all_pred.py in sme_opt_utils to generate the prediction of the opt mols
import pandas as pd

task_names = ['Prompt_MolOpt_no_embed_ADMET', 'Prompt_MolOpt_ADMET']
task_list = [['hERG', 'ESOL'],['Mutagenicity', 'BBBP'], ['hERG', 'Mutagenicity'], ['Mutagenicity', 'ESOL'], ['Mutagenicity', 'lipop']]


for task in task_list:
    origin_task = task[0]
    add_tag_task = task[1]

    for task_name in task_names:
        data_t_all_num = 0
        data_t_succeed = 0
        for i in range(1, 6):
            data_d = pd.read_csv(f'{task_name}_result_all_pred.csv')
            data_d.drop_duplicates(subset=['src smi', 'task name'], inplace=True)
            if add_tag_task in ['BBBP']:
                data_d = data_d[data_d['task name'] == f'{origin_task} {add_tag_task}']
                data_d = data_d[data_d[f'src pred {add_tag_task}']<0.5]
                data_d = data_d[data_d[f'src pred {origin_task}']>0.5]
                data_d_i = data_d[(data_d[f'opt smi {i}'] != 'CC')]
                len_data_d = len(data_d)
                if add_tag_task == '':
                    len_data_d = 10 ** 10
                data_d_i_succeed = data_d_i[data_d_i[f'opt smi {i} pred {origin_task}'] < 0.5]
                data_d_i = data_d_i[data_d_i[f'src pred {add_tag_task}']<0.5]
                # print(len(data), len(data_d))
                add_data_d_i_succeed = data_d_i_succeed[data_d_i_succeed[f'opt smi {i} pred {add_tag_task}']>=0.5]
            if add_tag_task in ['Mutagenicity', 'hERG']:
                # print(f'*************************Top {task_name} {i} opt result********************')
                data_d = data_d[data_d['task name'] == f'{origin_task} {add_tag_task}']
                # print(len(data), len(data_d))
                data_d = data_d[data_d[f'src pred {add_tag_task}']>0.5]
                data_d = data_d[data_d[f'src pred {origin_task}']>0.5]
                data_d_i = data_d[(data_d[f'opt smi {i}'] != 'CC')]
                len_data_d = len(data_d)
                if add_tag_task == '':
                    len_data_d = 10 ** 10
                data_d_i_succeed = data_d_i[data_d_i[f'opt smi {i} pred {origin_task}'] < 0.5]
                data_d_i = data_d_i[data_d_i[f'src pred {add_tag_task}']>0.5]
                add_data_d_i_succeed = data_d_i_succeed[data_d_i_succeed[f'opt smi {i} pred {add_tag_task}']<0.5]

            if add_tag_task in ['ESOL', 'lipop']:
                data_d = data_d[data_d['task name'] == f'{origin_task} {add_tag_task}']
                data_d = data_d[data_d[f'src pred {origin_task}']>0.5]
                data_d = data_d[data_d[f'src pred {origin_task}']>0.5]
                data_d_i = data_d[(data_d[f'opt smi {i}'] != 'CC')]
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
        print(f'{task_name} {origin_task} {add_tag_task} {round(data_t_succeed/data_t_all_num, 3)} {data_t_all_num/5}')
