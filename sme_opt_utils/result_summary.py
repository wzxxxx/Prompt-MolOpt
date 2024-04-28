import pandas as pd
from rdkit import Chem


def return_atom_num(sub_smi):
    try:
        sub_mol_atom_num = Chem.MolFromSmarts(sub_smi).GetNumAtoms()
        vir_atom_num = str(sub_smi).count("*")
        return sub_mol_atom_num-vir_atom_num
    except:
        return 0.00001


# data_name = 'drugbank'
# model_name = 'ESOL'
sub_list = ['fg', 'brics', 'murcko']
sub_list = ['mol']
group = 'test'

for model_name in ['hERG']:
    data_name = 'drug'
    all_data_sub = pd.DataFrame()
    for sub in sub_list:
        summary_data = pd.DataFrame()
        for i in range(10):
            data_i = pd.read_csv('../prediction/{}/{}_{}_{}_{}_{}_prediction.csv'.format(sub, data_name, model_name, sub, i+1, group))
            mol_data_i = pd.read_csv('../prediction/{}/{}_{}_{}_{}_{}_prediction.csv'.format('mol', data_name, model_name, 'mol', i+1, group))
            data_i = data_i.rename(columns={'pred':'pred_{}'.format(i+1)})
            new_order = ['smiles', 'label', 'sub_smi', 'pred_{}'.format(i+1)]
            data_i = data_i.reindex(columns=new_order)
            mol_pred_i_list = []
            for j in range(len(data_i)):
                if j%10000 == 0:
                    print('Model {}, Sub {}, seed {}, {}'.format(model_name, sub, i+1, j))
                smiles_i = data_i['smiles'].tolist()[j]
                mol_pred_i = mol_data_i[mol_data_i['smiles']==smiles_i].pred.tolist()[0]
                mol_pred_i_list.append(mol_pred_i)
            data_i['mol_pred_{}'.format(i+1)] = mol_pred_i_list
            summary_data = pd.concat([summary_data, data_i], axis=1)
        summary_data = summary_data.loc[:, ~summary_data.columns.duplicated()]
        for i in range(10):
            summary_data['attri_{}'.format(i+1)] = summary_data['mol_pred_{}'.format(i+1)] - summary_data['pred_{}'.format(i+1)]
        mol_pred_mean = summary_data[['mol_pred_{}'.format(i+1) for i in range(10)]].mean(axis=1)
        mol_pred_std = summary_data[['mol_pred_{}'.format(i+1) for i in range(10)]].std(axis=1)
        attri_mean = summary_data[['attri_{}'.format(i+1) for i in range(10)]].mean(axis=1)
        attri_std = summary_data[['attri_{}'.format(i+1) for i in range(10)]].std(axis=1)
        sub_connect_num = [str(sub_smi).count("*") for sub_smi in summary_data.sub_smi.tolist()]
        summary_data['sub_connect_num'] = sub_connect_num

        # 计算每个重原子的平均权重贡献值
        sub_smi_list = summary_data.sub_smi.tolist()
        smi_he_atom_num = [return_atom_num(smi) for smi in sub_smi_list]
        attri_per_he_atom = [attri_mean[i]/smi_atom_num for i, smi_atom_num in enumerate(smi_he_atom_num)]
        summary_data['attri_per_atom'] = attri_per_he_atom
        summary_data['mol_pred_mean'] = mol_pred_mean
        summary_data['mol_pred_std'] = mol_pred_std
        summary_data['attri_mean'] = attri_mean
        summary_data['attri_std'] = attri_std
        summary_data['sub_atom_num'] = smi_he_atom_num
        summary_data = summary_data[summary_data['sub_atom_num']>=1]
        summary_data = summary_data[summary_data['sub_connect_num']>=1]
        summary_data = summary_data[summary_data['sub_smi']!='NaN']
        summary_data = summary_data[abs(summary_data['attri_mean'])>abs(summary_data['attri_std'])]
        summary_data.to_csv('../prediction/summary/{}_{}_{}_{}_summary_prediction.csv'.format(data_name, model_name, sub, group), index=False)
        all_data_sub = pd.concat([all_data_sub, summary_data], axis=0)
    all_data_sub.to_csv('../prediction/summary/{}_{}_sub.csv'.format(data_name, model_name), index=False)