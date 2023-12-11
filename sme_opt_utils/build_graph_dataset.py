from build_data import built_mol_graph_data_and_save
import argparse

task_list = ['ESOL', 'Mutagenicity', 'hERG', 'BBBP']
for task in task_list:
    input_csv = '../data/origin_data/' + task + '.csv'
    output_g_path = '../data/graph_data/' + task + '.bin'
    output_g_group_path = '../data/graph_data/' + task + '_group.csv'

    output_g_for_brics_path = '../data/graph_data/' + task + '_for_brics.bin'
    output_g_group_for_brics_path = '../data/graph_data/' + task + '_group_for_brics.csv'
    output_g_smask_for_brics_path = '../data/graph_data/' + task + '_smask_for_brics.npy'

    output_g_for_murcko_path = '../data/graph_data/' + task + '_for_murcko.bin'
    output_g_group_for_murcko_path = '../data/graph_data/' + task + '_group_for_murcko.csv'
    output_g_smask_for_murcko_path = '../data/graph_data/' + task + '_smask_for_murcko.npy'


    output_g_for_fg_path = '../data/graph_data/' + task + '_for_fg.bin'
    output_g_group_for_fg_path = '../data/graph_data/' + task + '_group_for_fg.csv'
    output_g_smask_for_fg_path = '../data/graph_data/' + task + '_smask_for_fg.npy'
    
    built_mol_graph_data_and_save(
        origin_data_path=input_csv,
        labels_name=task,
        save_g_path=output_g_path,
        save_g_group_path=output_g_group_path,
    
        save_g_for_brics_path=output_g_for_brics_path,
        save_g_smask_for_brics_path=output_g_smask_for_brics_path,
        save_g_group_for_brics_path=output_g_group_for_brics_path,
    
        save_g_for_murcko_path=output_g_for_murcko_path,
        save_g_smask_for_murcko_path=output_g_smask_for_murcko_path,
        save_g_group_for_murcko_path=output_g_group_for_murcko_path,

        save_g_for_fg_path=output_g_for_fg_path,
        save_g_smask_for_fg_path=output_g_smask_for_fg_path,
        save_g_group_for_fg_path=output_g_group_for_fg_path
    )
