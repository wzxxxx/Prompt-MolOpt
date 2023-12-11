from build_data import built_mol_graph_data_and_save
import argparse

property_name_list = ['basic_physicochemical', 'absorption', 'metabolism', 'toxicity', 'tox21', 'Caco-2', 'T12', 'CL', 'mol_opt']
specific_name_list = [['LogS', 'LogD', 'LogP'], ['Pgp-inh', 'Pgp-sub', 'HIA', 'F(20%)', 'F(30%)'], ['CYP1A2-inh', 'CYP1A2-sub', 'CYP2C19-inh', 'CYP2C19-sub', 'CYP2C9-inh', 'CYP2C9-sub', 'CYP2D6-inh',
                       'CYP2D6-sub', 'CYP3A4-inh', 'CYP3A4-sub'], ['hERG', 'H-HT', 'DILI', 'Ames', 'ROA', 'FDAMDD', 'SkinSen', 'Carcinogenicity', 'EC', 'EI',
                       'Respiratory', 'BCF', 'IGC50', 'LC50', 'LC50DM'], ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5',
                       'SR-HSE', 'SR-MMP', 'SR-p53'], ['Caco-2'], ['T12'], ['CL'], ['Mutagenicity', 'hERG', 'BBBP', 'ESOL', 'lipop']]

property_name_list = ['mol_opt_relabel']
specific_name_list = [['Mutagenicity', 'hERG', 'BBBP', 'ESOL', 'lipop']]

property_name_list = ['fs_d_comparison_mol_opt_relabel']
specific_name_list = [['drd2', 'qed']]

for i, property_name in enumerate(property_name_list):
    print(property_name)
    input_csv = '../data/origin_data/' + property_name + '.csv'
    output_g_path = '../data/graph_data/' + property_name + '.bin'
    output_g_group_path = '../data/graph_data/' + property_name + '_group.csv'

    output_g_for_brics_path = '../data/graph_data/' + property_name + '_for_brics.bin'

    output_g_group_for_brics_path = '../data/graph_data/' + property_name + '_group_for_brics.csv'
    output_g_smask_for_brics_path = '../data/graph_data/' + property_name + '_smask_for_brics.npy'

    output_g_for_brics_emerge_path = '../data/graph_data/' + property_name + '_for_brics_emerge.bin'
    output_g_group_for_brics_emerge_path = '../data/graph_data/' + property_name + '_group_for_brics_emerge.csv'
    output_g_smask_for_brics_emerge_path = '../data/graph_data/' + property_name + '_smask_for_brics_emerge.npy'

    output_g_for_murcko_path = '../data/graph_data/' + property_name + '_for_murcko.bin'
    output_g_group_for_murcko_path = '../data/graph_data/' + property_name + '_group_for_murcko.csv'
    output_g_smask_for_murcko_path = '../data/graph_data/' + property_name + '_smask_for_murcko.npy'

    output_g_for_murcko_emerge_path = '../data/graph_data/' + property_name + '_for_murcko_emerge.bin'
    output_g_group_for_murcko_emerge_path = '../data/graph_data/' + property_name + '_group_for_murcko_emerge.csv'
    output_g_smask_for_murcko_emerge_path = '../data/graph_data/' + property_name + '_smask_for_murcko_emerge.npy'

    output_g_for_fg_path = '../data/graph_data/' + property_name + '_for_fg.bin'
    output_g_group_for_fg_path = '../data/graph_data/' + property_name + '_group_for_fg.csv'
    output_g_smask_for_fg_path = '../data/graph_data/' + property_name + '_smask_for_fg.npy'

    specific_name = specific_name_list[i]

    built_mol_graph_data_and_save(origin_data_path=input_csv,
                                  labels_list=specific_name,
                                  save_g_path=output_g_path,
                                  save_g_group_path=output_g_group_path, save_g_for_brics_path=output_g_for_brics_path,
                                  save_g_smask_for_brics_path=output_g_smask_for_brics_path,
                                  save_g_group_for_brics_path=output_g_group_for_brics_path,
                                  save_g_for_brics_emerge_path=output_g_for_brics_emerge_path,
                                  save_g_smask_for_brics_emerge_path=output_g_smask_for_brics_emerge_path,
                                  save_g_group_for_brics_emerge_path=output_g_group_for_brics_emerge_path,
                                  save_g_for_murcko_path=output_g_for_murcko_path,
                                  save_g_smask_for_murcko_path=output_g_smask_for_murcko_path,
                                  save_g_group_for_murcko_path=output_g_group_for_murcko_path,
                                  save_g_for_murcko_emerge_path=output_g_for_murcko_emerge_path,
                                  save_g_smask_for_murcko_emerge_path=output_g_smask_for_murcko_emerge_path,
                                  save_g_group_for_murcko_emerge_path=output_g_group_for_murcko_emerge_path,
                                  save_g_for_fg_path=output_g_for_fg_path,
                                  save_g_smask_for_fg_path=output_g_smask_for_fg_path,
                                  save_g_group_for_fg_path=output_g_group_for_fg_path,
                                  fg=False,
                                  brics=False,
                                  murcko=False,
                                  combination=False
                                  )
