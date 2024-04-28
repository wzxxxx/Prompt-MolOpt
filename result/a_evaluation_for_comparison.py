import pandas as pd
task = 'fs_comparison'
for task in ['comparison', 'fs_comparison']:
    for iteration in [1, 5]:
        for cutoff_sim in [0.4, 0.5, 0.6]:
            data = pd.read_csv(f'5_{cutoff_sim}_{task}_result.csv')
            smis = list(set(data['origin smi'].tolist()))
            if iteration==1:
                data = data[data['iteration']==1]
            data = data[data['opt smi']!='CC']
            succeed_num = 0

            best_result = pd.DataFrame()
            for smi in smis:
                data_i = data[(data['origin smi']==smi)&(data['sim_score']>=cutoff_sim)]
                data_s = data_i[(data_i['opt_drd2_score']>=0.5)&(data_i['opt_qed_score']>=0.6)]
                data_s.sort_values(by='opt_drd2_qed_score', ascending=False)
                if len(data_s)>0:
                    succeed_num = succeed_num+1
                    best_result = pd.concat([best_result, data_s[:1]], axis=0)
            print(f'Task: {task}, Iteration number: {iteration}, cutoff_sim: {cutoff_sim}, success rate: {round(succeed_num/800, 4)}')