
# origin smi,src smi,opt smi,sim_score,opt_drd2_score,opt_qed_score,opt_drd2_qed_score,iteration,src_drd2_score,src_qed_score,task_name
# CCN1CCN(C(=O)Cn2ncc3c4ccccc4n(Cc4ccccc4F)c3c2=O)CC1,O=C(Cn1ncc2ccccc2c1=O)N1CCN(CCCc2ccccc2)CC1,O=C(Cn1ncc2ccccc2c1=O)N1CCN(CCCc2ccc(F)cc2)CC1,0.4342105263157895,0.8358744520657256,0.6281919276231378,1.464066379688863,3,0.02135461948932419,0.4720452921055482,drd2 qed

src_smi = 'N#CC(Cc1cc2c(=O)[nH]c3c(F)c(F)ccc3c2cc1F)NC(=O)C1NC2CCC1C2'
opt_smi = 'N#CC(Cc1cc2c(=O)[nH]c3c(F)c(F)ccc3c2cc1F)NC(=O)C1NC2CCC1C2'
from properties import similarity, get_prop

sim_socre = similarity(src_smi, opt_smi, sim_type='binary')
src_drd2 = get_prop(src_smi, prop='drd2')
src_qed = get_prop(src_smi, prop='qed')

opt_drd2 = get_prop(opt_smi, prop='drd2')
opt_qed = get_prop(opt_smi, prop='qed')

print(sim_socre, src_drd2, src_qed, opt_drd2, opt_qed)