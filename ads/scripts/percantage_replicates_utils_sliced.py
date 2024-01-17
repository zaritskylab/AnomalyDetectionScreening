import random
from collections import defaultdict
from statistics import median
import numpy as np
from tqdm import tqdm
import sys
import os

currentdir = '/sise/home/alonshp/AnomalyDetectionScreening'
print(currentdir)

# currentdir = os.path.dirname('home/alonshp/AnomalyDetectionScreeningLocal/')
# print(currentdir)
sys.path.insert(0, os.getcwd())
sys.path.insert(0, currentdir)

from utils.general import revise_exp_name, set_configs, set_paths, add_exp_suffix
from utils.global_variables import DS_INFO_DICT
from utils.data_utils import set_index_fields
from utils.readProfiles import get_cp_path, get_cp_dir
from utils.reproduce_funcs import get_median_correlation, get_duplicate_replicates, get_replicates_score
from utils.eval_utils import load_zscores
from dataset_paper_repo.utils.normalize_funcs import standardize_per_catX



if __name__ == '__main__':

    import sys
    import pickle
    import os
    import pandas as pd
    import numpy as np
    import glob
    from tqdm import tqdm
    import random
    from collections import defaultdict
    from statistics import median

    # new_exp = False

    # if new_exp:
    #     today = date.today()
    #     date = today.strftime("%d_%m")
    #     date
    # else:
    #     date = '29_05'

    # ds_info_dict={'CDRP':['CDRP-BBBC047-Bray',['Metadata_Sample_Dose','pert_sample_dose']],
    #           'CDRP-bio':['CDRPBIO-BBBC036-Bray',['Metadata_Sample_Dose','pert_sample_dose']],
    #           'TAORF':['TA-ORF-BBBC037-Rohban',['Metadata_broad_sample','pert_id',]],
    #           'LUAD':['LUAD-BBBC041-Caicedo',['x_mutation_status','allele']],
    #           'LINCS':['LINCS-Pilot1',['Metadata_pert_id_dose','pert_id_dose']]}
    
    # configs = set_configs()
    # if len(sys.argv) <2:
        # exp_name = 'report'
        # configs.general.exp_name = exp_name
        # configs = set_configs(exp_name)
        # configs.general.dataset = 'LINCS'
    # else:
        # exp_name = sys.argv[1]
        # print(configs.general.exp_name)
    
    configs = set_configs()
    if len(sys.argv) <2:
        exp_name = 'report_911_t'
        configs.general.exp_name = exp_name
        configs.general.dataset = 'LINCS'
        # configs.general.dataset = 'CDRP-bio'

        configs.data.profile_type = 'normalized_variable_selected'
        configs.data.profile_type = 'augmented'
        configs.data.modality = 'CellPainting'
        configs.eval.by_dose = True
        run_parallel = True

        if run_parallel:
            slice_id = 0
        # configs = set_configs(exp_name)

    else:
        # exp_name = sys.argv[1]
        # configs = set_configs()
        profileType = configs.data.profile_type
        dataset = configs.general.dataset
        slice_id = configs.eval.slice_id
        exp_name = configs.general.exp_name
        modality = configs.data.modality
        by_dose = configs.eval.by_dose

    configs.general.flow = 'pr_sliced'
    configs.general.exp_num= slice_id
    configs = set_paths(configs)

    slice_id = int(slice_id)
    ################################################
    # dataset options: 'CDRP' , 'LUAD', 'TAORF', 'LINCS', 'CDRP-bio'
    # datasets=['LUAD','TAORF','LINCS','CDRP-bio'];
    # datasets=['LINCS', 'CDRP-bio','CDRP'];
    # datasets=['TAORF','LUAD','LINCS', 'CDRP-bio']
    datasets=['CDRP-bio']
    datasets = [configs.general.dataset]
    # DT_kfold={'LUAD':10, 'TAORF':5, 'LINCS':25, 'CDRP-bio':6,'CDRP':40}


    # from IPython.display import clear_output
    ################################################
    # CP Profile Type options: 'augmented', 'augmented_after_fs' , 'normalized', 'normalized_variable_selected'
    # profileType='normalized_variable_selected'
    # profileType ='normalized_variable_selected'

    base_dir= '/sise/assafzar-group/assafzar/genesAndMorph'
    # data_dir = get_cp_dir(configs)
    # data_dir=base_dir+'/preprocessed_data/'+ds_info_dict[datasets[0]][0]+'/'
    z_trim = 8
    # exp_name= 'ae_12_09_fs'


    output_dir = configs.general.output_exp_dir
    # data_dir = f'{base_dir}/preprocessed_data/{ds_info_dict[datasets[0]][0]}'
    null_base_dir = f'{base_dir}/results/{datasets[0]}/'
    save_base_dir = configs.general.res_dir
    exp_save_dir = configs.general.res_dir

    if configs.data.modality == 'CellPainting':
        modality_str = 'cp'
    else:
        modality_str = 'l1k'
    
    # include profile type in list if it is a file in folder 'output_dir'
    profile_types = [p for p in ['augmented','normalized_variable_selected'] if any(p in string for string in os.listdir(output_dir))]
    # profile_types = [p for p in ['augmented','normalized_variable_selected'] if any(p in string for string in os.listdir(output_dir))]


    new_ss = False
    ss_calc = 'median'
    num_rand = 1000
    by_doses = [True,False]

    if configs.eval.run_dose_if_exists and DS_INFO_DICT[configs.general.dataset]['has_dose']:
        by_doses = [False,True]
    else:
        by_doses = [False]
    normalize_by_alls = [True,False]

    for n in normalize_by_alls:
        configs.eval.normalize_by_all = n
        for d in by_doses:
            for p in profile_types:
                data_dir =  get_cp_dir(base_dir,datasets[0],p)

                exp_suffix = add_exp_suffix(p,d,configs.eval.normalize_by_all)
                exp_save_dir = f'{save_base_dir}/{exp_suffix}'
                if d:
                    null_dist_path = f'{null_base_dir}/null_distribution_replicates_{num_rand}_d.pkl'
                else:
                    null_dist_path = f'{null_base_dir}/null_distribution_replicates_{num_rand}.pkl'


                print(f'calculating for profile type:{p}')
                data_dir =  get_cp_dir(base_dir,configs.general.dataset,p)

                methods = {
                    # '4to1':{'path':f'/storage/users/g-and-n/tabular_models_results/41/ALL/zscores'},
                    # '2to1':{'path':f'/storage/users/g-and-n/tabular_models_results/21/ALL/zscores'},
                    # 'anomaly':{'name':'anomaly','path':os.path.join(output_dir,f'replicate_level_{modality_str}_{p}_ae.csv')},
                    'anomaly_err':{'name':'anomaly_err','path':os.path.join(output_dir,f'replicate_level_{modality_str}_{p}_ae_diff.csv')},
                    # 'anomaly_emb':{'name':'anomaly_emb','path':os.path.join(output_dir,f'replicate_level_{modality_str}_{p}_ae_embeddings.csv')},
                    'raw':{'name':'raw','path': os.path.join(output_dir,f'replicate_level_{modality_str}_{p}_baseline.csv')},
                    # 'raw_unchanged':{'name':'raw_unchanged','path': os.path.join(data_dir,f'replicate_level_cp_{p}.csv.gz')}
                    # '5to5':{'path':f'/storage/users/g-and-n/tabular_models_results/55/ALL/zscores'},
                }

                os.makedirs(save_base_dir,exist_ok=True)
                os.makedirs(exp_save_dir,exist_ok=True)
                print(f'loading from path {output_dir}')


                # ####################### loading zscores #########################
                # for m in methods.keys():
                #     try:
                #         print(f'loading zscores for method: {m}')
                #     except Exception as e:
                #         print(f'failed to load zscores for method {m}')
                #         print(e)
                #         # methods[m]['zscores'] = None
                #         methods.pop(m)
                #         continue
                methods = load_zscores(methods,base_dir,configs.general.dataset,p,by_dose=d,normalize_by_all =n,z_trim=configs.eval.z_trim)

                cpds_med_score = {}

                
                for m in methods.keys():    
                    cpds_med_score[m] = get_replicates_score(methods[m]['zscores'],methods[m]['features'])
                cpds_score_df_trt = pd.DataFrame({k[:]: v for k, v in cpds_med_score.items()})


                # ####################### creating null distribution replicates #########################
                # if not os.path.exists(null_dist_path):

                #     replicates_df, cpds = get_duplicate_replicates(methods[m]['zscores'],min_num_reps=4)
                #     null_distribution_replicates = get_null_distribution_replicates(replicates_df, cpds, rand_num=num_rand)
                    
                #     # null_distribution_replicates.to_csv(f'null_distribution_replicates_{num_plates}.csv')
                #     with open(null_dist_path, 'wb') as handle:
                #         pickle.dump(null_distribution_replicates, handle, protocol=pickle.HIGHEST_PROTOCOL)

                # else:
                try:
                    with open(null_dist_path, 'rb') as f:
                        null_distribution_replicates = pickle.load(f)
                        
                    len(null_distribution_replicates)


                    ####################### calculating null distribution medians #########################
                    # if run_parallel:
                
                    null_distribution_medians = {}

                    all_cpds = list(null_distribution_replicates.keys())
                    print(len(all_cpds))
                    all_cpds.sort()
                    slice_size = int(len(all_cpds)/90)
                    print(slice_id)
                    # print(slice_size)
                    start = slice_id * slice_size
                    end = min(len(all_cpds),(slice_id + 1) * slice_size)

                    cpds = all_cpds[start:end]

                    # if len(cpds[0].split('-'))>4:
                    #     cpds = [f"{c.split('-')[0]}-{c.split('-')[1]}" for c in cpds]
                    # print(f'calculating cpds index from from {start} to {end}')

                    if len(cpds) > 0:
                        for m in methods.keys():
                            
                            save_dir = exp_save_dir

                            null_distribution_medians_path = f'{save_dir}/null_distribution_medians_{num_rand}_{m}.pkl'
                            
                            if not os.path.exists(null_distribution_medians_path):
                                print(f'path {null_distribution_medians_path} doesn\'t exist, calculating null_distribution_medians for {m}')

                                print(len(cpds), m)
                                cur_dest = os.path.join(save_dir, f'{m}')
                                os.makedirs(cur_dest, exist_ok=True)
                                cur_dest = os.path.join(cur_dest, f'{slice_id}.pickle')
                                res = {}

                                for cpd in cpds:
                                    
                                    print(cpd)

                                    null_dist = null_distribution_replicates[cpd]
                                    null_dist_scores = []

                                    for null_idxs in null_dist:
                                        cur_null = methods[m]['zscores'][methods[m]['zscores'].index.isin(null_idxs,2)].copy()
                                        curr_null_score = get_median_correlation(cur_null)
                                        null_dist_scores.append(curr_null_score)
                                        del cur_null
                                    # res[cpd] = (cpd_med_score, null_dist_scores)
                                    res[cpd] = null_dist_scores
                                    print(null_dist_scores)


                                print(f'saving results for slice {slice_id} to {cur_dest}')
                                with open(cur_dest, 'wb') as handle:
                                    pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)

                except:
                    print(f'null replictates not found for {p}, dose={d}')

