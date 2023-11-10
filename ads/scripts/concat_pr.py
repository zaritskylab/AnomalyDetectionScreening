import os
import pickle
import sys
import shutil
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
import random
from collections import defaultdict
from statistics import median

currentdir = '/sise/home/alonshp/AnomalyDetectionScreening'
code_dir = '/sise/home/alonshp/AnomalyDetectionScreening/ads'

print(currentdir)


# currentdir = os.path.dirname('home/alonshp/AnomalyDetectionScreeningLocal/')
# print(currentdir)
sys.path.insert(0, os.getcwd())
sys.path.insert(0, currentdir)
sys.path.insert(0, code_dir)
from ads.utils.readProfiles import get_cp_path, get_cp_dir
# from utils.percantage_replicates_utils import get_null_dist_median_scores, get_moa_p_vals, get_replicates_score
from utils.general import set_configs, set_paths, add_exp_suffix
from utils.global_variables import DS_INFO_DICT
from utils.data_utils import set_index_fields
from utils.eval_utils import load_zscores
from utils.reproduce_funcs import get_median_correlation, get_replicates_score, get_moa_p_vals
from utils.file_utils import load_dict_pickles_and_concatenate
from dataset_paper_repo.utils.replicateCorrs import replicateCorrs
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from dataset_paper_repo.utils.normalize_funcs import standardize_per_catX



# def set_index_fields(df, index_fields=None):

#     if index_fields is None:
#         index_fields = ['Metadata_Plate', 'Metadata_ASSAY_WELL_ROLE', 'Metadata_broad_sample', 'Metadata_Well',
#                         'Metadata_mmoles_per_liter']
#     df = df.set_index(
#         index_fields)
#     return df




if __name__ == "__main__":


    # ds_info_dict={'CDRP':['CDRP-BBBC047-Bray',['Metadata_Sample_Dose','pert_sample_dose']],
    #         'CDRP-bio':['CDRPBIO-BBBC036-Bray',['Metadata_Sample_Dose','pert_sample_dose']],
    #         'TAORF':['TA-ORF-BBBC037-Rohban',['Metadata_broad_sample','pert_id',]],
    #         'LUAD':['LUAD-BBBC041-Caicedo',['x_mutation_status','allele']],
    #         'LINCS':['LINCS-Pilot1',['Metadata_pert_id_dose','pert_id_dose']]}

    configs = set_configs()
    if len(sys.argv) <2:
        exp_name = 'report_911_t'
        configs.general.exp_name = exp_name
        configs.general.dataset = 'LINCS'
        # configs.general.dataset = 'CDRP-bio'

        # configs.data.profile_type = 'normalized_variable_selected'
        # configs.data.profile_type = 'augmented'
        # configs.data.modality = 'CellPainting'
        # configs.eval.by_dose = False
    # else:
        
        # configs = set_configs()
    # else:
        # exp_name = sys.argv[1]
        # configs = set_configs()
        # print(configs.general.exp_name)
    
    
    configs = set_paths(configs)

    ################################################
    # dataset options: 'CDRP' , 'LUAD', 'TAORF', 'LINCS', 'CDRP-bio'
    # datasets=['LUAD','TAORF','LINCS','CDRP-bio'];
    # datasets=['LINCS', 'CDRP-bio','CDRP'];
    # datasets=['TAORF','LUAD','LINCS', 'CDRP-bio']
    # datasets=['CDRP-bio']
    datasets = [configs.general.dataset]
    
    # DT_kfold={'LUAD':10, 'TAORF':5, 'LINCS':25, 'CDRP-bio':6,'CDRP':40}


    # from IPython.display import clear_output
    ################################################
    # CP Profile Type options: 'augmented', 'augmented_after_fs' , 'normalized', 'normalized_variable_selected'
    # profileType='normalized_variable_selected'
    # profileType ='normalized_variable_selected'

    base_dir= '/sise/assafzar-group/assafzar/genesAndMorph'
    data_dir = get_cp_dir(base_dir, configs.general.dataset, configs.data.profile_type)
    # data_dir=base_dir+'/preprocessed_data/'+ds_info_dict[datasets[0]][0]+'/'
    # exp_name= 'ae_12_09_fs'

    output_dir = configs.general.output_exp_dir
    null_base_dir = f'{base_dir}/results/{datasets[0]}/'
    save_base_dir = configs.general.res_dir
    exp_save_dir = f'{save_base_dir}'
    fig_dir = f'{save_base_dir}/figs'
    if configs.data.modality == 'CellPainting':
        modality_str = 'cp'
    else:
        modality_str = 'l1k'

    num_rand = 1000

    normalize_by_alls = [True,False]
    # include profile type in list if it is a file in folder 'output_dir'
    profile_types = [p for p in ['augmented','normalized_variable_selected'] if any(p in string for string in os.listdir(output_dir))]
    if configs.eval.run_dose_if_exists and DS_INFO_DICT[configs.general.dataset]['has_dose']:
        doses = [False,True]
    else:
        doses = [False]

    
    for d in doses:
        if d:
            null_dist_path = f'{null_base_dir}/null_distribution_replicates_{num_rand}_d.pkl'
        else:
            null_dist_path = f'{null_base_dir}/null_distribution_replicates_{num_rand}.pkl'
        for n in normalize_by_alls:
            # configs.eval.normalize_by_all = n
            for p in profile_types:
                exp_suffix = add_exp_suffix(p,d, normalize_by_all=n)
                save_pr_path =f'{exp_save_dir}/pr{exp_suffix}.csv'

                if not  os.path.exists(save_pr_path):
                # if p == 'augmented':
                    # continue
                # null_dist_path = f'/sise/assafzar-group/assafzar/genesAndMorph/results/{dataset}/CellPainting/{p}/null_distribution_replicates_1000.pkl'

                    print(f'calculating for profile type:{p}')
                    methods = {
                        # '4to1':{'path':f'/storage/users/g-and-n/tabular_models_results/41/ALL/zscores'},
                        # '2to1':{'path':f'/storage/users/g-and-n/tabular_models_results/21/ALL/zscores'},
                        # 'anomaly':{'name':'anomaly','path':os.path.join(output_dir,f'replicate_level_{modality_str}_{p}_ae.csv')},
                        'anomaly_err':{'name':'anomaly_err','path':os.path.join(output_dir,f'replicate_level_{modality_str}_{p}_ae_diff.csv')},
                        # 'anomaly_emb':{'name':'anomaly_emb','path':os.path.join(output_dir,f'replicate_level_{modality_str}_{p}_ae_embeddings.csv')},
                        'raw':{'name':'raw','path': os.path.join(output_dir,f'replicate_level_{modality_str}_{p}_baseline.csv')},
                        # 'raw_unchanged':{'name':'raw_unchanged','path': os.path.join(output_dir,f'replicate_level_{modality_str}_{p}_baseline_unchanged.csv')},
                        # 'raw':{'name':'raw_unchanged','path': os.path.join(data_dir,'CellPainting',f'replicate_level_cp_{p}.csv.gz')}
                        # '5to5':{'path':f'/storage/users/g-and-n/tabular_models_results/55/ALL/zscores'},
                    }

                    os.makedirs(save_base_dir,exist_ok=True)
                    os.makedirs(exp_save_dir,exist_ok=True)

                    # ####################### loading zscores #########################
                    # for m in methods.keys():
                    #     print(f'loading zscores for method: {m}')
                    #     if m == 'l1k':
                    #         methods[m]['modality'] ='L1000'
                    #         scaled_zscores, features = load_data(configs,modality=methods[m]['modality'], plate_normalize_by_all =1)
                    #         methods[m]['features']=list(features)

                    #         scaled_zscores = scaled_zscores.query(f"{DS_INFO_DICT[configs.general.dataset]['L1000']['role_col']} != '{DS_INFO_DICT[configs.general.dataset]['L1000']['mock_val']}' ")
                            
                    #         # methods[m]['zscores'] = zscores.loc[:,methods[m]['features']]
                    #     else:
                    #         methods[m]['modality']='CellPainting'
                    #         zscores = pd.read_csv(methods[m]['path'], compression = 'gzip')     
                    #         if m == 'anomaly_emb':
                    #             meta_features = [c for c in zscores.columns if 'Metadata_' in c]
                    #             features = [c for c in zscores.columns if 'Metadata_' not in c]
                    #         else:
                    #             features = zscores.columns[zscores.columns.str.contains("Cells_|Cytoplasm_|Nuclei_")]
                    #         methods[m]['features']=list(features)

                    #         if configs.eval.normalize_by_all:
                    #             scaled_zscores = standardize_per_catX(zscores,DS_INFO_DICT[configs.general.dataset]['CellPainting']['plate_col'],methods[m]['features'])
                    #     scaled_zscores = scaled_zscores.query(f"{DS_INFO_DICT[configs.general.dataset][methods[m]['modality']]['role_col']} != '{DS_INFO_DICT[configs.general.dataset][methods[m]['modality']]['mock_val']}' ")
                    #     # zscores = zscores.query('Metadata_ASSAY_WELL_ROLE == "treated"')
                    #     scaled_zscores = set_index_fields(scaled_zscores,configs.general.dataset,by_dose=d, modality=methods[m]['modality'])
                    #     if z_trim is not None:
                    #         scaled_zscores[features] = scaled_zscores[features].clip(-z_trim, z_trim)
                    #     methods[m]['zscores'] = scaled_zscores.loc[:,methods[m]['features']]
                    # methods = load_zscores(methods, configs, d)
                    methods = load_zscores(methods,base_dir,configs.general.dataset,p,by_dose=d,normalize_by_all =n,z_trim=configs.eval.z_trim)


                    ##################### calc median correlation per compound #####################
                    cpds_med_score = {}
                    for m in methods.keys():    
                        cpds_med_score[m] = get_replicates_score(methods[m]['zscores'],methods[m]['features'])
                    cpds_score_df_trt = pd.DataFrame({k[:]: v for k, v in cpds_med_score.items()})

                    
                    ##################### calc null distribution #####################
                    for m in methods.keys():
                        
                        save_dir = exp_save_dir

                        # methods[m]['null_distribution_medians'] = 
                        null_distribution_medians_dir = f'{save_dir}/{m}{exp_suffix}'
                        null_distribution_medians_path = f'{save_dir}/null_distribution_medians_{num_rand}_{m}{exp_suffix}.pkl'

                        if not os.path.exists(null_distribution_medians_path):
                            
                            print(f'concating null medians for: {m}, {p}, dose={d}')

                            # print(f'path {null_distribution_medians_path} doesn\'t exist, calculating null_distribution_medians for {m}')
                            null_dist_medians = load_dict_pickles_and_concatenate(null_distribution_medians_dir)

                            # shutil.rmtree(null_distribution_medians_dir, ignore_errors=False, onerror=None)

                            methods[m]['null_distribution_medians'] = null_dist_medians
                            with open(null_distribution_medians_path, 'wb') as handle:
                                pickle.dump(methods[m]['null_distribution_medians'], handle, protocol=pickle.HIGHEST_PROTOCOL)
                            
                        else:
                            print(f'loading null medians for: {m}, {p}, dose={d}')
                            with open(null_distribution_medians_path, 'rb') as f:
                                methods[m]['null_distribution_medians'] = pickle.load(f)
                        

                    ##################### calc p-values #####################
                    df_null_p_vals = pd.DataFrame([])

                    for m in methods.keys():
                        
                        methods[m]['null_p_vals']= get_moa_p_vals(methods[m]['null_distribution_medians'], cpds_score_df_trt, method=m)
                        method_df_null_p_vals = pd.DataFrame.from_dict(methods[m]['null_p_vals'],orient='index',columns=[m])
                        if len(df_null_p_vals)>0:
                            df_null_p_vals = df_null_p_vals.join(method_df_null_p_vals)
                        else:
                            df_null_p_vals = method_df_null_p_vals.copy()
                        
                        # cpds_score_df_trt = cpds_score_df_trt.join(df_null_p_vals)
                    
                    medians = cpds_score_df_trt[methods.keys()].melt(var_name=["method"],value_name='median_score',ignore_index=False).reset_index()
                    p_vals = df_null_p_vals[methods.keys()].melt(var_name=["method"],value_name='p_val',ignore_index=False).reset_index()
                    # cpds_score_df_trt.to_csv(f'cpds_score_df_trt_{num_plates}.csv')
                    df_null_p_vals.to_csv(f'{exp_save_dir}/df_null_p_vals{exp_suffix}.csv')
                    # a = pd.merge(medians,p_vals, left_on=['index','median_score'], right_on=['index','p_val'])
                    percantage_replicating_df = pd.merge(medians,p_vals, on=['index','method'])
                    percantage_replicating_df.to_csv(f'{exp_save_dir}/pr{exp_suffix}.csv') #, index=False, sep="\t")
                    print('completed running PR successfully!')
                else:
                    print(f'PR already exists for profile type:{p} and dose:{d}, normalize_by_all:{n}')
