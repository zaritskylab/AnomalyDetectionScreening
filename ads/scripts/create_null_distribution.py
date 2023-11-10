import random
from collections import defaultdict
from statistics import median
import numpy as np
from tqdm import tqdm
import sys
import os


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
import matplotlib.pyplot as plt
import seaborn as sns

currentdir = '/sise/home/alonshp/AnomalyDetectionScreening'
print(currentdir)

# currentdir = os.path.dirname('home/alonshp/AnomalyDetectionScreeningLocal/')
# print(currentdir)
sys.path.insert(0, os.getcwd())
sys.path.insert(0, currentdir)

from utils.general import revise_exp_name, set_configs, set_paths, add_exp_suffix, write_dataframe_to_excel
from utils.global_variables import DS_INFO_DICT
from utils.data_utils import set_index_fields, load_data
from utils.eval_utils import load_zscores
from utils.readProfiles import get_cp_path, get_cp_dir
from utils.reproduce_funcs import get_duplicate_replicates, get_null_distribution_replicates,get_replicates_score
from dataset_paper_repo.utils.replicateCorrs import replicateCorrs
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from dataset_paper_repo.utils.normalize_funcs import standardize_per_catX


def main(configs):

    base_dir= '/sise/assafzar-group/assafzar/genesAndMorph'
    # data_dir = get_cp_dir(configs)
    data_dir =  get_cp_dir(base_dir,configs.general.dataset,configs.data.profile_type)
    l1k_data_dir = get_cp_dir(base_dir,configs.general.dataset,configs.data.profile_type,modality='L1000')
    num_rand = 1000
    run_dist_plots = False
    # exp_name= 'ae_12_09_fs'
    

    output_dir = configs.general.output_exp_dir
    null_base_dir = f'{base_dir}/results/{configs.general.dataset}/'
    save_base_dir = configs.general.res_dir
    exp_save_dir = configs.general.res_dir
    fig_dir = configs.general.fig_dir
    debug = configs.general.debug_mode
    # debug = False
    l1k_data_dir = os.path.dirname(data_dir) + '/L1000'
    if configs.data.modality == 'CellPainting':
        modality_str = 'cp'
    else:
        modality_str = 'l1k'

    if configs.eval.run_dose_if_exists and DS_INFO_DICT[configs.general.dataset]['has_dose']:
        by_doses = [False,True]
    elif configs.eval.by_dose and DS_INFO_DICT[configs.general.dataset]['has_dose']:
        by_doses = [True]
    else:
        by_doses = [False]
        
    
    # include profile type in list if it is a file in folder 'output_dir'
    profile_types = [p for p in ['augmented','normalized_variable_selected'] if any(p in string for string in os.listdir(output_dir))]

    for d in by_doses:

        if d:
            null_dist_path = f'{null_base_dir}/null_distribution_replicates_{num_rand}_d.pkl'
        else:
            null_dist_path = f'{null_base_dir}/null_distribution_replicates_{num_rand}.pkl'

            
        for p in profile_types:

            exp_suffix = add_exp_suffix(p,d, configs.eval.normalize_by_all)

            configs.general.logger.info(f'calculating for profile type:{p}')
            methods = {
                # '4to1':{'path':f'/storage/users/g-and-n/tabular_models_results/41/ALL/zscores'},
                # '2to1':{'path':f'/storage/users/g-and-n/tabular_models_results/21/ALL/zscores'},
                # 'anomaly':{'name':'anomaly','path':os.path.join(output_dir,f'replicate_level_{modality_str}_{p}_ae.csv')},
                'l1k':{'name':'l1k','path': os.path.join(l1k_data_dir,f'replicate_level_l1k_{p}.csv.gz')},
                'anomalyCP':{'name':'anomaly_err','path':os.path.join(output_dir,f'replicate_level_{modality_str}_{p}_ae_diff.csv')},
                # 'anomaly_emb':{'name':'anomaly_emb','path':os.path.join(output_dir,f'replicate_level_{modality_str}_{p}_ae_embeddings.csv')},
                'CP':{'name':'raw','path': os.path.join(output_dir,f'replicate_level_{modality_str}_{p}_baseline.csv')},
                # 'raw_unchanged':{'name':'raw_unchanged','path': os.path.join(data_dir,f'replicate_level_cp_{p}.csv.gz')}
                # '5to5':{'path':f'/storage/users/g-and-n/tabular_models_results/55/ALL/zscores'},
            }

            os.makedirs(save_base_dir,exist_ok=True)
            os.makedirs(exp_save_dir,exist_ok=True)
            configs.general.logger.info(f'loading from path {output_dir}')

            methods = load_zscores(methods,base_dir,configs.general.dataset,p,by_dose=d,normalize_by_all =n,z_trim=configs.eval.z_trim)
            # ####################### loading zscores #########################
            # for m in methods.keys():
            #     print(f'loading zscores for method: {m}')
            #     if m == 'l1k':
            #         methods[m]['modality'] ='L1000'
            #         scaled_zscores, features = load_data(configs,modality=methods[m]['modality'], plate_normalize_by_all =True)
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

            #         scaled_zscores = standardize_per_catX(zscores,DS_INFO_DICT[configs.general.dataset]['CellPainting']['plate_col'],methods[m]['features'])
            #     scaled_zscores = scaled_zscores.query(f"{DS_INFO_DICT[configs.general.dataset][methods[m]['modality']]['role_col']} != '{DS_INFO_DICT[configs.general.dataset][methods[m]['modality']]['mock_val']}' ")
            #     # zscores = zscores.query('Metadata_ASSAY_WELL_ROLE == "treated"')
            #     scaled_zscores = set_index_fields(scaled_zscores,configs.general.dataset,by_dose=d, modality=methods[m]['modality'])
            #     if z_trim is not None:
            #         scaled_zscores[features] = scaled_zscores[features].clip(-z_trim, z_trim)
            #     methods[m]['zscores'] = scaled_zscores.loc[:,methods[m]['features']]

            #         # by_dose = False

            if d:
                # pertColName='Metadata_Sample_Dose'
                # pertColName = DS_INFO_DICT[configs.general.dataset]['CellPainting']['dose_col']
                col = 'dose_col'
                # l1k_pert_col_name = 'pert_id_dose'
            else:
                col = 'cpd_col'
                # pertColName=DS_INFO_DICT[configs.general.dataset]['CellPainting']['cpd_col']
                # l1k_pert_col_name = 'pert_id'



        # run rep correlation measurements
        # for p in profile_types:
            corr_path = f'{exp_save_dir}/RepCorrDF.xlsx'
            # os.makedirs(corr_dir,exist_ok=True)
            for m in methods.keys():   
                # scaler = MinMaxScaler()

                pertColName = DS_INFO_DICT[configs.general.dataset][methods[m]['modality']][col]
                # sclr = StandardScaler()
                # scaled_zscores = methods[m]['zscores'].copy()
                # scaled_zscores[methods[m]['features']] = sclr.fit_transform(scaled_zscores[methods[m]['features']])
                # scaled_zscores.groupby(['Metadata_Plate']).transform(lambda x: (x - x.mean()) / x.std())

                # scaler = MinMaxScaler(feature_range=(0, 1))
                # scaled_zscores[[methods[m]['features']] = sclr.fit_transform(methods[m]['zscores'][methods[m]['features']])
                configs.general.logger.info(f'calculating replicate scores for method: {m}') 
                # if not os.path.exists(corr_path):
                if debug:
                    [methods[m]['rand_corr'],methods[m]['rep_corr'],methods[m]['corr_df']] = replicateCorrs(methods[m]['zscores'][0:5000].reset_index(),pertColName,methods[m]['features'],plotEnabled=False)
                else:    
                    [methods[m]['rand_corr'],methods[m]['rep_corr'],methods[m]['corr_df']] = replicateCorrs(methods[m]['zscores'].reset_index(),pertColName,methods[m]['features'],plotEnabled=False)
                methods[m]['corr_df'] = methods[m]['corr_df'].dropna()
                sheetname = f'{m}-{configs.general.dataset.lower()}{exp_suffix}'
                # if configs.eval.by_dose:
                # sheetname += exp_suffix
                df_for_saving = methods[m]['corr_df'].reset_index().rename(columns={'index':'Unnamed: 0'})
                if not debug:
                    write_dataframe_to_excel(corr_path,sheetname,df_for_saving, append_new_data_if_sheet_exists=False)

            # repCorrFilePath =  f'{base_dir}/results/RepCorrDF.xlsx'
            # repCorDF=pd.read_excel(repCorrFilePath, sheet_name=None)
            # # cpRepDF=repCorDF['cp-'+dataset.lower()]
            # # cpHighList=cpRepDF[cpRepDF['RepCor']>cpRepDF['Rand90Perc']]['Unnamed: 0'].tolist()
            # # print('CP: from ',cpRepDF.shape[0],' to ',len(cpHighList))
            # l1kRepDF=repCorDF['l1k-'+configs.general.dataset.lower()]
            # l1kHighList=l1kRepDF[l1kRepDF['RepCor']>l1kRepDF['Rand90Perc']]['Unnamed: 0'].tolist()
            # print('L1K: from ',l1kRepDF.shape[0],' to ',len(l1kHighList))
            # sheetname = f'l1k-{configs.general.dataset.lower()}{exp_suffix}'
            # if not debug:
            #     write_dataframe_to_excel(corr_path,sheetname,l1kRepDF, append_new_data_if_sheet_exists=False)

            for m in methods.keys():

                # plot distribution of replicate and random correlations

                repC = methods[m]['rep_corr']
                randC_v2 = methods[m]['rand_corr']

                repC = [repC for repC in repC if str(repC) != 'nan']
                randC_v2 = [randC_v2 for randC_v2 in randC_v2 if str(randC_v2) != 'nan']  

                perc90=np.percentile(randC_v2, 90);
                perc80=np.percentile(randC_v2, 80);
                perc75=np.percentile(randC_v2, 75);
                rep10=np.percentile(repC, 10);
                    
                repCorrDf = methods[m]['corr_df']
                fig, axes = plt.subplots(figsize=(5,3))
                sns.kdeplot(methods[m]['rand_corr'], bw_method=.1, label="random pairs",ax=axes,color='darkgrey')
                sns.kdeplot(methods[m]['rep_corr'], bw_method=.1, label="replicate pairs",ax=axes,color='tab:green');axes.set_xlabel('CC');
                # sns.kdeplot(randC_v2, bw=.1, label="random pairs",ax=axes);axes.set_xlabel('CC');
                #         perc5=np.percentile(repCC, 50);axes.axvline(x=perc5,linestyle=':',color='darkorange');
                #         perc95=np.percentile(randCC, 90);axes.axvline(x=perc95,linestyle=':');
                axes.legend();#axes.set_title('');
                axes.set_xlim(-1.1,1.1)

                axes.axvline(x=perc90,linestyle=':',color = 'red', label='threshold');
                # axes.axvline(x=perc75,linestyle=':',color = 'r');

                axes.axvline(x=0,linestyle=':',color='k');
                axes.legend(loc=2);#axes.set_title('');
                axes.set_xlim(-1,1);

                repre90 = np.sum(repCorrDf['RepCor']>perc90)/len(repCorrDf)*100
                repre75 = np.sum(repCorrDf['RepCor']>perc75)/len(repCorrDf)*100
                # axes.text(0.1+perc75, 1.5,str(int(np.round(repre75,2)))+'%>.75', fontsize=12) #add text
                axes.text(0.1+perc90, 0.8,str(int(np.round(repre90,2)))+'%>t', fontsize=12) #add text

                savename = f'{m}-{configs.general.dataset}-{configs.general.exp_name}{exp_suffix}'
                
                plt.tight_layout() 
                if not debug:
                    plt.savefig(f'{fig_dir}/{savename}.png',dpi=300)
                plt.close()

                # plot distribution of replicate and random correlations
                
                if run_dist_plots:
                    a = methods[m]['zscores'][0:2000].to_numpy().flatten()
                    sns.histplot(a, bins=100)
                    plt.axvline(x=0.5,color='r')
                    plt.title(f'{m}_{p} zscores distribution')
                    # plt.xlim(-5,5)
                    # plt.show()

                    savename = f'{m}_{p}_zscores_dist{exp_suffix}'
                    # if configs.eval.by_dose:
                        # savename+='_d'
                    if not debug:
                        plt.savefig(f'{fig_dir}/{savename}.png',dpi=300)
                    plt.close()
            configs.general.logger.info('completed running RC successfully!')

            ####################### creating null distribution replicates #########################
            
            # for m in methods.keys():    
            #     cpds_med_score[m] = get_replicates_score(methods[m]['zscores'],methods[m]['features'])
            # cpds_score_df_trt = pd.DataFrame({k[:]: v for k, v in cpds_med_score.items()})

            if not os.path.exists(null_dist_path):

                replicates_df, cpds = get_duplicate_replicates(methods[m]['zscores'],min_num_reps=4)
                null_distribution_replicates = get_null_distribution_replicates(replicates_df, cpds, rand_num=num_rand)
                
                # null_distribution_replicates.to_csv(f'null_distribution_replicates_{num_plates}.csv')
                with open(null_dist_path, 'wb') as handle:
                    pickle.dump(null_distribution_replicates, handle, protocol=pickle.HIGHEST_PROTOCOL)

            else:
                
                with open(null_dist_path, 'rb') as f:
                    null_distribution_replicates = pickle.load(f)
                    
            len(null_distribution_replicates)



if __name__ == '__main__':

    # new_exp = False

    # if new_exp:
    #     today = date.today()
    #     date = today.strftime("%d_%m")
    #     date
    # else:
    #     date = '29_05'

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
        configs.data.corr_threshold = 0.9
        configs.general.debug_mode = False

    configs = set_paths(configs)
    main(configs)

        # exp_name, dataset, profileType, slice_id = sys.argv[1:]
    # configs = set_configs(exp_name)
    # slice_size = int(slice_size)
    # slice_id = int(slice_id)
    ################################################
    # dataset options: 'CDRP' , 'LUAD', 'TAORF', 'LINCS', 'CDRP-bio'
    # datasets=['TAORF','LUAD','LINCS', 'CDRP-bio']
    # datasets=['CDRP-bio']
    # datasets = [configs.general.dataset]
    # dataset = datasets[0]
    # configs.general.dataset = dataset
    
    # DT_kfold={'LUAD':10, 'TAORF':5, 'LINCS':25, 'CDRP-bio':6,'CDRP':40}



    