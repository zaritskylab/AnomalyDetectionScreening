import random
from collections import defaultdict
from statistics import median
import numpy as np
from tqdm import tqdm
import sys
import pickle
import os
import pandas as pd
import glob
from collections import defaultdict
from statistics import median
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn2, venn2_circles, venn3, venn3_circles
currentdir = '/sise/home/alonshp/AnomalyDetectionScreening'

# currentdir = os.path.dirname('home/alonshp/AnomalyDetectionScreeningLocal/')
# print(currentdir)
sys.path.insert(0, os.getcwd())
sys.path.insert(0, currentdir)

from utils.general import revise_exp_name, set_configs, set_paths, add_exp_suffix, write_dataframe_to_excel
from utils.global_variables import DS_INFO_DICT
from utils.data_utils import set_index_fields, load_data, load_zscores
from utils.readProfiles import get_cp_path, get_cp_dir, filter_data_by_highest_dose
from utils.reproduce_funcs import get_duplicate_replicates, get_null_distribution_replicates,get_replicates_score
from utils.plotting import get_color_from_palette
from utils.metrics import extract_ss_score, extract_new_score_for_compound, extract_ss_score_for_compound
from dataset_paper_repo.utils.replicateCorrs import replicateCorrs, calc_rand_corr
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
            fig_dir = f'{configs.general.fig_dir}/{exp_suffix}'
            os.makedirs(fig_dir,exist_ok=True)

            configs.general.logger.info(f'calculating for profile type:{p}')
            methods = {
                # 'anomaly':{'name':'anomaly','path':os.path.join(output_dir,f'replicate_level_{modality_str}_{p}_ae.csv')},
                'anomalyCP':{'name':'Anomaly','path':os.path.join(output_dir,f'replicate_level_{modality_str}_{p}_ae_diff.csv')},
                # 'anomaly_emb':{'name':'anomaly_emb','path':os.path.join(output_dir,f'replicate_level_{modality_str}_{p}_ae_embeddings.csv')},
                'CP':{'name':'CellProfiler','path': os.path.join(output_dir,f'replicate_level_{modality_str}_{p}_baseline.csv')},
                # 'l1k':{'name':'L1000','path': os.path.join(l1k_data_dir,f'replicate_level_l1k_{p}.csv.gz')}
                # 'raw_unchanged':{'name':'raw_unchanged','path': os.path.join(data_dir,f'replicate_level_cp_{p}.csv.gz')}
            }

            os.makedirs(save_base_dir,exist_ok=True)
            os.makedirs(exp_save_dir,exist_ok=True)
            configs.general.logger.info(f'loading from path {output_dir}')

            methods = load_zscores(methods,base_dir,configs.general.dataset,p,by_dose=d,normalize_by_all =configs.eval.normalize_by_all,z_trim=configs.eval.z_trim,set_index=False,debug=debug)
            if 'l1k' in methods.keys():
                meta_features_l1k = [c for c in methods['l1k']['zscores'].columns if '_at' in c]
            if d:
                cpd_col = 'dose_col'
                # l1k_pert_col_name = 'pert_id_dose'
            else:
                cpd_col = 'cpd_col'

        # run rep correlation measurements
        # for p in profile_types:
            corr_path = f'{exp_save_dir}/RepCorrDF.xlsx'

            loaded_corr_df = False
            if os.path.exists(corr_path):
                loaded_corr_df = True
                corr_df = pd.read_excel(corr_path, sheet_name=None)
            # os.makedirs(corr_dir,exist_ok=True)
            for m in methods.keys():   
                # scaler = MinMaxScaler()

                pertColName = DS_INFO_DICT[configs.general.dataset][methods[m]['modality']][cpd_col]
                # sclr = StandardScaler()
                # scaled_zscores = methods[m]['zscores'].copy()
                # scaled_zscores[methods[m]['features']] = sclr.fit_transform(scaled_zscores[methods[m]['features']])
                # scaled_zscores.groupby(['Metadata_Plate']).transform(lambda x: (x - x.mean()) / x.std())

                # scaler = MinMaxScaler(feature_range=(0, 1))
                # scaled_zscores[[methods[m]['features']] = sclr.fit_transform(methods[m]['zscores'][methods[m]['features']])
                sheetname = f'{m}-{configs.general.dataset.lower()}{exp_suffix}'

                if loaded_corr_df and sheetname in corr_df.keys():
                    configs.general.logger.info(f'loading replicate scores for method: {m}') 
                    methods[m]['corr_df'] = corr_df[sheetname]
                    methods[m]['rep_corr'] = methods[m]['corr_df']['RepCor'].to_list()
                    methods[m]['rand_corr'] = calc_rand_corr(methods[m]['zscores'],pertColName,methods[m]['features'])
                else:

                    configs.general.logger.info(f'calculating replicate scores for method: {m}') 

                    zscores = methods[m]['zscores']

                    # if there is dose information and not running by dose, use only highest dose
                    if not configs.eval.by_dose and DS_INFO_DICT[configs.general.dataset]['has_dose'] and configs.eval.filter_by_highest_dose:
                            methods[m]['zscores'] = filter_data_by_highest_dose(methods[m]['zscores'], configs.general.dataset, modality = methods[m]['modality'])
                            configs.general.logger.info(f'filtered by highest dose: {len(zscores)} -> {len(methods[m]["zscores"])}')

                    # if not os.path.exists(corr_path):
                    # if debug:
                        # [methods[m]['rand_corr'],methods[m]['rep_corr'],methods[m]['corr_df']] = replicateCorrs(methods[m]['zscores'][0:5000].reset_index(),pertColName,methods[m]['features'],plotEnabled=False)
                    # else:    
                    [methods[m]['rand_corr'],methods[m]['rep_corr'],methods[m]['corr_df']] = replicateCorrs(methods[m]['zscores'],pertColName,methods[m]['features'],plotEnabled=False)
                    methods[m]['corr_df'] = methods[m]['corr_df'].dropna()
                    
                    # if configs.eval.by_dose:
                    # sheetname += exp_suffix
                    df_for_saving = methods[m]['corr_df'].reset_index().rename(columns={'index':'Unnamed: 0'})
                    if not debug:
                        write_dataframe_to_excel(corr_path,sheetname,df_for_saving, append_new_data_if_sheet_exists=False)

            methods_for_plot = ['anomalyCP','CP']
            ncols = len(methods_for_plot)

            sns.set_context("paper",font_scale = 1.5, rc={"font.size":4,"axes.titlesize":16,"axes.labelsize":13})

            ############
            # plot distribution of replicate and random correlations
            max_y_val = 0
            fig, axes = plt.subplots(nrows=1,ncols=ncols,figsize=(4.5*ncols,4),sharey=True, sharex=True)
            for i, m in enumerate(methods_for_plot):

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

                rgb_values = get_color_from_palette("Set2", i)
                # sns.kdeplot(methods[m]['rep_corr'], bw_method=.1, label="replicate pairs",ax=axes[i],color=rgb_values);
                sns.kdeplot(methods[m]['rep_corr'], bw_method=.15, label=f"{methods[m]['name']}",ax=axes[i],color=rgb_values,linewidth=2, fill =True);
                sns.kdeplot(methods[m]['rep_corr'], bw_method=.15, ax=axes[i],color=rgb_values,linewidth=1, fill =False);
                curr_max_y_val = np.max(axes[i].lines[0].get_data())
                max_y_val = max(max_y_val,curr_max_y_val)
                axes[i].set_xlabel('Correlation');

                if i == 0:
                    sns.kdeplot(methods[m]['rand_corr'], bw_method=.2, ax=axes[i],color='darkgrey')
                    axes[i].axvline(x=perc90,linestyle=':',color = 'firebrick', linewidth=2)
                else:
                    sns.kdeplot(methods[m]['rand_corr'], bw_method=.2, label="Random pairs",ax=axes[i],color='darkgrey')
                    axes[i].axvline(x=perc90,linestyle=':',color = 'firebrick', linewidth=2, label='corr threshold')
                axes[i].axvline(x=0,linestyle=':',color='k');
                # axes[i].legend(loc=2);#axes.set_title('');

                if 'Unnamed: 0' not in repCorrDf.columns:
                    repCorrDf = repCorrDf.reset_index().rename(columns={'index':'Unnamed: 0'})
                    # repCorrDf = repCorrDf.set_index('Unnamed: 0')

                methods[m]['reproducible_cpds'] = repCorrDf[repCorrDf['RepCor']>perc90]['Unnamed: 0']
                configs.general.logger.info(f"total number of cpds for {m}: {len(methods[m]['corr_df'])}")
                configs.general.logger.info(f"number of reproducible cpds for {m}: {len(methods[m]['reproducible_cpds'])}")
                methods[m]['rs'] = np.sum(repCorrDf['RepCor']>perc90)/len(repCorrDf)*100
                methods[m]['repre90'] = perc90
                repre75 = np.sum(repCorrDf['RepCor']>perc75)/len(repCorrDf)*100
                # axes.text(0.1+perc75, 1.5,str(int(np.round(repre75,2)))+'%>.75', fontsize=12) #add text

            y_lim = np.max(max_y_val * 1.33)

            if configs.general.dataset == 'LUAD':
                text_y_loc = y_lim*0.2
            else:
                text_y_loc = y_lim*0.65
            for i, m in enumerate(methods_for_plot):
                axes[i].text(0.08+methods[m]['repre90'], text_y_loc,str(int(np.round(methods[m]['rs'],2)))+ '%>t',fontsize=13) #add text

            # lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
            lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
            if configs.general.dataset == 'CDRP-bio':
                axes[0].legend(lines, labels, loc='upper left', borderaxespad=0, edgecolor='black',fontsize=12)
            savename = f'{m}_RC'
            plt.ylim(0,y_lim)
            plt.xlim(-1,1.2)
            # plt.title(f'{m} {p} {configs.general.dataset}{exp_suffix}')
            plt.tight_layout() 
            if not debug:
                plt.savefig(f'{fig_dir}/{savename}.png',dpi=500)
            plt.close()

            ####################### plot venn diageram of shared cpds of two methods #########################
            out = venn2([set(methods['anomalyCP']['reproducible_cpds']), set(methods['CP']['reproducible_cpds'])], set_labels = ('Anomaly','CellProfiler'),  alpha=0.6,set_colors=("tab:green","tab:orange"))

            for text in out.set_labels:
                text.set_fontsize(16)
            for text in out.subset_labels:
                text.set_fontsize(16)
            if not debug:
                venn_path= f'{fig_dir}/venn{exp_suffix}.png'
                plt.savefig(venn_path)
            plt.close()

            ####################### compute "Signature Strength" param for both methods #########################
            # for m in methods_for_plot:
            #     methods[m]['ss'] = extract_ss_score(methods[m]['zscores'],cpd_id_fld = DS_INFO_DICT[configs.general.dataset][methods[m]['modality']][cpd_col], th_range=[2], abs_zscore=False, new_ss = False,cpd_id_fld = methods[m]['ind_col']) / len(methods[m]['features'])


                # methods[m]['sig_strength'] = len(methods[m]['reproducible_cpds'])/len(methods[m]['corr_df'])
            if 'l1k' in methods.keys():
                m='l1k'
                ncols =1 
                fig, axes = plt.subplots(nrows=1,ncols=ncols,figsize=(4.5*ncols,4))

                repC = methods[m]['rep_corr']
                randC_v2 = methods[m]['rand_corr']

                repC = [repC for repC in repC if str(repC) != 'nan']
                randC_v2 = [randC_v2 for randC_v2 in randC_v2 if str(randC_v2) != 'nan']  

                perc90=np.percentile(randC_v2, 90);
                perc80=np.percentile(randC_v2, 80);
                perc75=np.percentile(randC_v2, 75);
                rep10=np.percentile(repC, 10);
                    
                repCorrDf = methods[m]['corr_df']
                
                sns.kdeplot(methods[m]['rand_corr'], bw_method=.1, label="random pairs",ax=axes,color='darkgrey')

                rgb_values = get_color_from_palette("Set2", i)
                sns.kdeplot(methods[m]['rep_corr'], bw_method=.1, label="replicate pairs",ax=axes,color=rgb_values);
                axes.set_xlabel('Correlation');
                # sns.kdeplot(randC_v2, bw=.1, label="random pairs",ax=axes);axes.set_xlabel('CC');
                #         perc5=np.percentile(repCC, 50);axes.axvline(x=perc5,linestyle=':',color='darkorange');
                #         perc95=np.percentile(randCC, 90);axes.axvline(x=perc95,linestyle=':');
                # axes[i].legend();#
                axes.set_title(methods[m]['name'])

                axes.set_xlim(-1.1,1.1)
                axes.axvline(x=perc90,linestyle=':',color = 'black', label='corr threshold');
                # axes.axvline(x=perc75,linestyle=':',color = 'r');

                # axes.axvline(x=0,linestyle=':',color='k');
                axes.legend(loc=2);#axes.set_title('');
                axes.set_xlim(-1,1);

                repre90 = np.sum(repCorrDf['RepCor']>perc90)/len(repCorrDf)*100
                repre75 = np.sum(repCorrDf['RepCor']>perc75)/len(repCorrDf)*100
                
                # axes.text(0.1+perc75, 1.5,str(int(np.round(repre75,2)))+'%>.75', fontsize=12) #add text
                axes.text(0.1+perc90, 0.8,str(int(np.round(repre90,2)))+'%>t', fontsize=12) #add text

                if 'Unnamed: 0' not in repCorrDf.columns:
                    repCorrDf = repCorrDf.reset_index().rename(columns={'index':'Unnamed: 0'})
                methods[m]['reproducible_cpds'] = repCorrDf[repCorrDf['RepCor']>perc90]['Unnamed: 0']

                savename = f'l1k_RC'
                # plt.title(f'{m} {p} {configs.general.dataset}{exp_suffix}')
                plt.tight_layout() 
                if not debug:
                    plt.savefig(f'{fig_dir}/{savename}.png',dpi=300)
                plt.close()

                            ####################### plot venn diageram of shared cpds of two methods #########################
                out_l1k = venn3([set(methods['anomalyCP']['reproducible_cpds']), set(methods['CP']['reproducible_cpds']), set(methods['l1k']['reproducible_cpds'])], set_labels = ('Anomaly','CellProfiler', 'L1000'),  alpha=0.6,set_colors=("tab:green","tab:orange","tab:blue"))

                for text in out_l1k.set_labels:
                    text.set_fontsize(16)
                for text in out_l1k.subset_labels:
                    text.set_fontsize(16)

                if not debug:
                    venn_path= f'{fig_dir}/venn3_{exp_suffix}.png'
                    plt.savefig(venn_path)
                plt.close()

        # plot distribution of replicate and random correlations
                
            if run_dist_plots:
                for m in methods.keys():
                    a = methods[m]['zscores'][0:2000].to_numpy().flatten()
                    sns.histplot(a, bins=100)
                    plt.axvline(x=0.5,color='r')
                    plt.title(f'{m}_{p} zscores distribution')
                    # plt.xlim(-5,5)
                    # plt.show()

                    savename = f'{m}_zscores_dist'
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

                replicates_df, cpds = get_duplicate_replicates(methods['CP']['zscores'],min_num_reps=4)
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
        exp_name = '101_t_le'
        configs.general.exp_name = exp_name
        # configs.general.dataset = 'LINCS'
        configs.general.dataset = 'CDRP-bio'
        configs.general.dataset = 'TAORF'

        configs.data.profile_type = 'normalized_variable_selected'
        configs.data.profile_type = 'augmented'
        configs.data.modality = 'CellPainting'
        configs.eval.by_dose = False
        configs.data.corr_threshold = 0.9
        configs.general.debug_mode = False
        configs.eval.normalize_by_all = True
        configs.eval.run_dose_if_exists = True
        configs.eval.filter_by_highest_dose = True

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



    