import random
from collections import defaultdict, Counter
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

# sys.path.insert(0, os.getcwd())
# sys.path.insert(0, currentdir)

from utils.general import revise_exp_name, set_configs, set_paths, add_exp_suffix, write_dataframe_to_excel
from utils.global_variables import DS_INFO_DICT, METHOD_NAME_MAPPING
from data_layer.data_utils import load_zscores
from data_layer.data_utils import get_cp_path, get_cp_dir, filter_data_by_highest_dose
# from utils.reproduce_funcs import get_duplicate_replicates, get_null_distribution_replicates,get_replicates_score
from utils.eval_utils import get_color_from_palette
# from utils.metrics import extract_ss_score, extract_new_score_for_compound, extract_ss_score_for_compound
from utils.eval_utils import replicateCorrs, calc_rand_corr
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# adapted from https://github.com/carpenter-singh-lab/2022_Haghighi_NatureMethods/blob/main/utils/replicateCorrs.py 


def calc_percent_replicating(configs,data_reps = ['ae_diff','baseline']):
    
    base_dir= '/sise/assafzar-group/assafzar/genesAndMorph'
    # data_dir = get_cp_dir(configs)
    data_dir =  get_cp_dir(base_dir,configs.general.dataset,configs.data.profile_type)
    l1k_data_dir = get_cp_dir(base_dir,configs.general.dataset,configs.data.profile_type,modality='L1000')
    run_dist_plots = False
    # exp_name= 'ae_12_09_fs'
    

    output_dir = configs.general.output_exp_dir
    null_base_dir = f'{base_dir}/results/{configs.general.dataset}/'
    exp_save_dir = configs.general.res_dir

    debug = configs.general.debug_mode
    # debug = False
    l1k_data_dir = os.path.dirname(data_dir) + '/L1000'
    if configs.data.modality == 'CellPainting':
        modality_str = 'cp'
    else:
        modality_str = 'l1k'

    # include profile type in list if it is a file in folder 'output_dir'
    profile_types = [p for p in ['augmented','normalized_variable_selected'] if any(p in string for string in os.listdir(output_dir))]
    data_reps = [dr for dr in data_reps if any(dr in string for string in os.listdir(output_dir))]

    for p in profile_types:

        exp_suffix = add_exp_suffix(p,configs.eval.by_dose, configs.eval.normalize_by_all,configs.eval.z_trim,configs.eval.min_max_norm)
        fig_dir = f'{configs.general.fig_dir}/{exp_suffix}'
        os.makedirs(fig_dir,exist_ok=True)

        configs.general.logger.info(f'calculating for profile type:{p}')
        methods = {}
        for dr in data_reps:
            methods[dr] = {'name':dr,'path':os.path.join(output_dir,f'replicate_level_{modality_str}_{p}_{dr}.csv')}
        
        if configs.eval.calc_l1k:
            methods['l1k'] = {'name':'L1000','path': os.path.join(l1k_data_dir,f'replicate_level_l1k_{p}.csv.gz')}

        os.makedirs(exp_save_dir,exist_ok=True)
        configs.general.logger.info(f'loading from path {output_dir}')

        methods = load_zscores(methods,base_dir,configs.general.dataset,p,by_dose=configs.eval.by_dose,normalize_by_all =configs.eval.normalize_by_all,z_trim=configs.eval.z_trim,set_index=False,debug=debug,min_max_norm=configs.eval.min_max_norm, filter_by_highest_dose=configs.eval.filter_by_highest_dose)

        # for m in methods.keys():                            
            # zscores = methods[m]['zscores']
            # if there is dose information and not running by dose, use only highest dose
            # if not configs.eval.by_dose and DS_INFO_DICT[configs.general.dataset]['has_dose'] and configs.eval.filter_by_highest_dose:
                    # methods[m]['zscores'] = filter_data_by_highest_dose(methods[m]['zscores'], configs.general.dataset, modality = methods[m]['modality'])
                    # configs.general.logger.info(f'filtered by highest dose: {len(zscores)} -> {len(methods[m]["zscores"])}')

        if 'l1k' in methods.keys():
            meta_features_l1k = [c for c in methods['l1k']['zscores'].columns if '_at' in c]
        if configs.eval.by_dose:
            cpd_col = 'dose_col'
            # l1k_pert_col_name = 'pert_id_dose'
        else:
            cpd_col = 'cpd_col'

        ################ run rep correlation measurements ####################
        corr_path = f'{exp_save_dir}/RepCorrDF.xlsx'

        methods = calc_replicate_corrs(methods,configs.general.dataset,cpd_col,exp_suffix,corr_path,debug,rand_reps=configs.eval.rand_reps,overwrite_experiment=configs.general.overwrite_experiment)
        for m in methods.keys():
            configs.general.logger.info(f"total number of cpds for {m}: {len(methods[m]['corr_df'])}")
            configs.general.logger.info(f"number of reproducible cpds for {m}: {len(methods[m]['reproducible_cpds'])}, {np.round(len(methods[m]['reproducible_cpds'])/len(methods[m]['corr_df'])*100,2)}%")
        methods_for_plot = methods.keys()
        if configs.eval.calc_l1k:
            methods_for_plot = [m for m in methods_for_plot if m != 'l1k']
        # ncols = len(methods_for_plot)

        run_pr_plot(methods,methods_for_plot,fig_dir)
        

        ############
        # find best method 
        comp_methods = data_reps[1:]
        compiting_methods = [m for m in comp_methods if m in methods.keys()]
        best_comp_method = max(compiting_methods, key=lambda key: methods[key]['percent_replicating'])
    
        # best_method = max(methods, key=lambda key: methods[key]['percent_replicating'])
        configs.general.logger.info(f'best method: {best_comp_method} with {methods[best_comp_method]["percent_replicating"]}% replicating')
        ####################### plot venn diageram of shared cpds of two methods #########################
        out = venn2([set(methods[data_reps[0]]['reproducible_cpds']), set(methods[best_comp_method]['reproducible_cpds'])], set_labels = ('Anomaly',METHOD_NAME_MAPPING[methods[best_comp_method]['name']]),  alpha=0.6,set_colors=("tab:green","tab:orange"))
        sets = compute_venn2_subsets(set(methods[data_reps[0]]['reproducible_cpds']), set(methods[best_comp_method]['reproducible_cpds']))
        sum_sets = sum(sets)
        shared_cpds_pr = sum_sets/len(methods[data_reps[0]]['corr_df'])*100
        configs.general.logger.info(f"number of reproducible cpds total: {sum_sets}, {np.round(shared_cpds_pr,2)}%")
        
        for text in out.set_labels:
            if text is not None:
                text.set_fontsize(20)
        for text in out.subset_labels:
            if text is not None:
                text.set_fontsize(20)
        if not debug:
            venn_path= f'{fig_dir}/venn{exp_suffix}.png'
            plt.tight_layout()
            plt.savefig(venn_path)
        plt.close()

        ####################### compute L1000 replicate correlation for both methods #########################
        if 'l1k' in methods.keys():
            run_pr_plot(methods,['l1k'],fig_dir)

            ####################### plot venn diageram of shared cpds of two methods #########################
            out_l1k = venn3([set(methods[data_reps[0]]['reproducible_cpds']), set(methods[best_comp_method]['reproducible_cpds']), set(methods['l1k']['reproducible_cpds'])], set_labels = ('Anomaly',METHOD_NAME_MAPPING[methods[best_comp_method]['name']], 'L1000'),  alpha=0.6,set_colors=("tab:green","tab:orange","tab:blue"))

            for text in out_l1k.set_labels:
                if text is not None:
                    text.set_fontsize(20)
            for text in out_l1k.subset_labels:
                if text is not None:
                    text.set_fontsize(20)
            plt.tight_layout()
            if not debug:
                venn_path= f'{fig_dir}/venn3{exp_suffix}.png'
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

        # create a dataframe summarizing reproducibile compounds
        cpd_col = DS_INFO_DICT[configs.general.dataset][methods[data_reps[0]]['modality']]['cpd_col']
        df_reproduce = pd.DataFrame()
        df_reproduce['cpd'] = methods[data_reps[0]]['zscores'][cpd_col]
        # df_reproduce['reproducible'] = 0
        # df_reproduce['reproducible'] = df_reproduce['reproducible'].astype(bool)
        # for m in methods.keys():
            # df_reproduce['reproducible'] = df_reproduce['reproducible'] | df_reproduce['cpd'].isin(methods[m]['reproducible_cpds'])
        df_reproduce['reproducible_anomaly'] = df_reproduce['cpd'].isin(methods[data_reps[0]]['reproducible_cpds'])
        df_reproduce['reproducible_raw'] = df_reproduce['cpd'].isin(methods[data_reps[0]]['reproducible_cpds'])
        if 'l1k' in methods.keys():
            df_reproduce['reproducible_l1k'] = df_reproduce['cpd'].isin(methods['l1k']['reproducible_cpds'])
            df_reproduce['reproducible_l1k'] = df_reproduce['cpd'].isin(methods['l1k']['reproducible_cpds'])
            df_reproduce['reproducible_cl'] = df_reproduce['reproducible_l1k'] | df_reproduce['reproducible_raw']
            df_reproduce['reproducible_acl'] = df_reproduce['reproducible_cl'] | df_reproduce['reproducible_anomaly']
            
        df_reproduce.to_csv(f'{exp_save_dir}/reproducible_cpds{exp_suffix}.csv',index=False)
                
        configs.general.logger.info('completed running RC successfully!')

        results = {}
        for m in methods.keys():
            results[m] = methods[m]['percent_replicating']
        results['shared_pr'] = shared_cpds_pr

        return results


########################### calc replicate correlations ############################
    
def calc_replicate_corrs(methods, dataset,cpd_col = 'cpd_col',exp_suffix = '',corr_path = None,debug = False, overwrite_experiment=False,rand_reps=5):
    
    if corr_path is not None and os.path.exists(corr_path):
        corr_df = pd.read_excel(corr_path, sheet_name=None)
        load_df=True
    else:
        load_df=False

    for m in methods.keys():   

        pertColName = DS_INFO_DICT[dataset][methods[m]['modality']][cpd_col]
        sheetname = f'{m}-{dataset.lower()}{exp_suffix}'

        if load_df and sheetname in corr_df.keys() and not overwrite_experiment:
            print(f'loading replicate scores for method: {m}') 
            methods[m]['corr_df'] = corr_df[sheetname]
            methods[m]['rep_corr'] = methods[m]['corr_df']['RepCor'].to_list()
            print(f'calculating rand corr for method {m}')
            methods[m]['rand_corr'] = calc_rand_corr(methods[m]['zscores'],pertColName,methods[m]['features'],reps=rand_reps)
        else:

            print(f'calculating replicate scores for method: {m}') 
            [rand_corr, rep_corr, corr_df] = replicateCorrs(methods[m]['zscores'],pertColName,methods[m]['features'],plotEnabled=False,reps=rand_reps)
            rand_corr = [rand_corr for rand_corr in rand_corr if str(rand_corr) != 'nan']  

            if 'index' not in corr_df.columns:
                corr_df = corr_df.reset_index().rename(columns={'Unnamed: 0':'index'})

            methods[m]['rand_corr'] = rand_corr
            methods[m]['rep_corr'] = rep_corr
            methods[m]['corr_df'] = corr_df

            df_for_saving = methods[m]['corr_df'].reset_index().rename(columns={'Unnamed: 0':'index'})
            if not debug:
                write_dataframe_to_excel(corr_path,sheetname,df_for_saving, append_new_data_if_sheet_exists=False)
        
        ######### calculate reproducible cpds and percent replicating ############
        perc90=np.percentile(methods[m]['rand_corr'], 90);
        methods[m]['reproducible_cpds'] = methods[m]['corr_df'][methods[m]['corr_df']['RepCor']>perc90]['index']
        
        methods[m]['percent_replicating'] = np.sum(methods[m]['corr_df']['RepCor']>perc90)/len(methods[m]['corr_df'])*100
        methods[m]['rand90'] = perc90

    return methods


################################################

def run_pr_plot(methods,data_reps,fig_dir,savename=None):

    sns.set_context("paper",font_scale = 1.8, rc={"font.size":8,"axes.titlesize":16,"axes.labelsize":16,"axes.facecolor": "white","axes.style": "ticks"})
    sns.set_style("ticks")

    ncols = len(data_reps)
    # plot distribution of replicate and random correlations
    max_y_val = 0
    fig, axes = plt.subplots(nrows=1,ncols=ncols,figsize=(5*ncols,3.8),sharey=True, sharex=True)

    if len(data_reps)>1:
        for i, m in enumerate(data_reps):

            # plot distribution of replicate and random correlations
            # repC = methods[m]['rep_corr']
            # randC_v2 = methods[m]['rand_corr']
            # repCorrDf = methods[m]['corr_df']   
        
            perc90=np.percentile(methods[m]['rand_corr'], 90);

            rgb_values = get_color_from_palette("Set2", i)
            sns.kdeplot(methods[m]['rep_corr'], bw_method=.15, label=f"{methods[m]['name']}",ax=axes[i],color=rgb_values,linewidth=2, fill =True);
            sns.kdeplot(methods[m]['rep_corr'], bw_method=.15, ax=axes[i],color=rgb_values,linewidth=1, fill =False);

            curr_max_y_val = np.max(axes[i].lines[0].get_data())
            max_y_val = max(max_y_val,curr_max_y_val)
            axes[i].set_xlabel('Correlation');
            
            # axes[i].set_title(METHOD_NAME_MAPPING[methods[m]['name']])

            if i == 1:
                axes[i].axvline(x=methods[m]['rand90'],linestyle=':',color = 'firebrick', linewidth=2, label='corr threshold')
                sns.kdeplot(methods[m]['rand_corr'], bw_method=.2, label="Random pairs",ax=axes[i],color='darkgrey')
            else:
                axes[i].axvline(x=methods[m]['rand90'],linestyle=':',color = 'firebrick', linewidth=2)
                sns.kdeplot(methods[m]['rand_corr'], bw_method=.2, ax=axes[i],color='darkgrey')
            axes[i].axvline(x=0,linestyle=':',color='k', linewidth=1);

        y_lim = np.max(max_y_val * 1.33)
        text_y_loc = y_lim*0.65
        for i, m in enumerate(data_reps):
            axes[i].text(0.08+methods[m]['rand90'], text_y_loc,str(int(np.round(methods[m]['percent_replicating'],2)))+ '%>t',fontsize=14) #add text

    else:
        m='l1k'
        ncols =1 
        fig, axes = plt.subplots(nrows=1,ncols=ncols,figsize=(4.5*ncols,4))
                

        # repCorrDf = methods[m]['corr_df']
        sns.kdeplot(methods[m]['rep_corr'], bw_method=.15, label="replicate pairs",ax=axes);
        sns.kdeplot(methods[m]['rep_corr'], bw_method=.15, label=f"{methods[m]['name']}",ax=axes,linewidth=2, fill =True);
        axes.axvline(x=methods[m]['rand90'],linestyle=':',color = 'firebrick', linewidth=2, label='corr threshold')
        sns.kdeplot(methods[m]['rand_corr'], bw_method=.2, label="Random pairs",ax=axes,color='darkgrey')
        axes.axvline(x=0,linestyle=':',color='k', linewidth=1);

        axes.set_xlabel('Correlation');

        curr_max_y_val = np.max(axes.lines[0].get_data())
        max_y_val = max(max_y_val,curr_max_y_val)
        y_lim = np.max(max_y_val * 1.33)
        text_y_loc = y_lim*0.65
        # for i, m in enumerate(data_reps):
        axes.text(0.08+methods[m]['rand90'], text_y_loc,str(int(np.round(methods[m]['percent_replicating'],2)))+ '%>t',fontsize=14) #add text

        
    if savename is None:
        savename = f'{m}_RC'

    plt.ylim(0,y_lim)
    plt.xlim(-1,1.2)
    plt.tight_layout() 
    debug = False
    if not debug:
        plt.savefig(f'{fig_dir}/{savename}.png',dpi=500)
    plt.close()



def compute_venn2_subsets(a, b):

    if not (type(a) == type(b)):
        raise ValueError("Both arguments must be of the same type")
    set_size = len if type(a) != Counter else lambda x: sum(x.values())   # We cannot use len to compute the cardinality of a Counter
    return (set_size(a - b), set_size(b - a), set_size(a & b))


if __name__ == '__main__':

    configs = set_configs()
    if len(sys.argv) <2:
        exp_name = 'interpret_2603'
        configs.general.exp_name = exp_name
        configs.general.dataset = 'LINCS'
        configs.general.dataset = 'CDRP-bio'
        # configs.general.dataset = 'TAORF'

        configs.data.profile_type = 'augmented'
        configs.data.modality = 'CellPainting'
        configs.eval.by_dose = False
        configs.data.corr_threshold = 0.9
        configs.general.debug_mode = False
        configs.eval.normalize_by_all = True
        configs.eval.run_dose_if_exists = True
        configs.eval.filter_by_highest_dose = True
        configs.eval.calc_l1k = True
        configs.general.overwrite_experiment = False
        data_reps = ['ae_diff','baseline']
    configs = set_paths(configs)
    calc_percent_replicating(configs,data_reps)

    # DT_kfold={'LUAD':10, 'TAORF':5, 'LINCS':25, 'CDRP-bio':6,'CDRP':40}



    