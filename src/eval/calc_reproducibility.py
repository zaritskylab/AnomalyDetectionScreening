
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn2, venn2_circles, venn3, venn3_circles
currentdir = '/sise/home/alonshp/AnomalyDetectionScreening'

# sys.path.insert(0, os.getcwd())
# sys.path.insert(0, currentdir)

from utils.config_utils import set_configs, add_exp_suffix
from utils.file_utils import write_dataframe_to_excel

from utils.global_variables import DS_INFO_DICT, METHOD_NAME_MAPPING, MODALITY_STR, BASE_DIR
from data.data_utils import load_zscores, get_cp_dir
from eval.eval_utils import run_pr_plot, compute_venn2_subsets



def calc_percent_replicating(configs,data_reps = ['ae_diff','baseline']):
    """
    Calculate reproducibility of compounds and save results to a CSV file.

    Args:
        configs: Configuration object containing general settings.
        data_reps: List of data representations.
        debug_mode: Boolean indicating if debug mode is enabled.
        exp_suffix: Suffix for the experiment.
        fig_dir: Directory to save figures.
        methods: Dictionary containing method details.
        best_comp_method: Best comparison method.
        res_dir: Directory to save experiment results.
        shared_cpds_with_baseline: Shared compounds with baseline.

    Returns:
        Dictionary containing results of percent replicating for each method.
    """

    # data_dir = get_cp_dir(configs)
    base_dir = configs.general.base_dir
    output_dir = configs.general.output_dir
    res_dir = configs.general.res_dir

    data_dir = get_cp_dir(base_dir,configs.general.dataset,configs.data.profile_type)
    # l1k_data_dir = get_cp_dir(base_dir,configs.general.dataset,configs.data.profile_type,modality='L1000')
    # exp_name= 'ae_12_09_fs'

    debug_mode = configs.general.debug_mode
    l1k_data_dir = os.path.dirname(data_dir) + '/L1000'

    # include profile type in list if it is a file in folder 'output_dir'
    profile_types = [p for p in ['augmented','normalized_variable_selected'] if any(p in string for string in os.listdir(output_dir))]
    data_reps = [dr for dr in data_reps if any(dr in string for string in os.listdir(output_dir))]
    assert len(profile_types) > 0, 'no profile types found in output_dir'

    for p in profile_types:

        exp_suffix = add_exp_suffix(p,configs.eval.by_dose, configs.eval.normalize_by_all,configs.eval.z_trim,configs.eval.min_max_norm)
        fig_dir = f'{configs.general.fig_dir}/{exp_suffix}'
        os.makedirs(fig_dir,exist_ok=True)

        configs.general.logger.info(f'calculating for profile type:{p}')
        methods = {}
        for dr in data_reps:
            methods[dr] = {'name':dr,'path':os.path.join(output_dir,f'replicate_level_{MODALITY_STR[configs.data.modality]}_{p}_{dr}.csv')}
        
        if configs.eval.with_l1k:
            methods['l1k'] = {'name':'L1000','path': os.path.join(l1k_data_dir,f'replicate_level_l1k_{p}.csv.gz')}

        os.makedirs(res_dir,exist_ok=True)
        configs.general.logger.info(f'loading from path {output_dir}')

        methods = load_zscores(methods,base_dir,configs.general.dataset,p,by_dose=configs.eval.by_dose,normalize_by_all =configs.eval.normalize_by_all,z_trim=configs.eval.z_trim,set_index=False,debug_mode=debug_mode,min_max_norm=configs.eval.min_max_norm, filter_by_highest_dose=configs.eval.filter_by_highest_dose)

        if 'l1k' in methods.keys():
            meta_features_l1k = [c for c in methods['l1k']['zscores'].columns if '_at' in c]
        if configs.eval.by_dose:
            cpd_col = 'dose_col'
            # l1k_pert_col_name = 'pert_id_dose'
        else:
            cpd_col = 'cpd_col'

        ################ run rep correlation measurements ####################
        corr_path = f'{res_dir}/RepCorrDF.xlsx'

        methods = calc_replicate_corrs(methods,configs.general.dataset,cpd_col,exp_suffix,corr_path,debug_mode,rand_reps=configs.eval.rand_reps,overwrite_experiment=configs.general.overwrite_experiment)
        for m in methods.keys():
            configs.general.logger.info(f"total number of cpds for {m}: {len(methods[m]['corr_df'])}")
            configs.general.logger.info(f"number of reproducible cpds for {m}: {len(methods[m]['reproducible_cpds'])}, {np.round(len(methods[m]['reproducible_cpds'])/len(methods[m]['corr_df'])*100,2)}%")
        methods_for_plot = methods.keys()
        if configs.eval.with_l1k:
            methods_for_plot = [m for m in methods_for_plot if m != 'l1k']
        # ncols = len(methods_for_plot)

        run_pr_plot(methods,methods_for_plot,fig_dir)

        if len(data_reps) > 1:
            # find best method in case of comparison with other representations
            comp_methods = data_reps[1:]
            compiting_methods = [m for m in comp_methods if m in methods.keys()]

            best_comp_method = max(compiting_methods, key=lambda key: methods[key]['percent_replicating'])
            methods_to_compare = [data_reps[0],best_comp_method]
            configs.general.logger.info(f'best method: {best_comp_method} with {methods[best_comp_method]["percent_replicating"]}% replicating')

            ####################### plot venn diageram of shared cpds of two methods #########################   
            plot_venn2(methods, methods_to_compare, configs, debug_mode, exp_suffix, fig_dir)
            sets = compute_venn2_subsets(set(methods[methods_to_compare[0]]['reproducible_cpds']), set(methods[methods_to_compare[1]]['reproducible_cpds']))
            sum_sets = sum(sets)
            shared_cpds_with_baseline = sum_sets/len(methods[data_reps[0]]['corr_df'])*100
            configs.general.logger.info(f"number of reproducible cpds total: {sum_sets}, {np.round(shared_cpds_with_baseline,2)}%")

        ####################### compute L1000 replicate correlation for both methods #########################
        if configs.eval.with_l1k:
            
            run_pr_plot(methods,['l1k'],fig_dir)

            if len(data_reps) == 1:
                methods_to_compare = [data_reps[0], 'l1k']
                plot_venn2(methods, methods_to_compare, configs, debug_mode, exp_suffix, fig_dir)
            else:
                methods_to_compare = [data_reps[0], best_comp_method, 'l1k']
                plot_venn3(methods, methods_to_compare, configs, debug_mode, exp_suffix, fig_dir)
    
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
            
        df_reproduce.to_csv(f'{res_dir}/reproducible_cpds{exp_suffix}.csv',index=False)
                
        configs.general.logger.info('completed running RC successfully!')

        results = {}
        for m in methods.keys():
            results[m] = methods[m]['percent_replicating']
        if len(data_reps) > 1:
            results['shared_pr'] = shared_cpds_with_baseline

        return results

def plot_venn2(methods, methods_to_compare, configs, debug_mode=False, exp_suffix='', fig_dir=''):

    assert len(methods_to_compare) == 2, 'only two methods can be compared'
    # methods = methods[methods_to_compare]
    out = venn2([set(methods[methods_to_compare[0]]['reproducible_cpds']), set(methods[methods_to_compare[1]]['reproducible_cpds'])], set_labels = (methods_to_compare),  alpha=0.6,set_colors=("tab:green","tab:orange"))
            
    for text in out.set_labels:
        if text is not None:
            text.set_fontsize(20)
    for text in out.subset_labels:
        if text is not None:
            text.set_fontsize(20)
    if not debug_mode:
        venn_path= f'{fig_dir}/venn{exp_suffix}.png'
        plt.tight_layout()
        plt.savefig(venn_path)
    plt.close()


def plot_venn3(methods, methods_to_compare, configs, debug_mode=False, exp_suffix='', fig_dir=''):

    assert len(methods_to_compare) == 3, 'only three methods can be compared'
    # methods = methods[methods_to_compare]
    out = venn3([set(methods[methods_to_compare[0]]['reproducible_cpds']), set(methods[methods_to_compare[1]]['reproducible_cpds']), set(methods[methods_to_compare[2]]['reproducible_cpds'])],
                           set_labels = (methods_to_compare),  alpha=0.6,set_colors=("tab:green","tab:orange","tab:blue"))
    ####################### plot venn diageram of shared cpds of two methods #########################

    for text in out.set_labels:
        if text is not None:
            text.set_fontsize(20)
    for text in out.subset_labels:
        if text is not None:
            text.set_fontsize(20)
    plt.tight_layout()

    if not debug_mode:
        venn_path= f'{fig_dir}/venn3{exp_suffix}.png'
        plt.savefig(venn_path)
    plt.close()



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

# adapted from https://github.com/carpenter-singh-lab/2022_Haghighi_NatureMethods/blob/main/utils/replicateCorrs.py

def replicateCorrs(inDf,pertColName,featColNames,plotEnabled,reps=5):
    
    """ 
    Calculates replicate correlation versus across purtburtion correlations
  
    This function takes the input dataframe and output/plot replicate correlations. 
  
    Parameters: 
    inDf   (pandas df): input dataframe contains metadata and features
    pertColName  (str): The column based on which we define replicates of a purturbation
    featColNames(list): The list of all columns corresponding to features
    plotEnabled (bool): If True or 1, plots the curves 
    
    Returns: 
    repCorrDf   (list):  
  
    """
    
    
    df=inDf.copy()
    df[featColNames]=inDf[featColNames].interpolate();
    uniqPert=df[pertColName].unique().tolist()
    repC=[]
    randC=[]
    
    repCorrDf=pd.DataFrame(index = uniqPert,columns=['RepCor']) 
    
    
    repSizeDF=df.groupby([pertColName]).size().reset_index()
    highRepComp=repSizeDF[repSizeDF[0]>1][pertColName].tolist()

    
    for u in highRepComp:
        df1=df[df[pertColName]==u].drop_duplicates().reset_index(drop=True)
#         df2=df[df[pertColName]!=u].drop_duplicates().reset_index(drop=True)

        repCorrPurtbs=df1.loc[:,featColNames].T.corr()
        repCorr=list(repCorrPurtbs.values[np.triu_indices(repCorrPurtbs.shape[0], k = 1)])
#         print(repCorr)
        repCorrDf.loc[u,'RepCor']=np.nanmean(repCorr)
#         print(repCorr)
#         repCorr=np.sort(np.unique(df1.loc[:,featColNames].T.corr().values))[:-1].tolist()
#         repC=repC+repCorr
        repC=repC+[np.nanmedian(repCorr)]

    randC_v2=calc_rand_corr(inDf,pertColName,featColNames,reps=reps)

        
    if 0:
        fig, axes = plt.subplots(figsize=(5,3))
        sns.kdeplot(randC, bw=.1, label="random pairs",ax=axes)
        sns.kdeplot(repC, bw=.1, label="replicate pairs",ax=axes);axes.set_xlabel('CC');
        sns.kdeplot(randC_v2, bw=.1, label="random v2 pairs",ax=axes);axes.set_xlabel('CC');
#         perc5=np.percentile(repCC, 50);axes.axvline(x=perc5,linestyle=':',color='darkorange');
#         perc95=np.percentile(randCC, 90);axes.axvline(x=perc95,linestyle=':');
        axes.legend();#axes.set_title('');
        axes.set_xlim(-1.1,1.1)
        
    repC = [repC for repC in repC if str(repC) != 'nan']
    
    perc95=np.percentile(randC_v2, 90);
    rep10=np.percentile(repC, 10);
    
    if plotEnabled:
        fig, axes = plt.subplots(figsize=(5,4))
#         sns.kdeplot(randC_v2, bw=.1, label="random pairs",ax=axes);axes.set_xlabel('CC');
#         sns.kdeplot(repC, bw=.1, label="replicate pairs",ax=axes,color='r');axes.set_xlabel('CC');
        sns.distplot(randC_v2,kde=True,hist=True,bins=100,label="random pairs",ax=axes,norm_hist=True);
        sns.distplot(repC,kde=True,hist=True,bins=100,label="replicate pairs",ax=axes,norm_hist=True,color='r');   

        #         perc5=np.percentile(repCC, 50);axes.axvline(x=perc5,linestyle=':',color='darkorange');
        axes.axvline(x=perc95,linestyle=':');
        axes.axvline(x=0,linestyle=':');
        axes.legend(loc=2);#axes.set_title('');
        axes.set_xlim(-1,1);
        plt.tight_layout() 
        
    repCorrDf['Rand90Perc']=perc95
    repCorrDf['Rep10Perc']=rep10
#     highRepPertbs=repCorrDf[repCorrDf['RepCor']>perc95].index.tolist()
#     return repCorrDf
    return [randC_v2,repC,repCorrDf]

def calc_rand_corr(inDf,pertColName,featColNames,reps = 5):
    randC_v2=[]    
    # reps = 5
    random_states = [42+i for i in range(reps)]
    # random_integers = [np.random.randint(0, 1000) for i in range(reps)]
    for i in range(reps):
        uniqeSamplesFromEachPurt=inDf.groupby(pertColName)[featColNames].apply(lambda s: s.sample(1,random_state=random_states[i]))
        corrMatAcrossPurtbs=uniqeSamplesFromEachPurt.loc[:,featColNames].T.corr()
        randCorrVals=list(corrMatAcrossPurtbs.values[np.triu_indices(corrMatAcrossPurtbs.shape[0], k = 1)])
        randC_v2=randC_v2+randCorrVals
    randC_v2 = [randC_v2 for randC_v2 in randC_v2 if str(randC_v2) != 'nan']    
    return randC_v2



if __name__ == '__main__':

    configs = set_configs()
