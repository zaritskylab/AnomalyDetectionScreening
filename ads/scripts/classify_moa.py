import random
from collections import defaultdict
from statistics import median
import numpy as np
from tqdm import tqdm
import sys
import os
from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn.utils import class_weight
from sklearn.naive_bayes import GaussianNB,ComplementNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, cross_val_predict, GroupKFold,StratifiedGroupKFold,GridSearchCV,KFold
from sklearn.neural_network import MLPClassifier

from imblearn.over_sampling import RandomOverSampler
import pandas as pd
from tqdm import tqdm
import xgboost as xgb
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.model_selection import LeaveOneOut,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import pairwise_distances,mean_absolute_error, mean_squared_error

import matplotlib.pyplot as plt
import pickle

currentdir = '/sise/home/alonshp/AnomalyDetectionScreening'
print(currentdir)

# currentdir = os.path.dirname('home/alonshp/AnomalyDetectionScreeningLocal/')
# print(currentdir)
sys.path.insert(0, os.getcwd())
sys.path.insert(0, currentdir)

from utils.general import revise_exp_name, set_configs, set_paths,set_seed, add_exp_suffix
from utils.global_variables import DS_INFO_DICT, ABRVS
from utils.data_utils import set_index_fields
# from utils.eval_utils import filter_data_by_highest_dose
from utils.readProfiles import get_cp_path, get_cp_dir, read_paired_treatment_level_profiles, read_treatment_level_profiles, filter_data_by_highest_dose
from utils.reproduce_funcs import get_median_correlation, get_duplicate_replicates, get_replicates_score
from utils.general import saveAsNewSheetToExistingFile, write_dataframe_to_excel
from utils.file_utils import load_df_pickles_and_concatenate
from dataset_paper_repo.utils.normalize_funcs import standardize_per_catX


############################################################################################################


def main(configs, data_reps = None):

    if data_reps is None:
        Types = ['baseline','ae_diff']
        data_reps = [f'{configs.data.profile_type}_{t}' for t in Types]
        

    if configs.general.flow in ['run_moa']:
        run_classifier(configs, data_reps = data_reps)
    elif configs.general.flow in ['concat_moa_res']:
        # print('skipping classify_moa')
        concat_exps_results(configs, data_reps = data_reps)
    else:
        raise ValueError(f'flow {configs.general.flow} not supported for MoA classification script')

############################################################################################################

def concat_exps_results(configs, data_reps = None):

    if configs.moa.do_all_filter_groups:
            filter_combs = [
            # ['CP','anomalyCP'],
            ['CP','l1k'],
            ['anomalyCP','l1k'],
            ['CP','anomalyCP','l1k']
        ]
    else:
        filter_combs =  [['CP','l1k']]

    for fg in filter_combs:
        configs.moa.filter_groups = fg
        moa_csv_fileName, exp_suffix, filter_abrv = get_moa_filename(configs.data.profile_type, configs.eval.by_dose, configs.moa.filter_groups, configs.moa.folds, configs.moa.min_samples,configs.eval.normalize_by_all)

        save_dir = os.path.join(configs.general.res_dir,'MoAprediction', moa_csv_fileName)
        # os.makedirs(save_dir,exist_ok=True)
        save_path = os.path.join(save_dir, 'pred_moa_res_all.xlsx')

        dirs = glob.glob(f'{configs.general.res_dir}/MoAprediction/{moa_csv_fileName}/*')
        exp_dirs = [d for d in dirs if any(f in d.split('/')[-1] for f in ['lr','mlp'])]

        for e in exp_dirs:
            moa_pred_res = load_df_pickles_and_concatenate(e)
            # moa_pred_res = pd.DataFrame(res)
            sheetname = e.split('/')[-1]

            print(f'writing results for {sheetname} to {save_path}, number of vals is {moa_pred_res.shape[0]}')
            write_dataframe_to_excel(save_path,sheetname,moa_pred_res)


def get_moa_filename(profile_type, by_dose, filter_groups, folds, min_samples,normalize_by_all):

    filter_abrv = ''
    for f in filter_groups:
        filter_abrv += f[0].lower()
    # filter_abrv = [configs.moa.filter_groups[i][0] for i in range(len(configs.moa.filter_groups))]
    exp_suffix = add_exp_suffix(profile_type=profile_type,by_dose= by_dose, normalize_by_all=normalize_by_all)
    
    exp_params_str = f'n{min_samples}_f{folds}{exp_suffix}_{filter_abrv}'
    exp_fileName = f'{profile_type}_{exp_params_str}'
    return exp_fileName, exp_suffix, filter_abrv

############################################################################################################

def get_filters_abrv(filter_groups):

    filter_abrv = ''
    for f in filter_groups:
        filter_abrv += f[0].lower()
    return filter_abrv

############################################################################################################

def get_filter_rep_params(dataset, exp_suffix,repCorrFilePath, filter_groups,filter_perts= 'highRepUnion'):
    sheet_names = []
    if 'CP' in filter_groups:
        # filter_perts = 'onlyCP'
        sheet_names.append(f'CP-{dataset.lower()}{exp_suffix}')
    if 'anomalyCP' in filter_groups:
        sheet_names.append(f'anomalyCP-{dataset.lower()}{exp_suffix}')
    if 'l1k' in filter_groups:
        #TODO: add l1k by dose and not only general
        sheet_names.append(f'l1k-{dataset.lower()}{exp_suffix}')

    filter_repCorr_params=[filter_perts,repCorrFilePath, sheet_names]

    return filter_repCorr_params

############################################################################################################

def run_classifier(configs, data_reps = None):

    # repCorrFilePath=f'{configs.general.res_dir}/{rep_corr_fileName}.xlsx'
    # repCorrFilePath=f'{base_dir}/RepCorrDF_{profileType}.xlsx'
    
    # if filter_perts=='onlyCP':  
    if data_reps is None:
        Types = ['ae_diff', 'baseline']
        data_reps = [f'{configs.data.profile_type}_{t}' for t in Types]

    repCorrFilePath=f'{configs.general.res_dir}/{configs.moa.rep_corr_fileName}.xlsx'
    # else:
        # repCorrFilePath=f'{configs.general.base_dir}/results/RepCorrDF.xlsx'
    # filter_abrv = [configs.moa.filter_groups[i][0] for i in range(len(configs.moa.filter_groups))]
    # filter_abrv = configs.moa.filter_groups[0][0] + configs.moa.filter_groups[1][0]
    # exp_suffix = add_exp_suffix(profile_type=configs.data.profile_type,by_dose= configs.eval.by_dose)
    # exp_params_str = f'n{configs.moa.nSamplesMOA}_f{configs.moa.folds}{exp_suffix}_{filter_abrv}'
    # configs.moa.moa_csv_fileName = f'{configs.data.profile_type}_{exp_params_str}'
    if configs.moa.do_all_filter_groups:
            filter_combs = [
            # ['CP','anomalyCP'],
            ['CP','l1k'],
            ['anomalyCP','l1k'],
            ['CP','anomalyCP','l1k'],
            ['CP'],
            ['anomalyCP']
        ]
    else:
        filter_combs =  [['CP','l1k']]

    for fg in filter_combs:

        configs.moa.filter_groups = fg

        moa_csv_fileName, exp_suffix, filter_abrv = get_moa_filename(configs.data.profile_type, configs.eval.by_dose, configs.moa.filter_groups, configs.moa.folds, configs.moa.min_samples,configs.eval.normalize_by_all)
        filter_repCorr_params = get_filter_rep_params(configs.general.dataset, exp_suffix,repCorrFilePath, configs.moa.filter_groups,filter_perts= configs.moa.filter_perts)

        moa_dir = os.path.join(configs.general.res_dir,configs.moa.moa_dirname)
        summary_csv_path = os.path.join(moa_dir, 'moa_summary.xlsx')
        save_dir = os.path.join(moa_dir, moa_csv_fileName)
        os.makedirs(save_dir,exist_ok=True)
        save_path = os.path.join(save_dir, 'pred_moa_res.xlsx')
        # filter_repCorr_params=[filter_perts,repCorrFilePath]

        ################################################
        # pertColName='PERT'
        
        # if filter_by_pr:
        #     repreducible_path= f'{base_dir}/{dataset}/{modality}/ldim_8/reproducible_cpds_1000.csv'
        #     pr_cpds = pd.read_csv(repreducible_path)
        #     profile_type_pr = pr_cpds[pr_cpds['method']==profTypeAbbrev[0]] 
        
        res_base_dir = os.path.dirname(configs.general.res_dir)
        profTypeAbbrev = [ABRVS[profileType] for profileType in data_reps]

        # try:
        methods, filteredMOAs = load_data_filtered_by_rep_corr(configs, data_reps, filter_repCorr_params,run_l1k=configs.moa.run_l1k)
        # except:
            # print(f'failed to load data for moa prediction for {exp_suffix}')
            # return

        if configs.moa.do_fusion:
            # methods = create_fused_ad_rep(configs, methods)
            profTypeAbbrev.append(f'fuse')

        
        # # logo = LeaveOneGroupOut()

        test_index_gen = []
        res= {}

        if configs.moa.tune:

            parameter_space_MLP = {
                'hidden_layer_sizes': [(256,),(128,),(64), (128,64),(256,64), (128,32), (64,16),(128,64,32),(256,128,64)],
                'activation': ['tanh', 'relu'],
                'alpha': [0.0001,0.05],
                'learning_rate': ['constant','adaptive'],
                'random_state': [configs.moa.exp_seed]
            }

            parameter_space_logistic={"C":[1,10,100],
            'random_state': [configs.moa.exp_seed]
            }
            
        coefs_all_cvs = {}
        coefs_all = {}

        for j, p in enumerate(methods.keys()):
            # coefs[p] = {}
            coefs_all_cvs[p] = pd.DataFrame()
            coefs_all[p] = []
            for cls_model in ['lr','mlp']:

            # for cls_model in ['lr','mlp','xgboost']:
            # for cls_model in ['xgboost']:
                # coefs_all_cvs = pd.DataFrame()

                moa_pred_res=pd.DataFrame(index=filteredMOAs.index,columns=['CP','GE','Early Fusion',
                                # 'RGCCA_CP','RGCCA_GE','RGCCA_EarlyFusion',
                                'Late Fusion',
                                'Metadata_moa_num'])


                moa_pred_res['PERT']=filteredMOAs['PERT']
                moa_pred_res['Metadata_moa_with_n']=filteredMOAs['Metadata_moa_with_n']
                moa_pred_res['Compounds']=filteredMOAs['Compounds']
                
                n_classes =  len(np.unique(filteredMOAs['Metadata_moa_num'].values))
                splits = min(configs.moa.folds,configs.moa.min_samples+1)
                sgkf = StratifiedGroupKFold(n_splits=splits,shuffle=True,random_state=configs.moa.exp_seed)

                leG = preprocessing.LabelEncoder()
                group_labels=leG.fit_transform(filteredMOAs['Compounds'].values)

                i=0
                # for train_index0, test_index in gkf.split(filteredMOAs, groups=group_labels):
                for train_index, test_index in tqdm(sgkf.split(filteredMOAs,filteredMOAs['Metadata_moa_num'].values, groups=group_labels)):
                # for train_index0, test_index in gkf.split(filteredMOAs, groups=group_labels):
                # for train_index, test_index in tqdm(sgkf_split):
                    print('rand ',i)
                    i+=1
                    if configs.general.debug_mode and i == 2:
                            break
                    test_index_gen.append(test_index)
                #     data_train = filteredMOAs.loc[train_index,domXfeats].values;
                    labels_train=filteredMOAs.loc[train_index,'Metadata_moa_num'].tolist()
                #     print(filteredMOAs.loc[train_index,'Metadata_moa_num'].unique().shape)

                #     data_test = filteredMOAs.loc[test_index,domXfeats].values;
                    labels_test=filteredMOAs.loc[test_index,'Metadata_moa_num'].tolist()
                #     print(filteredMOAs.loc[test_index,'Metadata_moa_num'].unique().shape)    

                #     class_weightt = class_weight.compute_class_weight(class_weight='balanced',classes=np.unique(labels_train),y=labels_train)    
                #         model_tr = RandomForestClassifier(n_estimators=10,max_features=100,class_weight="balanced")

                #     overSampleMinorityFirstClassTo=np.max([labels_train.count(i) for i in set(labels_train)])

                #     ratios = {l: overSampleMinorityFirstClassTo for l in set(labels_train) \
                #       if labels_train.count(l)<overSampleMinorityFirstClassTo}
                #     sm1=RandomOverSampler(ratio=ratios)
                    moa_pred_res.loc[test_index,'Fold']=i

                    sm1=RandomOverSampler(sampling_strategy='not majority',random_state=configs.moa.exp_seed)

                    probs=[]
                    if configs.moa.run_l1k:
                        reps_for_training = ['CP','Early Fusion','GE']
                    else:
                        reps_for_training = ['CP','Early Fusion']

                    for n,dt_modality,col in zip([0,1,2,3],methods[p]["data4eval"],reps_for_training):
                                                                        #   'RGCCA_CP',\
                                # 'RGCCA_GE','RGCCA_EarlyFusion',
                        # if col =='GE' and n>0:
                            # continue
                        print(f'training represntation {p} on modality {col} with model {cls_model} fold {i}')

                        data_m=dt_modality[0]

                        dt_train=data_m.loc[train_index,dt_modality[1]].values;
                        dt_test=data_m.loc[test_index,dt_modality[1]].values; 

                        if cls_model=='lr':            
                            if configs.moa.tune:
                                if (configs.moa.tune_first_fold and i==1) or not configs.moa.tune_first_fold:
                                    # if (configs.moa.tune_first_fold and i==1) or not configs.moa.tune_first_fold:
                                    model_logistic = LogisticRegression(random_state=configs.moa.exp_seed,multi_class='multinomial',class_weight="balanced",n_jobs=4, max_iter=1000) 
                                    model_tr = GridSearchCV(model_logistic, parameter_space_logistic, n_jobs=4, cv=5)
                                # else:
                            elif not configs.moa.tune:
                                model_tr = LogisticRegression(multi_class='multinomial',class_weight="balanced", max_iter=1000, random_state=configs.moa.exp_seed) 

                        elif cls_model=='mlp':
                #             model_MLP = MLPClassifier(random_state=5,max_iter=100,alpha=0.0001,activation='tanh')
                            if configs.moa.tune:
                                if (configs.moa.tune_first_fold and i==1) or not configs.moa.tune_first_fold:
                                    model_MLP = MLPClassifier(random_state=configs.moa.exp_seed,max_iter=1000)
                                    model_tr = GridSearchCV(model_MLP, parameter_space_MLP, n_jobs=4, cv=5)
                            elif not configs.moa.tune:
                                # model_tr = MLPClassifier(random_state=5,max_iter=200,alpha=0.0001,activation='tanh')
                                model_tr = MLPClassifier(random_state=configs.moa.exp_seed,max_iter=1000,alpha=0.0001,activation='relu',hidden_layer_sizes=(200,),learning_rate='constant')
                        # elif cls_model=='mlp_torch':
                        elif cls_model=='mlp_pytorch':
                            # trainer.fit(model=model, datamodule=datamodule)
                            model_tr = ElasticNetMLP(in_features = dt_train.shape[1], num_classes = nClasses,epochs=600, hidden_size = 100)
                        elif cls_model=='xgboost':
                            model_tr = xgb.XGBClassifier(n_estimators=500,learning_rate=0.05)
                            # model_tr = xgb.XGBRegressor(tree_method="hist", device="cuda")

                        dt_train_balanced,labels_train_balanced = sm1.fit_resample(dt_train,labels_train)

                        model_tr.fit(dt_train_balanced,labels_train_balanced)
                        
                        if  'mlp' in cls_model:
                            print("Training set score: %f" % model_tr.score(dt_train_balanced,labels_train_balanced))
                            
                            # if tune:
                                # print(model_tr.best_params_)
                            # print(model_tr.best_score_)
                        # else:
                        #     print("Training set loss: %f" % model_tr.loss_)
                            
                        accc=model_tr.score(dt_test,labels_test)
                        print("Test score: %f" % accc)
                        if cls_model=='lr' and col == 'CP':
                            # print(model_tr.coef_)
                            coefs = model_tr.best_estimator_.coef_
                            coefs_all[p].append(coefs)
                            coefs_mean = abs(coefs.mean(axis=0))
                            coefs_mean = coefs_mean / np.sum(coefs_mean)
                            coefs = pd.DataFrame(coefs_mean.reshape(1,len(dt_modality[1])), columns = dt_modality[1], index = ['Fold_'+str(i)])
                            coefs_all_cvs[p] = pd.concat([coefs_all_cvs[p], coefs], axis=1)
                        # print(model_tr.predict(dt_test))
                        # accc=f1_score(labels_test,model_tr.predict(dt_test), average='weighted')        
                        moa_pred_res.loc[test_index,col]=model_tr.predict(dt_test)
                #         moa_pred_res.loc[filteredMOAs.loc[test_index,'Compounds'].unique()[0],col]=accc*100
                        probs.append(model_tr.predict_proba(dt_test))

                    if configs.moa.run_l1k:
                        prob_sum = probs[0]
                        for pro in range(1,len(probs)):
                            prob_sum+=probs[pro]                    
                        labels_lateFusion=model_tr.classes_[np.argmax(prob_sum/len(probs),axis=1)]
                        moa_pred_res.loc[test_index,'Late Fusion']=labels_lateFusion

                    moa_pred_res.loc[test_index,'Metadata_moa_num']=labels_test

                    # del model_tr
                    if 'torch' in cls_model:
                        torch.cuda.empty_cache()
                        
                #     f1_score(labels_test,labels_lateFusion, average='weighted')*100
                #     accuracy_score(labels_test,labels_lateFusion)*100

                moa_pred_res['Metadata_moa_num']=moa_pred_res.Metadata_moa_num.apply(lambda x: int(x[0]) if type(x)==list else x)  
                moa_pred_res['Metadata_moa_with_n']=moa_pred_res.Metadata_moa_with_n.apply(lambda x: int(x[0]) if type(x)==list else x)    
                moa_pred_res['exp_num']=configs.general.exp_num
                # print(moa_pred_res.mean())
                if configs.moa.tune:
                    if configs.moa.tune_first_fold:
                        tuneStr='-t'
                    else:
                        tuneStr='-ta'
                else:
                    tuneStr=''
                
                profile_abrv = profTypeAbbrev[j]
                sheetname = f'{profile_abrv}-{cls_model}{tuneStr}'
                
                f1_exp_name = f'{moa_csv_fileName}-{cls_model}'
                f1s = compute_f1_scores(moa_pred_res)
                for m in f1s.keys():
                    f1s[m]['exp_name'] = f1_exp_name
                    # f[m]['data_rep'] = m
                    f1s[m]['cls_model'] = cls_model
                    f1s[m]['n_features'] = len(methods[p]['cp_scaled'].columns)
                    f1s[m]['n_samples'] = len(methods[p]['cp_scaled'].index)
                    f1s[m]['n_classes'] = len(np.unique(filteredMOAs['Metadata_moa_num'].values))
                    f1s[m]['profile_type'] = p
                    f1s[m]['filter_groups'] = filter_abrv
                    f1s[m]['by_dose'] = configs.eval.by_dose
                    f1s[m]['exp_num'] = configs.general.exp_num
                    f1s[m]['min_samples'] = configs.moa.min_samples
                    f1s[m]['folds'] = splits
                    f1s[m]['normalize_by_all'] = configs.eval.normalize_by_all
                    
    
                if f1_exp_name is not None:
                    index_names = [f1_exp_name]
                else:
                    index_names = [0]

                f1s_df = pd.DataFrame.from_dict(f1s).T.reset_index().rename(columns={'index':'data_rep'})
                # f1s_df = f1s_df.rename(index ={0:exp_name}).reset_index()
                


                if not configs.general.debug_mode:
                    
                    # save slice_id results to pickle
                    pkl_dir = f'{save_dir}/{sheetname}'
                    os.makedirs(pkl_dir,exist_ok=True)
                    cur_dest = os.path.join(pkl_dir,f'{configs.general.exp_num}.pickle')
                    print(f'saving results for slice {configs.general.exp_num} to {cur_dest}')
                    with open(cur_dest, 'wb') as handle:
                        pickle.dump(moa_pred_res, handle, protocol=pickle.HIGHEST_PROTOCOL)

                    write_dataframe_to_excel(save_path,sheetname,moa_pred_res,append_new_data_if_sheet_exists=True)
                    write_dataframe_to_excel(summary_csv_path,'results',f1s_df.reset_index(),append_new_data_if_sheet_exists=True)

                    coefs_dir = f'{save_dir}/coefs'
                    os.makedirs(coefs_dir,exist_ok=True)

                    # for p in coefs_all_cvs.keys():

                    coefs_method = coefs_all_cvs[p]
                    coefs_for_plot = coefs_method.mean().sort_values(ascending=False)[:15]
                    coefs_for_plot = coefs_for_plot.sort_values(ascending=True)
                    coefs_for_plot.plot(kind='barh', title='Feature Importances',figsize=(8,9))
                    # coef_mean_sorted = np.sort(coef_mean)[::-1]
                    # coefs = pd.DataFrame(coef_mean, columns = dt_modality[1], index = "coefs")
                    # coefs = coefs.sort_values(by = "coefs", ascending = False)
                    # coefs_for_plot.plot(kind='barh', title='Feature Importances',figsize=(8,9))
                    plt.ylabel('Feature Importance Score')
                    # plt.xticks(rotation=45)
                    plt.tight_layout()
                    # a = coefs_df.mean().sort_values(ascending=False)[:10]
                    # plt.ylabel('Feature Importance Score')
                    
                    plt.savefig(f'{coefs_dir}/coefs_{p}_{configs.eval.slice_id}.png')
                    plt.close()
                    coefs_method.to_pickle(f'{coefs_dir}/coefs_{p}.pickle')
                    coefs_method.to_csv(f'{coefs_dir}/coefs_{p}.csv')
                    
                    # save dict as pickle
                    pickle.dump(coefs_all, open(f'{coefs_dir}/coefs_all.pickle', 'wb'))
                else:
                    res[sheetname] = moa_pred_res
                    # saveAsNewSheetToExistingFile(filename,moa_pred_res,'fC-'+dataset+'-'+profTypeAbbrev+'-'+f+'-preds-'+cls_model+'-ht-sgkf-10f')

############################################################################################################
    
def compute_f1_scores(moa_pred_res, exp_name = None,run_l1k = False):
    if run_l1k:
        methods= ['CP','GE','Early Fusion','Late Fusion']
    else:
        methods= ['CP','Early Fusion']
    f1s={}
    # f1s_mean={}
    # f1_std = {}
    print('before nan removal', moa_pred_res.shape)
    moa_pred_res.dropna(subset=['Fold'],inplace=True)
    print('after nan removal', moa_pred_res.shape)
    folds=moa_pred_res['Fold'].unique().tolist()
    for dd in methods:
        mean_name = f'{dd}_mean'

        f1s[dd] = {}
        # f1s_mean[dd] = {}
        # f1_std[dd] = {}
        f1_methods_folds = []
        for f in folds:
                df_f=moa_pred_res[moa_pred_res['Fold']==f];
                    #    ,'RGCCA_EarlyFusion']:
                scor_vals=df_f[dd].apply(lambda x: int(eval(x)[0]) if type(x)==str else x)    
                # print(f'data type: {data_type}, model: {model_name}, modality: {dd}, fold: {f}')
                f1=f1_score(scor_vals,df_f.Metadata_moa_num.values, average='macro')
                f1_methods_folds.append(f1)
        
        f1s_mean = np.mean(f1_methods_folds)
        f1_std = np.std(f1_methods_folds)
        # f1s[dd]['f1_all'] = f1_methods_folds
        f1s[dd]['f1_mean'] = f1s_mean
        f1s[dd]['f1_std'] = f1_std

    return f1s

############################################################################################################
def load_data_filtered_by_rep_corr(configs, data_reps, filter_repCorr_params, pertColName='PERT',run_l1k = False):

    moa_col='Metadata_MoA'
    moa_col_name='Metadata_moa_with_n'
    methods = {}
    

    for p in data_reps:
        methods[p] = {}

        # if there is dose information and not running by dose, use only highest dose
        # loads merged data by doses if filter_by_highest_dose=True, and use only highest dose
        if not configs.eval.by_dose and DS_INFO_DICT[configs.general.dataset]['has_dose'] and configs.eval.filter_by_highest_dose:
            mergProf_treatLevel, cp_features,l1k_features = \
                read_paired_treatment_level_profiles(configs.general.base_dir,configs.general.dataset,p,filter_repCorr_params,configs.eval.normalize_by_all,configs.general.exp_name,by_dose = False, filter_by_highest_dose=configs.eval.filter_by_highest_dose)
        else:
            mergProf_treatLevel, cp_features,l1k_features = \
                read_paired_treatment_level_profiles(configs.general.base_dir,configs.general.dataset,p,filter_repCorr_params,configs.eval.normalize_by_all,configs.general.exp_name,by_dose = configs.eval.by_dose)


        ##################################
        if configs.general.dataset=='LINCS':
            mergProf_treatLevel[moa_col]=mergProf_treatLevel['Metadata_moa']
            mergProf_treatLevel.loc[mergProf_treatLevel['Metadata_moa'].isnull(),moa_col]=\
            mergProf_treatLevel.loc[mergProf_treatLevel['Metadata_moa'].isnull(),'moa'].str.lower()    
            mergProf_treatLevel['Compounds']=mergProf_treatLevel['PERT'].str[0:13]
        
        elif configs.general.dataset=='CDRP-bio':
            mergProf_treatLevel[moa_col]=mergProf_treatLevel['Metadata_moa'].str.lower()
            mergProf_treatLevel['Compounds']=mergProf_treatLevel['PERT'].str[0:13]

        # if configs.moa.filter_by_pr:
        #     mergProf_treatLevel = mergProf_treatLevel[mergProf_treatLevel['PERT'].isin(profile_type_pr['old_ind'])]
        #     print('filtered by pr', mergProf_treatLevel.shape[0])
        #     print('number of features', len(cp_features))

        # mergProf_repLevel,mergProf_treatLevel,l1k_features,cp_features,pertColName=readMergedProfiles(dataset,profileType,nRep)
        # cp_features,l1k_features=cp_features.tolist(),l1k_features.tolist()
        # mergProf_repLevel['Compounds']=mergProf_repLevel['PERT'].str[0:13]

        # if profileLevel=='replicate':
        #     l1k=mergProf_repLevel[[pertColName]+l1k_features]
        #     cp=mergProf_repLevel[[pertColName]+cp_features]
        # elif profileLevel=='treatment':

        l1k=mergProf_treatLevel[[pertColName,'Compounds',moa_col]+l1k_features]
        cp=mergProf_treatLevel[[pertColName,'Compounds',moa_col]+cp_features]
        methods[p]['l1k']= l1k.copy()
        methods[p]['cp']= cp.copy()
        
        # profiles[p]['cp']= pd.read_csv(profiles[p]['cp_path'],compression = 'gzip')
        # cp_features = profiles[p]['cp'].columns[profiles[p]['cp'].columns.str.contains("Cells_|Cytoplasm_|Nuclei_")]

        ################## stadanrdize the data (alreay done in the preprocessing step) ##################
        # scaler_ge = preprocessing.StandardScaler()
        # scaler_cp = preprocessing.StandardScaler()

        l1k_scaled=methods[p]["l1k"].copy()
        # l1k_scaled[l1k_features] = scaler_ge.fit_transform(l1k_scaled[l1k_features].values)
        cp_scaled=methods[p]["cp"].copy()
        # cp_scaled[cp_features] = scaler_cp.fit_transform(cp_scaled[cp_features].values.astype('float64'))

        # cp_scaled[cp_scaled[cp_features]>z_trim]=z_trim
        cp_features = list(cp_scaled.columns[cp_scaled.columns.str.contains("Cells_|Cytoplasm_|Nuclei_")])
        if configs.eval.z_trim is not None:
            cp_scaled[cp_features] = np.where(cp_scaled[cp_features]>configs.eval.z_trim,configs.eval.z_trim ,cp_scaled[cp_features])
            cp_scaled[cp_features] = np.where(cp_scaled[cp_features]<-configs.eval.z_trim,-configs.eval.z_trim ,cp_scaled[cp_features])
        # cp_scaled[cp_scaled[cp_features]<-z_trim]=-z_trim 
        if 1:
            cp_scaled[cp_features] =preprocessing.MinMaxScaler(feature_range=(0, 1)).fit_transform(cp_scaled[cp_features].values)   
            l1k_scaled[l1k_features] =preprocessing.MinMaxScaler(feature_range=(0, 1)).fit_transform(l1k_scaled[l1k_features].values) 

        methods[p]["l1k_scaled"] = l1k_scaled
        methods[p]["cp_scaled"] = cp_scaled      

        merged_scaled=pd.concat([methods[p]["cp_scaled"], methods[p]["l1k_scaled"]], axis=1)
        merged_scaled = merged_scaled.loc[:,~merged_scaled.columns.duplicated()]    
        merged_scaled['Compounds']=merged_scaled['PERT'].str[0:13]
        methods[p]["merged_scaled"]=merged_scaled

    ################ Filter MoAs with less than nSamplesMOA samples
    # ATTENTION: filtered MoAs is computed using the last profileType!!!! this means that all data_reps should have the same MoAs.

    nSamplesforEachMOAclass=mergProf_treatLevel.groupby(['Compounds']).sample(1).groupby([moa_col]).size().\
    reset_index().rename(columns={0:'size'}).sort_values(by=['size'],ascending=False).reset_index(drop=True)


    nSamplesforEachMOAclass2=mergProf_treatLevel.groupby([moa_col]).size().reset_index().rename(columns={0:'size'}).sort_values(by=['size'],ascending=False).reset_index(drop=True)

    listOfSelectedMoAs=nSamplesforEachMOAclass[nSamplesforEachMOAclass['size']>configs.moa.min_samples][moa_col].tolist()
    configs.general.logger.info('If we filter to MoAs which have more than',configs.moa.min_samples+1,' compounds in their category, ',\
        len(listOfSelectedMoAs),' out of ',nSamplesforEachMOAclass.shape[0] ,' MoAs remain.')



    multi_label_MoAs=[l for l in listOfSelectedMoAs if '|' in l]
    configs.general.logger.info('There are ',len(listOfSelectedMoAs),'MoA categories, which out of them ',len(multi_label_MoAs),\
        ' have multi labels and is removed')

    listOfSelectedMoAs=[ele for ele in listOfSelectedMoAs if ele not in multi_label_MoAs]

    
    filteredMOAs=merged_scaled[merged_scaled[moa_col].isin(listOfSelectedMoAs)].reset_index(drop=True)

    n_samples = filteredMOAs.groupby(['Metadata_MoA']).size().sort_values(ascending=False)
    filteredMOAs['Metadata_moa_n'] = filteredMOAs['Metadata_MoA'].map(n_samples)
    filteredMOAs[moa_col_name] = filteredMOAs['Metadata_MoA'] + ' (n=' + filteredMOAs['Metadata_moa_n'].astype(str) + ')'

    listOfSelectedMoAs_with_n=filteredMOAs[moa_col_name].unique().tolist()
    le = preprocessing.LabelEncoder()
    le.fit(listOfSelectedMoAs_with_n)

    for p in data_reps:
        merged_scaled=methods[p]["merged_scaled"]
        filteredMOAs=merged_scaled[merged_scaled[moa_col].isin(listOfSelectedMoAs)].reset_index(drop=True)
        filteredMOAs['Metadata_moa_n'] = filteredMOAs['Metadata_MoA'].map(n_samples)
        filteredMOAs[moa_col_name] = filteredMOAs['Metadata_MoA'] + ' (n=' + filteredMOAs['Metadata_moa_n'].astype(str) + ')'

        filteredMOAs['Metadata_moa_num']=le.transform(filteredMOAs[moa_col_name].tolist())

        # min_samples=0 and union
        configs.general.logger.info("There are ", filteredMOAs.shape[0],"samples across different doses of ",filteredMOAs['Compounds'].unique().shape[0] ,\
            "compounds", ", for ",filteredMOAs["Metadata_MoA"].unique().shape[0], "MoAs")

        nClasses = filteredMOAs['Compounds'].unique().shape[0]

        n_samples = filteredMOAs.groupby(['Metadata_MoA']).size().sort_values(ascending=False)
        filteredMOAs['Metadata_moa_n'] = filteredMOAs['Metadata_MoA'].map(n_samples)
        filteredMOAs['Metadata_moa_with_n'] = filteredMOAs['Metadata_MoA'] + ' (n=' + filteredMOAs['Metadata_moa_n'].astype(str) + ')'
        filteredMOAs.head()

        ########################################################### Prepare data for evaluation ###########################################################
        
        methods[p]["data4eval"] =[
            # run classification based on cp data only
            [methods[p]["cp_scaled"][methods[p]["cp_scaled"][moa_col].isin(listOfSelectedMoAs)].reset_index(drop=True),cp_features],
            # run classification based on fused data of l1k and cp
            [filteredMOAs,cp_features+l1k_features]
            ]
         
    if configs.moa.do_fusion:
        methods = create_fused_ad_rep(configs, methods, listOfSelectedMoAs, l1k_features, cp_features, moa_col=moa_col)
        # if run_l1k:
            # methods[p]["data4eval"].append([methods[p]["l1k_scaled"][methods[p]["l1k_scaled"][moa_col].isin(listOfSelectedMoAs)].reset_index(drop=True),l1k_features])

            # methods['fusion']['data4eval'].append([filteredMOAs,cp_features+l1k_features])
    
    for p in methods.keys():

        # run classification based on l1k data only
        if run_l1k:
            methods[p]['data4eval'].append([methods[p]['l1k_scaled'][methods[p]['l1k_scaled'][moa_col].isin(listOfSelectedMoAs)].reset_index(drop=True),l1k_features])

        
        # methods.append([filteredMOAs,cp_features+l1k_features])
        

    return methods, filteredMOAs

#########################data4###################################################################################
   
def create_fused_ad_rep(configs, methods, listOfSelectedMoAs, l1k_features, cp_features, moa_col='Metadata_MoA'):

    profileTypes = list(methods.keys())
    methods['fusion'] = {}
    meta_feautres = [feat for feat in methods[profileTypes[0]]['cp_scaled'].columns if 'Metadata' in feat]
    methods['fusion']['cp_scaled'] = pd.merge(methods[profileTypes[0]]["cp_scaled"], methods[profileTypes[1]]["cp_scaled"], on=['PERT','Compounds','Metadata_MoA'], how='inner', suffixes=(f'_{ABRVS[profileTypes[0]][0]}', f'_{ABRVS[profileTypes[1]][0]}'))
    # profiles['fusion']['cp_scaled']['PERT'] = profiles['fusion']['cp_scaled'][f'PERT_{ABRVS[profileTypes[0]][0]}']
    # profiles['fusion']['cp_scaled']['Compounds'] = profiles[profileTypes[0]]['cp_scaled']['Compounds']
    
    fused_features = [feat for feat in methods['fusion']['cp_scaled'].columns if 'Cells_' in feat or 'Cytoplasm_' in feat or 'Nuclei_' in feat]
    
    # profiles['fusion']['cp_scaled'] = pd.concat([profiles[p]["cp_scaled"][cp_features] for p in profileTypes], axis=1)
    # cp_meta_feautres = [feat for feat in profiles['fusion']['cp_scaled'].columns if 'Metadata' in feat]
    # profiles['fusion']['cp_scaled'][meta_feautres]= profiles['normalized_variable_selected_baseline']['cp_scaled'][meta_feautres]
    methods['fusion']['l1k_scaled'] = methods[profileTypes[0]]["l1k_scaled"].copy()
    # l1k_meta_features = [feat for feat in profiles['fusion']['l1k_scaled'].columns if 'Metadata' in feat]
    # profiles['fusion']['l1k_scaled'][meta_feautres]= profiles['normalized_variable_selected_baseline']['l1k_scaled'][meta_feautres]
    # profiles['fusion']['merged_scaled'] = pd.concat([profiles[p]["merged_scaled"] for p in profileTypes], axis=1)
    fused_merged_scaled = pd.merge(methods['fusion']['cp_scaled'], methods['fusion']['l1k_scaled'], on=['PERT','Compounds','Metadata_MoA'], how='inner')
    methods['fusion']['data4eval'] = [[methods['fusion']["cp_scaled"][methods['fusion']["cp_scaled"][moa_col].isin(listOfSelectedMoAs)].reset_index(drop=True),fused_features],\
        # [methods[profileTypes[0]]["l1k_scaled"][methods['fusion']["l1k_scaled"][moa_col].isin(listOfSelectedMoAs)].reset_index(drop=True),l1k_features]]
            [fused_merged_scaled[fused_merged_scaled[moa_col].isin(listOfSelectedMoAs)].reset_index(drop=True),fused_features+l1k_features]]
    return methods


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

    configs = set_configs()
    if len(sys.argv) <2:
        exp_name = '0512_t_dd'
        configs.general.exp_name = exp_name
        configs.general.dataset = 'LINCS'
        # configs.general.dataset = 'CDRP-bio'
        configs.data.profile_type = 'normalized_variable_selected'
        configs.data.profile_type = 'augmented'
        configs.data.modality = 'CellPainting'
        configs.eval.by_dose = False
        configs.eval.filter_by_highest_dose = True
        slice_id = 0
        configs.general.flow ='concat_moa_res'
        # configs.general.flow ='run_moa'
        configs.moa.do_all_filter_groups = True
        
        configs.general.debug_mode = False
        configs.moa.min_samples = 4
        configs.moa.tune = True
        configs.eval.normalize_by_all = True
        configs.eval.run_dose_if_exists = True
        configs.moa.moa_dirname = 'moa_single'
        configs.moa.folds = 5
        configs.eval.z_trim = 8

    else:
        # exp_name = sys.argv[1]
        # configs = set_configs()
        profileType = configs.data.profile_type
        dataset = configs.general.dataset
        slice_id = configs.eval.slice_id
        exp_name = configs.general.exp_name
        modality = configs.data.modality
        by_dose = configs.eval.by_dose
        configs.general.exp_num = slice_id
        configs.general.flow = 'run_moa'


    configs = set_paths(configs)

    slice_id = int(slice_id)
    ################################################
    # datasets=['TAORF','LUAD','LINCS', 'CDRP-bio']
    
    datasets = [configs.general.dataset]
    # DT_kfold={'LUAD':10, 'TAORF':5, 'LINCS':25, 'CDRP-bio':6,'CDRP':40}

    base_dir= '/sise/assafzar-group/assafzar/genesAndMorph'

    data_dir =  get_cp_dir(base_dir,datasets[0],configs.data.profile_type)

    ################################################
    # filtering to compounds which have high replicates for both GE and CP datasets

    # 'highRepUnion','highRepOverlap','', 'onlyCP'
    filter_perts='highRepUnion'
    # filter_perts=''
    # filter_by_pr = False
    configs.moa.rep_corr_fileName='RepCorrDF'

    # configs.moa.filter_by_pr = filter_by_pr
    configs.moa.filter_perts = filter_perts
    # debug=False

    # Types = ['ae','ae_error', 'baseline']
    Types = ['ae_diff', 'baseline']
    
    methods = [f'{configs.data.profile_type}_{t}' for t in Types]
        
    rand_ints_list = random.sample(range(0,1000), configs.moa.n_exps)
    print(rand_ints_list)
    # exp_num = slice_id
    configs.moa.exp_seed = rand_ints_list[configs.general.exp_num]

    set_seed(configs.moa.exp_seed)

    print(f'slice_id is {slice_id}, exp_seed is {configs.moa.exp_seed}')

    profile_types = [p for p in ['augmented','normalized_variable_selected'] if any(p in string for string in os.listdir(configs.general.output_exp_dir))]
    # do_dose_if_exists = True
    # normalize_by_alls = [True]

    if DS_INFO_DICT[configs.general.dataset]['has_dose']:
    
        if configs.eval.run_dose_if_exists:
            doses = [False,True]
        else:
            doses = [configs.eval.by_dose]
    else:
        doses = [False]
        
    # for n in normalize_by_alls:
        
        # for f in filter_combs:
    for d in doses:
        for p in profile_types:
            # if configs.general.dataset == 'CDRP-bio' and d:
                # continue
            if configs.general.dataset == 'LINCS' and p == 'normalized_variable_selected':
                continue
            # print( f'running {p} {d} normalize{n}')
            
        # try:
            # configs.eval.normalize_by_all = n
            configs.data.profile_type = p
            configs.eval.by_dose = d
            # configs.moa.filter_groups = f
        # configs = set_paths(configs)
            main(configs, methods)
            # except:
                # print(f'failed for {p}, {d}')
                    
        