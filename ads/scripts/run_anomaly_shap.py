import numpy as np
# import scipy.spatial
import pandas as pd
import sys
# import sklearn.decomposition
import matplotlib.pyplot as plt
# import keras
from sklearn import preprocessing
import os
import copy
# %matplotlib inline
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.style as style
import shap
import pickle

currentdir = '/sise/home/alonshp/AnomalyDetectionScreening'

# style.use('seaborn-whitegrid')
sys.path.insert(0, '../') 
print(sys.path)
sys.path.insert(0, os.getcwd())
sys.path.insert(0, currentdir)
from interpret_layer.shap_anomalies import ExplainAnomaliesUsingSHAP,run_anomaly_shap
from utils.readProfiles import *
from utils.global_variables import ABRVS, DS_INFO_DICT
from utils.general import write_dataframe_to_excel, add_exp_suffix
from data_layer.data_utils import load_data, pre_process,to_dataloaders
from old.Classifier import *
from scripts.classify_moa import get_moa_dirname, remove_classes_with_few_moa_samples, remove_multi_label_moa
from scripts.create_null_distribution import main as create_null_distributions


from utils.general import set_configs, set_paths,get_configs
from scripts.run_ad import train_autoencoder, test_autoencoder, post_process_anomaly_and_save, load_checkpoint, test_autoencoder2

plt.rcParams.update(plt.rcParamsDefault)
plt.style.use('seaborn-v0_8-colorblind')
sns.set_theme(style='ticks',context='paper', font_scale=1.5)

#### from https://github.com/ronniemi/explainAnomaliesUsingSHAP/blob/master/ExplainAnomaliesUsingSHAP.py 

if len(sys.argv)>1:
    run_parrallel =True
    configs = set_configs()
    # print(sys.argv)
    slice_id = configs.general.slice_id
    print('slice_id: ',slice_id)
    only_plot = False
    exp_name = configs.general.exp_name
else:
    exp_name = 'final_154_nvs_t'
    exp_name = 'final_184_t'
    exp_name = '2142'
    run_parrallel =False
    configs = set_configs(exp_name)

    # if not configs.general.from_file:
    configs.general.run_all_datasets = False
    configs.general.debug_mode = False
    
    configs.data.run_data_process = False
    configs.model.tune_hyperparams = False
    configs.data.feature_select = True
    configs.general.overwrite_experiment = False
    configs.model.encoder_type = 'deep'
    # configs.data.overwrite_data_creation = True

    # dataset : CDRP, CDRP-bio, LINCS, LUAD, TAORF
    configs.general.flow = 'run_ad'
    # configs.general.flow = 'calc_metrics'
    configs.general.dataset = 'LINCS'
    configs.general.dataset = 'CDRP-bio'
    # configs.general.dataset = 'TAORF'
    configs.data.corr_threshold = 0.9


    configs.data.modality = 'CellPainting'
        # configs.data.modality = 'L1000'

        # configs.data.norm_method = 'mad_robustize'
    configs.data.profile_type = 'normalized_variable_selected'
    # configs.data.profile_type = 'augmented'
    
    configs.eval.rand_reps=1


configs = set_paths(configs)
config_path = os.path.join(configs.general.output_exp_dir,'args.pkl')
if os.path.exists(config_path):
    print('Experiment already exists! Loading configs from file')
    configs = get_configs(configs.general.output_exp_dir)
# configs = get_configs(configs.general.output_exp_dir)

# configs.eval.normalize_by_all = True
run_anomaly_shap(configs,filter_non_reproducible=True)

stop = True

if stop:
    sys.exit()


if run_parrallel:
    slice_id = configs.general.slice_id
    print('slice_id: ',slice_id)
    datasets = ['CDRP-bio','LINCS','LUAD','TAORF']
    configs.general.dataset = datasets[slice_id]
# configs.moa.moa_dirname = 'MoAprediction_single'

###### load data #######
data , __ = load_data(configs.general.base_dir,configs.general.dataset,configs.data.profile_type, modality=configs.data.modality)
data_preprocess,features =  pre_process(data,configs,data_reps=['ae_diff','baseline'])
dataloaders = to_dataloaders(data_preprocess,configs.model.batch_size,features)
test_dataloaders = list(dataloaders.keys())[2:]


##### run autoencoder ######
model =None
if load_model:
    model = load_checkpoint(configs.model.ckpt_dir)

if model is not None:
    print('Loaded model')
else:
    print('No model found,training new model')
    model = train_autoencoder(dataloaders, features, configs)


#### run autoencoder on test data ####    
test_path = os.path.join( configs.general.output_exp_dir,f'replicate_level_{configs.data.modality_str}_{configs.data.profile_type}_ae_diff.csv')
if os.path.exists(test_path) and load_model:
    print('Loading predictions from file')
    diffs_normalized = pd.read_csv(test_path, compression='gzip')

else:
    print('Running autoencoder on test data')


# preds = test_autoencoder(model, dataloaders)
# preds_ctrl = preds['test_ctrl']
# preds_treat = preds['test_treat']
# # preds_treat = preds.loc[preds['Metadata_set']=='test_treat',:][features]
# preds_ctrl2 = test_autoencoder2(model, data_preprocess[data_preprocess['Metadata_set'] == 'test_ctrl'], features, configs)
# preds_treat2 = test_autoencoder2(model, data_preprocess[data_preprocess['Metadata_set'] == 'test_treat'], features, configs)

# # assert np.allclose(preds_ctrl.sum(axis=1), preds_ctrl2[features].values.sum(axis=1))
# # assert np.allclose(preds_treat[features].sum(axis=1), preds_treat2[features].values.sum(axis=1))
# # sys.exit()
# # diffs_ctrl = np.power(preds['test_ctrl'] - data[data['Metadata_set'] == 'test_ctrl'][features],2)
# # diffs_treat = np.power(preds['test_treat'] - data[data['Metadata_set'] == 'test_treat'][features],2)
# diffs_ctrl = preds_ctrl - data_preprocess[data_preprocess['Metadata_set'] == 'test_ctrl'][features]
# diffs_treat = preds_treat - data_preprocess[data_preprocess['Metadata_set'] == 'test_treat'][features]

# diffs = save_treatments(data_preprocess, diffs_ctrl,diffs_treat, configs.general.output_exp_dir, 
#     f'replicate_level_{configs.data.modality_str}_{configs.data.profile_type}_recon_errors', configs, features, normalize_reps=False)

# diffs_normalized= save_treatments(data_preprocess, diffs_ctrl,diffs_treat, configs.general.output_exp_dir, 
#     f'replicate_level_{configs.data.modality_str}_{configs.data.profile_type}_ae_diff', configs, features)


plt.close('all')

# sys.exit()
X_control =  data_preprocess[data_preprocess['Metadata_set'] == 'test_ctrl'].reset_index(drop=True)
X_test = data_preprocess[data_preprocess['Metadata_set'] == 'test_treat'].dropna(subset=['Metadata_moa'])

# if group_by_col =='moa_col':
cpd_col = DS_INFO_DICT[configs.general.dataset][configs.data.modality]['cpd_col']
group_by = 'moa'
# elif group_by == 'cpd':
if not DS_INFO_DICT[configs.general.dataset]['has_moa']:
    group_by = 'cpd'
group_by_col = DS_INFO_DICT[configs.general.dataset][configs.data.modality][f'{group_by}_col']

X_control_normalized = X_control.copy()
X_test_normalized = X_test.copy()

# scaler = preprocessing.StandardScaler().fit(X_control_normalized[features])
# X_control_normalized.loc[:, features] = scaler.transform(X_control_normalized[features])
# X_test_normalized.loc[:, features] = scaler.transform(X_test_normalized[features])

# set all moa cols with lowercase
X_test[group_by_col] = X_test[group_by_col].str.lower()
X_test_processed, _, _ = remove_classes_with_few_moa_samples(X_test, min_samples = 5)  # remove classes with less than 5 samples
X_test_processed, _ = remove_multi_label_moa(X_test_processed)
X_test_processed = X_test_processed.reset_index(drop=True)


# save a .txt file of all features
# with open(f'{configs.general.res_dir}/features.txt', 'w') as f:
    # for item in features:
        # f.write("%s\n" % item)

do_non_reproducible = False
if do_non_reproducible:
    
    # save_dir = os.path.join(configs.general.res_dir, 'shap_vis_single')
    # os.makedirs(save_dir, exist_ok=True)

    # exp_model = ExplainAnomaliesUsingSHAP(model,features=features, num_anomalies_to_explain=1)
    # all_sets_explaining_features = exp_model.explain_unsupervised_data(x_train=X_control, 
    #                                                             x_explain=X_test_processed,
    #                                                             autoencoder=model,
    #                                                             return_shap_values=True,
    #                                                             # group_by='Metadata_moa',
    #                                                             save_dir=save_dir)

    # for key, value in all_sets_explaining_features.items():
    #     print(f"Record index: {key}")
    #     print(f"Explanation features set: {value}")

    # save_dir = os.path.join(configs.general.res_dir, 'shap_vis_moa_all')
    # os.makedirs(save_dir, exist_ok=True)

    # group_by = 'moa'
    # group_by_col = DS_INFO_DICT[configs.general.dataset][configs.data.modality][f'{group_by}_col']
    
    # exp_model = ExplainAnomaliesUsingSHAP(model,features=features, num_anomalies_to_explain=1)

    # all_sets_explaining_features = exp_model.explain_unsupervised_data(x_train=X_control, 
    #                                                             x_explain=X_test_processed,
    #                                                             autoencoder=model,
    #                                                             return_shap_values=True,
    #                                                             group_by=group_by_col,
    #                                                             save_dir=save_dir)
    # exp_model = ExplainAnomaliesUsingSHAP(model,features=features, num_anomalies_to_explain=1)
    
    only_top_moa_data = X_test_processed[X_test_processed[group_by_col].isin(top_moa_cdrp)]
    only_top_moa_data = only_top_moa_data.reset_index(drop=True)

    save_dir = os.path.join(configs.general.res_dir, 'shap_vis_moa_top')
    os.makedirs(save_dir, exist_ok=True)

    exp_model = ExplainAnomaliesUsingSHAP(model,features=features, num_anomalies_to_explain=3)
    all_sets_explaining_features = exp_model.explain_unsupervised_data(x_train=X_control, 
                                                                x_explain=only_top_moa_data,
                                                                autoencoder=model,
                                                                return_shap_values=True,
                                                                group_by=group_by_col,
                                                                save_dir=save_dir)
    
    
    only_new_moa_data = X_test_processed[X_test_processed[group_by_col].isin(new_moa_cdrp)]
    only_new_moa_data = only_new_moa_data.reset_index(drop=True)
    save_dir = os.path.join(configs.general.res_dir, 'shap_vis_moa_new')
    os.makedirs(save_dir, exist_ok=True)

    exp_model = ExplainAnomaliesUsingSHAP(model,features=features, num_anomalies_to_explain=3)
    all_sets_explaining_features = exp_model.explain_unsupervised_data(x_train=X_control, 
                                                                x_explain=only_new_moa_data,
                                                                autoencoder=model,
                                                                return_shap_values=True,
                                                                group_by=group_by_col,
                                                                save_dir=save_dir)
    


exp_suffix = add_exp_suffix(configs.data.profile_type,configs.eval.by_dose,configs.eval.normalize_by_all)
corr_path = f'{configs.general.res_dir}reproducible_cpds{exp_suffix}.csv'

if not os.path.exists(corr_path):
    create_null_distributions(configs,data_reps=['ae_diff','baseline'])
    # sys.exit()
repcorr_df = pd.read_csv(corr_path)

if 'reproducible_acl' not in repcorr_df.columns:
    reproduce_col = 'reproducible_cl'
else:
    reproduce_col = 'reproducible_acl'
only_reproducible = list(repcorr_df[repcorr_df[reproduce_col] == True]['cpd'])
X_test_reproducible = X_test_processed[X_test_processed[cpd_col].isin(only_reproducible)]
X_test_reproducible = X_test_reproducible.reset_index(drop=True)

if group_by != 'moa':
    save_dir = os.path.join(configs.general.res_dir, 'shap_vis_reproducible')
    os.makedirs(save_dir, exist_ok=True)


    exp_model = ExplainAnomaliesUsingSHAP(model,features=features, num_anomalies_to_explain=6)
    all_sets_explaining_features = exp_model.explain_unsupervised_data(x_train=X_control, 
                                                                x_explain=X_test_reproducible,
                                                                autoencoder=model,
                                                                return_shap_values=True,
                                                                # group_by=group_by_col,
                                                                save_dir=save_dir)


    save_dir = os.path.join(configs.general.res_dir, 'shap_vis_reproducible_moa')
    os.makedirs(save_dir, exist_ok=True)

    exp_model = ExplainAnomaliesUsingSHAP(model,features=features, num_anomalies_to_explain=4)
    all_sets_explaining_features = exp_model.explain_unsupervised_data(x_train=X_control, 
                                                                x_explain=X_test_reproducible,
                                                                autoencoder=model,
                                                                return_shap_values=True,
                                                                group_by=group_by_col,
                                                                save_dir=save_dir)
else:
    only_top_moa_data = X_test_reproducible[X_test_reproducible[group_by_col].isin(top_moa_cdrp)]
    only_top_moa_data = only_top_moa_data.reset_index(drop=True)

    save_dir = os.path.join(configs.general.res_dir, 'shap_vis_reproducible_moa_top')
    os.makedirs(save_dir, exist_ok=True)

    exp_model = ExplainAnomaliesUsingSHAP(model,features=features, num_anomalies_to_explain=3)
    all_sets_explaining_features = exp_model.explain_unsupervised_data(x_train=X_control, 
                                                                x_explain=only_top_moa_data,
                                                                autoencoder=model,
                                                                return_shap_values=True,
                                                                group_by=group_by_col,
                                                                save_dir=save_dir)
    
    
    only_new_moa_data = X_test_reproducible[X_test_reproducible[group_by_col].isin(new_moa_cdrp)]
    only_new_moa_data = only_new_moa_data.reset_index(drop=True)
    save_dir = os.path.join(configs.general.res_dir, 'shap_vis_reproducible_moa_new')
    os.makedirs(save_dir, exist_ok=True)

    exp_model = ExplainAnomaliesUsingSHAP(model,features=features, num_anomalies_to_explain=2)
    all_sets_explaining_features = exp_model.explain_unsupervised_data(x_train=X_control, 
                                                                x_explain=only_new_moa_data,
                                                                autoencoder=model,
                                                                return_shap_values=True,
                                                                group_by=group_by_col,
                                                                save_dir=save_dir)
    
    group_by = 'cpd'
    group_by_col = DS_INFO_DICT[configs.general.dataset][configs.data.modality][f'{group_by}_col']
    save_dir = os.path.join(configs.general.res_dir, 'shap_vis_reproducible_cpd')
    os.makedirs(save_dir, exist_ok=True)

    x_explain = only_top_moa_data.copy()
    exp_model = ExplainAnomaliesUsingSHAP(model,features=features, num_anomalies_to_explain=5)
    all_sets_explaining_features = exp_model.explain_unsupervised_data(x_train=X_control, 
                                                                x_explain=only_top_moa_data,
                                                                autoencoder=model,
                                                                return_shap_values=True,
                                                                group_by=group_by_col,
                                                                save_dir=save_dir)
    


stop = True

if stop:
    sys.exit()

# test_data = data[data['Metadata_set'] == 'test_treat'].reset_index()
# X = torch.tensor(test_data[features].values,dtype=torch.float32).to(model.device)
error = diffs_normalized[diffs_normalized['Metadata_set'] == 'test_treat'][features].reset_index(drop=True)
# add suffix to error columns
error.columns = [col + '_error' for col in error.columns]

error_total = np.linalg.norm(error, axis=1)

# X_norm = StandardScaler().fit_transform(X)
# X_reconstruction = model.predict(X_norm)
# error = np.linalg.norm(X_norm - X_reconstruction)

# We transform data into dataframes
# df_norm = pd.DataFrame(X, columns = X.columns)
# X_only_features = X.drop(meta_features, axis=1)
# df_error_per_col = pd.DataFrame(error, columns = [col + '_error' for col in X.columns])

df_error_rec = pd.DataFrame(error_total, columns = ['error_rec'], index = X.index)
df = pd.concat([test_data,error, df_error_rec],axis=1)
error_features = [col for col in df.columns if '_error' in col]
# We select the data point with the highest reconstruction error
top_error = df.loc[df.error_rec == df.error_rec.max()].T

# We extract the feature names, ranked by error

featuresRankedByError = df[error_features].max().sort_values(ascending=False).index
featuresRankedByError_argsort = df[error_features].max().sort_values(ascending=False).argsort()
# get column number in the original dataframe
featuresRankedByError_argsort = [error.columns.get_loc(c) for c in featuresRankedByError]
topMfeatures = featuresRankedByError_argsort[:10]
shap_values = [[0]*len(features)]*len(topMfeatures)

for feature_index in topMfeatures:

    explainer = shap.KernelExplainer(func_predict_feature, backgroungd_set)
    shap_values = explainer.shap_values(top_error, nsamples='auto')
    # # set the weight for the current feature to 0
    # # weights_feature = weights.copy()
    # weights = model.get_weights(module='encoder', layer_num=1)
    # weights_feature = weights.data.clone()
    # weights_feature[:,feature_index] = 0
    
    # # model weights are updated
    # model_feature = copy.deepcopy(model)        
    # model_feature.update_weights(weights_feature,module='encoder', layer_num=1)

    # ## SHAP values for the feature are saved in shap_values
    # explainer_autoencoder = shap.DeepExplainer(model_feature, X_train)
    # shap_values[feature_index] = explainer_autoencoder.shap_values(X)

#     def mad_score(points):
#         m = np.median(points)
#         ad = np.abs(points - m)
#         mad = np.median(ad)
#         return 0.6745 * ad / mad

#     THRESHOLD = 3
#     z_scores = mad_score(df_error_rec)
#     outliers = z_scores > THRESHOLD


# # shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
# # test_numpy = np.swapaxes(np.swapaxes(test_images.numpy(), 1, -1), 1, 2)


# # from the genomic example
# # seqs_to_explain = onehot_data[[0, 3, 9]]  # these three are positive for task 0
# # dinuc_shuff_explainer = shap.DeepExplainer(
# #     (keras_model.input, keras_model.output[:, 0]), shuffle_several_times
# # )
# # raw_shap_explanations = dinuc_shuff_explainer.shap_values(
# #     seqs_to_explain, check_additivity=False
# # )


