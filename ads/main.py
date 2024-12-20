import pandas as pd
import torch
from sklearn import preprocessing
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import sys
import os
import transformers
import argparse
import logging
import pickle
import pandas as pd 
from collections import defaultdict
from utils.general import save_configs,get_configs
from eval.shap_anomalies import run_anomaly_shap
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update(plt.rcParamsDefault)
plt.style.use('seaborn-v0_8-colorblind')
sns.set_theme(style='ticks',context='paper', font_scale=1.5)

# currentdir = '/sise/home/alonshp/AnomalyDetectionScreening'
# code_dir = '/sise/home/alonshp/AnomalyDetectionScreening/ads'

# sys.path.insert(0, os.getcwd())
# sys.path.insert(0, currentdir)
# sys.path.insert(0, code_dir)

from anomaly_pipeline import anomaly_pipeline as run_ad
# from utils.configuration import DataArguments, GeneralArguments, ModelArguments, EvalArguments
from utils.general import set_seed, set_logger, output_path, set_paths, assert_configs, set_configs, revise_exp_name
# from scripts.calc_repreducibility import main as calc_reproducibility
# from scripts.classify_moa import main as run_moa
from utils.global_variables import ABRVS,DS_INFO_DICT, HYPER_PARAMS
from scripts.calc_reproducibility import calc_percent_replicating as calc_percent_replicating
from scripts.classify_moa import run_classifier

pd.set_option('display.max_rows', 100)
# sns.set_context("paper",font_scale = 1.8, rc={"font.size":8,"axes.titlesize":16,"axes.labelsize":16})
logger = logging.getLogger(__name__)


def main(configs):

    configs.general.logging_dir = configs.general.output_dir

    configs.general.logger.info(os.getcwd())
    configs.general.logger.info('*' * 10)
    configs.general.logger.info(configs)
    configs.general.logger.info('*' * 10)

    if configs.general.flow in ['run_ad']:
        eval_model = True
        diff_filename = f'replicate_level_cp_{configs.data.profile_type}_ae_diff.csv' 
        run_flag = not os.path.exists(os.path.join(configs.general.output_dir, diff_filename)) or configs.general.overwrite_experiment
        if run_flag or configs.general.debug_mode:
            # save_profiles(test_treat_out_normalized_diff, output_dir, diff_filename)
            if configs.general.run_both_profiles:
                profile_types = ['augmented', 'normalized_variable_selected']
                for p in profile_types:
                    configs.data.profile_type = p
                    run_ad(configs)
            else:
                run_ad(configs)

                        
        # configs.eval.res(losses)
        if eval_model and not configs.general.debug_mode:
            data_reps = configs.data.data_reps
            res = eval_results(configs,data_reps=configs.data.data_reps)
            if configs.eval.run_shap:
                run_anomaly_shap(configs, filter_non_reproducible=configs.eval.filter_non_reproducible)
            # run only vs. cell profiler baseline
            if data_reps != ['ae_diff','baseline']:
                _ = eval_results(configs)
    elif configs.general.flow in ['eval_model']:
        res = eval_results(configs)

    else:
        raise Exception('Not recognized flow')
        
    return res
    # exit(0)

def eval_results(configs,data_reps = None):
    
        data_reps = configs.data.data_reps

        if configs.eval.run_dose_if_exists and DS_INFO_DICT[configs.general.dataset]['has_dose']:
            doses = [False,True]
        else:
            doses = [False]

        rc = defaultdict(dict)
        for d in doses:
                configs.eval.by_dose = d
                configs.eval.z_trim = None
                configs.eval.normalize_by_all = True
                configs.general.logger.info(f'Running null distributions for dose {d} and normalize_by_all = True')
                rc = calc_percent_replicating(configs,data_reps=data_reps)
                # configs.eval.normalize_by_all = False
                # configs.general.logger.info(f'Running null distributions for dose {d} and normalize_by_all = False')
                # _ =  create_null_distributions(configs,data_reps=data_reps)
                # configs.eval.z_trim = 8
                # configs.general.logger.info(f'Running null distributions for dose {d} and z_trim = 8')
                # _ =  create_null_distributions(configs,data_reps=data_reps)
        



        run_moa = False
        if run_moa:
            for d in doses:
                configs.eval.by_dose = d
                configs.eval.normalize_by_all = True
                run_classifier(configs, data_reps = data_reps)
                # configs.eval.normalize_by_all = False
                # run_classifier(configs, data_reps = data_reps)

        return rc
            

if __name__ == "__main__":


    if len(sys.argv) < 2:
        
        # exp_name = 'test9_1704'
        exp_name = 'try0612'
        configs = set_configs(exp_name) 

        # CP Profile Type options: 'augmented' , 'normalized', 'normalized_variable_selected'
        configs.general.debug_mode = False
        configs.general.run_all_datasets = False
        configs.data.run_data_process = False
        configs.model.tune_hyperparams = False
        configs.model.n_tuning_trials = 10
        configs.data.feature_select = True
        configs.model.deep_decoder = False
        configs.general.overwrite_experiment = False
        # dataset : CDRP, CDRP-bio, LINCS, LUAD, TAORF
        configs.general.flow = 'run_ad'
        # configs.general.flow = 'calc_metrics'
        # configs.general.dataset = 'LINCS'
        configs.general.dataset = 'CDRP-bio'
        configs.data.corr_threshold = 0.9
        configs.data.plate_normalized = True
        configs.eval.run_dose_if_exists = True
        # configs.general.dataset = 'LUAD'
        configs.general.dataset = 'TAORF'

        configs.data.modality = 'CellPainting'
        # configs.data.modality = 'L1000'
        
        # configs.data.norm_method = 'mad_robustize'
        configs.data.profile_type = 'normalized_variable_selected'
        configs.data.profile_type = 'normalized'
        configs.data.profile_type = 'augmented'
        configs.moa.moa_dirname = 'MoAprediction_single'
        configs.eval.with_l1k = True
        configs.eval.rand_reps = 1
        configs.general.run_all_datasets = False

    else:
        configs = set_configs()
        exp_name = configs.general.exp_name

    if configs.general.run_parallel:
        datasets = ['CDRP-bio','LUAD','TAORF','LINCS']
        datasets = [datasets[configs.general.slice_id]]
    elif configs.general.run_all_datasets:
        datasets = ['TAORF','LUAD','CDRP-bio','LINCS']
    else:
        datasets = [configs.general.dataset]

    if configs.data.plate_normalized:
        # data_reps = ['ae_diff','baseline', 'PCA',  'ZCA']
        # data_reps = ['ae_diff','baseline','baseline_dmso']
        # data_reps = ['ae_diff','baseline', 'baseline_unchanged']
        data_reps = ['ae_diff','baseline']
    else:
        # data_reps = ['ae_diff','baseline', 'PCA',  'ZCA', 'PCA-cor','ZCA-cor']
        # data_reps = ['ae_diff','baseline', 'baseline_unchanged']
        data_reps = ['ae_diff','baseline']
    

    configs.data.data_reps = data_reps
    print(f'Running datasets: {datasets}')

    for d in datasets:
        configs.general.dataset = d
        configs = set_paths(configs)
        config_path = os.path.join(configs.general.output_dir,'args.pkl')
        if os.path.exists(config_path):
            print('Experiment already exists! Loading configs from file')
            configs = get_configs(configs.general.output_dir)
            configs.general.from_file = True

        for param in HYPER_PARAMS[d].keys():
            # check if attribute not is sys.args 
            if param not in sys.argv[1:]:
                setattr(configs.model, param, HYPER_PARAMS[d][param])
                configs.general.logger.info(f'Overriding {param} with {HYPER_PARAMS[d][param]} from default hyperparams')
            else:
                configs.general.logger.info(f'{param} set to {getattr(configs.model, param)} from argument')

        if configs.data.profile_type == 'normalized_variable_selected':
            configs.model.latent_dim = 16
            configs.model.encoder_type = 'default'
        

            
        # configs.model.latent_dim = HYPER_PARAMS[d]['latent_dim']
        # configs.model.encoder_type = HYPER_PARAMS[d]['encoder_type']
        # configs.model.deep_decoder = HYPER_PARAMS[d]['deep_decoder']
        # configs.model.batch_size = HYPER_PARAMS[d]['batch_size']
        # if configs.data.modality == 'CellPainting':
            # MODALITY_STR[configs.data.modality] = 'cp'
        elif configs.data.modality == 'L1000':
            # configs.data.MODALITY_STR[configs.data.modality] = 'l1k'
            configs.data.do_fs = False
        # configs.general.exp_name = revise_exp_name(configs)
        # if configs.data.feature_selection:
            # configs.general.exp_name += f'_fs'

        if configs.general.debug_mode:
            # configs.general.exp_name += '_debug'
            configs.model.n_tuning_trials = 10

        save_configs(configs)

        main(configs)
