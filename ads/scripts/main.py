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
import json
import pandas as pd 
from collections import defaultdict

currentdir = '/sise/home/alonshp/AnomalyDetectionScreening'
code_dir = '/sise/home/alonshp/AnomalyDetectionScreening/ads'

print(currentdir)


# currentdir = os.path.dirname('home/alonshp/AnomalyDetectionScreeningLocal/')
# print(currentdir)
sys.path.insert(0, os.getcwd())
sys.path.insert(0, currentdir)
sys.path.insert(0, code_dir)

from run_ad import train_autoencoder
# from ads.utils.configuration import DataArguments, GeneralArguments, ModelArguments, EvalArguments
from ads.utils.general import set_seed, set_logger, output_path, set_paths, assert_configs, set_configs, revise_exp_name
# from ads.scripts.calc_repreducibility import main as calc_reproducibility
# from ads.scripts.classify_moa import main as run_moa
from ads.utils.global_variables import ABRVS,DS_INFO_DICT
from ads.utils.plotting import plot_latent_effect
from create_null_distribution import main as create_null_distributions
from classify_moa import run_classifier
# from qa_generator.utils.data import get_dataloader, generate_ocr
# from ads.utils.general import output_path, set_seed, set_logger
# from readProfiles import *
# from pred_models import *
# from data import load_data

pd.set_option('display.max_rows', 100)
logger = logging.getLogger(__name__)

  

def main(configs):

    # configs.general.logging_dir = configs.general.output_exp_dir = output_path(configs)
    # configs = set_paths(configs)
    # assert_configs(configs)
    configs.general.logging_dir = configs.general.output_exp_dir
    # print(configs.general.output_exp_dir)
    # configs = set_logger(configs)
    # os.makedirs(configs.general.output_exp_dir, exist_ok=True)
    # os.makedirs(configs.general.res_dir, exist_ok=True)

    if configs.general.debug_mode == False:
        # with open('commandline_args.txt', 'w') as f:
            # json.dump(args.__dict__, f, indent=2)
        with open(os.path.join(configs.general.output_exp_dir, 'configs.json'), 'w', encoding='utf-8') as f:
            json.dump(str(configs), f, ensure_ascii=False, indent=4)

    configs.general.logger.info(os.getcwd())
    configs.general.logger.info('*' * 100)
    configs.general.logger.info(configs)
    configs.general.logger.info('*' * 100)

    # if configs.general.flow in ['eval']:
    #     # model = load_model(configs)
    #     dataset = load_datasets(configs)
    #     dataloader = get_dataloader(dataset, configs)
    #     eval_model(model, dataloader, configs)
    if configs.general.flow in ['run_ad']:
        eval_model = True
        diff_filename = f'replicate_level_cp_{configs.data.profile_type}_ae_diff.csv'
        if not configs.general.debug_mode and not os.path.exists(os.path.join(configs.general.output_exp_dir, diff_filename)):
            # save_profiles(test_treat_out_normalized_diff, output_dir, diff_filename)
            if configs.general.run_both_profiles:
                profile_types = ['augmented', 'normalized_variable_selected']
                for p in profile_types:
                    configs.data.profile_type = p
                    model,losses = train_autoencoder(configs)      
            else:
                model,losses = train_autoencoder(configs)          
        # configs.eval.res(losses)
        if eval_model:
            eval_results(configs)
    elif configs.general.flow in ['eval_model']:
        eval_results(configs)
    else:
        raise Exception('Not recognized flow')
        
    return configs
    # exit(0)

def eval_results(configs):

        Types = ['ae_diff', 'baseline']
        data_reps = [f'{configs.data.profile_type}_{t}' for t in Types]

        if configs.eval.run_dose_if_exists and DS_INFO_DICT[configs.general.dataset]['has_dose']:
            doses = [False,True]
        else:
            doses = [False]

        for d in doses:
            configs.eval.by_dose = d
            configs.eval.normalize_by_all = True
            create_null_distributions(configs)
            configs.eval.normalize_by_all = False
            create_null_distributions(configs)

        run_moa = False
        if run_moa:
            for d in doses:
                configs.eval.by_dose = d
                configs.eval.normalize_by_all = True
                run_classifier(configs, data_reps = data_reps)
                configs.eval.normalize_by_all = False
                run_classifier(configs, data_reps = data_reps)
            

if __name__ == "__main__":


    # if len(sys.argv) <2:

        # exp_name = 'base_fs'
        # configs = set_configs()
    # else:
    # configs = set_configs()
    
    if len(sys.argv) < 2:
        
        # configs.general.exp_name = 'base'
        exp_name = 'report_911_t'

        configs = set_configs(exp_name)

        # CP Profile Type options: 'augmented' , 'normalized', 'normalized_variable_selected'
        
        # configs.general.debug_mode = True
        configs.model.tune_hyperparams = True
        configs.data.feature_select = True
        configs.general.tune_ldims = False
        configs.model.deep_decoder = False
        # configs.data.overwrite_data_creation = True
        # dataset type: CDRP, CDRP-bio, LINCS, LUAD, TAORF
        configs.general.flow = 'run_ad'
        # configs.general.flow = 'calc_metrics'
        configs.general.dataset = 'LINCS'
        configs.general.dataset = 'CDRP-bio'
        configs.data.corr_threshold = 0.9
        # configs.general.dataset = 'CDRP'


        configs.data.modality = 'CellPainting'
        # configs.data.modality = 'L1000'
        
        # configs.data.norm_method = 'mad_robustize'
        configs.data.profile_type = 'normalized_variable_selected'
        configs.data.profile_type = 'augmented'
        # configs.data.run_data_process = True

        print("Usage: python main.py <flow>")
    else:
        configs = set_configs()

    configs = set_paths(configs)

    if configs.data.modality == 'CellPainting':
        configs.data.modality_str = 'cp'
    elif configs.data.modality == 'L1000':
        configs.data.modality_str = 'l1k'
        configs.data.do_fs = False
    # configs.general.exp_name = revise_exp_name(configs)
    # if configs.data.feature_selection:
        # configs.general.exp_name += f'_fs'

    if configs.general.debug_mode:
        configs.general.exp_name += '_debug'
        configs.model.n_tuning_trials = 10

    # if configs.general.flow in ['calc_metrics']:
    #     # configs.general.tune_ldims = True
    #     if configs.general.tune_ldims:
    #         configs.eval.scores = defaultdict(list)
    #         ldims = [8,16,32,64]
    #         for i in ldims:
    #             configs.model.latent_dim = i
    #             configs.general.exp_name = f'ldim_{i}'
    #             if configs.model.tune_hyperparams:
    #                 configs.general.exp_name += f'_{ABRVS["tune"]}'
    #             if configs.data.norm_method == 'mad_robustize':
    #                 configs.general.exp_name += f'_{ABRVS[configs.data.norm_method]}'
    #                 # configs.data.profile_type += f'_{ABRVS[configs.data.norm_method]}'
    #             configs = main(configs)
    #     else:
    #         configs = main(configs)
    #         # scores = pd.DataFrame(configs.eval.scores)

    # if configs.general.flow in ['analyze']:
    #     pass

    # if configs.general.flow in ['run_moa']:
    #     pass
        # run_moa(configs)


    main(configs)
