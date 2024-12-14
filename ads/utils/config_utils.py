import sys
from datetime import datetime
import os
import torch
import numpy as np
import random
import logging
import argparse
from data_layer.data_utils import get_cp_path, get_cp_dir
from utils.args import DataArguments, GeneralArguments, ModelArguments, EvalArguments, MoaArguments
from utils.global_variables import DS_INFO_DICT
from utils.global_variables import ABRVS
# import transformers
import pandas as pd
import openpyxl as pxl
from openpyxl.utils import get_column_letter
import os
import json

from argparse import ArgumentParser
from pytorch_lightning import seed_everything
import pickle
import yaml


def load_configs(config_path: str) -> dict:
    """Load configurations from a YAML file."""

    with open(config_path, "r") as file:
        configs = yaml.safe_load(file)
    
    return configs

################################################################################

def merge_cli_and_config(cli_args: dict, config: dict) -> None:

    """ Merge command-line arguments into the configuration (CLI args take precedence)"""
    config_keys = ['GeneralArguments', 'DataArguments', 'ModelArguments', 'EvalArguments', 'MoaArguments']
    for config_key in config_keys:
        for key, value in cli_args.items():
            if value is not None:  # Override only if CLI argument is provided
                keys = key.split('.')  # Support nested keys if needed
                temp = config[config_key]            
                for k in keys[:-1]:
                    temp = temp.get(k, {})
                if keys[-1] in temp:
                    temp[keys[-1]] = value

    # Instantiate dataclasses with merged config
    general_args = GeneralArguments(**config['GeneralArguments'])
    data_args = DataArguments(**config['DataArguments'])
    model_args = ModelArguments(**config['ModelArguments'])
    eval_args = EvalArguments(**config['EvalArguments'])
    moa_args = MoaArguments(**config['MoaArguments'])
    
    configs = argparse.Namespace(**{
        'general': GeneralArguments(**config['GeneralArguments']),
        'data':  DataArguments(**config['DataArguments']),
        'model': ModelArguments(**config['ModelArguments']), 
        'eval':EvalArguments(**config['EvalArguments']),
        'moa':  MoaArguments(**config['MoaArguments'])
    })


    return configs

################################################################################

def set_configs(cli_args):

    # # Revise and prepare configurations
    # configs['general']['exp_name'] = revise_exp_name(configs)
    # set_paths(configs)
    # set_seed(configs['general']['seed'])

    # # Setup logger
    # set_logger(configs)

    
    configs = load_configs(cli_args["config"])
    if cli_args is not None:
        configs = merge_cli_and_config(cli_args, configs)

    # # Instantiate dataclasses with merged config
    # general_args = GeneralArguments(**config['general'])
    # data_args = DataArguments(**config['data'])
    # model_args = ModelArguments(**config['model'])
    # eval_args = EvalArguments(**config['eval'])

    # parser = transformers.HfArgumentParser((GeneralArguments, DataArguments,ModelArguments, EvalArguments,MoaArguments))
    # general_args, data_args,model_args, eval_args, moa_args = parser.parse_args_into_dataclasses()
    
    # configs = argparse.Namespace(**{
    #     'general': general_args,
    #     'model': model_args,
    #     'data': data_args,
    #     'eval':eval_args,
    #     'moa':moa_args
    # })
    # if len(exp_name)>0:
        # configs.general.exp_name = exp_name
    # exp_name = revise_exp_name(configs)
    # configs.general.exp_name = exp_name
    configs.general.output_dir = f"{configs.general.base_dir}/anomaly_output/{configs.general.dataset}/{configs.data.modality}/{configs.general.exp_name}/"

    set_paths(configs)
    set_seed(configs.general.seed)
    set_logger(configs)

    save_yaml_config(configs, os.path.join(configs.general.output_dir, f'configs_{configs.general.flow}.yaml'))
    return configs

################################################################################

# def revise_exp_name(configs):
#     if configs.model.tune_hyperparams:
#         configs.general.exp_name += f'_{ABRVS["tune"]}'
#     if configs.data.norm_method == 'mad_robustize':
#         configs.general.exp_name += f'_{ABRVS[configs.data.norm_method]}'
#     # if configs.model.l2_lambda>0.01:
#         # configs.general.exp_name += f'_l2_{configs.model.l2_lambda}'
#     if configs.model.deep_decoder:
#         configs.general.exp_name += f'_dd'
        
#     if configs.model.encoder_type == 'shallow':
#         configs.general.exp_name += f'_se'
#     elif configs.model.encoder_type == 'deep':
#         configs.general.exp_name += f'_de'
#     return configs.general.exp_name 

################################################################################

def add_exp_suffix(profile_type, by_dose,normalize_by_all=False, z_trim=None, min_max_norm=False):

    suffix = ''
    
    if profile_type == 'augmented':
        suffix+= '_a'
    elif profile_type == 'normalized_variable_selected':
        suffix+= '_nvs'
    if by_dose:
        suffix+= '_d'
    if normalize_by_all:
        suffix+= '_ba'
    if z_trim is not None:
        suffix+= f'_z{str(z_trim)}'
    if min_max_norm:
        suffix+= '_mm' 
        
    return suffix




################################################################################

def save_yaml_config(config, file_path: str):
    """Save the configuration dictionary to a YAML file."""
    with open(file_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)


################################################################################

# def save_configs(configs):
#     config_path = os.path.join(configs.general.output_dir,'args.pkl')
    
#     with open(config_path, 'wb') as f:
#         pickle.dump(configs, f)
    
#     # save text file with all params    
#     with open(os.path.join(configs.general.output_dir, 'params.txt'), 'w') as f:
#         # skip line for each parameter
#         for k,v in configs.__dict__.items():
#             f.write(f'{k}:\n')

#             for k2,v2 in v.__dict__.items():
#                 # add indetation for each sub-parameter
#                 f.write(f'     {k2}: {v2}\n')


################################################################################

# def get_configs(exp_dir):
#     # path = get_cp_dir(DATASET_ROOT_DIR,DS_INFO_DICT[DATASET]['name'],exp_name)
#     config_path = os.path.join(exp_dir,'args.pkl')
    
#     if os.path.exists(config_path):  
#         parser = ArgumentParser()

#         args, unknown = parser.parse_known_args()
#         # args = parser.parse_args()  

#         with open(config_path, 'rb') as f:
#             return pickle.load(f)
        
#     else:
#         return None
    

################################################################################

def set_configs_jupyter(exp_name = ''):

    parser = transformers.HfArgumentParser((GeneralArguments, DataArguments,ModelArguments, EvalArguments,MoaArguments))
    general_args = GeneralArguments()
    data_args = DataArguments()
    model_args = ModelArguments()
    eval_args = EvalArguments()
    moa_args = MoaArguments()

    # general_args, data_args,model_args, eval_args = parser.parse_args_into_dataclasses()
    # general_args, data_args,model_args, eval_args = parser.parse_known_args()
    # configs = set_paths(configs)
    # assert_configs(configs)
    # configs.general.exp_name = exp_name


    # cpds_med_scores, null_distribution_medians = calc_reproducibility(configs)
    # general_args.flow
    configs = argparse.Namespace(**{
            'general': general_args,
            'model': model_args,
            'data': data_args,
            'eval':eval_args,
            'moa':moa_args
        })
    
    if len(exp_name)>0:
        configs.general.exp_name = exp_name
    exp_name = revise_exp_name(configs)
    configs.general.exp_name = exp_name
    # configs = set_paths(configs)

    # configs_from_file = get_configs(configs.general.output_dir)
    # if configs_from_file is not None:
        # configs = configs_from_file

    set_seed(configs.general.seed)
    
    return configs

def set_logger(configs):
    if configs.general.debug_mode == True:
        logging.basicConfig(filename=os.path.join(configs.general.res_dir, 'debug_log.log'),
                            datefmt='%H:%M:%S',
                            level=logging.INFO)
    else:
        log_dir = os.path.join(configs.general.output_dir,'logs', configs.general.flow)
        os.makedirs(log_dir, exist_ok=True)
        logging.basicConfig(filename=os.path.join(log_dir, f'log_exp_num={configs.general.exp_num}.log'),
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.INFO)
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    configs.general.logger = logging.getLogger(__name__)
    return configs


def assert_configs(configs):
    assert configs.general.dataset in ['CDRP-bio','CDRP', 'LINCS', 'LUAD', 'TAORF'], "dataset not supported"
    assert configs.data.modality in ['CellPainting', 'L1000'], "modality not supported"
    assert configs.data.profile_type in ['augmented', 'normalized', 'normalized_variable_selected'], "profile_type not supported"
    # assert configs.general.flow in ['run_ad', 'calc_metrics', 'analyze'], "flow not supported"
    assert configs.general.exp_name != '', "exp_name is empty"
    assert configs.general.base_dir != '', "base_dir is empty"
    # assert configs.general.output_dir != '', "output_dir is empty"
    assert configs.general.res_dir != '', "res_dir is empty"

def set_paths(configs):
    configs.general.data_path = get_cp_path(configs.general.base_dir, configs.general.dataset,configs.data.profile_type)
    configs.general.processed_data_dir = get_cp_dir(configs.general.base_dir, configs.general.dataset, configs.data.profile_type,processed=True)
    configs.general.output_dir = get_cp_dir(configs.general.base_dir, configs.general.dataset, configs.data.profile_type, exp_name=configs.general.exp_name,processed=True)
    configs.general.res_dir = f"{configs.general.base_dir}/results/{configs.general.dataset}/{configs.data.modality}/{configs.general.exp_name}/"
    configs.general.fig_dir = f"{configs.general.res_dir}/figs"
    configs.model.ckpt_dir = f"{configs.general.res_dir}/ckpt"
    configs.model.tb_logs_dir = f"{configs.general.res_dir}/logs"

    os.makedirs(configs.general.output_dir, exist_ok=True)
    os.makedirs(configs.general.res_dir, exist_ok=True)
    os.makedirs(configs.general.fig_dir, exist_ok=True)

    assert_configs(configs)
    if not hasattr(configs.general,'logger'):
        configs = set_logger(configs)
    # os.makedirs(configs.general.processed_data_dir, exist_ok=True)

    return configs
    

def output_path(configs):
    output_path = 'Not created'
    if configs.general.debug_mode == False:
        now = datetime.now()
        output_name = now.strftime("%d_%m_%Y____%H_%M_%S")
        output_path = os.path.join(
            configs.general.output_dir,
            configs.model.name,
            configs.data.dataset_name,
            output_name
        )
        if configs.general.debug_mode == False and os.path.exists(output_path) == False:
            print('path is: {}'.format(output_path))
            os.makedirs(output_path, exist_ok=True)

    return output_path


def set_seed(seed):
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    seed_everything(seed)
