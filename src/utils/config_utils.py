import sys
from datetime import datetime
import os
import torch
import numpy as np
import random
import logging
import argparse
from data.data_utils import get_cp_path, get_cp_dir
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



def merge_cli_and_config(cli_args: dict, config: dict) -> argparse.Namespace:
    """
    Merge command-line arguments into the configuration (CLI args take precedence)
    """
    # First, create a copy of the config to avoid modifying the original
    merged_config = config.copy()

    # Iterate through CLI arguments
    for key, value in cli_args.items():
        if value is not None:  # Only process if CLI argument was provided
            # Check each config section
            for config_key in ['GeneralArguments', 'DataArguments', 'ModelArguments', 'EvalArguments', 'MoaArguments']:
                if config_key in merged_config:
                    # If the key exists in the current section's keys, update it
                    if key in merged_config[config_key]:
                        merged_config[config_key][key] = value
                        break  # Stop once we've found and updated the matching key

    # Create namespace with updated configurations
    configs = argparse.Namespace(
        general=GeneralArguments(**merged_config['GeneralArguments']),
        data=DataArguments(**merged_config['DataArguments']),
        model=ModelArguments(**merged_config['ModelArguments']),
        eval=EvalArguments(**merged_config['EvalArguments']),
        moa=MoaArguments(**merged_config['MoaArguments'])
    )

    return configs

################################################################################

def set_configs(cli_args):

    
    configs = load_configs(cli_args["config"])
    if cli_args is not None:
        configs = merge_cli_and_config(cli_args, configs)

    if configs.general.output_dir is not None:
        configs.general.output_dir = f"{configs.general.output_dir}/{configs.general.dataset}/{configs.data.modality}/{configs.general.exp_name}/"
    else:
        configs.general.output_dir = f"{configs.general.base_dir}/anomaly_output/{configs.general.dataset}/{configs.data.modality}/{configs.general.exp_name}/"

    if configs.general.res_dir is not None:
        configs.general.res_dir = f"{configs.general.res_dir}/{configs.general.dataset}/{configs.data.modality}/{configs.general.exp_name}/"
    else:
        configs.general.res_dir = f"{configs.general.base_dir}/results/{configs.general.dataset}/{configs.data.modality}/{configs.general.exp_name}/"
    set_paths(configs)
    set_seed(configs.general.seed)
    set_logger(configs)

    save_yaml_config(configs, os.path.join(configs.general.output_dir, f'configs_{configs.general.flow}.yaml'))
    return configs

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
    seed_everything(seed)
