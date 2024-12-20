import argparse
import sys
import logging
import yaml
from ads.utils.global_variables import BASE_DIR
from ads.utils.config_utils import load_configs, set_configs
from ads.utils.logger import setup_logger
# from ads.pipeline.anomaly_pipeline import anomaly_pipeline
from ads.pipeline.eval_pipeline import eval_pipeline
from ads.data.data_processing import load_data, pre_process, construct_dataloaders
from ads.pipeline.ProfilingAnomalyDetector import ProfilingAnomalyDetector
from utils.global_variables import MODALITY_STR
import os
import time

sys.path.insert(0, os.path.join(os.getcwd(),'ads'))
# sys.path.insert(0, currentdir)


def main(args):
    """
    Entry point for the application.
    Args:
        config_path (str): Path to the configuration file.
    """
    # configs = load_configs(config_path)
    configs = set_configs(args)


    if configs.general.flow == "train":
        anomaly_pipeline2(configs)
    elif configs.general.flow == "eval":
        eval_pipeline(configs)
    else:
#         logger.error(f"Unknown flow: {configs.general.flow}")
        raise ValueError("Invalid flow provided in configuration.")

def anomaly_pipeline2(configs):
    """
    Entry point for the anomaly detection pipeline.
    Args:
        configs (dict): Configuration dictionary.
    """
    # Run the anomaly detection pipeline

    data , __ = load_data(configs.general.base_dir,configs.general.dataset,configs.data.profile_type, modality=configs.data.modality)
    data_preprocess,features =  pre_process(data,configs)
    dataloaders = construct_dataloaders(data_preprocess,configs.model.batch_size,features)
    features, configs.general.logger, 
    anomaly_detector = ProfilingAnomalyDetector(features,  **vars(configs.model))
    anomaly_detector.fit(dataloaders['train'])
    anomaly_detector.forward(dataloaders['test'], configs.general.output_dir)

    save_path = os.path.join(configs.general.output_dir,  f'replicate_level_{MODALITY_STR[configs.data.modality]}_{configs.data.profile_type}_ae_diff')
    anomaly_detector.preds_to_anomalies(data_preprocess, save_path)


    # model = train_autoencoder(dataloaders, features, configs)

    # preds = test_autoencoder(model, dataloaders)
    # preds = test_autoencoder(model, dataloaders, features, configs)
    

    # __ =post_process_anomaly_and_save(data_preprocess, preds['test_ctrl'],preds['test_treat'], configs.general.output_dir,  f'replicate_level_{MODALITY_STR[configs.data.modality]}_{configs.data.profile_type}_preds', configs, features)
    # z_preds_normalized = save_treatments(data, z_preds['test_ctrl'],z_preds['test_treat'], configs.general.output_dir,  f'replicate_level_{configs.data.MODALITY_STR[configs.data.modality]}_{configs.data.profile_type}_ae_embeddings', configs, features, embeddings=True)
    # __ = post_process_anomaly_and_save(data_preprocess, diffs_ctrl,diffs_treat, 
        # , configs, features)
            
        

################################################################################
# Main Script
################################################################################

# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Run profiling anomaly detection pipeline")
    
    # Add arguments corresponding to dataclass fields
    parser.add_argument("--base_dir", type=str, default=BASE_DIR, help="Base directory for the project")
    parser.add_argument("--exp_name", type=str, help="Experiment name")
    parser.add_argument("--config", type=str, default = 'configs/default_config.yaml',help="Path to the configuration file")
    parser.add_argument("--flow", type=str, default = 'train',help="Flow of the experiment")
    parser.add_argument("--debug_mode", action="store_true", help="Enable debug mode")
    parser.add_argument("--use_cache", action="store_true", help="upload data from cache")
    # parser.add_argument("--seed", type=int, default=42, help="Random seed for the experiment")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--dataset", type=str, default='TAORF', help="Dataset to use")
    parser.add_argument("--exp_num", type=int, default=0, help="Experiment number")
    parser.add_argument("--run_all_datasets", type=bool, default=False, help="Run all datasets")
    parser.add_argument("--run_parallel", type=bool, default=False, help="Run multiple gpus")
    # Add more arguments as needed for other dataclasses
    args = parser.parse_args()
    return vars(args)  # Return as a dictionary for merging


if __name__ == "__main__":


    args = parse_args()
    main(args)