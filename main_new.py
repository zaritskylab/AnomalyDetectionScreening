import argparse
import sys
import logging
import yaml
from ads.utils.global_variables import BASE_DIR
from ads.utils.config_utils import load_configs, set_configs
from ads.utils.logger import setup_logger
from ads.pipeline.anomaly_pipeline import anomaly_pipeline
from ads.pipeline.eval_pipeline import eval_pipeline
import os
import time

sys.path.insert(0, os.path.join(os.getcwd(),'ads'))
# sys.path.insert(0, currentdir)


# # Function to load and merge YAML files
# def load_config(default_path: str, experiment_path: str) -> dict:
#     with open(default_path, 'r') as default_file:
#         default_config = yaml.safe_load(default_file)
    
#     with open(experiment_path, 'r') as experiment_file:
#         experiment_config = yaml.safe_load(experiment_file)
    
#     # Merge configs: experiment_config values override default_config
#     def merge_dicts(default, override):
#         if not isinstance(override, dict):
#             return override
#         result = default.copy()
#         for key, value in override.items():
#             if key in result:
#                 result[key] = merge_dicts(result[key], value)
#             else:
#                 result[key] = value
#         return result
    
#     return merge_dicts(default_config, experiment_config)


def main(args):
    """
    Entry point for the application.
    Args:
        config_path (str): Path to the configuration file.
    """
    # configs = load_configs(config_path)
    configs = set_configs(args)


    # setup_logger(configs.general.output_dir)
    # logger = logging.getLogger(__name__)

    # logger.info(f"Loaded configurations: {configs}")

    if configs.general.flow == "train":
        anomaly_pipeline(configs)
    elif configs.general.flow == "eval":
        eval_pipeline(configs)
    else:
#         logger.error(f"Unknown flow: {configs.general.flow}")
        raise ValueError("Invalid flow provided in configuration.")

# # if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Run anomaly detection pipeline.")
#     parser.add_argument(
#         "--config",
#         type=str,
#         default="configs/experiment.yaml",
#         help="Path to the configuration file.",
#     )
#     args = parser.parse_args()
#     main(args.config)


    # def set_logger(configs):
    # """Configure logging."""
    # general = configs['general']
    # debug_mode = general.get('debug_mode', False)
    # res_dir = general['res_dir']

    # if debug_mode:
    #     log_file = os.path.join(res_dir, 'debug_log.log')
    # else:
    #     log_dir = os.path.join(res_dir, 'logs', general.get('flow', 'default'))
    #     os.makedirs(log_dir, exist_ok=True)
    #     log_file = os.path.join(log_dir, f"log_exp_num={general.get('exp_num', 0)}.log")
    
    # logging.basicConfig(
    #     filename=log_file,
    #     filemode='a',
    #     format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    #     datefmt='%H:%M:%S',
    #     level=logging.INFO,
    # )
    # logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    # general['logger'] = logging.getLogger(__name__)

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