import argparse
from src.utils.global_variables import BASE_DIR, DS_INFO_DICT
from src.utils.config_utils import set_configs
from src.data.data_processing import pre_process, construct_dataloaders
from data.data_utils import load_data
from ProfilingAnomalyDetector import ProfilingAnomalyDetector
from eval.calc_reproducibility import calc_percent_replicating
from eval.classify_moa import run_moa_classifier
from eval.shap_anomalies import run_anomaly_shap
from utils.global_variables import MODALITY_STR


# sys.path.insert(0, os.path.join(os.getcwd(),'src'))
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
        anomaly_pipeline(configs)
    elif configs.general.flow == "eval":
        eval_pipeline(configs)
    else:
#         logger.error(f"Unknown flow: {configs.general.flow}")
        raise ValueError("Invalid flow provided in configuration.")

def anomaly_pipeline(configs):
    """
    Entry point for the anomaly detection pipeline.
    Args:
        configs (dict): Configuration dictionary.
    """
    # Run the anomaly detection pipeline

    data , __ = load_data(configs.general.base_dir,configs.general.dataset,configs.data.profile_type, modality=configs.data.modality)
    data_preprocess,features = pre_process(data,configs)
    dataloaders = construct_dataloaders(data_preprocess,configs.model.batch_size,features)

    anomaly_detector = ProfilingAnomalyDetector(features,  **vars(configs.model))
    anomaly_detector.fit(dataloaders, features)
    anomaly_detector.forward(dataloaders)

    # save_path = os.path.join(configs.,  )
    filename = f'replicate_level_{MODALITY_STR[configs.data.modality]}_{configs.data.profile_type}_ae_diff'
    anomaly_detector.save_anomalies(data_preprocess,save_dir=configs.general.output_dir, filename=filename)


def eval_pipeline(configs):
    """
    Entry point for the evaluation pipeline.
    Args:
        configs (dict): Configuration dictionary.
    """

    res = {}
    res['rc'] = calc_percent_replicating(configs)

    if configs.eval.run_shap:
        res['shap'] = run_anomaly_shap(configs, filter_non_reproducible=configs.eval.filter_non_reproducible)
    if configs.eval.run_moa_classifier and DS_INFO_DICT[configs.general.dataset]['has_moa']:
        res['moa'] = run_moa_classifier(configs)


# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Run profiling anomaly detection pipeline")
    
    # Add arguments corresponding to dataclass fields
    parser.add_argument("--base_dir", type=str, help="Base directory for the project")
    parser.add_argument("--output_dir", type=str, help="Base directory for saving project output")
    parser.add_argument("--res_dir", type=str, help="Base directory for saving experiment results")
    parser.add_argument("--exp_name", type=str, help="Experiment name")
    parser.add_argument("--config", type=str, default = 'configs/default_config.yaml',help="Path to the configuration file")
    parser.add_argument("--flow", type=str, default = 'eval',help="Flow of the experiment")
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
