import os
from dataclasses import dataclass, field
from typing import Optional
from datetime import date
from typing_extensions import Literal
from utils.global_variables import BASE_DIR
from collections import defaultdict


# # Function to parse command-line arguments
# def parse_args():
#     parser = argparse.ArgumentParser(description="Argument parser with YAML support")
    
#     # Add arguments corresponding to dataclass fields
#     parser.add_argument("--flow", type=str, help="Flow of the experiment")
#     parser.add_argument("--debug_mode", type=bool, help="Enable debug mode")
#     parser.add_argument("--exp_name", type=str, help="Experiment name")
#     parser.add_argument("--seed", type=int, help="Random seed for the experiment")
#     parser.add_argument("--gpus", type=int, help="Number of GPUs to use")
#     parser.add_argument("--dataset", type=str, help="Dataset to use")
#     parser.add_argument("--exp_num", type=int, help="Experiment number")
#     parser.add_argument("--run_all_datasets", type=bool, help="Run all datasets")
    
#     # Add more arguments as needed for other dataclasses
#     args = parser.parse_args()
#     return vars(args)  # Return as a dictionary for merging


# @dataclass
# class GeneralArguments:
#     flow: Optional[str] = field(default="train")
#     debug_mode: Optional[bool] = field(default=False)
#     exp_name: Optional[str] = field(default=date.today().strftime("%d_%m"))
#     seed: Optional[int] = field(default=42)
#     gpus: Optional[int] = field(default=1)
#     base_dir: Optional[str] = field(default=BASE_DIR)
#     dataset: Optional[str] = field(default="CDRP-bio")
#     exp_num: Optional[int] = field(default=0)
#     run_all_datasets: Optional[bool] = field(default=False)


@dataclass
class GeneralArguments:
    flow: Optional[str] = field(default="train", metadata={"help": ""})
    debug_mode: Optional[bool] = field(default=False, metadata={"help": ""})
    exp_name: Optional[str] = field(default=date.today().strftime("%d_%m"), metadata={"help": "experiment name"})
    seed: Optional[int] = field(default=42, metadata={"help": "experiment seed"})
    gpus: Optional[int] = field(default=1, metadata={"help": "number of gpus to use"})
    base_dir: Optional[str] = field(default=BASE_DIR, metadata={"help": "base directory"})
        # supported datasets =['CDRP-bio','CDRP', 'LINCS', 'LUAD', 'TAORF'])
    exp_num: Optional[int] = field(default=0, metadata={"help": "experiment number"})
    tune_ldims: Optional[bool] = field(default=False, metadata={"help": "if True, tune latent dims"})
    dataset: Optional[str] = field(default='CDRP-bio', metadata={"help": "The name of the dataset"})
    # run_both_profiles: Optional[bool] = field(default=False, metadata={"help": "if True, run both profiles"})
    
    run_all_datasets: Optional[bool] = field(default=False, metadata={"help": "if True, run all datasets"})
    overwrite_experiment: Optional[bool] = field(default=False, metadata={"help": "if True, overwrite output results"})
    run_parallel: Optional[bool] = field(default=False, metadata={"help": "if True, run different datasets on different GPUS"})
    slice_id: Optional[int] = field(default=None, metadata={"help": "slice id in case of sbatch run"})

@dataclass
class DataArguments:
    modality: Optional[str] = field(default='CellPainting', metadata={"help": "The name of the modality"})
    # modality_str: Optional[str] = field(default='cp', metadata={"help": "The name of the modality"})
    profile_type: Optional[str] = field(default='augmented', metadata={"help": "The name of the dataset"})
                                    #   choices=['augmented', 'normalized', 'normalized_variable_selected'])
    test_split_ratio: Optional[float] = field(default=0.5, metadata={"help": "split ratio for test set"})
    val_split_ratio: Optional[float] = field(default=0.1, metadata={"help": "split ratio for between train set and val set"})
    normalize_condition: Optional[str] = field(default='train', metadata={"help": "if train, normalize by train set. \
                                            if DMSO, normalize by DMSO. else, normalize by all data"})
    plate_normalized: Optional[bool] = field(default=True, metadata={"help": "if True, normalize by plate"})
    # by_plate: Optional[bool] = field(default=True, metadata={"help": "if True, normalize by plate"})
    norm_method: Optional[str] = field(default='standardize', metadata={"help": "normalization method."}) \

    use_cache: Optional[bool] = field(default=False, metadata={"help": "if True, run data process even if data already exists"})
    feature_select: Optional[bool] = field(default=True, metadata={"help": "if True, run feature selection. \
                                            if False, use features selected in 'normalized_variable_selected' profile"})    
    corr_threshold: Optional[float] = field(default=0.9, metadata={"help": "correlation threshold for feature selection"})
    n_samples_for_training: Optional[int] = field(default=None, metadata={"help": "number of data samples for training, all if None"})
    var_threshold: Optional[float] = field(default=10e-4, metadata={"help": "variance threshold for feature selection"})
    

@dataclass
class ModelArguments:
    lr: Optional[float] = field(default=0.005, metadata={"help": "learning rate"})
    dropout: Optional[float] = field(default=0.0, metadata={"help": "dropout rate"})
    latent_dim: Optional[int] = field(default=16, metadata={"help": "latent dim size"})
    # l2_lambda: Optional[float] = field(default=0.0, metadata={"help": "l2 regularization lambda"})
    l2_lambda: Optional[float] = field(default=0.007, metadata={"help": "l2 regularization lambda"})
    # tune_l2: Optional[bool] = field(default=True, metadata={"help": "if True, tune l2 regularization lambda as part of hyperparam tuning"})
    l1_latent_lambda: Optional[float] = field(default=0, metadata={"help": "l1 regularization lambda for latent dim"})
    # tune_l1: Optional[bool] = field(default=False, metadata={"help": "if True, tune l1 regularization lambda as part of hyperparam tuning"})
    batch_size: Optional[int] = field(default=32, metadata={"help": "batch size"})
    max_epochs: Optional[int] = field(default=500, metadata={"help": "maximal number of epochs"})
    # use_16bit: Optional[bool] = field(default=False, metadata={"help": "use 16bit precision"})
    # ckpt_dir: Optional[str] = field(default="ckpt/", metadata={"help": "path to save checkpoints"})
    # tb_logs_dir: Optional[str] = field(default="logs/", metadata={"help": "path to save tensorboard logs"})
    save_top_k: Optional[int] = field(default=1, metadata={"help": "save top k checkpoints"})
    tune_hyperparams: Optional[bool] = field(default=False, metadata={"help": "tune hyperparams"})
    n_tuning_trials: Optional[int] = field(default=150, metadata={"help": "number of tuning trials"})
    deep_decoder: Optional[bool] = field(default=False, metadata={"help": "if True, use deep decoder"})
    encoder_type: Optional[str] = field(default='default', metadata={"help": "choice of encoder, options: ['default', 'shallow', 'deep']"})
    max_epochs_in_trial: Optional[int] = field(default=100, metadata={"help": "maximal number of epochs in each trial"})


@dataclass
class EvalArguments:
    num_simulations: Optional[int] = field(default=1000, metadata={"help": "number of simulations"})
    do_baseline: Optional[bool] = field(default=True, metadata={"help": "if True, run baseline"})
    do_original: Optional[bool] = field(default=False, metadata={"help": "if True, run original"})
    by_dose: Optional[bool] = field(default=False, metadata={"help": "if True, run by dose"})
    z_trim: Optional[int] = field(default=None, metadata={"help": "z-score threshold for trimming"})
    normalize_by_all: Optional[bool] = field(default=True, metadata={"help": "if True, normalize by all data including treatment"})
    run_dose_if_exists: Optional[bool] = field(default=False, metadata={"help": "if True, run dose if exists"})
    filter_by_highest_dose: Optional[bool] = field(default=True, metadata={"help": "if True, run only highest dose"})
    with_l1k: Optional[bool] = field(default=False, metadata={"help": "if True, calculate l1k metrics"})
    latent_exp: Optional[bool] = field(default=False, metadata={"help": "if True, run latent dim experiment"})
    load_corr_if_exists: Optional[bool] = field(default=True, metadata={"help": "if True, load correlation if exists"})
    min_max_norm: Optional[bool] = field(default=False, metadata={"help": "if True, min-max normalize"})
    rand_reps: Optional[int] = field(default=3, metadata={"help": "Number of sampling for random correlations"})
    filter_non_reproducible: Optional[bool] = field(default=True, metadata={"help": "if True, filter non-reproducible compounds for SHAP evaluation"})
    run_shap: Optional[bool] = field(default=True, metadata={"help": "if True, run SHAP evaluation"})
    run_moa_classifier: Optional[bool] = field(default=True, metadata={"help": "if True, run moa classification"})


@dataclass
class MoaArguments:
    # models = field(default_factory=list, metadata={"help": "list of models to run"})
    # models: Optional[list] = field(default_factory=['lr','mlp'], metadata={"help": "list of models to run"})
    tune: Optional[bool] = field(default=False, metadata={"help": "if True, tune hyperparams"})
    tune_first_fold: Optional[bool] = field(default=False, metadata={"help": "if True, tune hyperparams only for first fold"})
    filter_perts: Optional[str] = field(default='HighRepUnion', metadata={"help": "options: ['HighRepUnion', 'onlyCP', '']"})
    n_exps: Optional[int] = field(default=1, metadata={"help": "number of experiments"})
    do_fusion: Optional[bool] = field(default=True, metadata={"help": "if True, run fusion"})
    # filter_by_pr: Optional[bool] = field(default=False, metadata={"help": "if True, filter by percentage replication"})
    folds: Optional[int] = field(default=10, metadata={"help": "number of folds for cross validation"})
    # flow: Optional[str] = field(default="run_moa", metadata={"help": ""})
    min_samples: Optional[int] = field(default=4, metadata={"help": "number of samples for each perturbation"})
    # run_dose_if_exists: Optional[bool] = field(default=True, metadata={"help": "if True, run dose if exists"})
    exp_seed: Optional[int] = field(default=42, metadata={"help": "experiment seed"})
    rep_corr_fileName: Optional[str] = field(default='RepCorrDF', metadata={"help": "replicate correlation file name"})
    do_all_filter_groups: Optional[bool] = field(default=False, metadata={"help": "if True, run all filter groups"})
    moa_dirname: Optional[str] = field(default='MoAprediction', metadata={"help": "moa directory name"})
    with_l1k: Optional[bool] = field(default=True, metadata={"help": "if True, run l1k"})
    moa_plate_normalized: Optional[bool] = field(default=True, metadata={"help": "if True, normalize by plate"})


    # filter_groups = field(default_factory=['CP','l1k'], metadata={"help": "if True, filter groups with less than min_samples"})

