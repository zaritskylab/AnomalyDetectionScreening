import os
from dataclasses import dataclass, field
from typing import Optional
from datetime import date
from typing_extensions import Literal
from ads.utils.global_variables import BASE_DIR
from collections import defaultdict
# from transformers import TrainingArguments
# import transformers


@dataclass
class GeneralArguments:
    flow: Optional[str] = field(default="run_ad", metadata={"help": ""})
    debug_mode: Optional[bool] = field(default=False, metadata={"help": ""})
    exp_name: Optional[str] = field(default=date.today().strftime("%d_%m"), metadata={"help": "experiment name"})
    seed: Optional[int] = field(default=42, metadata={"help": "experiment seed"})
    gpus: Optional[int] = field(default=1, metadata={"help": "number of gpus to use"})
    base_dir: Optional[str] = field(default=BASE_DIR, metadata={"help": "base directory"})
    # supported datasets =['CDRP-bio','CDRP', 'LINCS', 'LUAD', 'TAORF'])

    tune_ldims: Optional[bool] = field(default=False, metadata={"help": "if True, tune latent dims"})
    dataset: Optional[str] = field(default='CDRP-bio', metadata={"help": "The name of the dataset"})
    run_both_profiles: Optional[bool] = field(default=False, metadata={"help": "if True, run both profiles"})
    exp_num: Optional[int] = field(default=0, metadata={"help": "experiment number"})
    run_all_datasets: Optional[bool] = field(default=False, metadata={"help": "if True, run all datasets"})

    # modality: Optional[str] = field(default='CellPainting', metadata={"help": "The name of the modality"})

    # output_dir: Optional[str] = field(default=f"/sise/assafzar-group/assafzar/genesAndMorph/processed_data/{dataset}/{modality}/{exp_name}", metadata={"help": "output directory"})
    # res_dir: Optional[str] = field(default=f"/sise/assafzar-group/assafzar/genesAndMorph/results/{dataset}/{modality}/{exp_name}", metadata={"help": "results directory"})
    # fig_dir: Optional[str] = field(default=f"/sise/assafzar-group/assafzar/genesAndMorph/results/{dataset}/{modality}/{exp_name}/figs", metadata={"help": "figures directory"})


@dataclass
class DataArguments:
    modality: Optional[str] = field(default='CellPainting', metadata={"help": "The name of the modality"})
    modality_str: Optional[str] = field(default='cp', metadata={"help": "The name of the modality"})
    profile_type: Optional[str] = field(default='augmented', metadata={"help": "The name of the dataset"})
                                    #   choices=['augmented', 'normalized', 'normalized_variable_selected'])
    test_split_ratio: Optional[float] = field(default=0.5, metadata={"help": "split ratio for test set"})
    val_split_ratio: Optional[float] = field(default=0.2, metadata={"help": "split ratio for between train set and val set"})
    normalize_condition: Optional[str] = field(default='train', metadata={"help": "if train, normalize by train set. \
                                            if DMSO, normalize by DMSO. else, normalize by all data"})
    plate_normalized: Optional[bool] = field(default=True, metadata={"help": "if True, normalize by plate"})
    # by_plate: Optional[bool] = field(default=True, metadata={"help": "if True, normalize by plate"})
    norm_method: Optional[str] = field(default='standardize', metadata={"help": "normalization method."}) \

    run_data_process: Optional[bool] = field(default=False, metadata={"help": "if True, run data process even if data already exists"})
    overwrite_data_creation: Optional[bool] = field(default=False, metadata={"help": "if True, overwrite data even if already exists"})
    feature_select: Optional[bool] = field(default=True, metadata={"help": "if True, run feature selection. \
                                            if False, use features selected in 'normalized_variable_selected' profile"})    
    corr_threshold: Optional[float] = field(default=0.9, metadata={"help": "correlation threshold for feature selection"})
    remove_non_normal_features: Optional[bool] = field(default=True, metadata={"help": "if True, remove non-normal features"})           

@dataclass
class ModelArguments:
    model_type: Optional[str] = field(default="AE", metadata={"help": "model type"})
    model_path: Optional[str] = field(default="None",
                                      metadata={"help": "If we want to load pre-trained model from path"})
    lr: Optional[float] = field(default=0.005, metadata={"help": "learning rate"})
    dropout: Optional[float] = field(default=0.0, metadata={"help": "dropout rate"})
    latent_dim: Optional[int] = field(default=16, metadata={"help": "latent dim size"})
    # l2_lambda: Optional[float] = field(default=0.0, metadata={"help": "l2 regularization lambda"})
    l2_lambda: Optional[float] = field(default=0.007, metadata={"help": "l2 regularization lambda"})
    batch_size: Optional[int] = field(default=32, metadata={"help": "batch size"})
    max_epochs: Optional[int] = field(default=500, metadata={"help": "maximal number of epochs"})
    use_16bit: Optional[bool] = field(default=False, metadata={"help": "use 16bit precision"})
    # ckpt_dir: Optional[str] = field(default="ckpt/", metadata={"help": "path to save checkpoints"})
    # tb_logs_dir: Optional[str] = field(default="logs/", metadata={"help": "path to save tensorboard logs"})
    save_top_k: Optional[int] = field(default=1, metadata={"help": "save top k checkpoints"})
    tune_hyperparams: Optional[bool] = field(default=False, metadata={"help": "tune hyperparams"})
    n_tuning_trials: Optional[int] = field(default=150, metadata={"help": "number of tuning trials"})
    deep_decoder: Optional[bool] = field(default=False, metadata={"help": "if True, use deep decoder"})
    encoder_type: Optional[str] = field(default='default', metadata={"help": "choice of encoder, options: ['default', 'small', 'large']"})


@dataclass
class EvalArguments:
    num_simulations: Optional[int] = field(default=1000, metadata={"help": "number of simulations"})
    do_baseline: Optional[bool] = field(default=True, metadata={"help": "if True, run baseline"})
    do_original: Optional[bool] = field(default=False, metadata={"help": "if True, run original"})
    by_dose: Optional[bool] = field(default=False, metadata={"help": "if True, run by dose"})
    z_trim: Optional[int] = field(default=8, metadata={"help": "z-score threshold for trimming"})
    slice_id: Optional[int] = field(default=None, metadata={"help": "slice id in case of sbatch run"})
    normalize_by_all: Optional[bool] = field(default=True, metadata={"help": "if True, normalize by all data including treatment"})
    run_dose_if_exists: Optional[bool] = field(default=True, metadata={"help": "if True, run dose if exists"})
    filter_by_highest_dose: Optional[bool] = field(default=True, metadata={"help": "if True, run only highest dose"})


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
    run_l1k: Optional[bool] = field(default=True, metadata={"help": "if True, run l1k"})
    

    # filter_groups = field(default_factory=['CP','l1k'], metadata={"help": "if True, filter groups with less than min_samples"})

