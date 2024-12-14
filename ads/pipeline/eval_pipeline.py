from utils.global_variables import ABRVS,DS_INFO_DICT, HYPER_PARAMS
from eval_layer.calc_reproducibility import calc_percent_replicating as calc_percent_replicating
from eval_layer.classify_moa import run_moa_classifier
from eval_layer.shap_anomalies import run_anomaly_shap


def eval_pipeline(configs):
    """
    Entry point for the evaluation pipeline.
    Args:
        configs (dict): Configuration dictionary.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting evaluation pipeline.")
    
    res = {}
    res['rc'] = calc_percent_replicating(configs,data_reps=data_reps)
    
    if configs.eval.run_shap:
        res['shap'] = run_anomaly_shap(configs, filter_non_reproducible=configs.eval.filter_non_reproducible)
    if configs.eval.run_moa_classifier:
        res['moa'] = run_moa_classifier(configs, data_reps = data_reps)

    logger.info("Evaluation pipeline completed.")


# def eval_results(configs):
#     """
#     Evaluate the results of the pipeline.
#     Args:
#         configs (dict): Configuration dictionary.
#     """
#     data_reps = configs.data.data_reps

#     if configs.eval.run_dose_if_exists and DS_INFO_DICT[configs.general.dataset]['has_dose']:
#         doses = [False,True]
#     else:
#         doses = [False]

#     rc = defaultdict(dict)
#     for d in doses:
#             configs.eval.by_dose = d
#             configs.eval.normalize_by_all = True
#             configs.general.logger.info(f'Running null distributions for dose {d} and normalize_by_all = {configs.eval.normalize_by_all})
#             rc = calc_percent_replicating(configs,data_reps=data_reps)

#     run_moa = False
#     if run_moa:
#         for d in doses:
#             configs.eval.by_dose = d
#             configs.eval.normalize_by_all = True
#             run_moa_classifier(configs, data_reps = data_reps)
#             # configs.eval.normalize_by_all = False
#             # run_classifier(configs, data_reps = data_reps)

