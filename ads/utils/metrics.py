from itertools import cycle
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from glob import glob

import os
import sys

sys.path.append(os.path.abspath('..'))


def extract_ss_score_for_compound(cpdf, abs_zscore=True, th_range=range(2, 21)):
    res_cols = ['Rep_Cnt', *sum([[f'SS_{t}'] for t in th_range], [])]
    res = {}
    rep_cnt, fet_cnt = cpdf.shape

    corr = cpdf.astype('float64').T.corr(method='pearson').values
    if len(corr) == 1:
        med_corr = 1
    else:
        med_corr = np.median(list(corr[np.triu_indices(len(corr), k=1)]))

    res['Med_Corr'] = med_corr

    cpdf_norm = cpdf * np.sqrt(rep_cnt)

    if abs_zscore:
        cpdf_norm = abs(cpdf_norm.T)

    for t in th_range:
        gtr_t_cnt = (cpdf_norm >= t).sum().sum()
        ss_norm = gtr_t_cnt / rep_cnt
        mas = np.sqrt((max(med_corr, 0) * ss_norm) / fet_cnt)
        res[f'SS_{t}'] = ss_norm
        res[f'MAS_{t}'] = mas

    return pd.Series(res)


def extract_new_score_for_compound(cpdf, abs_zscore=True, th_range=range(2, 21), sqrt_norm=True, max_rep=None, n=1, value='median'):
    res_cols = ['Rep_Cnt', *sum([[f'SS_{t}'] for t in th_range], [])]
    res = []

    rep_cnt, _ = cpdf.shape
    if max_rep and max_rep >= rep_cnt:
        n = 1

    for _ in range(n):
        cur_res = []

        cur_cpdf = cpdf
        rep_cnt, fet_cnt = cur_cpdf.shape
        if max_rep and max_rep < rep_cnt:
            cur_cpdf = cpdf.sample(max_rep)
            rep_cnt = max_rep

        cur_res.append(rep_cnt)

        cpdf_norm = cur_cpdf
        if abs_zscore:
            cpdf_norm = abs(cpdf_norm)
        
        if value =='mean':
            cpd = cpdf_norm.mean()
        else:
            cpd = cpdf_norm.median()
        if sqrt_norm:
            cpd = cpd * np.sqrt(rep_cnt)

        for t in th_range:
            gtr_t_cnt = (cpd >= t).sum()
            ss_norm = gtr_t_cnt / rep_cnt
            # Add Normalizations for feature count
            ss_norm = ss_norm / fet_cnt
            cur_res.append(ss_norm)

        res.append(cur_res)

    return pd.DataFrame(res, columns=res_cols)


def extract_ss_score(df, th_range=[2, 6, 10, 14], cpd_id_fld='Metadata_broad_sample',new_ss = True, value='mean',abs_zscore=False, save_path=None):

    if new_ss:
        print(f'calc with {value}')
        cur_res = df.groupby(cpd_id_fld).apply(extract_new_score_for_compound, abs_zscore=abs_zscore,
                                               th_range=th_range, value=value)
        # cur_res.to_csv(os.path.join(expr_fld, f'ss-new-scores-{value}.csv'))
        # print(f'saved to {expr_fld}')
    else:
        cur_res = df.groupby(cpd_id_fld).apply(extract_ss_score_for_compound, abs_zscore=abs_zscore,
                                               th_range=th_range)
        # cur_res.to_csv(os.path.join(expr_fld, f'ss-scores.csv'))
        
    # del df
    if save_path is not None:
        cur_res.to_csv(save_path)
    return cur_res


# def extract_ss_score(df, th_range=[2, 6, 10, 14], cpd_id_fld='Metadata_broad_sample',new_ss = True, value='mean',abs_zscore=False):
#     if new_ss:
#         print(f'calc with {value}')
#         cur_res = df.groupby(cpd_id_fld).apply(extract_new_score_for_compound, abs_zscore=abs_zscore,
#                                                th_range=th_range, value=value)
#     else:
#         cur_res = df.groupby(cpd_id_fld).apply(extract_ss_score_for_compound, abs_zscore=abs_zscore,
#                                                th_range=th_range)   
#     return cur_res
#     # del df
    # del cur_res

def extract_dist_score(plate_csv, well_type='treated', **kwargs):
    df = load_plate_csv(plate_csv)
    df = df.groupby(by=['Plate', LABEL_FIELD, 'Metadata_broad_sample', 'Image_Metadata_Well']).apply(
        lambda g: g.mean())

    def calculate_distance_from(v):
        return lambda x: np.linalg.norm(x - v)

    _, _, channels = list_columns(df)
    all_cols = [col for ch_cols in channels.values() for col in ch_cols]
    channels['ALL'] = all_cols

    scores = []
    for channel, cols in channels.items():
        df_mck = df[df.index.isin(['mock'], 1)][cols]
        mck_profile = df_mck.median()
        df_trt = df[df.index.isin([well_type], 1)][cols]

        dist_func = calculate_distance_from(mck_profile)
        trt_dist = df_trt.apply(dist_func, axis=1)
        del df_trt

        trt_dist.name = channel

        scores.append(trt_dist)

    del df

    scores_df = pd.concat(scores, axis=1)
    return scores_df


def extract_dist_score_norm_before(plate_csv, well_type='treated', **kwargs):
    df = load_pure_zscores(plate_csv, kwargs['raw'], kwargs['inter_channel'])

    def calculate_distance_from(v):
        return lambda x: np.linalg.norm(x - v)

    _, _, channels = list_columns(df)
    all_cols = [col for ch_cols in channels.values() for col in ch_cols]
    channels['ALL'] = all_cols

    scores = []
    for channel, cols in channels.items():
        df_mck = df[df.index.isin(['mock'], 1)][cols]
        mck_profile = df_mck.median()
        df_trt = df[df.index.isin([well_type], 1)][cols]

        dist_func = calculate_distance_from(mck_profile)
        trt_dist = df_trt.apply(dist_func, axis=1)
        del df_trt

        trt_dist.name = channel

        scores.append(trt_dist)

    del df

    scores_df = pd.concat(scores, axis=1)
    return scores_df


def extract_dist_score_norm_after(plate_csv, well_type='treated', **kwargs):
    df = load_plate_csv(plate_csv)
    df = df.groupby(by=['Plate', LABEL_FIELD, 'Metadata_broad_sample', 'Image_Metadata_Well']).apply(
        lambda g: g.mean())

    def calculate_distance_from(v):
        return lambda x: np.linalg.norm(x - v)

    _, _, channels = list_columns(df)
    all_cols = [col for ch_cols in channels.values() for col in ch_cols]
    channels['ALL'] = all_cols

    scores = []
    for channel, cols in channels.items():
        df_mck = df[df.index.isin(['mock'], 1)][cols]
        mck_profile = df_mck.median()
        df_trt = df[df.index.isin([well_type], 1)][cols]

        dist_func = calculate_distance_from(mck_profile)
        mck_dist = df_trt.apply(dist_func, axis=1)
        del df_mck
        trt_dist = df_trt.apply(dist_func, axis=1)
        del df_trt

        scaler = StandardScaler()
        scaler.fit(mck_dist.to_numpy().reshape(-1, 1))
        del mck_dist

        cur_scores = pd.Series(scaler.transform(trt_dist.to_numpy().reshape(-1, 1)).reshape(-1),
                               index=trt_dist.index,
                               name=channel)
        del trt_dist

        scores.append(cur_scores)

    del df

    scores_df = pd.concat(scores, axis=1)
    return scores_df


def extract_raw_and_err(score_func, plate_csv, by_well=True, well_type='treated', threshold=None):
    print('.', sep='', end='')

    params = {'by_well': by_well,
              'well_type': well_type,
              }
    if threshold is not None:
        params['thresh'] = threshold

    err = score_func(f'{err_fld}/{plate_csv}', abs_zscore=False, raw=False, inter_channel=True, **params)
    raw = score_func(f'{raw_fld}/{plate_csv}', abs_zscore=True, raw=True, inter_channel=True, **params)

    res = err.join(raw, how='inner', lsuffix='_map', rsuffix='_raw')
    del err
    del raw

    raw1to1 = score_func(f'{raw1to1_fld}/{plate_csv}', abs_zscore=False, raw=True, inter_channel=False, **params)
    res = res.join(raw1to1.add_suffix('_raw1to1'), how='inner')
    del raw1to1

    return res

def parse_boolean(value):
    value = value.lower()

    if value in ["true", "yes", "y", "1", "t"]:
        return True
    elif value in ["false", "no", "n", "0", "f"]:
        return False

    return False

def extract_scores_from_all(score_func, by_well=True, well_type='treated', threshold=None):
    p = Pool(3)

    plates = [f[1] for f in files]
    score_results = p.starmap(extract_raw_and_err,
                              zip(cycle([score_func]), plates, cycle([by_well]), cycle([well_type]),
                                  cycle([threshold])))
    p.close()
    p.join()

    scores = {plate_number: score_results[i] for i, plate_number in enumerate([f[0] for f in files])}
    return scores


## Taken from:
## https://gist.github.com/bwhite/3726239
## Note that this code assumes that the ranking lists provided as
## parameters contain all the relevant documents that need to be
## retrived. In other words, it assumes that the list has no misses.
## It's important when using this code to do that, otherwise the
## results will be artificially inflated. If there is need to compute
## recall metrics, other implementations need to be used.
## MAP slides: https://slideplayer.com/slide/2295316/

"""Information Retrieval metrics
Useful Resources:
http://www.cs.utexas.edu/~mooney/ir-course/slides/Evaluation.ppt
http://www.nii.ac.jp/TechReports/05-014E.pdf
http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
http://hal.archives-ouvertes.fr/docs/00/72/67/60/PDF/07-busa-fekete.pdf
Learning to Rank for Information Retrieval (Tie-Yan Liu)
"""
import numpy as np


def interpolated_precision_recall_curve(Y_true_matrix, Y_predict_matrix):
    """ Compute the average precision / recall curve over all queries in a matrix.
    That is, consider each point in the graph as a query and evaluate all nearest neighbors until
    all positives have been found. Y_true_matrix is a binary matrix, Y_predict_matrix is a
    continuous matrix. Each row in the matrices is a query, and for each one PR curve can be computed.
    Since each PR curve has a different number of recall points, the curves are interpolated to 
    cover the max number of recall points. This is standard practice in Information Retrieval research """

    from sklearn.metrics import precision_recall_curve

    # Suppress self matching
    Y_predict_matrix[np.diag_indices(Y_predict_matrix.shape[0])] = -1 # Assuming Pearson correlation as the metric
    Y_true_matrix[np.diag_indices(Y_true_matrix.shape[0])] = False    # Assuming a binary matrix

    # Prepare axes
    recall_axis = np.linspace(0.0, 1.0, num=Y_true_matrix.shape[0])[::-1]
    precision_axis = []

    # Each row in the matrix is one query
    is_query = Y_true_matrix.sum(axis=0) > 1
    for t in range(Y_true_matrix.shape[0]):
        if not is_query[t]: 
            continue
        # Compute precision / recall for each query
        precision_t, recall_t, _ = precision_recall_curve(Y_true_matrix[t,:], Y_predict_matrix[t,:])

        # Interpolate max precision at all recall points
        max_precision = np.maximum.accumulate(precision_t)
        interpolated_precision = np.zeros_like(recall_axis)

        j = 0
        for i in range(recall_axis.shape[0]):
            interpolated_precision[i] = max_precision[j]
            while recall_axis[i] < recall_t[j]:
                j += 1

        # Store interpolated results for query
        precision_axis.append(interpolated_precision[:,np.newaxis])
    
    return recall_axis, np.mean( np.concatenate(precision_axis, axis=1), axis=1)


def mean_reciprocal_rank(rs):
    """Score is reciprocal of the rank of the first relevant item
    First element is 'rank 1'.  Relevance is binary (nonzero is relevant).
    Example from http://en.wikipedia.org/wiki/Mean_reciprocal_rank
    >>> rs = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.61111111111111105
    >>> rs = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    >>> mean_reciprocal_rank(rs)
    0.5
    >>> rs = [[0, 0, 0, 1], [1, 0, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.75
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean reciprocal rank
    """
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])


def r_precision(r):
    """Score is precision after all relevant documents have been retrieved
    Relevance is binary (nonzero is relevant).
    >>> r = [0, 0, 1]
    >>> r_precision(r)
    0.33333333333333331
    >>> r = [0, 1, 0]
    >>> r_precision(r)
    0.5
    >>> r = [1, 0, 0]
    >>> r_precision(r)
    1.0
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        R Precision
    """
    r = np.asarray(r) != 0
    z = r.nonzero()[0]
    if not z.size:
        return 0.
    return np.mean(r[:z[-1] + 1])


def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    >>> r = [0, 0, 1]
    >>> precision_at_k(r, 1)
    0.0
    >>> precision_at_k(r, 2)
    0.0
    >>> precision_at_k(r, 3)
    0.33333333333333331
    >>> precision_at_k(r, 4)
    Traceback (most recent call last):
        File "<stdin>", line 1, in ?
    ValueError: Relevance score length < k
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)


def average_precision(r):
    """Score is average precision (area under PR curve)
    Relevance is binary (nonzero is relevant).
    >>> r = [1, 1, 0, 1, 0, 1, 0, 0, 0, 1]
    >>> delta_r = 1. / sum(r)
    >>> sum([sum(r[:x + 1]) / (x + 1.) * delta_r for x, y in enumerate(r) if y])
    0.7833333333333333
    >>> average_precision(r)
    0.78333333333333333
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Average precision
    """
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)


def mean_average_precision(rs):
    """Score is mean average precision
    Relevance is binary (nonzero is relevant).
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1]]
    >>> mean_average_precision(rs)
    0.78333333333333333
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1], [0]]
    >>> mean_average_precision(rs)
    0.39166666666666666
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean average precision
    """
    return np.mean([average_precision(r) for r in rs])


def dcg_at_k(r, k, method=0):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> dcg_at_k(r, 1)
    3.0
    >>> dcg_at_k(r, 1, method=1)
    3.0
    >>> dcg_at_k(r, 2)
    5.0
    >>> dcg_at_k(r, 2, method=1)
    4.2618595071429155
    >>> dcg_at_k(r, 10)
    9.6051177391888114
    >>> dcg_at_k(r, 11)
    9.6051177391888114
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=0):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> ndcg_at_k(r, 1)
    1.0
    >>> r = [2, 1, 2, 0]
    >>> ndcg_at_k(r, 4)
    0.9203032077642922
    >>> ndcg_at_k(r, 4, method=1)
    0.96519546960144276
    >>> ndcg_at_k([0], 1)
    0.0
    >>> ndcg_at_k([1], 2)
    1.0
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


if __name__ == "__main__":
    import doctest
    doctest.testmod()