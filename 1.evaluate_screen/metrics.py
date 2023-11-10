from itertools import cycle
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from glob import glob

import os
import sys

sys.path.append(os.path.abspath('..'))
# from hit_finding.constants import *
# from learning_tabular.constants import CHANNELS, LABEL_FIELD
# from learning_tabular.preprocessing import load_plate_csv, list_columns, aggregate_by_well


# def generate_raw_zscores_based_on_test_control(plate_csv, by_well=True, index_fields=None, well_index=None,
#                       dest=None, normalized=True):

#     plate_num = os.path.basename(plate_csv).replace('.csv', '')

#     if dest is None:
#         if normalized:
#             dest = f'/sise/assafzar-group/g-and-n/plates/raw_normalized/'
#         else:
#             dest = f'/sise/assafzar-group/g-and-n/plates/raw/'

#     if os.path.exists(dest):
#         index_size = 4 if by_well else 6
#         return pd.read_csv(dest, index_col=list(range(index_size)))

#     if normalized:
#         control_test_path = f'/sise/assafzar-group/g-and-n/plates/csvs_processed_normalized/{plate_num}_mock_predict.csv'
#         treated_path = f'/sise/assafzar-group/g-and-n/plates/csvs_processed_normalized/{plate_num}_treated_predict.csv'

#     else:
#         control_test_path = f'/sise/assafzar-group/g-and-n/plates/csvs_processed/{plate_num}_mock_predict.csv'
#         treated_path = f'/sise/assafzar-group/g-and-n/plates/csvs_processed/{plate_num}_treated_predict.csv'

#     df_treated = load_plate_csv(treated_path, index_fields=index_fields)
#     df_treated = aggregate_by_well(df_treated,by_well=by_well, well_index=well_index)

#     infs = np.isinf(df_treated)
#     df_treated[infs] = 0
#     print(f'number of infs in treated: {infs.sum().sum()}')
#     print(f'number of na in treated: {df_treated.isna().sum().sum()}')

#     df_mock = load_plate_csv(control_test_path, index_fields=index_fields)
#     df_mock = aggregate_by_well(df_mock, by_well=by_well, well_index=well_index)

#     infs = np.isinf(df_mock)
#     print(f'number of infs in mock: {infs.sum().sum()}')
#     print(f'number of na in mock: {df_mock.isna().sum().sum()}')

#     df_mock[infs] = 0
#     mean_mock = df_mock.mean()
#     std_mock = df_mock.std()
#     std_mock[std_mock == 0] = 1
#     std_mock[std_mock.isna()] = 1

#     # scaler = StandardScaler()
#     # scaler.fit(df_mock)
#     del df_mock

#     # df_treated_normalized = scaler.transform(df_treated)
#     df_treated_normalized = (df_treated - mean_mock)/std_mock
#     df_zscores = pd.DataFrame(df_treated_normalized, index=df_treated.index, columns=df_treated.columns)

#     del df_treated_normalized
#     del df_treated

#     df_zscores.to_csv(dest)

#     return df_zscores


# def load_pure_zscores(plate_csv, raw=False, by_well=True, inter_channel=True, index_fields=None, well_index=None,
#                       dest=None):
#     if dest is None:
#         if raw:
#             if inter_channel:
#                 dest = 'raw'
#             else:
#                 dest = 'raw1to1'
#         else:
#             dest = 'err'
#     # dest = f'{out_fld}/{dest}/{os.path.basename(plate_csv)}'

#     if os.path.exists(dest):
#         index_size = 4 if by_well else 6
#         return pd.read_csv(dest, index_col=list(range(index_size)))

#     df = load_plate_csv(plate_csv, index_fields=index_fields)
#     df = aggregate_by_well(df,by_well=by_well, well_index=well_index)

#     df_mock = df[df.index.isin(['mock'], 1)]
#     if not df_mock.shape[0]:
#         print(f'no mock wells in {os.path.basename(plate_csv)}')
#         return None

#     scaler = StandardScaler()
#     scaler.fit(df_mock)
#     del df_mock

#     df_zscores = pd.DataFrame(scaler.transform(df), index=df.index, columns=df.columns)
#     del df

#     df_zscores.to_csv(dest)

#     return df_zscores


# def extract_pure(plate_csv, index_fields=None, well_index=None):
#     print('.', end='')

#     err = load_pure_zscores(f'{err_fld}/{plate_csv}', raw=False,
#                             index_fields=index_fields, well_index=well_index)
#     del err
#     # raw = load_pure_zscores(f'{raw_fld}/{plate_csv}', raw=True)
#     # del raw
#     raw1to1 = load_pure_zscores(f'{raw1to1_fld}/{plate_csv}', raw=True, inter_channel=False,
#                                 index_fields=index_fields, well_index=well_index)
#     del raw1to1


# def extract_z_score(plate_csv, by_well=True, by_channel=True, abs_zscore=True, well_type='treated', raw=False,
#                     inter_channel=True):
#     df = load_pure_zscores(plate_csv, raw, inter_channel)

#     if well_type in ['treated', 'mock']:
#         df_selected = df[df.index.isin([well_type], 1)]
#         del df
#     else:
#         df_selected = df

#     if abs_zscore:
#         df_selected = df_selected.abs()

#     if by_channel:
#         _, _, channels = list_columns(df_selected)
#         for channel, cols in channels.items():
#             df_selected[channel] = df_selected[cols].mean(axis=1)

#         channels_cols = [col for ch_cols in channels.values() for col in ch_cols]
#         df_selected["ALL"] = df_selected[channels_cols].mean(axis=1)

#         data = df_selected[CHANNELS + ["ALL"]]
#         del df_selected
#     else:
#         data = df_selected

#     if by_well:
#         gb = data.groupby(by=['Plate', 'Metadata_broad_sample', 'Image_Metadata_Well'])
#         del data

#         by_trt = gb.apply(lambda g: g.mean())
#         return by_trt

#     return data


# def extract_score(plate_csv, by_well=True, by_channel=True, abs_zscore=True, well_type='treated', raw=False, thresh=4,
#                   inter_channel=True):
#     df = load_pure_zscores(plate_csv, raw, inter_channel)

#     if well_type in ['treated', 'mock']:
#         df_selected = df[df.index.isin([well_type], 1)]
#         del df
#     else:
#         df_selected = df

#     if abs_zscore:
#         df_selected = df_selected.abs()

#     df_selected = df_selected.apply(lambda x: x.apply(lambda y: 0 if y < thresh else 1))

#     if by_channel:
#         _, _, channels = list_columns(df_selected)
#         for channel, cols in channels.items():
#             df_selected[channel] = df_selected[cols].sum(axis=1) / len(cols)

#         channels_cols = [col for ch_cols in channels.values() for col in ch_cols]
#         df_selected["ALL"] = df_selected[channels_cols].sum(axis=1) / len(channels_cols)

#         data = df_selected[CHANNELS + ["ALL"]]
#         del df_selected
#     else:
#         data = df_selected

#     if by_well:
#         gb = data.groupby(by=['Plate', 'Metadata_broad_sample', 'Image_Metadata_Well'])
#         del data

#         by_trt = gb.apply(lambda g: g.mean())
#         return by_trt

#     return data


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


def extract_ss_score(df, expr_fld, th_range=[2, 6, 10, 14], cpd_id_fld='Metadata_broad_sample',new_ss = True, value='mean',abs_zscore=False):
    if new_ss:
        print(f'calc with {value}')
        cur_res = df.groupby(cpd_id_fld).apply(extract_new_score_for_compound, abs_zscore=abs_zscore,
                                               th_range=th_range, value=value)
        cur_res.to_csv(os.path.join(expr_fld, f'ss-new-scores-{value}.csv'))
        print(f'saved to {expr_fld}')
    else:
        cur_res = df.groupby(cpd_id_fld).apply(extract_ss_score_for_compound, abs_zscore=abs_zscore,
                                               th_range=th_range)
        cur_res.to_csv(os.path.join(expr_fld, f'ss-scores.csv'))
        
    del df
    del cur_res


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


if __name__ == '__main__':
    print('metrics main')
    print("Usage: metrics.py -p [experiment_path] [plate_index]")
    print("Usage: metrics.py -s [experiment_path] [channel_index]")

    # if run directly from
    if len(sys.argv) == 1:
        exp_fld = "/storage/users/g-and-n/tabular_models_results/5012"

        channels = glob(os.path.join(exp_fld, '*'))
        num_channels = len(channels)
        num_plates = len(glob(os.path.join(exp_fld, channels[0], 'results', '*')))

        # try:
        channel_idx = 0
        compute_zscores = False
        compute_raw = False

        if compute_raw:
            normalized =True
            # if normalized:
            exp_fld = "/storage/users/g-and-n/plates/"
            # else:
            #     exp_fld = "/storage/users/g-and-n/plates/csv_processed"

            plates = glob(os.path.join(exp_fld, 'csvs', '*'))
            plate_idx = 0
            plate = plates[plate_idx]
            if normalized:
                out_fld = os.path.join(exp_fld + 'raw_normalized')
            else:
                out_fld = os.path.join(exp_fld + 'raw_normalized')

            os.makedirs(out_fld, exist_ok=True)

            dest = os.path.join(out_fld, os.path.basename(plate))
            generate_raw_zscores_based_on_test_control(plate, by_well=True,
                              index_fields=None,
                              well_index=None,normalized=True,
                              dest=dest)

        if compute_zscores:  # Means to run pure zscores run
            plates = glob(os.path.join(exp_fld, '*', 'results', '*'))
            plate_idx = 0
            plate = plates[plate_idx]
            out_fld = os.path.join(exp_fld, 'zscores')
            os.makedirs(out_fld, exist_ok=True)
            dest = os.path.join(out_fld, os.path.basename(plate))
            print(f'Extract pure z-scores for plate {plate}')
            load_pure_zscores(plate, by_well=True,
                              index_fields=None,
                              well_index=None,
                              dest=dest)
            print('Done!')

        else:  # Means to extract ss scores

            # try:
            channel_idx = 0
            plates = glob(os.path.join(exp_fld, 'zscores', '*'))
            cur_fld = channels[0]
            cur_df = pd.concat([pd.read_csv(pth, index_col=[0, 1, 2, 3]) for pth in plates])
            cur_df = cur_df.query('Metadata_ASSAY_WELL_ROLE == "treated"').droplevel(1)
            print(f'start running for {cur_fld}')
            extract_ss_score(cur_df, cur_fld, th_range=range(2, 21), cpd_id_fld='Metadata_broad_sample', new_ss=True,
                             value='median')
            # except:
            #     print(f'Error while reading channel {sys.argv[3]}')


    else:
        # when running -z from terminal - use tasks = plates * channels
        if sys.argv[1] == '-z':  # Means to run zscores on results

            exp_fld = sys.argv[2]
            plates = glob(os.path.join(exp_fld, '*', 'results', '*'))
            try:
                plate_idx = int(sys.argv[3])
                plate = plates[plate_idx]
                z_fld = os.path.join(exp_fld,plate.split('/')[-3], 'zscores')
                os.makedirs(z_fld, exist_ok=True)
                dest = os.path.join(z_fld, os.path.basename(plate))
                print(f'Extract pure z-scores for plate {plate}')
                load_pure_zscores(plate, by_well=True,
                                  index_fields=None,
                                  well_index=None,
                                  dest=dest)
                print('Done!')
            except:
                print(f'Error while reading plate {sys.argv[3]}')

        elif sys.argv[1] == '-p':  # Means to run pure zscores run (for single channel) (The working one!)

            exp_fld = sys.argv[2]
            plates = glob(os.path.join(exp_fld, '*', 'results', '*'))
            try:
                plate_idx = int(sys.argv[3])
                plate = plates[plate_idx]
                out_fld = os.path.join(exp_fld, 'zscores')
                os.makedirs(out_fld, exist_ok=True)
                dest = os.path.join(out_fld, os.path.basename(plate))
                print(f'Extract pure z-scores for plate {plate}')
                load_pure_zscores(plate, by_well=True,
                                  index_fields=None,
                                  well_index=None,
                                  dest=dest)
            except:
                print(f'Error while reading plate {sys.argv[3]}')

        # running raw zscores
        elif sys.argv[1] == '-r':

            normalized = True
            # if normalized:
            exp_fld = "/storage/users/g-and-n/plates/"
            # else:
            #     exp_fld = "/storage/users/g-and-n/plates/csv_processed"
            plates = glob(os.path.join(exp_fld, 'csvs', '*'))
            # plate_idx = 0
            plate = plates[int(sys.argv[3])]
            if normalized:
                out_fld = os.path.join(exp_fld + 'raw_normalized')
            else:
                out_fld = os.path.join(exp_fld + 'raw')

            os.makedirs(out_fld, exist_ok=True)

            dest = os.path.join(out_fld, os.path.basename(plate))
            generate_raw_zscores_based_on_test_control(plate, by_well=True,
                              index_fields=None,
                              well_index=None,normalized=True,
                              dest=dest)


        # when running -z from terminal - use tasks = channels
        elif sys.argv[1] == '-s':  # Means to extract ss scores
            exp_fld = sys.argv[2]
            new_ss=True
            value = 'median'

            channels = glob(os.path.join(exp_fld, '*'))
            if 'zscores' in channels:
                channels.remove('zscores')
            # try:
            channel_idx = int(sys.argv[3])

            if len(sys.argv) > 4:
                new_ss = parse_boolean(sys.argv[4])
                # new_ss = False
                value = sys.argv[5]
                print(new_ss)
                print(value)


            # new_ss = sys.argv[4]
            # method = sys.argv[5]
            cur_fld = channels[channel_idx]
            plates = glob(os.path.join(cur_fld, 'zscores', '*'))
            cur_df = pd.concat([pd.read_csv(pth, index_col=[0, 1, 2, 3]) for pth in plates])
            cur_df = cur_df.query('Metadata_ASSAY_WELL_ROLE == "treated"').droplevel(1)
            print(f'start running for {cur_fld}')
            extract_ss_score(cur_df, cur_fld, th_range=range(2, 10), cpd_id_fld='Metadata_broad_sample',new_ss = new_ss, value=value)
            # except:
            #     print(f'Error while reading channel {sys.argv[3]}')

    # For Visual Results
    # idx_fld = ['Plate', 'Well_Role', 'Broad_Sample', 'Well', 'Site', 'ImageNumber']
    # wll_idx = ['Plate', 'Well_Role', 'Broad_Sample', 'Well']
