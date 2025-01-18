import os

import pandas as pd
import pycytominer
from pycytominer import feature_select
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from torch.utils.data import DataLoader
from data.data_utils import save_profiles, get_features
from data.tabular_dataset import TabularDataset
from utils.global_variables import DS_INFO_DICT, MODALITY_STR


##########################################################

def construct_dataloaders(data, batch_size,features):
  '''
  construct dataloaders from the given data

  Args:
  data: pandas dataframe. rows are samples and columns are features
  batch_size: int. size of the batch
  features: list of strings. features to be used by the model
  '''

  datasets = to_datasets(data,features)

  # construct dataloaders
  dataloaders = {}
  # num_workers = int(min(os.cpu_count(), batch_size)/4)
  # num_workers = 2
  for key in datasets.keys():
    if key == 'train':
      dataloaders[key] = DataLoader(datasets[key],batch_size,shuffle=True)
    else:
      dataloaders[key] = DataLoader(datasets[key], batch_size)

  return dataloaders


##########################################################


def to_datasets(data, features):
  '''
  separate data to train, val, test_ctrl, and test_treat sets
  '''

  sets = {
      'train': data.loc[data['Metadata_set'] == 'train', :],
      'val': data.loc[data['Metadata_set'] == 'val', :],
      'test_ctrl': data.loc[data['Metadata_set'] == 'test_ctrl', :],
      'test_treat': data.loc[data['Metadata_set'] == 'test_treat', :]
    }
  # construct dataset
  datasets = {}
  for key in sets.keys():
    datasets[key] = TabularDataset(sets[key],features)

  return datasets



##########################################################

def pre_process(data, configs,data_reps = ['ae_diff','baseline']):

  features, meta_features = get_features(data,configs.data.modality)

  ######### data normalization by training set #########
  train_filename = f'replicate_level_{MODALITY_STR[configs.data.modality]}_{configs.data.profile_type}_normalized_by_train'
  train_path = os.path.join(configs.general.output_dir, f'{train_filename}.csv')
  save_data_flag = not (os.path.exists(train_path) and configs.data.use_cache)

  # load from cache if exists
  if os.path.exists(train_path) and configs.data.use_cache:
      print('loading normalized data from file...')
      df_normalized_feature_selected = pd.read_csv(train_path,compression='gzip')
      features = get_features(df_normalized_feature_selected,configs.data.modality)[0]

  else:
    # split data with equal samples from different plates (DS_INFO_DICT[configs.general.dataset[3]])
    data_splitted = split_data(data, configs.general.dataset, configs.data.test_split_ratio, modality=configs.data.modality, n_samples_for_training=configs.data.n_samples_for_training,random_state=configs.general.seed)
    meta_features += ['Metadata_set']
    
    log_data_stats(data_splitted, configs)

    if configs.data.profile_type == 'augmented':

      print('normalizing to training set...')
      # normalized_df_by_train = normalize(data_splitted,features, configs.data.modality,norm_method="standardize", normalize_condition = configs.data.normalize_condition,plate_normalized=configs.data.plate_normalized, clip_outliers=False)
      df_normalized = normalize(data_splitted,features, configs.data.modality,norm_method="standardize", normalize_condition = configs.data.normalize_condition,plate_normalized=configs.data.plate_normalized)

    ########## feature selection based on all samples ##########
    if configs.data.profile_type in ['normalized','augmented']:
      
      print('feature selection after normalization...')
      configs.general.logger.info(f'number of features before feature selection: {len(features)}')

      fs_ops = ['drop_na_columns', 'variance_threshold', 'correlation_threshold', 'drop_outliers']
      df_normalized_by_all_feature_selected = feature_select(
        df_normalized,
        features=features,
        operation=fs_ops,
        samples='all',
        na_cutoff=0.05,
        corr_threshold=configs.data.corr_threshold,
        corr_method="pearson",
        freq_cut=0.05,
        unique_cut=0.01,
        outlier_cutoff=500,
      )
      
      new_features = [f for f in df_normalized_by_all_feature_selected.columns if f.startswith('Cells_') or f.startswith('Cytoplasm_') or f.startswith('Nuclei_')]
      cols_to_drop = [f for f in features if f not in new_features]

      # add all granulirity features back for the CDRP test
      # if configs.general.dataset == 'CDRP-bio':
        # granurality_agp_features = sorted([f for f in features if ('Granularity' in f) and ('AGP' in f)])
        
        # features = list(set(features_selected + granurality_agp_features))
        # all_features = meta_features + features
        # df_normalized_feature_selected = df_normalized[all_features]
      # else:
        # features = features_selected
      features = new_features
      df_normalized_feature_selected =  df_normalized.drop(cols_to_drop, axis=1)

    elif configs.data.profile_type == 'normalized_variable_selected':
      
      df_normalized_feature_selected = data_splitted.copy()
      
    # normalized_df_by_train = normalized_df_by_train.drop(cols_to_drop, axis=1)


    if not configs.general.debug_mode:
      save_profiles(df_normalized_feature_selected, configs.general.output_dir, train_filename)

    raw_filename = f'replicate_level_{MODALITY_STR[configs.data.modality]}_{configs.data.profile_type}_baseline'
    raw_path = os.path.join(os.path.join(configs.general.output_dir, raw_filename))

    save_data_flag = not (os.path.exists(raw_path) or configs.data.use_cache)

    if save_data_flag:

      test_indices = df_normalized_feature_selected['Metadata_set'].isin(['test_ctrl','test_treat'])
      test_data = df_normalized_feature_selected.loc[test_indices, :]
      # if not 'variable_selected' in configs.data.profile_type:
        # test_data = test_data.drop(cols_to_drop, axis=1)
      # test_data.drop(cols_to_drop, axis=1)

      for d in data_reps[1:]:

        print(f'calclating raw measurements for {d}...')
        # try:
        if 'baseline' in d:
          raw_filename = f'replicate_level_{MODALITY_STR[configs.data.modality]}_{configs.data.profile_type}_{d}'
          raw_data = normalize(test_data,features, configs.data.modality, normalize_condition = 'test_ctrl',plate_normalized=configs.data.plate_normalized)

        if save_data_flag:
          raw_filename = f'replicate_level_{MODALITY_STR[configs.data.modality]}_{configs.data.profile_type}_{d}'
          save_profiles(raw_data, configs.general.output_dir, raw_filename)


  return df_normalized_feature_selected, features

##########################################################

def log_data_stats(data, configs):

    configs.general.logger.info(f'number of plates: {len(data[data["Metadata_set"] == "test_ctrl"]["Metadata_Plate"].unique())}')
    configs.general.logger.info(f'number of samples in train set: {len(data[data["Metadata_set"] == "train"])}')
    configs.general.logger.info(f'number of samples in val set: {len(data[data["Metadata_set"] == "val"])}')
    configs.general.logger.info(f'number of samples in test_ctrl set: {len(data[data["Metadata_set"] == "test_ctrl"])}')
    configs.general.logger.info(f'number of samples in test_treat set: {len(data[data["Metadata_set"] == "test_treat"])}')
    configs.general.logger.info(f'median number of replicates per compound: {data[data["Metadata_set"] == "test_treat"].groupby(DS_INFO_DICT[configs.general.dataset][configs.data.modality]["cpd_col"]).size().median()}')
    if DS_INFO_DICT[configs.general.dataset]['has_dose']:
      configs.general.logger.info(f'median number of replicates per compound and dose: {data[data["Metadata_set"] == "test_treat"].groupby([DS_INFO_DICT[configs.general.dataset][configs.data.modality]["cpd_col"],DS_INFO_DICT[configs.general.dataset][configs.data.modality]["dose_col"]]).size().median()}')

##########################################################

def split_data(data_preprocess, dataset, test_split_ratio, val_split_ratio=.2, modality = 'CellPainting', n_samples_for_training = None,random_state = 42):

    """ 
    Split data to train, val, test_mocks, and test_treated sets.
    data: pandas dataframe
    dataset: string. options: ['CDRP','CDRP-bio','TAORF','LUAD','LINCS']
    test_split_ratio: float. ratio of test set from all data
    val_split_ratio: float. ratio of validation set from train set
    """
    if modality == 'CellPainting':
      pertColName = 'PERT'
    else:
      pertColName = 'pert_id'
    # control_data = data_preprocess[data_preprocess[ds_info_dict[dataset][2][0]] == ds_info_dict[dataset][2][1]]
    if dataset == 'TAORF':
      # control_data = data_preprocess[data_preprocess[DS_INFO_DICT[dataset][modality]['role_col']].isin(['CTRL','Untreated'])] 
      # control_data = data_preprocess[data_preprocess[DS_INFO_DICT[dataset][modality]['role_col']].isin(['Untreated'])]
      control_data = data_preprocess[data_preprocess[DS_INFO_DICT[dataset][modality]['role_col']].isin(['CTRL'])]
    else:
      control_data = data_preprocess[data_preprocess[DS_INFO_DICT[dataset][modality]['role_col']] == DS_INFO_DICT[dataset][modality]['mock_val']]
    # split data with equal samples from different plates (DS_INFO_DICT[configs.general.dataset[3]])
    splitter = StratifiedShuffleSplit(test_size=test_split_ratio, n_splits=1,random_state=random_state)

    split = splitter.split(X = control_data, y=control_data[DS_INFO_DICT[dataset][modality]['plate_col']])
    train_all_inds, test_inds = next(split)

    train_data_all = control_data.iloc[train_all_inds]
    test_data_mocks = control_data.iloc[test_inds]

    # train_data, val_data = train_test_split(mock_data, test_size=0.4)


    if modality == 'CellPainting':
      splitter = StratifiedShuffleSplit(test_size=.1, n_splits=1, random_state=random_state)
      split = splitter.split(X = train_data_all, y=train_data_all[DS_INFO_DICT[dataset][modality]['plate_col']])
    else:
      # not enough samples in each plate for stratified split 
      splitter = ShuffleSplit(test_size=.2, n_splits=1, random_state=random_state)
      split = splitter.split(X = train_data_all)

    train_inds, val_inds = next(split)

    train_data = train_data_all.iloc[train_inds]
    val_data = train_data_all.iloc[val_inds]

    # for analysis with different number of samples
    if n_samples_for_training is not None and n_samples_for_training < len(train_data):
      n_plates = len(train_data[DS_INFO_DICT[dataset][modality]['plate_col']].unique())
      if n_plates < n_samples_for_training:
        samples_percentage = 1 - n_samples_for_training/len(train_data)
        splitter = ShuffleSplit(test_size=samples_percentage, n_splits=1,random_state=random_state)
        split = splitter.split(X = train_data)
        train_inds, inds_to_leave_out = next(split)
        train_data = train_data.iloc[train_inds]
      else:
        train_data = train_data.sample(n=n_samples_for_training)
      
    data_splitted = data_preprocess.copy()

    #### assert there are no overlapping indices
    assert len(set(train_data.index.values).intersection(set(val_data.index.values))) == 0
    assert len(set(train_data.index.values).intersection(set(test_data_mocks.index.values))) == 0
    assert len(set(val_data.index.values).intersection(set(test_data_mocks.index.values))) == 0

    data_splitted.loc[train_data.index.values, 'Metadata_set'] = 'train'
    data_splitted.loc[val_data.index.values, 'Metadata_set'] = 'val'
    data_splitted.loc[test_data_mocks.index.values, 'Metadata_set'] = 'test_ctrl'
    # treat_indices = data_splitted[DS_INFO_DICT[dataset][modality]['role_col']] != DS_INFO_DICT[dataset][modality]['mock_val']
    if dataset == 'TAORF':
      treat_indices = (data_splitted[DS_INFO_DICT[dataset][modality]['role_col']] != 'Untreated') & (data_splitted[DS_INFO_DICT[dataset][modality]['role_col']] != 'CTRL')
    else:
      treat_indices = data_splitted[DS_INFO_DICT[dataset][modality]['role_col']] != DS_INFO_DICT[dataset][modality]['mock_val']
    
    data_splitted.loc[treat_indices, 'Metadata_set'] = 'test_treat'

    return data_splitted


##########################################################

def normalize(df, features=None,modality='CellPainting',dataset='CDRP-bio', normalize_condition = "train", plate_normalized=False, norm_method = "standardize"):
    """ Normalize by plate and return normalized df.

    norm_method: str. normalization method. default: "mad_robustize", options: ["mad_robustize", "standardize", "normalize", "spherize"]
    plate_col: str. plate column name. default: "Metadata_Plate"
    well_col: str. well column name. default: "Metadata_Well"
  
    
    
    """
    plate_col = DS_INFO_DICT[dataset][modality]['plate_col']
    cpd_col = DS_INFO_DICT[dataset][modality]['cpd_col']
    
    if features is None:
      features, meta_features = get_features(df, modality)

    if normalize_condition == "train":
        query = f"Metadata_set == 'train'"
    elif normalize_condition == "test_ctrl":
        query = f"Metadata_set == 'test_ctrl'"
    elif normalize_condition == "DMSO":
      query = f"{cpd_col} == 'DMSO'"
    else:
      query = "all"
    # strata = [plate_col, well_col]
    
    if plate_normalized:
        normalized_data = []
        for plate_name in df[plate_col].unique():

          # Normalize Profiles (DMSO Control) - Level 4A Data
          plate_data = df[df[plate_col]==plate_name]      # norm_dmso_file = pathlib.PurePath(data_path, f"{plate_name}_normalized_train.csv.gz")
          normalized_plate = pycytominer.normalize(
            profiles=plate_data,
            samples=query,
            features=features,
            # samples="Metadata_broad_sample == 'DMSO'",
            method=norm_method,
            # output_file=norm_dmso_file,
            # float_format=float_format,
            # compression_options=compression,
            # spherize_epsilon=1e-3,
            # spherize_method=spherize_method
            # mad_robustize_epsilon = 1e-6
          )
          normalized_data.append(normalized_plate)

        normalized_df = pd.concat(normalized_data)
    else:
        # plate_data = df[df[plate_col]==plate_name]
          # norm_dmso_file = pathlib.PurePath(data_path, f"{plate_name}_normalized_train.csv.gz")
        
        normalized_df = pycytominer.normalize(
          profiles=df,
          samples=query,
          features=features,
          # samples="Metadata_broad_sample == 'DMSO'",
          method=norm_method,
          # output_file=norm_dmso_file,
          # float_format=float_format,
          # compression_options=compression,
          # spherize_method=spherize_method
          # mad_robustize_epsilon = 1e-6
        )


    isna = normalized_df[features].isna().sum().sum()/(normalized_df.shape[0]*len(features))
    print(f'ratio of null values after normalization: {isna}')
    normalized_df.loc[:,features] = normalized_df.loc[:,features].interpolate()
    isna = normalized_df[features].isna().sum().sum()
    if isna > 0:
      raise ValueError('null values after interpolation')

    return normalized_df


