import os

# from readProfiles import read_replicate_level_profiles
import numpy as np
import pandas as pd
import pathlib as pathlib
import pycytominer
from pycytominer import feature_select
from pycytominer.operations import variance_threshold, get_na_columns, correlation_threshold
from sklearn import preprocessing
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit, train_test_split,ShuffleSplit
from torch.utils.data import DataLoader, Dataset
from utils.readProfiles import read_replicate_level_profiles, save_profiles
from utils.global_variables import DS_INFO_DICT
from utils.file_utils import load_list_from_txt, save_list_to_txt
#TODO: transform each dataset info to dicts
# ds_info_dict={'CDRP':['CDRP-BBBC047-Bray',['Metadata_Sample_Dose','pert_sample_dose'],['Metadata_ASSAY_WELL_ROLE','mock'],'Metadata_Plate'],
#               'CDRP-bio':['CDRPBIO-BBBC036-Bray',['Metadata_Sample_Dose','pert_sample_dose'],['Metadata_ASSAY_WELL_ROLE','mock'],'Metadata_Plate'],
#               'TAORF':['TA-ORF-BBBC037-Rohban',['Metadata_broad_sample','pert_id',]],
#               'LUAD':['LUAD-BBBC041-Caicedo',['x_mutation_status','allele']],
#               'LINCS':['LINCS-Pilot1',['Metadata_pert_id_dose','pert_id_dose'],['Metadata_pert_type','control'],'Metadata_plate_map_name']}


# index_fields =


class TabularDataset(Dataset):
  def __init__(self, data):
    self.data = data.to_numpy().astype(np.float32)

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    return self.data[idx]


def load_data(base_dir,dataset, profile_type,plate_normalize_by_all = False, modality = 'CellPainting'):
# def load_data(configs, plate_normalize_by_all = False, modality = 'CellPainting'):

  # if 'variable_selected' in configs.data.profile_type:
  #   profileType = 'normalized_variable_selected'
  # else:
  #   profileType = 'augmented'
  [cp_data_repLevel, cp_features], [l1k_data_repLevel, l1k_features] = \
    read_replicate_level_profiles(base_dir, dataset, profile_type, plate_normalize_by_all)

  if modality == 'CellPainting':
    data = cp_data_repLevel.copy()
  else:
    data = l1k_data_repLevel.copy()

  features, meta_features = get_features(data, modality)
  # data = remove_null_features(data, features)
  # l1k = l1k_data_repLevel[[pertColName] + l1k_features]
  # cp = cp_data_repLevel[[pertColName] + cp_features]
  #
  # if dataset == 'LINCS':
  #   cp['Compounds'] = cp['PERT'].str[0:13]
  #   l1k['Compounds'] = l1k['PERT'].str[0:13]
  # else:
  #   cp['Compounds'] = cp['PERT']
  #   l1k['Compounds'] = l1k['PERT']

  return data, features



def to_datasets(data):

  datasets = {
      'train': data.loc[data['Metadata_set'] == 'train', :],
      'val': data.loc[data['Metadata_set'] == 'val', :],
      'test_ctrl': data.loc[data['Metadata_set'] == 'test_ctrl', :],
      'test_treat': data.loc[data['Metadata_set'] == 'test_treat', :]
    }
  return datasets

def to_dataloaders(data, batch_size, features):

  datasets = to_datasets(data)
  # construct dataset
  dataset_modules = {}
  for key in datasets.keys():
    dataset_modules[key] = TabularDataset(datasets[key][features])

  # construct dataloaders
  dataloaders = {}
  # num_workers = int(min(os.cpu_count(), batch_size)/4)
  # num_workers = 2
  for key in datasets.keys():
    if key == 'train':
      dataloaders[key] = DataLoader(dataset_modules[key],batch_size,shuffle=True)
      # dataloaders[key] = DataLoader(dataset_modules[key],batch_size, num_workers=num_workers,shuffle=True)

    else:
      dataloaders[key] = DataLoader(dataset_modules[key], batch_size)

  return dataloaders



def pre_process(data, configs, overwrite = False):

  #TODO: debug why not working with LINCS - nan in data (see Zernike_3_3)
  # take control data for training

  data_path = f'{configs.general.base_dir}/preprocessed_data/{DS_INFO_DICT[configs.general.dataset]["name"]}/{configs.data.modality}/replicate_level_{configs.data.modality_str}_{configs.data.profile_type}.csv.gz'
  # features = data.columns[data.columns.str.contains("Cells_|Cytoplasm_|Nuclei_")].tolist()
  features, meta_features = get_features(data,configs.data.modality)

  features_filename = f'{configs.data.profile_type}_features.txt'
  if configs.data.profile_type == 'augmented' and configs.data.feature_select:
    features_filename = f'{configs.data.profile_type}_{configs.data.corr_threshold}_features.txt'
    features_path = os.path.join(configs.general.processed_data_dir, features_filename)

    if os.path.exists(features_path):
      print('loading features from file...')
      features_selected = load_list_from_txt(features_path)
      cols_to_drop = [col for col in features if col not in features_selected]
      features = [col for col in features if col in features_selected]
      
      

    else:
      # do feature selection - only only on mock data!!
      features, cols_to_drop = feature_selection(data[data[DS_INFO_DICT[configs.general.dataset][configs.data.modality]['role_col']] == DS_INFO_DICT[configs.general.dataset][configs.data.modality]['mock_val']],configs.data.corr_threshold,features)
    
      if not configs.general.debug_mode:
        save_list_to_txt(features, features_path)
      data = data.drop(cols_to_drop, axis=1)
      print(f'cols removed: {cols_to_drop}')


    print(f'number of features after feature selection: {len(features)}')
    
    data.loc[:,features] = data.loc[:,features].interpolate()
    # if not os.path.exists(data_path): 
      # data.to_csv(data_path, compression='gzip')
      # configs.data.overwrite_data_creation = True

  print(f'number of features for training is {len(features)}')

  # output_dir = os.path.join(configs.general.base_dir, 'anomaly_output', configs.general.dataset,'CellPainting',configs.general.exp_name)
  # os.makedirs(output_dir, exist_ok =True)
  train_filename = f'replicate_level_{configs.data.modality_str}_{configs.data.profile_type}_normalized_by_train'
  train_path = os.path.join(configs.general.output_exp_dir, f'{train_filename}.csv')
  
  data.loc[:,features] = data.loc[:,features].interpolate()

  if not os.path.exists(train_path) or configs.data.overwrite_data_creation:

    # split data with equal samples from different plates (DS_INFO_DICT[configs.general.dataset[3]])
    data_splitted = split_data(data, configs.general.dataset, configs.data.test_split_ratio, modality=configs.data.modality)

    # datasets_pre_normalization = {
    #   'train': data_splitted.loc[data_splitted['Metadata_set'] == 'train', :],
    #   'val': data_splitted.loc[data_splitted['Metadata_set'] == 'val', :],
    #   'test_ctrl': data_splitted.loc[data_splitted['Metadata_set'] == 'test_ctrl', :],
    #   'test_treat': data_splitted.loc[data_splitted['Metadata_set'] == 'test_treat', :]
    # }

    print('normalizing to training set...')

    normalized_df_by_train, cols_to_remove = normalize(data_splitted,features, configs.data.modality, normalize_condition = configs.data.normalize_condition,plate_normalized=1, norm_method = "standardize", remove_non_normal_features = True, clip_outliers=False)
    # features = features.drop(cols_to_remove)
    features = [col for col in features if col not in cols_to_remove]
    data_splitted = data_splitted.drop(cols_to_remove, axis=1)
    normalized_df_by_train = normalized_df_by_train.drop(cols_to_remove, axis=1)
    print(f'number of features training data: {len(normalized_df_by_train.columns)}')

    # filename = f'replicate_level_cp_{configs.data.profile_type}_normalized_by_train'

    save_profiles(normalized_df_by_train, configs.general.output_exp_dir, train_filename)

    raw_filename = f'replicate_level_{configs.data.modality_str}_{configs.data.profile_type}_baseline'
    raw_path = os.path.join(os.path.join(configs.general.output_exp_dir, raw_filename))
    if not os.path.exists(raw_path) or configs.data.overwrite_data_creation:

      print('calclating raw measurements...')

      raw_data, _ = normalize(data_splitted,features, configs.data.modality, normalize_condition = 'test_ctrl',plate_normalized=1, norm_method = "standardize", remove_non_normal_features = False, clip_outliers=False)
      # test_raw_data = raw_data.loc[raw_data['Metadata_set'] == 'test_treat', :]
      save_profiles(raw_data, configs.general.output_exp_dir, raw_filename)
      # test_raw_unchanged = data_splitted.loc[data_splitted['Metadata_set'] == 'test_treat', :]
      save_profiles(data_splitted, configs.general.output_exp_dir, f'replicate_level_{configs.data.modality_str}_{configs.data.profile_type}_baseline_unchanged')
      print(f'number of features raw data: {len(raw_data.columns)}')

  else:
    print('loading normalized data from file...')
    normalized_df_by_train = pd.read_csv(train_path,compression='gzip')


  return normalized_df_by_train, features

def split_data(data_preprocess, dataset, test_split_ratio, val_split_ratio=.2, modality = 'CellPainting'):

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
    control_data = data_preprocess[data_preprocess[DS_INFO_DICT[dataset][modality]['role_col']] == DS_INFO_DICT[dataset][modality]['mock_val']]
    # split data with equal samples from different plates (DS_INFO_DICT[configs.general.dataset[3]])
    splitter = StratifiedShuffleSplit(test_size=test_split_ratio, n_splits=1)

    split = splitter.split(X = control_data, y=control_data[DS_INFO_DICT[dataset][modality]['plate_col']])
    train_all_inds, test_inds = next(split)

    train_data_all = control_data.iloc[train_all_inds]
    test_data_mocks = control_data.iloc[test_inds]

    # train_data, val_data = train_test_split(mock_data, test_size=0.4)
    
    if modality == 'CellPainting':
      splitter = StratifiedShuffleSplit(test_size=.2, n_splits=1)
      split = splitter.split(X = train_data_all, y=train_data_all[DS_INFO_DICT[dataset][modality]['plate_col']])
    else:
      # not enough samples in each plate for stratified split 
      splitter = ShuffleSplit(test_size=.2, n_splits=1)
      split = splitter.split(X = train_data_all)

    train_inds, val_inds = next(split)

    train_data = train_data_all.iloc[train_inds]
    val_data = train_data_all.iloc[val_inds]

    data_splitted = data_preprocess.copy()
    data_splitted.loc[train_data.index.values, 'Metadata_set'] = 'train'
    data_splitted.loc[val_data.index.values, 'Metadata_set'] = 'val'
    data_splitted.loc[test_data_mocks.index.values, 'Metadata_set'] = 'test_ctrl'
    data_splitted.loc[data_splitted[DS_INFO_DICT[dataset][modality]['role_col']] != DS_INFO_DICT[dataset][modality]['mock_val'], 'Metadata_set'] = 'test_treat'

    return data_splitted


def scale_data_by_set(scale_by, sets, features):

  # scaler_ge = preprocessing.StandardScaler()
  scaler_cp = preprocessing.StandardScaler()
  # l1k_scaled = l1k.copy()
  # l1k_scaled[l1k_features] = scaler_ge.fit_transform(l1k[l1k_features].values)
  # cp_scaled = cp.copy()
  scale_by[features] = scaler_cp.fit_transform(scale_by[features].values.astype('float64')).fillna(0)

  scaled_sets = []
  scaled_sets.append(scale_by)
  for set in sets:
    scaled_set = set.copy()
    scaled_set[features] = scaler_cp.transform(set.values.astype('float64')).fillna(0)
    scaled_sets.append(scaled_set)

  return scaled_sets


# def scale_by_set(reference_data, sets_to_scale=None, features=None):

#   scaler = preprocessing.StandardScaler()
#   scaler = scaler.fit(reference_data[features].values.astype('float64'))

#   if scaled_sets is None:
#     return scaler.transform(reference_data[features].values.astype('float64')).fillna(0)
#   elif isinstance(sets_to_scale, list):
#     scaled_sets = []
#     # scaled_sets.append(scaler.transform(reference_data[features].values.astype('float64')).fillna(0))
#     for set in sets_to_scale:
#       scaled_sets.append(scaler.transform(set[features].values.astype('float64')).fillna(0))
#     return scaled_sets
#   else:
#     return scaler.transform(sets_to_scale[features].values.astype('float64')).fillna(0)

def remove_null_features(data, features,null_vals_ratio = 0.05):
  
  cols2remove_manyNulls = [i for i in features if
                            (data[i].isnull().sum(axis=0) / data.shape[0]) \
                            > null_vals_ratio]
  data = data.drop(cols2remove_manyNulls, axis=1)

  return data

def feature_selection(data, corr_threshold=0.95, features=None):
    """
    Perform feature selection by dropping columns with null or
    only zeros values, and highly correlated values from the data.

    params:
    dataset_link: string of github link to the consensus dataset

    Returns:
    data: returned consensus dataframe

    """
    null_vals_ratio = 0.05
    thrsh_std = 0.001
    # corr_threshold = corr_threshold

    # print(f'number of features before feature selection: {len(features)}')

    # cols2remove_manyNulls = [i for i in features if
    #                          (data[i].isnull().sum(axis=0) / data.shape[0]) \
    #                          > null_vals_ratio]
    # cols2remove_lowVars = data[features].std()[
    #   data[features].std() < thrsh_std].index.tolist()
    #
    cols2remove_manyNulls = get_na_columns(
      population_df=data,
      features=features,
      samples="all",
      cutoff=null_vals_ratio,
    )
    cols2remove_lowVars = variance_threshold(
      population_df=data,
      features=features,
      samples="all",
      freq_cut=0.05,
      unique_cut=0.01,
    )

    cols2removeCP = cols2remove_manyNulls + cols2remove_lowVars

    cp_features = list(set(features) - set(cols2removeCP))
    print(f'number of features after removing cols with nulls and low var: {len(cp_features)}')

    cp_data_repLevel = data.drop(cols2removeCP, axis=1)
    # cols2remove_highCorr = get_highly_correlated_features(cp_data_repLevel, cp_features,corr_threshold)
    
    cols2remove_highCorr = correlation_threshold(
      population_df=data,
      features=features,
      samples="all",
      threshold=corr_threshold
    )


    cp_features = list(set(cp_features) - set(cols2remove_highCorr))
    print(f'number of features after removing high correlated features: {len(cp_features)}')
    cp_data_repLevel = data.drop(cols2remove_highCorr, axis=1)
    # cp_data_repLevel[cp_features] = cp_data_repLevel[cp_features].interpolate()

    cols2removeCP += cols2remove_highCorr
    return cp_features, cols2removeCP


def get_highly_correlated_features(data, features,corr_threshold=0.9):

    # Compute the correlation matrix
    corr_matrix = data[features].corr(method='pearson').abs()

    # get upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # find features with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > corr_threshold)]

    return to_drop


##########################################################

def normalize(df, features,modality,dataset='CDRP-bio', normalize_condition = "train",plate_normalized=False, norm_method = "standardize", plate_col= "Metadata_Plate", well_col=  "Metadata_Well", float_format = "%.5g", compression={"method": "gzip", "mtime": 1}, remove_non_normal_features = True, clip_outliers=False):
    """ Normalize by plate and return normalized df.

    norm_method: str. normalization method. default: "mad_robustize", options: ["mad_robustize", "standardize", "normalize"]
    """

    if modality == 'CellPainting':
      plate_col = "Metadata_Plate"
      cpd_col = DS_INFO_DICT[dataset][modality]['cpd_col']
      mock_val = DS_INFO_DICT[dataset][modality]['mock_val']

    else:
      plate_col = "det_plate"
      cpd_col = DS_INFO_DICT[dataset][modality]['cpd_col']
      mock_val = DS_INFO_DICT[dataset][modality]['mock_val']
      normalize_condition='DMSO'
    # clip feature values above 95th quantile

    if clip_outliers:
      df = df.copy()
      df.loc[:,features] = df.loc[:,features].clip(upper=df.loc[:,features].quantile(0.99), axis=1)
      df.loc[:,features] = df.loc[:,features].clip(lower=df.loc[:,features].quantile(0.01), axis=1)

    if normalize_condition == "train":
      # if 'train' in df['Metadata_set'].unique():
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
            float_format=float_format,
            compression_options=compression,
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
          float_format=float_format,
          compression_options=compression,
          # mad_robustize_epsilon = 1e-6
        )


    isna = normalized_df[features].isna().sum().sum()/(normalized_df.shape[0]*len(features))
    print(f'ratio of null values after normalization: {isna}')
    normalized_df.loc[:,features] = normalized_df.loc[:,features].interpolate()
    if remove_non_normal_features:
      print(f'number of features before feature selection of only normally distributed features: {len(features)}')
      mean_condition = normalized_df[features].abs().mean()>1000
      cols_to_remove = mean_condition[mean_condition].index.tolist()
      normalized_df = normalized_df.drop(cols_to_remove, axis=1)
      features, meta_features = get_features(normalized_df)
      print(f'number of features after feature selection of only normally distributed features: {len(features)}')
      return normalized_df, cols_to_remove
    else:
      cols_to_remove = None
      return normalized_df, cols_to_remove

##########################################################

def get_features(df, modality = 'CellPainting'):
  # meta_features = df.columns[df.columns.str.contains("Metadata_")].tolist()
  # features = df.columns[~df.columns.isin(meta_features)].tolist()
  if modality == 'CellPainting':
    features = df.columns[df.columns.str.contains("Cells_|Cytoplasm_|Nuclei_")].tolist()
    meta_features = df.columns[df.columns.str.contains("Metadata_")].tolist()
  else:
    features = df.columns[df.columns.str.contains("_at")].tolist()
    meta_features = df.columns[~df.columns.str.contains("_at")].tolist()
  # features = df.columns[df.columns.str.contains("Cells_|Cytoplasm_|Nuclei_")].tolist()
  return features, meta_features


def list_columns(df):
    """
    This function split the plate's columns to different lists
    general columns, correlation columns and a dictionary channel -> channel columns
    """
    general_cols = [f for f in df.columns if all(c not in f for c in CHANNELS)]
    corr_cols = [f for f in df.columns if 'Correlation' in f]

    # Split columns by channel
    dict_channel_cols = {}
    for channel in CHANNELS:
        dict_channel_cols[channel] = [col for col in df.columns if channel in col and col not in corr_cols]

    return general_cols, corr_cols, dict_channel_cols


def set_index_fields(df, dataset, index_fields=None,by_dose = False, modality='CellPainting'):

    if by_dose:
      ind_col = 'dose_col'
    else:
      ind_col = 'cpd_col'
    
    if modality == 'CellPainting':
      if index_fields is None:
          index_fields = ['Metadata_Plate', DS_INFO_DICT[dataset][modality]['role_col'], DS_INFO_DICT[dataset][modality][ind_col], 
          'Metadata_mmoles_per_liter','Metadata_Well']
    else:
          index_fields = ['det_plate', DS_INFO_DICT[dataset][modality][ind_col],
                          'pert_dose']
    df = df.set_index(
        index_fields)
    return df

