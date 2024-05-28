import os

import numpy as np
import pandas as pd
import pathlib as pathlib
import pycytominer
from pycytominer import feature_select
from pycytominer.operations import variance_threshold, get_na_columns, correlation_threshold
from sklearn import preprocessing
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit, train_test_split,ShuffleSplit
from torch.utils.data import DataLoader, Dataset
from utils.readProfiles import read_replicate_level_profiles, save_profiles, filter_data_by_highest_dose, standardize_per_catX, read_replicate_single_modality_level_profiles
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
  def __init__(self, data,features, cpd_col = 'Metadata_broad_sample',dose_col = 'Metadata_Sample_Dose',role_col = 'Metadata_ASSAY_WELL_ROLE',plate_col = 'Metadata_Plate',mock_val = 'mock',modality = 'CellPainting'):
    self.features = features

    self.data = data
    self.cpd_col = cpd_col
    self.dose_col = dose_col
    self.role_col = role_col
    self.plate_col = plate_col
    self.mock_val = mock_val
    self.modality = modality
    self.dataset = data
    self.data = data[features].to_numpy().astype(np.float32)

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):

    # record = self.data.iloc[idx,:]
    # index = self.data.index[idx]
    # data = record[self.features].to_numpy().astype(np.float32)
    
    # return data, record[self.cpd_col], record[self.plate_col],index
    return self.data[idx]



def load_data(base_dir,dataset, profile_type,plate_normalize_by_all = False, modality = 'CellPainting'):

  [data, cp_features] = read_replicate_single_modality_level_profiles(base_dir, dataset, profile_type ,per_plate_normalized_flag=plate_normalize_by_all, modality=modality)
  # if modality == 'CellPainting':
    # data = cp_data_repLevel.copy()
  # else:
    # data = l1k_data_repLevel.copy()

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

##########################################################

def to_datasets(data):

  datasets = {
      'train': data.loc[data['Metadata_set'] == 'train', :],
      'val': data.loc[data['Metadata_set'] == 'val', :],
      'test_ctrl': data.loc[data['Metadata_set'] == 'test_ctrl', :],
      'test_treat': data.loc[data['Metadata_set'] == 'test_treat', :]
    }
  return datasets


##########################################################

def to_dataloaders(data, batch_size, features):

  datasets = to_datasets(data)
  # construct dataset
  dataset_modules = {}
  for key in datasets.keys():
    dataset_modules[key] = TabularDataset(datasets[key],features)

  # construct dataloaders
  dataloaders = {}
  # num_workers = int(min(os.cpu_count(), batch_size)/4)
  # num_workers = 2
  for key in datasets.keys():
    if key == 'train':
      dataloaders[key] = DataLoader(dataset_modules[key],batch_size,shuffle=True)
    else:
      dataloaders[key] = DataLoader(dataset_modules[key], batch_size)

  return dataloaders


##########################################################

def pre_process(data, configs,data_reps = ['ae_diff','baseline']):

  # data_path = f'{configs.general.base_dir}/preprocessed_data/{DS_INFO_DICT[configs.general.dataset]["name"]}/{configs.data.modality}/replicate_level_{configs.data.modality_str}_{configs.data.profile_type}.csv.gz'
  # features = data.columns[data.columns.str.contains("Cells_|Cytoplasm_|Nuclei_")].tolist()
  features, meta_features = get_features(data,configs.data.modality)

  # old - to delete
  if configs.data.profile_type == 'augmented2' and configs.data.feature_select:
    features_filename = f'{configs.data.profile_type}_{configs.data.corr_threshold}_features.txt'
    features_path = os.path.join(configs.general.processed_data_dir, features_filename)

    if os.path.exists(features_path):
      configs.general.logger.info('loading features from file...')
      features_selected = load_list_from_txt(features_path)
      cols_to_drop = [col for col in features if col not in features_selected]
      features = [col for col in features if col in features_selected]
      
      print(f'cols removed: {len(cols_to_drop)}')
    else:
      # do feature selection - only on control data
      if configs.general.dataset == 'TAORF':

        control_data = data[data[DS_INFO_DICT[configs.general.dataset][configs.data.modality]['role_col']].isin(['CTRL'])]
      else:
        control_data = data[data[DS_INFO_DICT[configs.general.dataset][configs.data.modality]['role_col']] == DS_INFO_DICT[configs.general.dataset][configs.data.modality]['mock_val']]
      # features, cols_to_drop = feature_selection(control_data,configs.data.corr_threshold,features, var_threshold = configs.data.var_threshold, blocklist_features = DS_INFO_DICT[configs.general.dataset][configs.data.modality]['blocklist_features'])
    
      if not configs.general.debug_mode:
        save_list_to_txt(features, features_path)

    configs.general.logger.info(f'number of features after feature selection: {len(features)}')
    
    data = data.drop(cols_to_drop, axis=1)
    print(f'cols removed: {len(cols_to_drop)}')
    data.loc[:,features] = data.loc[:,features].interpolate()


  ######### data normalization by training set #########
  train_filename = f'replicate_level_{configs.data.modality_str}_{configs.data.profile_type}_normalized_by_train'
  train_path = os.path.join(configs.general.output_exp_dir, f'{train_filename}.csv')
  save_data_flag = not os.path.exists(train_path) or configs.general.overwrite_experiment

  if save_data_flag or configs.data.run_data_process:

    # split data with equal samples from different plates (DS_INFO_DICT[configs.general.dataset[3]])
    data_splitted = split_data(data, configs.general.dataset, configs.data.test_split_ratio, modality=configs.data.modality, n_samples_for_training=configs.data.n_samples_for_training,random_state=configs.general.seed)
    meta_features += ['Metadata_set']
    
    configs.general.logger.info(f'number of plates: {len(data_splitted[data_splitted["Metadata_set"] == "test_ctrl"]["Metadata_Plate"].unique())}')
    configs.general.logger.info(f'number of samples in train set: {len(data_splitted[data_splitted["Metadata_set"] == "train"])}')
    configs.general.logger.info(f'number of samples in val set: {len(data_splitted[data_splitted["Metadata_set"] == "val"])}')
    configs.general.logger.info(f'number of samples in test_ctrl set: {len(data_splitted[data_splitted["Metadata_set"] == "test_ctrl"])}')
    configs.general.logger.info(f'number of samples in test_treat set: {len(data_splitted[data_splitted["Metadata_set"] == "test_treat"])}')
    configs.general.logger.info(f'median number of replicates per compound: {data_splitted[data_splitted["Metadata_set"] == "test_treat"].groupby(DS_INFO_DICT[configs.general.dataset][configs.data.modality]["cpd_col"]).size().median()}')
    if DS_INFO_DICT[configs.general.dataset]['has_dose']:
      configs.general.logger.info(f'median number of replicates per compound and dose: {data_splitted[data_splitted["Metadata_set"] == "test_treat"].groupby([DS_INFO_DICT[configs.general.dataset][configs.data.modality]["cpd_col"],DS_INFO_DICT[configs.general.dataset][configs.data.modality]["dose_col"]]).size().median()}')


    if configs.data.profile_type == 'augmented':
      
      # print(f'number of features before removing features with low variance: {len(features)}')
      # # Calculate the variance of each feature

      # variances = data_splitted[features].var()
      # Get the names of features with variance lower than var_threshold
      # low_variance_features = variances[variances <configs.data.var_threshold].index.tolist()
      # data_splitted.drop(low_variance_features, axis=1, inplace=True)
      # features = [f for f in features if f not in low_variance_features]
      # print(f'number of features with low variance: {len(low_variance_features)}')

      print('normalizing to training set...')
      # normalized_df_by_train = normalize(data_splitted,features, configs.data.modality,norm_method="standardize", normalize_condition = configs.data.normalize_condition,plate_normalized=configs.data.plate_normalized, clip_outliers=False)
      df_normalized = normalize(data_splitted,features, configs.data.modality,norm_method="standardize", normalize_condition = configs.data.normalize_condition,plate_normalized=configs.data.plate_normalized, clip_outliers=False)

    ########## feature selection ##########
    if configs.data.profile_type in ['normalized','augmented']:
      # normalized_df_by_train = data_splitted.copy()
      print('feature selection after normalization...')
      configs.general.logger.info(f'number of features before feature selection: {len(features)}')
      # features, cols_to_drop = feature_selection(normalized_df_by_train,configs.data.corr_threshold,features, var_threshold =  configs.data.var_threshold, blocklist_features = DS_INFO_DICT[configs.general.dataset][configs.data.modality]['blocklist_features'])
      feature_select_ops = ['drop_na_columns', 'variance_threshold', 'correlation_threshold', 'drop_outliers']

      # df_normalized_by_all = data_splitted.copy()
      # normalized_df_by_all = normalize(data_splitted,features, configs.data.modality,norm_method="standardize", normalize_condition = 'all',plate_normalized=True, clip_outliers=False)
      df_normalized_by_all = normalize(data_splitted,features, configs.data.modality,norm_method="standardize", normalize_condition ='all',plate_normalized=True, clip_outliers=False)
      df_normalized_by_all_feature_selected, features_selected, cols_to_drop = feature_selection(df_normalized_by_all, 
                                                          corr_threshold=configs.data.corr_threshold, 
                                                          features=features, 
                                                          # freq_cut =  10e-4, 
                                                          # unique_cut = 0.1,
                                                          # outlier_cutoff=15,
                                                          blocklist_features = DS_INFO_DICT[configs.general.dataset][configs.data.modality]['blocklist_features'],
                                                          # samples='Metadata_set == "train"',
                                                          samples = 'all', 
                                                          ops = feature_select_ops)
      
      # add all granulirity features back for the CDRP test
      # if configs.general.dataset == 'CDRP-bio':
        # granurality_agp_features = sorted([f for f in features if ('Granularity' in f) and ('AGP' in f)])
        
        # features = list(set(features_selected + granurality_agp_features))
        # all_features = meta_features + features
        # df_normalized_feature_selected = df_normalized[all_features]
      # else:
        # features = features_selected
      features = features_selected
      df_normalized_feature_selected =  df_normalized.drop(cols_to_drop, axis=1)
      # low_variance = [f for f in low_variance_features if f in a.columns]
      # print(f'cols removed: {len(cols_to_drop)}')
      # normalized_df_by_train.drop(cols_to_drop, axis=1, inplace=True)
      # configs.general.logger.info(f'number of features after feature selection: {len(features)}')
      # print(f'number of features after feature selection: {len(features)}')
      # normalized_df_by_train.loc[:,features] = normalized_df_by_train.loc[:,features].interpolate()
      
      # variances = normalized_df_by_train[features].var()
      # Get the names of features with variance lower than var_threshold
      # low_variance_features = variances[variances <configs.data.var_threshold].index.tolist()
      # print(f'number of features with low variance: {len(low_variance_features)}')
      # normalized_df_by_train.drop(low_variance_features, axis=1, inplace=True)
      # still_low_variance_features = [f for f in features if f in low_variance_features]
      
      # configs.general.logger.info(f'number of features after removing features with low variance: {len(features)}')

    elif configs.data.profile_type == 'normalized_variable_selected':
      # normalize by train population pre-training
      # normalized_df_by_train = normalize(data_splitted,features, configs.data.modality,norm_method="standardize", normalize_condition = configs.data.normalize_condition,plate_normalized=configs.data.plate_normalized, clip_outliers=False)
      df_normalized_feature_selected = data_splitted.copy()
      
    # normalized_df_by_train = normalized_df_by_train.drop(cols_to_drop, axis=1)


    if not configs.general.debug_mode:
      save_profiles(df_normalized_feature_selected, configs.general.output_exp_dir, train_filename)

    raw_filename = f'replicate_level_{configs.data.modality_str}_{configs.data.profile_type}_baseline'
    raw_path = os.path.join(os.path.join(configs.general.output_exp_dir, raw_filename))

    save_data_flag = not os.path.exists(raw_path) or configs.general.overwrite_experiment

    if save_data_flag:

      test_indices = df_normalized_feature_selected['Metadata_set'].isin(['test_ctrl','test_treat'])
      test_data = df_normalized_feature_selected.loc[test_indices, :]
      # if not 'variable_selected' in configs.data.profile_type:
        # test_data = test_data.drop(cols_to_drop, axis=1)
      # test_data.drop(cols_to_drop, axis=1)

      for d in data_reps[1:]:

        print(f'calclating raw measurements for {d}...')
        try:
          if 'baseline' in d:
            raw_filename = f'replicate_level_{configs.data.modality_str}_{configs.data.profile_type}_{d}'
            if d == 'baseline':
              
              # save_profiles(raw_data, configs.general.output_exp_dir, raw_filename)
              raw_data = normalize(test_data,features, configs.data.modality, normalize_condition = 'test_ctrl',plate_normalized=configs.data.plate_normalized, clip_outliers=False)
            elif d == 'baseline_unchanged':
              raw_data = data_splitted.copy()
            else:
              raw_data = normalize(data_splitted,features, configs.data.modality, normalize_condition = 'DMSO',plate_normalized=configs.data.plate_normalized, clip_outliers=False)
              raw_data = raw_data.loc[test_indices, :]
            # normalize data by DMSO
          else:
          # if configs.data.norm_method == 'spherize':
            raw_data = normalize(test_data,features, configs.data.modality, normalize_condition = 'test_ctrl',plate_normalized=configs.data.plate_normalized, norm_method = 'spherize', clip_outliers=False,spherize_method=d)
                      
          if save_data_flag:
            raw_filename = f'replicate_level_{configs.data.modality_str}_{configs.data.profile_type}_{d}'
            save_profiles(raw_data, configs.general.output_exp_dir, raw_filename)
        except:
          print(f'failed to normalize with {d}')
          continue
          
  else:
    print('loading normalized data from file...')
    df_normalized_feature_selected = pd.read_csv(train_path,compression='gzip')
    features = get_features(df_normalized_feature_selected,configs.data.modality)[0]

  return df_normalized_feature_selected, features

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


def feature_selection(data, corr_threshold=0.9, features=None, freq_cut = 0.05, unique_cut = 0.01, outlier_cutoff=500,blocklist_features = None,samples = 'all', ops = ['drop_na_columns', 'variance_threshold', 'correlation_threshold', 'drop_outliers']):
    """
    Perform feature selection by dropping columns with null or
    only zeros values, and highly correlated values from the data.

    Returns:
    data: returned consensus dataframe

    """

    print(f'number of features before feature selection: {len(features)}')
    null_vals_ratio = 0.05


    data_feature_selected = feature_select(
      data,
      features=features,
      operation=ops,
      samples=samples,
      na_cutoff=null_vals_ratio,
      corr_threshold=corr_threshold,
      corr_method="pearson",
      freq_cut=freq_cut,
      unique_cut=unique_cut,
      # compression_options=None,
      # float_format=None,
      # blocklist_file=None,
      outlier_cutoff=outlier_cutoff,
      # noise_removal_perturb_groups=None,
      # noise_removal_stdev_cutoff=None,
    )
    
    new_features = [f for f in data_feature_selected.columns if f.startswith('Cells_') or f.startswith('Cytoplasm_') or f.startswith('Nuclei_')]
    exluded_features = [f for f in features if f not in new_features]
    return data_feature_selected, new_features,exluded_features

##########################################################

def normalize(df, features=None,modality='CellPainting',dataset='CDRP-bio', normalize_condition = "train", plate_normalized=False, norm_method = "standardize", plate_col= "Metadata_Plate", well_col=  "Metadata_Well", float_format = "%.5g", compression={"method": "gzip", "mtime": 1}, clip_outliers=False,spherize_method='ZCA'):
    """ Normalize by plate and return normalized df.

    norm_method: str. normalization method. default: "mad_robustize", options: ["mad_robustize", "standardize", "normalize", "spherize"]
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

    if features is None:
      features, meta_features = get_features(df, modality)

    n_plates = len(df[plate_col].unique())
    n_normalization_population = df[df['Metadata_set'] == normalize_condition].shape[0]
    plate_normalized = plate_normalized and n_plates*10 < n_normalization_population
    # plate_normalized = plate_normalized and normalize_condition == 'train'
    print('normalizing by plate:', plate_normalized)
    
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
            # spherize_epsilon=1e-3,
            spherize_method=spherize_method
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
          spherize_method=spherize_method
          # mad_robustize_epsilon = 1e-6
        )
        # scaler = preprocessing.StandardScaler()
        # normalized_df = df.copy()
        # scaler.fit(df.loc[df['Metadata_set'] == 'test_ctrl',features].values.astype('float64'))
        # normalized_df.loc[:,features] = scaler.transform(df.loc[:,features].values.astype('float64'))
        # normalized_df = df.copy()
        # normalized_df.loc[:,features] = normalized_df.loc[:,features].interpolate()



    isna = normalized_df[features].isna().sum().sum()/(normalized_df.shape[0]*len(features))
    print(f'ratio of null values after normalization: {isna}')
    normalized_df.loc[:,features] = normalized_df.loc[:,features].interpolate()
    isna = normalized_df[features].isna().sum().sum()
    if isna > 0:
      raise ValueError('null values after interpolation')

    # if remove_non_normal_features:
    #   print(f'number of features before feature selection of only normally distributed features: {len(features)}')
    #   mean_condition = normalized_df[features].abs().mean()>1000
    #   cols_to_remove = mean_condition[mean_condition].index.tolist()
    #   normalized_df = normalized_df.drop(cols_to_remove, axis=1)
    #   features, meta_features = get_features(normalized_df)
    #   print(f'number of features after feature selection of only normally distributed features: {len(features)}')
    #   # return normalized_df, cols_to_remove
    # else:
    #   cols_to_remove = None

    return normalized_df

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

##########################################################


def list_cp_columns_type(df):
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
          index_fields = ['Metadata_Plate', DS_INFO_DICT[dataset][modality]['role_col'], DS_INFO_DICT[dataset][modality][ind_col], 'Metadata_Well']
    else:
          index_fields = ['det_plate', DS_INFO_DICT[dataset][modality][ind_col],
                          'pert_dose']
    

    df = df.set_index(
        index_fields)
    return df


def load_zscores(methods,base_dir,dataset, profile_type,normalize_by_all=False,by_dose=False,z_trim=None,sample='treated',set_index=True,debug=False, filter_by_highest_dose=False, min_max_norm=False):

    for m in methods.keys():
      print(f'loading zscores for method: {m}')
      if m == 'l1k':
          methods[m]['modality'] ='L1000'

          zscores, features = load_data(base_dir,dataset, profile_type,modality=methods[m]['modality'], plate_normalize_by_all =normalize_by_all)
          if debug:
            zscores = zscores.sample(2000)
          methods[m]['features']=list(features)
          
          # methods[m]['zscores'] = zscores.loc[:,methods[m]['features']]
      else:
          methods[m]['modality']='CellPainting'
          zscores = pd.read_csv(methods[m]['path'], compression = 'gzip', low_memory=False)   
          if debug:
            zscores = zscores.sample(2000)  
          if m == 'anomaly_emb':
              meta_features = [c for c in zscores.columns if 'Metadata_' in c]
              features = [c for c in zscores.columns if 'Metadata_' not in c]
          else:
              features = zscores.columns[zscores.columns.str.contains("Cells_|Cytoplasm_|Nuclei_")]
          methods[m]['features']=list(features)

          if normalize_by_all:
              zscores = standardize_per_catX(zscores,DS_INFO_DICT[dataset]['CellPainting']['plate_col'],methods[m]['features'])

      # zscores = zscores.query(f"{DS_INFO_DICT[dataset][methods[m]['modality']]['role_col']} != '{DS_INFO_DICT[dataset][methods[m]['modality']]['mock_val']}' ")

      # zscores = zscores.query('Metadata_ASSAY_WELL_ROLE == "treated"')
      if sample=='treated':
        zscores = zscores.query(f"{DS_INFO_DICT[dataset][methods[m]['modality']]['role_col']} != '{DS_INFO_DICT[dataset][methods[m]['modality']]['mock_val']}' ")

        
      elif sample=='mock':
        zscores = zscores.query(f"{DS_INFO_DICT[dataset][methods[m]['modality']]['role_col']} == '{DS_INFO_DICT[dataset][methods[m]['modality']]['mock_val']}' ")
      elif sample=='all':
        pass
      else:
        raise ValueError(f'invalid sample type: {sample}')

      if z_trim is not None:
          zscores.loc[:,features] = zscores.loc[:,features].clip(-z_trim, z_trim)
      if min_max_norm:
          scaler = preprocessing.MinMaxScaler()
          zscores.loc[:,features] = scaler.fit_transform(zscores.loc[:,features])
          
      if not by_dose and filter_by_highest_dose and DS_INFO_DICT[dataset]['has_dose']:
        trt_indices = zscores[DS_INFO_DICT[dataset][methods[m]['modality']]['role_col']] != DS_INFO_DICT[dataset][methods[m]['modality']]['mock_val']
        treated_zscores = filter_data_by_highest_dose(zscores.loc[trt_indices,:], dataset, modality=methods[m]['modality']).reset_index(drop=True)
        mock_zscores = zscores.loc[~trt_indices,:].reset_index(drop=True)
        zscores = pd.concat([mock_zscores, treated_zscores]).reset_index(drop=True)

      if set_index:
        zscores = set_index_fields(zscores,dataset,by_dose=by_dose, modality=methods[m]['modality'])
        methods[m]['zscores'] = zscores.loc[:,methods[m]['features']]
      else:
        methods[m]['zscores'] = zscores

    return methods

