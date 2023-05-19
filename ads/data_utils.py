import os

from readProfiles import read_replicate_level_profiles
from sklearn import preprocessing
from sklearn.model_selection import GroupShuffleSplit

ds_info_dict={'CDRP':['CDRP-BBBC047-Bray',['Metadata_Sample_Dose','pert_sample_dose']],
              'CDRP-bio':['CDRPBIO-BBBC036-Bray',['Metadata_Sample_Dose','pert_sample_dose']],
              'TAORF':['TA-ORF-BBBC037-Rohban',['Metadata_broad_sample','pert_id',]],
              'LUAD':['LUAD-BBBC041-Caicedo',['x_mutation_status','allele']],
              'LINCS':['LINCS-Pilot1',['Metadata_pert_id_dose','pert_id_dose']]}

# index_fields =

def load_data(procProf_dir, dataset, profileType, plate_normalized=0):
  [cp_data_repLevel, cp_features], [l1k_data_repLevel, l1k_features] = \
    read_replicate_level_profiles(procProf_dir, dataset, profileType, plate_normalized)

  # l1k = l1k_data_repLevel[[pertColName] + l1k_features]
  # cp = cp_data_repLevel[[pertColName] + cp_features]
  #
  # if dataset == 'LINCS':
  #   cp['Compounds'] = cp['PERT'].str[0:13]
  #   l1k['Compounds'] = l1k['PERT'].str[0:13]
  # else:
  #   cp['Compounds'] = cp['PERT']
  #   l1k['Compounds'] = l1k['PERT']

  return cp_data_repLevel, cp_features

def split_train_test(data,config, features = None):

  mock_data = data[data['Metadata_ASSAY_WELL_ROLE'] == 'mock']

  splitter = GroupShuffleSplit(test_size=config['test_split_ratio'], n_splits=2, random_state=7)
  split = splitter.split(mock_data, groups=mock_data['Metadata_Plate'])
  train_inds, test_inds = next(split)

  train_data_all = mock_data.iloc[train_inds]
  test_data_mocks = mock_data.iloc[test_inds]

  # train_data, val_data = train_test_split(mock_data, test_size=0.4)
  splitter = GroupShuffleSplit(test_size=.30, n_splits=2, random_state=7)
  split = splitter.split(train_data_all, groups=train_data_all['Metadata_Plate'])
  train_inds, val_inds = next(split)

  train_data = train_data_all.iloc[train_inds]
  val_data = train_data_all.iloc[val_inds]

  test_data_treated = data[data['Metadata_ASSAY_WELL_ROLE'] != 'mock']

  # scale to training set
  scaler_cp = preprocessing.StandardScaler()
  train_data.loc[:, features] = scaler_cp.fit_transform(train_data[features].values.astype('float64'))
  val_data.loc[:, features] = scaler_cp.transform(val_data[features].values.astype('float64'))
  test_data_mocks.loc[:, features] = scaler_cp.transform(test_data_mocks[features].values.astype('float64'))
  test_data_treated.loc[:, features] = scaler_cp.transform(test_data_treated[features].values.astype('float64'))

  datasets = {
    'train': train_data,
    'val': val_data,
    'test_ctrl': test_data_mocks,
    'test_treat': test_data_treated
  }

  for set in datasets.keys():
    datasets[set].to_csv(os.path.join(config['data_dir'], 'anomaly_output', config['dataset'], f'input_data_{set}.csv'))

  return datasets
    # l1k=mergProf_treatLevel[[pertColName]+l1k_features]
  # cp=mergProf_treatLevel[[pertColName]+cp_features]

  # le = preprocessing.LabelEncoder()
  # group_labels=le.fit_transform(l1k['Compounds'].values)

  # X_train, X_test = dt[0][0][cp_features].values, dt[1][0][cp_features].values
  # y_train, y_test = dt[0][1][l].values, dt[1][1][l]#.values

def scale_data_by_set(scale_by, sets, features):
  # scaler_ge = preprocessing.StandardScaler()
  scaler_cp = preprocessing.StandardScaler()
  # l1k_scaled = l1k.copy()
  # l1k_scaled[l1k_features] = scaler_ge.fit_transform(l1k[l1k_features].values)
  # cp_scaled = cp.copy()
  scale_by[features] = scaler_cp.fit_transform(scale_by[features].values.astype('float64'))

  scaled_sets = []
  scaled_sets.append(scale_by)
  for set in sets:
    scaled_set = set.copy()
    scaled_set[features] = scaler_cp.transform(set.values.astype('float64'))
    scaled_sets.append(scaled_set)

  return scaled_sets
