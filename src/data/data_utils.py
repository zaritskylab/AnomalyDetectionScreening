import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import pairwise_distances
from pycytominer.cyto_utils import output
import pickle

from utils.global_variables import DS_INFO_DICT


# adapted from https://github.com/carpenter-singh-lab/2022_Haghighi_NatureMethods/blob/main/utils/readProfiles.py
labelCol='PERT'

################################################################################

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


################################################################################


def load_zscores(methods,base_dir,dataset, profile_type,normalize_by_all=False,by_dose=False,z_trim=None,sample='treated',set_index=True,debug_mode=False, filter_by_highest_dose=False, min_max_norm=False):


    for m in methods.keys():
      print(f'loading zscores for method: {m}')
      if m == 'l1k':
          methods[m]['modality'] ='L1000'

          zscores, features = load_data(base_dir,dataset, profile_type,modality=methods[m]['modality'], whole_plate_normalization =normalize_by_all)
          if debug_mode:
                print('debug mode is on')
                sample_size = round(len(zscores) / 10)
                zscores = zscores.sample(sample_size)

          methods[m]['features']=list(features)
          
          # methods[m]['zscores'] = zscores.loc[:,methods[m]['features']]
      else:
        methods[m]['modality']='CellPainting'
        zscores = pd.read_csv(methods[m]['path'], compression = 'gzip', low_memory=False)   
        if debug_mode:
            print('debug mode is on')
            sample_size = round(len(zscores) / 10)
            zscores = zscores.sample(sample_size)
        
        if m == 'anomaly_emb':
        #   meta_features = [c for c in zscores.columns if 'Metadata_' in c]
            features = [c for c in zscores.columns if 'Metadata_' not in c]
        else:
            features = zscores.columns[zscores.columns.str.contains("Cells_|Cytoplasm_|Nuclei_")]
        methods[m]['features']=list(features)

        if normalize_by_all:
            zscores = standardize_per_catX(zscores,DS_INFO_DICT[dataset]['CellPainting']['plate_col'],methods[m]['features'])

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

      # if set_index:
      #   zscores = set_index_fields(zscores,dataset,by_dose=by_dose, modality=methods[m]['modality'])
      #   methods[m]['zscores'] = zscores.loc[:,methods[m]['features']]
      # else:
      methods[m]['zscores'] = zscores

    return methods



################################################################################

def get_cp_dir(dataset_rootDir,dataset,profileType,exp_name = '',processed=False, modality='CellPainting'):

    """get the path to the cp data directory"""
    if processed:
        # dataDir = os.path.join(dataset_rootDir,'anomaly_output', DS_INFO_DICT[dataset]['name'],modality)
        if len(exp_name)>0:
            dataDir = os.path.join(dataset_rootDir,'anomaly_output', dataset,modality,exp_name)
            os.makedirs(dataDir, exist_ok=True)
        else:
            dataDir = os.path.join(dataset_rootDir,'anomaly_output', dataset,modality)

        # if len(exp_name)>0:
            # dataDir = os.path.join(dataDir,)
    else:  
        dataDir = os.path.join(dataset_rootDir,'preprocessed_data', DS_INFO_DICT[dataset]['name'],modality)

    os.makedirs(dataDir, exist_ok=True)
    return dataDir

################################################################################

def get_cp_path(dataset_rootDir,dataset,profileType,exp_name = '',processed=False, modality='CellPainting'):
    # dataDir = os.path.join(dataset_rootDir,'processed_data', DS_INFO_DICT[dataset][0],'CellPainting',exp_name)
    cp_dir = get_cp_dir(dataset_rootDir,dataset,profileType,exp_name,processed,modality)

    if modality == 'L1000':
        cp_path = os.path.join(cp_dir,'replicate_level_l1k.csv.gz')
    else:
        cp_path = os.path.join(cp_dir,'replicate_level_cp_'+profileType+'.csv.gz')

    return cp_path


################################################################################
def save_profiles(df, dataset_rootDir, filename, float_format = "%.5g", compression={"method": "gzip", "mtime": 1}, output_type='.csv',exp_name='', modality='CellPainting'):

    output(
            df=df,
            output_filename=os.path.join(dataset_rootDir,filename+output_type),
            compression_options=compression,
            float_format=float_format,
        )


def read_replicate_single_modality_level_profiles(dataset_rootDir,dataset,profileType,per_plate_normalized_flag,exp_name='',modality='CellPainting'):
    """
    Reads replicate level CSV files in the form of a dataframe
    Extract measurments column names for a single modality
    Remove columns with low variance (<thrsh_var)
    Remove columns with more NaNs than a certain threshold (>null_vals_ratio)

    Inputs:
    dataset_rootDir: datasets root dir
    dataset: any from the available list of ['LUAD', 'TAORF', 'LINCS', 'CDRP-bio', 'CDRP']
    profileType:   Cell Painting profile type that can be 'augmented' , 'normalized', 'normalized_variable_selected'
    per_plate_normalized_flag: if True it will standardize data per plate

    Output:
    cp_data_repLevel: dataframes with all the annotations available in the raw data
    """

    dataDir=dataset_rootDir+'/preprocessed_data/'+DS_INFO_DICT[dataset]['name']+'/'

    if modality == 'L1000':
        filename = 'replicate_level_l1k.csv'
    else:
        filename = 'replicate_level_cp_'+profileType+'.csv'
    if len(exp_name)>0:
        cp_dataDir = dataset_rootDir+'/anomaly_output/'+dataset
        cp_path = os.path.join(cp_dataDir,modality,exp_name,filename)
    else:
        if dataset == 'LUAD':
            cp_path = dataDir+modality+'/'+filename
        else:
            cp_path = dataDir+modality+'/'+filename+'.gz'
        # cp_path = os.path.join(dataDir,modality,filename+'.gz')

    # print(f' loading {cp_path}')
    # if not os.path.exists(cp_path):
        # raise ValueError("File not found: {}".format(cp_path))
        # profileType = 'augmented'
        # cp_path = dataDir + '/CellPainting/replicate_level_cp_' + profileType + '.csv.gz'
    if dataset == 'LUAD' and len(exp_name) == 0:
        cp_data_repLevel = pd.read_csv(cp_path)
    else:
        cp_data_repLevel = pd.read_csv(cp_path, compression='gzip')

    if modality == 'L1000':
        features = cp_data_repLevel.columns[cp_data_repLevel.columns.str.contains("_at")].tolist()
    else:
        features = cp_data_repLevel.columns[cp_data_repLevel.columns.str.contains("Cells_|Cytoplasm_|Nuclei_")].tolist()
        
    ########## removes nan and inf values
    cp_data_repLevel=cp_data_repLevel.replace([np.inf, -np.inf], np.nan)

    #
    cols2remove=[i for i in features if cp_data_repLevel[i].isnull().sum(axis=0)>0.05]
    cp_data_repLevel.drop(cols2remove, axis=1, inplace=True);
    features = list(set(features) - set(cols2remove))
    cp_data_repLevel[features] = cp_data_repLevel[features].interpolate()
    #     cp=cp.fillna(cp.median())

    ################ Per plate scaling
    if per_plate_normalized_flag:
        plate_col = DS_INFO_DICT[dataset][modality]['plate_col']
        cp_data_repLevel = standardize_per_catX(cp_data_repLevel,plate_col,features);
        # l1k_data_repLevel = standardize_per_catX(l1k_data_repLevel,'det_plate',l1k_features);

        # cols2removeCP=[i for i in cp_features if (cp_data_repLevel[i].isnull().sum(axis=0)/cp_data_repLevel.shape[0])>0.05]
        # cp_data_repLevel=cp_data_repLevel.drop(cols2removeCP, axis=1);
        # cp_features = list(set(cp_features) - set(cols2removeCP))
        # cp_data_repLevel[cp_features] = cp_data_repLevel[cp_features].interpolate()

    return [cp_data_repLevel,features]

################################################################################
def read_replicate_level_profiles(dataset_rootDir,dataset,profileType,per_plate_normalized_flag,exp_name=''):
    """
    Reads replicate level CSV files in the form of a dataframe
    Extract measurments column names for each modalities
    Remove columns with low variance (<thrsh_var)
    Remove columns with more NaNs than a certain threshold (>null_vals_ratio)
    
    Inputs:
    dataset_rootDir: datasets root dir
    dataset: any from the available list of ['LUAD', 'TAORF', 'LINCS', 'CDRP-bio', 'CDRP']
    profileType:   Cell Painting profile type that can be 'augmented' , 'normalized', 'normalized_variable_selected'
    per_plate_normalized_flag: if True it will standardize data per plate 
    
    Output:
    cp_data_repLevel, l1k_data_repLevel: dataframes with all the annotations available in the raw data
    """
    
    dataDir=dataset_rootDir+'/preprocessed_data/'+DS_INFO_DICT[dataset]["name"]+'/'
    # dataDir=dataset_rootDir+'/anomaly_output/'+dataset

        
    # cp_data_repLevel=pd.read_csv(dataDir+'/CellPainting/replicate_level_cp_'+profileType+'.csv.gz')
    if len(exp_name)>0:
        cp_dataDir = dataset_rootDir+'/anomaly_output/'+dataset
        cp_path = os.path.join(cp_dataDir,'CellPainting',exp_name,'replicate_level_cp_'+profileType+'.csv')
    else:
        if dataset == 'LUAD':
            cp_path = dataDir+'CellPainting/replicate_level_cp_'+profileType+'.csv'
        else:
            cp_path = dataDir+'CellPainting/replicate_level_cp_'+profileType+'.csv.gz'
    print(f' loading {cp_path}')
    if not os.path.exists(cp_path):
        raise ValueError("File not found: {}".format(cp_path))
        # profileType = 'augmented'
        # cp_path = dataDir + '/CellPainting/replicate_level_cp_' + profileType + '.csv.gz'
    if dataset == 'LUAD' and len(exp_name) == 0:
        cp_data_repLevel = pd.read_csv(cp_path)
        l1k_data_repLevel=pd.read_csv(dataDir+'/L1000/replicate_level_l1k.csv')
    else:
        cp_data_repLevel=pd.read_csv(cp_path, compression='gzip')
        l1k_data_repLevel=pd.read_csv(dataDir+'/L1000/replicate_level_l1k.csv.gz')

    cp_features, l1k_features =  extract_feature_names(cp_data_repLevel, l1k_data_repLevel);
    
    ########## removes nan and inf values
    l1k_data_repLevel=l1k_data_repLevel.replace([np.inf, -np.inf], np.nan)
    cp_data_repLevel=cp_data_repLevel.replace([np.inf, -np.inf], np.nan)
    
    #
    cols2removeCP=[i for i in cp_features if cp_data_repLevel[i].isnull().sum(axis=0)>0.05]
    # print(cols2removeCP)
    cp_data_repLevel.drop(cols2removeCP, axis=1, inplace=True);
    cp_features = list(set(cp_features) - set(cols2removeCP))
    cp_data_repLevel[cp_features] = cp_data_repLevel[cp_features].interpolate()
#     cp=cp.fillna(cp.median())

    # cols2removeGE=[i for i in l1k.columns if l1k[i].isnull().sum(axis=0)>0]
    # print(cols2removeGE)
    # l1k_features = list(set(l1k_features) - set(cols2removeGE))
    # print(len(l1k_features))
    # l1k=l1k.drop(cols2removeGE, axis=1);
    l1k_data_repLevel[l1k_features] = l1k_data_repLevel[l1k_features].interpolate()
    # l1k=l1k.fillna(l1k.median())  
    # 
    print('ATTENTION: l1k_data_repLevel IS standardized per plate EVEN with flag "per_plate_normalized_flag"=False\n\
        To change this, move the standartization into the flag condition in "read_replicate_level_profiles function"')  
    l1k_data_repLevel = standardize_per_catX(l1k_data_repLevel,'det_plate',l1k_features);    

    ################ Per plate scaling 
    if per_plate_normalized_flag:
        cp_data_repLevel = standardize_per_catX(cp_data_repLevel,'Metadata_Plate',cp_features);
        # l1k_data_repLevel = standardize_per_catX(l1k_data_repLevel,'det_plate',l1k_features);    

        cols2removeCP=[i for i in cp_features if (cp_data_repLevel[i].isnull().sum(axis=0)/cp_data_repLevel.shape[0])>0.05]
        cp_data_repLevel=cp_data_repLevel.drop(cols2removeCP, axis=1);
        cp_features = list(set(cp_features) - set(cols2removeCP))
        cp_data_repLevel[cp_features] = cp_data_repLevel[cp_features].interpolate()
    
    return [cp_data_repLevel,cp_features], [l1k_data_repLevel,l1k_features]


################################################################################
def extract_feature_names(cp_data_repLevel, l1k_data_repLevel):
    """
    extract Cell Painting and L1000 measurments names among the column names
    
    Inputs:
    cp_data_repLevel, l1k_data_repLevel: dataframes with all the annotations available in the raw data
    
    Outputs: list of feature names for each modality
    
    """
    # features to analyse
    cp_features=cp_data_repLevel.columns[cp_data_repLevel.columns.str.contains("Cells_|Cytoplasm_|Nuclei_")].tolist()
    l1k_features=l1k_data_repLevel.columns[l1k_data_repLevel.columns.str.contains("_at")].tolist()

    return cp_features, l1k_features


################################################################################
def extract_metadata_column_names(cp_data, l1k_data):
    """
    extract metadata column names among the column names for any level of data
    
    Inputs:
    cp_data_repLevel, l1k_data_repLevel: dataframes with all the annotations available in the raw data
    
    Outputs: list of metadata column names for each modality
    
    """
    cp_meta_col_names=cp_data.columns[~cp_data.columns.str.contains("Cells_|Cytoplasm_|Nuclei_")].tolist()
    l1k_meta_col_names=l1k_data.columns[~l1k_data.columns.str.contains("_at")].tolist()

    return cp_meta_col_names, l1k_meta_col_names

################################################################################
def read_treatment_level_profiles(dataset_rootDir,dataset,profileType,filter_repCorr_params,per_plate_normalized_flag,exp_name='',by_dose=True, filter_by_highest_dose=False, save_reproducible=False, save_n_samples_per_MoA=False, filter_perts='highRepOverlap'):

    """
    Reads replicate level CSV files (scaled replicate level profiles per plate)
    Rename the column names to match across datasets to PERT in both modalities
    Remove perturbations with low rep corr across both (filter_perts='highRepOverlap') 
            or one of the modalities (filter_perts='highRepUnion')
    Form treatment level profiles by averaging the replicates
    Select and keep the metadata columns you want to keep for each dataset
    Merge treatment level profiles to its own metadata
    
    Inputs:
    dataset_rootDir: datasets root dir
    dataset: any from the available list of ['LUAD', 'TAORF', 'LINCS', 'CDRP-bio', 'CDRP']
    profileType:   Cell Painting profile type that can be 'augmented' , 'normalized', 'normalized_variable_selected'

    Output: 
    [cp_data_treatLevel,cp_features], [l1k_data_treatLevel,l1k_features]
    each is a list of dataframe and feature names for each of modalities
    """
    
    filter_perts=filter_repCorr_params[0]
    repCorrFilePath=filter_repCorr_params[1]
    if len(filter_repCorr_params)>2:
        profile_types_filtered = filter_repCorr_params[2]
    else:
        profile_types_filtered = None
    
    
    [cp_data_repLevel,cp_features], [l1k_data_repLevel,l1k_features] = read_replicate_level_profiles(dataset_rootDir,dataset,profileType,per_plate_normalized_flag,exp_name);
    
    if filter_by_highest_dose:
        print('filtering by highest dose, from ',cp_data_repLevel.shape[0],',',len(cp_features),  ',  ',l1k_data_repLevel.shape[0],',',len(l1k_features))
        cp_data_repLevel = filter_data_by_highest_dose(cp_data_repLevel, dataset, modality = 'CellPainting')
        l1k_data_repLevel = filter_data_by_highest_dose(l1k_data_repLevel, dataset, modality = 'L1000')
        print('to ',cp_data_repLevel.shape[0],',',len(cp_features),  ',  ',l1k_data_repLevel.shape[0],',',len(l1k_features))
        # cp_data_repLevel = cp_data_repLevel[cp_data_repLevel['Metadata_dose_recode'] == 'dose_6']
        # l1k_data_repLevel = l1k_data_repLevel[l1k_data_repLevel['Metadata_dose_recode'] == 'dose_6']

    ############ rename columns that should match to PERT
    labelCol='PERT'
    # DS_INFO_DICT[dataset]['CellPainting']['dose_col']
    if by_dose:
        cp_data_repLevel=cp_data_repLevel.rename(columns={DS_INFO_DICT[dataset]['CellPainting']['dose_col']:labelCol})
        l1k_data_repLevel=l1k_data_repLevel.rename(columns={DS_INFO_DICT[dataset]['L1000']['dose_col']:labelCol})    
    else:
        cp_data_repLevel=cp_data_repLevel.rename(columns={DS_INFO_DICT[dataset]['CellPainting']['cpd_col']:labelCol})
        l1k_data_repLevel=l1k_data_repLevel.rename(columns={DS_INFO_DICT[dataset]['L1000']['cpd_col']:labelCol})    
        # DS_INFO_DICT[dataset][1][0] = labelCol
    meta_features = [m for m in cp_data_repLevel.columns if m not in cp_features]

    # if '.' in cp_data_repLevel[labelCol].iloc[0]:
    #     cp_data_repLevel[labelCol] = cp_data_repLevel[labelCol].apply(lambda x: x.split('.')[0])
    # if '.' in l1k_data_repLevel[labelCol].iloc[0]:
    #     l1k_data_repLevel[labelCol] = l1k_data_repLevel[labelCol].apply(lambda x: x.split('.')[0])
            
    
    ###### print some data statistics
    print(dataset+': Replicate Level Shapes (nSamples x nFeatures): cp: ',\
          cp_data_repLevel.shape[0],',',len(cp_features),  ',  l1k: ',l1k_data_repLevel.shape[0],',',len(l1k_features))

    print('l1k n of rep: ',l1k_data_repLevel.groupby([labelCol]).size().median())
    print('cp n of rep: ',cp_data_repLevel.groupby([labelCol]).size().median())
    
    # add dose to sheetname if not included
    # profile_types_filtered = [a+'_d' for a in profile_types_filtered if by_dose and a.split('_')[-1] != 'd']
    # if save_n_samples_per_MoA:
        # n_samples_per_MoA = cp_data_repLevel.groupby('Metadata_moa').size().sort_values(ascending=False)
        # path = f'{dataset_rootDir}/results/{dataset}/CellPainting/{exp_name}/n_samples_per_MoA_pre_filtering.csv'
        # n_samples_per_MoA.to_csv(path)
    ###### remove perts with low rep corr
    if filter_perts=='highRepOverlap':    
        highRepPerts = highRepFinder(dataset,'intersection',repCorrFilePath,profile_types_filtered=profile_types_filtered,save_reproducible=save_reproducible) + ['negcon'];
        
        cp_data_repLevel=cp_data_repLevel[cp_data_repLevel['PERT'].isin(highRepPerts)].reset_index()
        l1k_data_repLevel=l1k_data_repLevel[l1k_data_repLevel['PERT'].isin(highRepPerts)].reset_index()  
        
    elif filter_perts=='highRepUnion':
        highRepPerts = highRepFinder(dataset,'union',repCorrFilePath,profile_types_filtered=profile_types_filtered, save_reproducible=save_reproducible) + ['negcon'];
        
        cp_data_repLevel=cp_data_repLevel[cp_data_repLevel['PERT'].isin(highRepPerts)].reset_index()
        l1k_data_repLevel=l1k_data_repLevel[l1k_data_repLevel['PERT'].isin(highRepPerts)].reset_index()      
    
    ####### form treatment level profiles
    l1k_data_treatLevel=l1k_data_repLevel.groupby(labelCol)[l1k_features].mean().reset_index();
    cp_data_treatLevel=cp_data_repLevel.groupby(labelCol)[cp_features].mean().reset_index();
    
    ###### define metadata and merge treatment level profiles
#     dataset:[[cp_columns],[l1k_columns]]
#     meta_dict={'CDRP':[['Metadata_moa','Metadata_target'],['CPD_NAME','CPD_TYPE','CPD_SMILES']],
#                'CDRP-bio':[['Metadata_moa','Metadata_target'],['CPD_NAME','CPD_TYPE','CPD_SMILES']],
#               'TAORF':[['Metadata_moa'],['pert_type']],
#               'LUAD':[['Metadata_broad_sample_type','Metadata_pert_type'],[]],
#               'LINCS':[['Metadata_moa', 'Metadata_alternative_moa'],['moa']]}

    if by_dose:
        meta_dict={'CDRP':[['Metadata_moa','Metadata_target'],[]],
                'CDRP-bio':[['Metadata_moa','Metadata_target'],[]],
                'TAORF':[[],[]],
                'LUAD':[[],[]],
                'LINCS':[['Metadata_moa', 'Metadata_alternative_moa'],['moa']]}
    else:
        meta_dict={'CDRP':[['Metadata_moa','Metadata_target'],[]],
                'CDRP-bio':[['Metadata_moa','Metadata_target'],[]],
                'TAORF':[[],[]],
                'LUAD':[[],[]],
                'LINCS':[['Metadata_moa', 'Metadata_alternative_moa'],['moa']]}    
    
    meta_cp=cp_data_repLevel[[labelCol]+meta_dict[dataset][0]].\
    drop_duplicates().reset_index(drop=True)
    meta_l1k=l1k_data_repLevel[[labelCol]+meta_dict[dataset][1]].\
    drop_duplicates().reset_index(drop=True)

    cp_data_treatLevel=pd.merge(cp_data_treatLevel,meta_cp, how='inner',on=[labelCol])
    l1k_data_treatLevel=pd.merge(l1k_data_treatLevel,meta_l1k, how='inner',on=[labelCol])
    
    return [cp_data_treatLevel,cp_features], [l1k_data_treatLevel,l1k_features]



################################################################################
def read_paired_treatment_level_profiles(dataset_rootDir,dataset,profileType,filter_repCorr_params,per_plate_normalized_flag,exp_name='',by_dose=True, filter_by_highest_dose=False, save_n_samples_per_MoA=False, filter_perts='highRepOverlap'):

    """
    Reads treatment level profiles
    Merge dataframes by PERT column
    
    Inputs:
    dataset_rootDir: datasets root dir
    dataset: any from the available list of ['LUAD', 'TAORF', 'LINCS', 'CDRP-bio', 'CDRP']
    profileType:   Cell Painting profile type that can be 'augmented' , 'normalized', 'normalized_variable_selected'
    per_plate_normalized_flag: True for scaling per plate

    Output: 
    mergedProfiles_treatLevel: paired treatment level profiles
    cp_features,l1k_features list of feature names for each of modalities
    """
    
    [cp_data_treatLevel,cp_features], [l1k_data_treatLevel,l1k_features]=\
    read_treatment_level_profiles(dataset_rootDir,dataset,profileType,filter_repCorr_params,per_plate_normalized_flag,exp_name,by_dose, filter_by_highest_dose, save_n_samples_per_MoA=save_n_samples_per_MoA);
    

    mergedProfiles_treatLevel=pd.merge(cp_data_treatLevel, l1k_data_treatLevel, how='inner',on=[labelCol])

    print('Treatment Level Shapes (nSamples x nFeatures+metadata):',cp_data_treatLevel.shape,l1k_data_treatLevel.shape,\
          'Merged Profiles Shape:', mergedProfiles_treatLevel.shape)

    
    return mergedProfiles_treatLevel,cp_features,l1k_features


################################################################################
def generate_random_match_of_replicate_pairs(cp_data_repLevel, l1k_data_repLevel,nRep):
    """
    Note that there is no match at the replicate level for this dataset, we either:
        - Forming ALL the possible pairs for replicate level data matching (nRep='all' - string)
        - Randomly sample samples in each modality and form pairs (nRep -> int)
        
    Inputs:
        cp_data_repLevel, l1k_data_repLevel: dataframes with all the annotations available in the raw data
    
    Outputs: 
        Randomly paired replicate level profiles
    
    """
    labelCol='PERT'
    
    if nRep=='all':
        cp_data_n_repLevel=cp_data_repLevel.copy()
        l1k_data_n_repLevel=l1k_data_repLevel.copy()
    else:
#         nR=np.min((cp_data_repLevel.groupby(labelCol).size().min(),l1k_data_repLevel.groupby(labelCol).size().min()))
#     cp_data_n_repLevel=cp_data_repLevel.groupby(labelCol).apply(lambda x: x.sample(n=nR,replace=True)).reset_index(drop=True)
        nR=nRep
        cp_data_n_repLevel=cp_data_repLevel.groupby(labelCol).\
        apply(lambda x: x.sample(n=np.min([nR,x.shape[0]]))).reset_index(drop=True)
        l1k_data_n_repLevel=l1k_data_repLevel.groupby(labelCol).\
        apply(lambda x: x.sample(n=np.min([nR,x.shape[0]]))).reset_index(drop=True)


    mergedProfiles_repLevel=pd.merge(cp_data_n_repLevel, l1k_data_n_repLevel, how='inner',on=[labelCol])

    return mergedProfiles_repLevel

################################################################################
def highRepFinder(dataset,how,repCorrFilePath, profile_types_filtered=None, save_reproducible=False):
    """
    This function reads pre calculated and saved Replicate Correlation values file and filters perturbations
    using one of the following filters:
        - intersection: intersection of high quality profiles across both modalities
        - union: union of high quality profiles across both modalities
        
    * A High Quality profile is defined as a profile having replicate correlation more than 90th percentile of
      its null distribution
        
    Inputs:
        dataset (str): dataset name
        how (str):  can be intersection or union
    
    Output: list of high quality perurbations
    
    """
    repCorDF=pd.read_excel(repCorrFilePath, sheet_name=None)
    
    repCorrs = {}
    all_profiles_acronym = ''
    if profile_types_filtered is not None:
        for s in profile_types_filtered:
            all_profiles_acronym += s[0]
            if 'index' in repCorDF[s].keys():
                repCorDF[s].rename(columns={'index':'Unnamed: 0'}, inplace=True)
            # repCorDF[s]=repCorDF[s].rename(columns={'Unnamed: 0':'pert_id'})
            # repCorDF[s]=repCorDF[s].set_index('pert_id')
            repCorrs[s] = {}
            repCorrs[s]['df'] =repCorDF[s]
            repCorrs[s]['highRepPerts'] = repCorDF[s][repCorDF[s]['RepCor']>repCorDF[s]['Rand90Perc']]['Unnamed: 0'].tolist()

        # if how=='intersection':
        highRepPerts= []
        for s in profile_types_filtered:
            print(f'{s} has {len(repCorrs[s]["highRepPerts"])} perturbations')
            if len(highRepPerts) == 0:
                highRepPerts = repCorrs[s]['highRepPerts']
            else:
                if how=='intersection':
                    highRepPerts = list(set(highRepPerts) & set(repCorrs[s]['highRepPerts']))
                elif how=='union':
                    highRepPerts = list(set(highRepPerts) | set(repCorrs[s]['highRepPerts']))
            print(f'{s}: from {repCorrs[s]["df"].shape[0]} to {len(repCorrs[s]["highRepPerts"])}')
            # highRepPerts += repCorrs[s]['high']
        print(f'CP and l1k high rep overlap: {len(highRepPerts)}')
    else:
        # repCorDF=repCorDF['anomalyCP-'+dataset.lower()]
        cpRepDF=repCorDF['cp-'+dataset.lower()]
        cpHighList=cpRepDF[cpRepDF['RepCor']>cpRepDF['Rand90Perc']]['Unnamed: 0'].tolist()
        print('CP: from ',cpRepDF.shape[0],' to ',len(cpHighList))
        cpRepDF=repCorDF['l1k-'+dataset.lower()]
        l1kHighList=cpRepDF[cpRepDF['RepCor']>cpRepDF['Rand90Perc']]['Unnamed: 0'].tolist()
    
    #     print("l1kHighList",l1kHighList)
    #     print("cpHighList",cpHighList)   
        if how=='intersection':
            highRepPerts=list(set(l1kHighList) & set(cpHighList))
            print('l1k: from ',cpRepDF.shape[0],' to ',len(l1kHighList))
            print('CP and l1k high rep overlap: ',len(highRepPerts))
            
        elif how=='union':
            highRepPerts=list(set(l1kHighList) | set(cpHighList))
            print('l1k: from ',cpRepDF.shape[0],' to ',len(l1kHighList))
            print('CP and l1k high rep union: ',len(highRepPerts))        

    if save_reproducible:
        pickle.dump(highRepPerts, open(f'highRepPerts_{all_profiles_acronym}_.pkl', 'wb'))
        
    return highRepPerts


def rename_affyprobe_to_genename(l1k_data_df,l1k_features,map_source_address):
    """
    map input dataframe column name from affy prob id to gene names
    
    """
    meta=pd.read_excel(map_source_address)  
    
#     meta=pd.read_csv("../affy_probe_gene_mapping.txt",delimiter="\t",header=None, names=["probe_id", "gene"])
    meta_gene_probID=meta.set_index('probe_id')
    d = dict(zip(meta_gene_probID.index, meta_gene_probID['symbol']))
    l1k_features_gn=[d[l] for l in l1k_features]
    l1k_data_df = l1k_data_df.rename(columns=d)   

    return l1k_data_df,l1k_features_gn



def rename_to_genename_list_to_affyprobe(l1k_features_gn,our_l1k_prob_list,map_source_address):
    """
    map a list of gene names to a list of affy prob ids
    
    """
#     map_source_address='../idmap.xlsx'
    meta=pd.read_excel(map_source_address) 
#     meta=pd.read_csv("../affy_probe_gene_mapping.txt",delimiter="\t",header=None, names=["probe_id", "gene"])
#     meta=meta[meta['probe_id'].isin(our_l1k_prob_list)].reset_index(drop=True)
    meta_gene_probID=meta.set_index('symbol')
    d = dict(zip(meta_gene_probID.index, meta_gene_probID['probe_id']))
    l1k_features=[d[l] for l in l1k_features_gn]
#     l1k_data_df = l1k_data_df.rename(columns=d)   


    return l1k_features
 

def filter_data_by_highest_dose(df, dataset, modality='CellPainting',label_col=None,dose_col=None):
    """
    Filter data to keep only high dose replicates
    """

    # if cpd_col is None:
    if label_col is None:
      if modality == 'CellPainting':

        label_col = DS_INFO_DICT[dataset][modality]['cpd_col']
        mock_val = DS_INFO_DICT[dataset][modality]['mock_val']
        n_per_dose = DS_INFO_DICT[dataset][modality]['n_per_dose']
        dose_col = 'Metadata_mg_per_ml'
      else:

        label_col = DS_INFO_DICT[dataset][modality]['cpd_col']
        n_per_dose = DS_INFO_DICT[dataset][modality]['n_per_dose']

        dose_col = 'pert_dose'

    df = df[df.groupby([label_col,dose_col]).transform('size') > (n_per_dose-1)]

    # take for each compound all the replicates with the highest dose. there can be more than one replicate with the highest dose
    df = df.sort_values(by=[label_col, dose_col], ascending=False).groupby([label_col]).head(n_per_dose)

    # keep only compounds with more than 4 replicates
    df = df[df.groupby([label_col])[label_col].transform('size') > (n_per_dose-1)]
    
    return df

def standardize_per_catX(df,column_name,cp_features):
# column_name='Metadata_Plate'
#     cp_features=df.columns[df.columns.str.contains("Cells_|Cytoplasm_|Nuclei_")]
    df_scaled_perPlate=df.copy()
    df_scaled_perPlate[cp_features]=\
    df[cp_features+[column_name]].groupby(column_name)\
    .transform(lambda x: (x - x.mean()) / x.std()).values
    return df_scaled_perPlate


def load_data(base_dir,dataset, profile_type,whole_plate_normalization = False, modality = 'CellPainting'):
  '''
  Load data from the given dataset and profile type

  Args:
  base_dir: string. path to the data directory
  dataset: string. options: ['CDRP','CDRP-bio','TAORF','LUAD','LINCS']
  profile_type: string. options: ['normalized','augmented','normalized_variable_selected']
  whole_plate_normalization: boolean. normalize plate-wise by all samples in the plate
  modality: string. options: ['CellPainting']. 'L1000' is not supported yet

  '''

  [data, cp_features] = read_replicate_single_modality_level_profiles(base_dir, dataset, profile_type ,per_plate_normalized_flag=whole_plate_normalization, modality=modality)
  features, meta_features = get_features(data, modality)

  return data, features
