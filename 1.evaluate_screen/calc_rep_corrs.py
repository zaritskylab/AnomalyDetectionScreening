import os
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


from AnomalyDetectionScreening.ads.utils.percantage_replicates_utils import get_null_dist_median_scores, get_moa_p_vals, get_replicates_score
from AnomalyDetectionScreening.dataset_paper_repo.utils.replicateCorrs import replicateCorrs
from sklearn.preprocessing import MinMaxScaler




def set_index_fields(df, index_fields=None):

    if index_fields is None:
        index_fields = ['Metadata_Plate', 'Metadata_ASSAY_WELL_ROLE', 'Metadata_broad_sample', 'Metadata_Well',
                        'Metadata_mmoles_per_liter']
    df = df.set_index(
        index_fields)
    return df


def load_pickles_and_concatenate(directory):
    # Initialize an empty dictionary to store the concatenated data.
    concatenated_dict = {}

    # List all files in the directory.
    files = os.listdir(directory)

    # Loop through each file in the directory.
    for filename in files:
        # Check if the file is a pickle file (ends with .pkl).
        if filename.endswith(".pickle"):
            file_path = os.path.join(directory, filename)
            
            # Load the pickle file and merge it into the concatenated_dict.
            with open(file_path, 'rb') as file:
                data = pickle.load(file)
                concatenated_dict.update(data)

    return concatenated_dict


if __name__ == "__main__":
    import sys
    import pickle
    import os
    import pandas as pd
    import numpy as np
    import glob
    from tqdm import tqdm
    import random
    from collections import defaultdict
    from statistics import median

    ds_info_dict={'CDRP':['CDRP-BBBC047-Bray',['Metadata_Sample_Dose','pert_sample_dose']],
            'CDRP-bio':['CDRPBIO-BBBC036-Bray',['Metadata_Sample_Dose','pert_sample_dose']],
            'TAORF':['TA-ORF-BBBC037-Rohban',['Metadata_broad_sample','pert_id',]],
            'LUAD':['LUAD-BBBC041-Caicedo',['x_mutation_status','allele']],
            'LINCS':['LINCS-Pilot1',['Metadata_pert_id_dose','pert_id_dose']]}


    if len(sys.argv) <2:
        exp_name = 'base_t'
        dataset = 'CDRP-bio'
        profileType = 'normalized_variable_selected'
        # profileType = 'augmented'

        slice_id = 0
    else:
        exp_name, dataset, profileType, slice_id = sys.argv[1:]

    null_dist_path = f'/sise/assafzar-group/assafzar/genesAndMorph/results/{dataset}/CellPainting/{profileType}/null_distribution_replicates_1000.pkl'
    # slice_size = int(slice_size)
    # slice_id = int(slice_id)
    ################################################
    # dataset options: 'CDRP' , 'LUAD', 'TAORF', 'LINCS', 'CDRP-bio'
    # datasets=['LUAD','TAORF','LINCS','CDRP-bio'];
    # datasets=['LINCS', 'CDRP-bio','CDRP'];
    # datasets=['TAORF','LUAD','LINCS', 'CDRP-bio']
    datasets=['CDRP-bio']
    # index_col = 'Metadata_broad_sample'
    datasets = [dataset]
    # DT_kfold={'LUAD':10, 'TAORF':5, 'LINCS':25, 'CDRP-bio':6,'CDRP':40}


    # from IPython.display import clear_output
    ################################################
    # CP Profile Type options: 'augmented', 'augmented_after_fs' , 'normalized', 'normalized_variable_selected'
    # profileType='normalized_variable_selected'
    # profileType ='normalized_variable_selected'
    base_dir= '/sise/assafzar-group/assafzar/genesAndMorph'
    data_dir=base_dir+'/preprocessed_data/'+ds_info_dict[datasets[0]][0]+'/'
    # exp_name= 'ae_12_09_fs'

    if 'variable_selected' in profileType:
        profileType = 'normalized_variable_selected'
    else:  
        profileType = 'augmented'

    # output_dir = f'{base_dir}/anomaly_output/{datasets[0]}/CellPainting/{exp_name}'  
    # save_base_dir = f'{base_dir}/results/{datasets[0]}/CellPainting'
    # exp_save_dir = f'{save_base_dir}/{exp_name}'

    by_dose = False
    cp = True

    if by_dose:
        pertColName='Metadata_Sample_Dose'
        l1k_pert_col_name = 'pert_id_dose'
        dose_str = 'd'
    else:
        pertColName='Metadata_broad_sample'
        l1k_pert_col_name = 'pert_id'
        dose_str = ''

    if cp:
        modality_str = 'cp'
    else:
        modality_str = 'l1k'

    sheet_name = f'{modality_str}-{dataset}-{profileType.split("_")[1]}-{dose_str}'


    output_dir = f'{base_dir}/anomaly_output/{datasets[0]}/CellPainting/{exp_name}/'  
    save_base_dir = f'{base_dir}/results/{datasets[0]}/CellPainting'
    exp_save_dir = f'{save_base_dir}/{exp_name}/figs'
    # null_base_dir = f'{base_dir}/results/{datasets[0]}/'

    methods = {
        # '4to1':{'path':f'/storage/users/g-and-n/tabular_models_results/41/ALL/zscores'},
        # '2to1':{'path':f'/storage/users/g-and-n/tabular_models_results/21/ALL/zscores'},
        'anomaly':{'name':'anomaly','path':os.path.join(output_dir,f'replicate_level_{modality_str}_{profileType}_ae.csv')},
        'anomaly_err':{'name':'anomaly_err','path':os.path.join(output_dir,f'replicate_level_{modality_str}_{profileType}_ae_diff.csv')},
        'anomaly_emb':{'name':'anomaly_emb','path':os.path.join(output_dir,f'replicate_level_{modality_str}_{profileType}_ae_embeddings.csv')},
        'raw':{'name':'raw','path': os.path.join(output_dir,f'replicate_level_{modality_str}_{profileType}_baseline.csv')},
        # 'raw_unchanged':{'name':'raw_unchanged','path': os.path.join(data_dir,'CellPainting',f'replicate_level_cp_{profileType}.csv.gz')}
        # '5to5':{'path':f'/storage/users/g-and-n/tabular_models_results/55/ALL/zscores'},
    }


    # os.makedirs(save_base_dir,exist_ok=True)
    os.makedirs(exp_save_dir,exist_ok=True)

    new_ss = True
    ss_calc = 'median'
    num_rand = 1000

    # This is a must, otherwise the MinMax scaler will not work properly
    z_trim = 5


    for m in methods.keys():
        print(f'loading zscores for method: {m}')
        zscores = pd.read_csv(methods[m]['path'], compression = 'gzip')
        zscores = zscores.query('Metadata_ASSAY_WELL_ROLE == "treated"')
        if m == 'anomaly_emb':
            meta_features = zscores.columns[zscores.columns.str.contains("Metadata_")]
            features = zscores.columns[~zscores.columns.str.contains("Metadata_")]
        else:
            features = zscores.columns[zscores.columns.str.contains("Cells_|Cytoplasm_|Nuclei_")]
        # zscores = set_index_fields(zscores)
        # methods[m]['zscores'] = zscores.loc[:,methods[m]['features']]
        if z_trim is not None:
            zscores[features] = np.where(zscores[features]>z_trim,z_trim ,zscores[features])
            zscores[features] = np.where(zscores[features]<-z_trim,-z_trim ,zscores[features])
        # del cp_data_repLevel, l1k_data_repLevel

        scale_minmax = True # scale minmax
        if scale_minmax:
            scaler = MinMaxScaler(feature_range=(0, 1))
            zscores[features] = scaler.fit_transform(zscores[features])
        methods[m]['features'] = features
        methods[m]['zscores'] = zscores


