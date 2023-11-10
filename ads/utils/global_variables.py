
BASE_DIR = '/sise/assafzar-group/assafzar/genesAndMorph'
DATA_DIR = f'{BASE_DIR}/preprocessed_data'
PROCESSED_DATA_DIR = f'{BASE_DIR}/processed_data'
RESULTS_DIR = f'{BASE_DIR}/results'

DS_INFO_DICT={
    'CDRP':{
        'name':'CDRP-BBBC047-Bray',
        'has_dose':False,
        'has_moa':True,
        'CellPainting':{
            'cpd_col':'Metadata_broad_sample',
            'dose_col':'Metadata_Sample_Dose',
            'role_col':'Metadata_ASSAY_WELL_ROLE',
            'plate_col':'Metadata_Plate',
            'mock_val':'mock'
        },
        'L1000':{
            'cpd_col':'pert_id',
            'dose_col':'pert_sample_dose',
            'role_col':'pert_id',
            'plate_col':'det_plate',
            'mock_val':'mock'}
        
    },

    'CDRP-bio':{
        'has_dose':False,
        'has_moa':True,
        'name':'CDRPBIO-BBBC036-Bray',
        'CellPainting':{
            'cpd_col':'Metadata_broad_sample',
            'dose_col':'Metadata_Sample_Dose',
            'role_col':'Metadata_ASSAY_WELL_ROLE',
            'plate_col':'Metadata_Plate',
            'mock_val':'mock'
        },
        'L1000':{
            'cpd_col':'pert_id',
            'dose_col':'pert_sample_dose',
            'role_col':'pert_id',
            'plate_col':'det_plate',
            'mock_val':'DMSO'}
    },

    'TAORF':{
        # 'has_dose':False,
        'has_moa':False,
        'name':'TA-ORF-BBBC037-Rohban',
        'CellPainting':{
            'cpd_col':'Metadata_broad_sample'},
        'L1000':{
            'cpd_col':'pert_id'}
    },

    'LUAD':{
        # 'has_dose':False,
        'has_moa':False,
        'name':'LUAD-BBBC041-Caicedo',
        'CellPainting':{
            'cpd_col':'x_mutation_status'},
        'L1000':{
            'cpd_col':'allele'},
    },

    'LINCS':{
        'has_dose':True,
        'has_moa':True,
        'name':'LINCS-Pilot1',
        'CellPainting':{
            'cpd_col': 'Metadata_pert_id',
            'dose_col':'Metadata_pert_id_dose',
            'role_col':'Metadata_pert_type',
            'plate_col':'Metadata_Plate',
            'mock_val':'control'},
        'L1000':{
            'cpd_col':'pert_id',
            'dose_col':'pert_id_dose',
            'role_col':'pert_id',
            'plate_col':'Metadata_plate_map_name',
            'mock_val':'DMSO'}
    }
}


# profileTypes_INFO_DICT ={
#         'normalized_variable_selected_ae':{'name':'anomaly_nvs',
#         'is_processed':True,
#         'abrv':'anomaly_n'},

#         'normalized_variable_selected_ae_error':{'name':'anomaly_err_nvs',
#         'is_processed':True,
#         'abrv':'anomaly_err_n'},

#         'normalized_variable_selected_baseline':{'name':'raw_nvs',
#         'is_processed':True,
#         'abrv':'raw_n'},

#         'normalized_variable_selected':{'name':'original_nvs',
#         'is_processed':False,
#         'abrv':'original_n'},

#         'normalized_variable_selected_normalized_by_train':{'name':'original_nvs_n',
#         'is_processed':True,
#         'abrv':'original_nvs_n'},

#         'normalized_variable_selected_normalized_by_DMSO':{'name':'original_nvs_n_DMSO',
#         'is_processed':True,
#         'abrv':'original_nvs_n_DMSO'},

#         'augmented_ae':{'name':'anomaly_aug',
#         'is_processed':True,
#         'abrv':'anomaly_a'},

#         'augmented_ae_error':{'name':'anomaly_err_aug',
#         'is_processed':True,
#         'abrv':'anomaly_err_a'},

#         'augmented_baseline':{'name':'raw_aug',
#         'is_processed':True,
#         'abrv':'raw_a'},

#         'augmented':{'name':'original_aug',
#         'is_processed':False,
#         'abrv':'original_pre'},

#         'augmented_normalized_by_train':{'name':'original_aug_n',
#         'is_processed':True,
#         'abrv':'original'},

#         'augmented_normalized_by_DMSO':{'name':'original_aug_n_DMSO',
#         'is_processed':True,
#         'abrv':'original_DMSO'},

#         'augmented_normalized_by_mad':{'name':'original_aug_mad',
#         'is_processed':True,
#         'abrv':'original_mad'},
#     }


ABRVS = {
    'normalized_variable_selected_ae':'anomaly_nvs',
    'normalized_variable_selected_ae_error':'anomaly_err_nvs',
    'normalized_variable_selected_ae_diff':'anomaly_diff_nvs',
    'normalized_variable_selected_baseline':'raw_nvs',
    'normalized_variable_selected':'original_nvs',
    'normalized_variable_selected_normalized_by_train':'original_nvs_n',
    'normalized_variable_selected_normalized_by_DMSO':'original_nvs_DMSO',
    'normalized_variable_selected_remove_non_normal_features_ae':'anomaly_nvs_r',
    'normalized_variable_selected_remove_non_normal_features_ae_error':'anomaly_err_nvs_r',
    'normalized_variable_selected_remove_non_normal_features_baseline':'raw_nvs_r',
    'normalized_variable_selected_remove_non_normal_features':'original_nvs_r',
    'normalized_variable_selected_fs_ae':'anomaly_nvs_fs',
    'normalized_variable_selected_fs_ae_error':'anomaly_err_nvs_fs',
    'normalized_variable_selected_fs_baseline':'raw_nvs_fs',
    'normalized_variable_selected_fs':'original_nvs_fs',
    'augmented_ae':'anomaly_a',
    'augmented_ae_error':'anomaly_err_a',
    'augmented_baseline':'raw_a',
    'augmented_baseline_mad':'raw_m',
    'augmented_fs_ae':'anomaly_fs',
    'augmented_fs_ae_error':'anomaly_err_fs',
    'augmented_ae_diff':'anomaly_diff',
    'augmented_fs_baseline':'raw_fs',
    'augmented_fs':'original_fs',
    'augmented':'original_a',
    'augmented_normalized_by_train':'original_n',
    'augmented_normalized_by_DMSO':'original_DMSO',
    'augmented_normalized_by_train_mad':'original_m',
    'dose':'d',
    'cpd':'c',
    'role':'r',
    'plate':'p',
    'mock':'m',
    'L1000':'L',
    'CellPainting':'CP',
    'tune':'t',
    'mad_robustize':'mad',

}
# DS_INFO_DICT={'CDRP':['CDRP-BBBC047-Bray',['Metadata_Sample_Dose','pert_sample_dose'],['Metadata_ASSAY_WELL_ROLE','mock'],'Metadata_Plate'],
#               'CDRP-bio':['CDRPBIO-BBBC036-Bray',['Metadata_Sample_Dose','pert_sample_dose'],['Metadata_ASSAY_WELL_ROLE','mock'],'Metadata_Plate'],
#               'TAORF':['TA-ORF-BBBC037-Rohban',['Metadata_broad_sample','pert_id',]],
#               'LUAD':['LUAD-BBBC041-Caicedo',['x_mutation_status','allele']],
#               'LINCS':['LINCS-Pilot1',['Metadata_pert_id_dose','pert_id_dose'],['Metadata_pert_type','control'],'Metadata_plate_map_name']}



# modality ='CellPainting'


# # Doc VQA
# ANSWER_NAME: str = 'answers'
# QUESTION_NAME: str = 'question'
# QUESTION_ID_NAME = 'questionId'
# INPUT_ID_NAME = 'input_ids'
# DECODER_INPUT_ID_NAME = 'decoder_input_ids'
# DECODER_ATTENTION_MASK_NAME = 'decoder_attention_mask'
# LOGITS_NAME = 'logits'
# PIXEL_VALUES_NAME = 'pixel_values'
# ATTENTION_MASK_NAME = 'attention_mask'
# BBOX_NAME = 'bbox'
# START_POSITION_NAME = 'start_positions'
# END_POSITION_NAME = 'end_positions'
# LABELS_NAME = 'labels'
# WORDS_NAME = 'words'
# ANSWER_PAGE_IDX_NAME = 'answer_page_idx'
# PAGE_NUMBER_NAME = 'page_number'
# LABELS_ATTENTION_MASK_NAME = 'labels_attention_mask'
# SAMPLE_IDX = 'sample_idx'
# IMAGE_PATH = 'image_path'
# # logger
# global logger_exits
# logger_exits = False

# # Datasets names
# DOC_VQA_NAME = 'DocVQA'
# MP_DOC_VQA_NAME = 'MPDocVQA'
# DOC_VGQA_NAME = 'DocVGQA'

# # Model names
# LAYOUTLM_V3_NAME = 'LayoutLMv3'
# LAYOUTLM_V3_MP_NAME = 'LayoutLMv3_MP'
# T5_OCR_BASE_NAME = 'T5_OCR_baseModel'
# T5_BBOX_PIXEL_SIZE = 1000
