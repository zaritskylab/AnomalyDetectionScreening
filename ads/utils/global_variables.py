from eval.eval_utils import get_color_from_palette

BASE_DIR = '/sise/assafzar-group/assafzar/genesAndMorph'
DATA_DIR = f'{BASE_DIR}/preprocessed_data'
PROCESSED_DATA_DIR = f'{BASE_DIR}/anomaly_output'
RESULTS_DIR = f'{BASE_DIR}/results'


MODALITY_STR = {
    'CellPainting':'cp',
    'L1000':'L1000'
}

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
            'mock_val':'mock',
            'dose':'Metadata_mg_per_ml',
            'blocklist_features':[]
        },
        'L1000':{
            'cpd_col':'pert_id',
            'dose_col':'pert_sample_dose',
            'role_col':'pert_id',
            'plate_col':'det_plate',
            'mock_val':'mock',
            'blocklist_features':[]
            }
        
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
            'mock_val':'mock',
            'moa_col':'Metadata_moa',
            'blocklist_features':['Nuclei_Correlation_Costes_AGP_DNA']
        },
        'L1000':{
            'cpd_col':'pert_id',
            'dose_col':'pert_sample_dose',
            'role_col':'pert_id',
            'plate_col':'det_plate',
            'mock_val':'DMSO',
            'blocklist_features':[]}
    },

    'TAORF':{
        'has_dose':False,
        'has_moa':False,
        'name':'TA-ORF-BBBC037-Rohban',
        'CellPainting':{
            'cpd_col':'Metadata_broad_sample',
            'role_col':'Metadata_ASSAY_WELL_ROLE',
            'mock_val':'Untreated',
            # has both 'Untreated' and 'CTRL', why?
            'plate_col':'Metadata_Plate',
            'blocklist_features':[]
            },
        'L1000':{
            'cpd_col':'pert_id',
            'dose_col':'pert_sample_dose',
            'role_col':'pert_id',
            'plate_col':'det_plate',
            'mock_val':'DMSO',
            'blocklist_features':[]
            }
    },

    'LUAD':{
        'has_dose':False,
        'has_moa':False,
        'name':'LUAD-BBBC041-Caicedo',
        'CellPainting':{
            'cpd_col':'Metadata_broad_sample',
            'role_col':'Metadata_broad_sample_type',
            'mock_val':'control',
            'plate_col':'Metadata_Plate',
            'blocklist_features':[]
            },
        'L1000':{
            'cpd_col':'pert_id',
            'dose_col':'pert_sample_dose',
            'role_col':'pert_id',
            'plate_col':'det_plate',
            'mock_val':'DMSO',
            'blocklist_features':[]}
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
            'mock_val':'control',
            'n_per_dose':5,
            'moa_col':'Metadata_moa',
            'blocklist_features':[]},
        'L1000':{
            'cpd_col':'pert_id',
            'dose_col':'pert_id_dose',
            'role_col':'pert_id',
            'plate_col':'det_plate',
            'mock_val':'DMSO',
            'n_per_dose':3,
            'blocklist_features':[]}
    }
}

method_labels = {
    'normalized_variable_selected_ae':'Anomaly',
    'normalized_variable_selected_ae_diff':'Anomaly',
    }

methods_colors = {
    'Anomaly':get_color_from_palette("Set2", 0),
    'CellProfiler':get_color_from_palette("Set2", 1),
    'Fusion': get_color_from_palette("Set2", 2),
}

methods_colors_list_for_moa = [get_color_from_palette("Set2", 1),get_color_from_palette("Set2", 0),get_color_from_palette("Set2", 2)]


ABRVS = {
    'ae_diff':'Anomaly',
    'baseline':'CellProfiler',
    'PCA':'CellProfiler spherized (PCA)',
    'ZCA':'CellProfiler spherized (ZCA)',
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
    'mad_robustize':'mad'
}

METHOD_NAME_MAPPING = {
    'ae_diff':'Anomaly',
    'baseline':'CellProfiler',
    'baseline_dmso':'CellProfiler (all DMSO)',
    'baseline_unchanged': 'CellProfiler (unchanged)',
    'ZCA':'CellProfiler spherized (ZCA)',
    'PCA':'CellProfiler spherized (PCA)',
    'ZCA-cor':'CellProfiler spherized (ZCA-cor)',
    'PCA-cor':'CellProfiler spherized (PCA-cor)',
}



#     'TAORF':{
#         'latent_dim': 8,
#         'encoder_type': 'default',
#         # 'encoder_type': 'deep',
#         'deep_decoder': False,
#         'batch_size': 16,
#         # 'l1_latent_lambda': 0.001,
#         'l1_latent_lambda': 0,
#     },
#     'CDRP-bio':{
#         'latent_dim': 32,
#         'encoder_type': 'deep',
#         'deep_decoder': False,
#         'batch_size': 32,
#         # 'l1_latent_lambda': 0.0001,
#         'l1_latent_lambda': 0,
#         'lr': 0.0005,
#         # 'lr': 0.0007,
#         # 'l2_lambda': 1e-06,
#         'l2_lambda': 0.1,
#         # 'dropout': 0.14,
#     },
#     'LUAD':{
#         'latent_dim': 8,
#         # 'encoder_type': 'deep',
#         'deep_decoder': False,
#         'batch_size': 16,
#         # 'l1_latent_lambda': 1,
#         'l1_latent_lambda': 0,

#     },
#     'LINCS':{
#         'latent_dim': 32,
#         'encoder_type': 'deep',
#         'deep_decoder': False,
#         'batch_size': 32,
#         # 'l1_latent_lambda': 0.0001,
#         'l1_latent_lambda': 0,
#     }
# }
HYPER_PARAMS = {
    'CDRP':{
        'latent_dim': 32,
        'encoder_type': 'deep',
        'deep_decoder': False,
        'batch_size': 32,
        'lr': 0.0005,
        'l1_latent_lambda': 0.0001,
        'l2_lambda': 0.1,
    },
    'CDRP-bio':{
        'latent_dim': 32,
        # 'encoder_type': 'default',
        'encoder_type': 'deep',
        'deep_decoder': False,
        'batch_size': 32,
        'lr': 0.0007,
        # 'l1_latent_lambda': 0.0001,
        'l1_latent_lambda': 0,
        'l2_lambda': 0.005,
    },
    'LUAD':{
        # 'latent_dim': 8,
        'latent_dim': 32,
        # 'encoder_type': 'default',
        'encoder_type': 'deep',
        'deep_decoder': False,
        'batch_size': 32,
        'lr': 0.0007,
        # 'l1_latent_lambda': 1,
        'l1_latent_lambda': 0,
        'l2_lambda': 0.005
    },
    'LINCS':{
        'latent_dim': 32,
        # 'encoder_type': 'default',
        'encoder_type': 'deep',
        'deep_decoder': False,
        'lr': 0.0007,
        'batch_size': 32,
        # 'l1_latent_lambda': 0.0001,
        'l1_latent_lambda': 0,
        'l2_lambda': 0.005,
    },
    'TAORF':{
        'latent_dim': 32,
        # 'encoder_type': 'default',
        'encoder_type': 'deep',
        'deep_decoder': False,
        'batch_size': 32,
        # 'l1_latent_lambda': 0.001,
        'l1_latent_lambda': 0,
        'lr': 0.0007,
        'l2_lambda': 0.005,
    }
}

NEW_MOAS_DICT = {
    'CDRP-bio':['p38 mapk inhibitor',
            #  'egfr inhibitor',
            #  'histamine receptor antagonist',
            'retinoid receptor agonist',
            #  'glutamate receptor antagonist',
            #  'dopamine receptor agonist']
            ],
    'LINCS':['monoamine oxidase inhibitor',
        'progesterone receptor agonist',
        'PKC inhibitor',
        'androgen receptor antagonist',
        'angiotensin converting enzyme inhibitor',
        'potassium channel blocker',
        'ribonucleotide reductase inhibitor',
        'nucleoside reverse transcriptase inhibitor',
        'carbonic anhydrase inhibitor'
        ]
}

TOP_MOAS_DICT = {
    'CDRP-bio':['atpase inhibitor',
        # 'tubulin polymerization inhibitor',
        'glucocorticoid receptor agonist',
        'cdk inhibitor',
        # 'retinoid receptor agonist',
        # 'p38 mapk inhibitor',
        # 'protein synthesis inhibitor',
        # 'dopamine receptor antagonist',
        # 'bacterial cell wall synthesis inhibitor',
        # 'adrenergic receptor agonist'
        ],
    'LINCS':['mek inhibitor',
        'hsp inhibitor',
        'proteasome inhibitor',
        'egfr inhibitor',
        'cdk inhibitor',
        'hdac inhibitor',
        'tubulin polymerization inhibitor',
        'glucocorticoid receptor agonist',
        'retinoid receptor agonist'     
        ]
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
