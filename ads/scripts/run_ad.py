import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from sklearn import preprocessing
from torch import nn
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping, TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger

import sys
import os
# sys.path.insert(0, os.path.join(os.getcwd(),'AnomalyDetectionScreening'))
# sys.path.insert(0, os.path.join(os.getcwd(),'AnomalyDetectionScreening','ads'))

from utils.eval_utils import plot_latent_effect
from models.AEModel import AutoencoderModel

from utils.data_utils import load_data, pre_process, to_dataloaders,normalize
from utils.readProfiles import save_profiles
import os
from typing import Dict, Union
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import seaborn as sns
import optuna


# from pytorch_lightning.utilities.cli import LightningCLI

# Define a PyTorch Lightning model evaluation function for Optuna
def objective(trial, data,features,hidden_size=None,deep_decoder=False,model_type='AE'):
    # Define hyperparameters to tune    

    if hidden_size is None:
        hidden_size = trial.suggest_categorical('hidden_size', [16, 32, 64, 128, 256])

    # dropout = trial.suggest_float('dropout', 0, 0.4)
    # output_size = 10
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    # hidden_size = trial.suggest_int('hidden_size', 32, 256, log=True)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2)
    l2_lambda = trial.suggest_float('l2_lambda', 1e-7, 1e-2)
        # Initialize model

    dataloaders = to_dataloaders(data,batch_size,features)

    hparams = {
        'input_size': len(features),
        'latent_size': hidden_size,
        'l2_lambda': l2_lambda,
        'lr': learning_rate,
        # 'dropout': dropout,
        'batch_size': batch_size,
        'deep_decoder': deep_decoder,
        'model_type': model_type
    }
    # Initialize the PyTorch Lightning model
    model = AutoencoderModel(hparams)

    # Create a PyTorch Lightning trainer
    trainer = pl.Trainer(
        max_epochs=100,
        callbacks=[EarlyStopping(monitor='val_loss')],
        logger=False  # Disable the logger to reduce overhead
    )


    # Perform training and validation
    trainer.fit(model, dataloaders['train'],dataloaders['val'])

    # Retrieve the best validation loss from the trainer
    best_val_loss = trainer.callback_metrics['val_loss'].item()

    # Return the best validation loss as the objective to minimize
    return best_val_loss



# def train_autoencoder(configs: Dict[str, Union[str, float, int]],losses = {}) -> pl.LightningModule:
def train_autoencoder(configs,losses = {}):

    # Read data
    #  data,features = load_data(configs.general.data_dir, configs.data.dataset, configs.data.profile_type)
    data , __ = load_data(configs.general.base_dir,configs.general.dataset,configs.data.profile_type, modality=configs.data.modality)
    data,features =  pre_process(data,configs,overwrite = False)
    # features = data.columns.tolist()
# 
# (data, configs, features = None, overwrite = False, do_fs = False, modality ='CellPainting'):
    # datasets, dataloaders,cp_features = prepare_data(cp, configs, cp_features, do_fs=configs.data.feature_select)

    # Initialize model
    hparams = {'input_size': len(features),
        'latent_size': configs.model.latent_dim,
        'l2_lambda': configs.model.l2_lambda,
        'lr': configs.model.lr,
        # 'dropout': configs.model.dropout,
        'batch_size': configs.model.batch_size,
        'deep_decoder': configs.model.deep_decoder,
        'model_type': configs.model.model_type
    }

    # Set up logger and checkpoint callbacks
    # logger = TensorBoardLogger(save_dir=configs.model.tb_logs_dir, name=configs.general.exp_name)
    logger = CSVLogger(save_dir=configs.model.tb_logs_dir),
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=70,
        mode="min"
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=configs.model.ckpt_dir,
        filename='autoencoder-{epoch:02d}-{val_loss:.2f}',
        save_top_k=configs.model.save_top_k,
        monitor='val_loss',
        mode='min'
    )
    progressbar_callback = TQDMProgressBar(refresh_rate=150)
    # Train model

    #TODO: move to lightning
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback,early_stop_callback,progressbar_callback],
        max_epochs=configs.model.max_epochs,
        accelerator="auto",
        # progress_bar_refresh_rate=50,

        # precision=16 if configs['use_16bit'] else 32

        # deterministic=True,
        # fast_dev_run=configs['fast_dev_run']
    )

    #TODO: add support to 'model' variable
    # if configs.model.name == 'AE':

    if configs.model.tune_hyperparams:
        configs.general.logger.info('Tuning hyperparams...')
        configs.general.logger.info(f'latent dim size is {configs.model.latent_dim}')

        # Optuna study
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: objective(trial, data, features,configs.model.latent_dim,configs.model.deep_decoder, configs.model.model_type), n_trials=configs.model.n_tuning_trials)


        # study.optimize(objective,dataloaders)

        configs.general.logger.info('Best trial:')
        trial = study.best_trial
        configs.general.logger.info('  Value: {}'.format(trial.value))
        configs.general.logger.info('  Params: ')
        for key, value in trial.params.items():
            configs.general.logger.info('    {}: {}'.format(key, value))

        # Initialize model
        hparams = {'input_size': len(features),
            'model_type': configs.model.model_type,
            # 'latent_size': trial.params['hidden_size'],
            'latent_size': configs.model.latent_dim,
            'l2_lambda': trial.params['l2_lambda'],
            'lr': trial.params['learning_rate'],
            # 'dropout': trial.params['dropout'],
            'batch_size': trial.params['batch_size'],
            'deep_decoder': configs.model.deep_decoder
        }

    dataloaders = to_dataloaders(data,hparams['batch_size'],features)
    model = AutoencoderModel(hparams)
    trainer.fit(model, dataloaders['train'],dataloaders['val'])

    # # NOT SUPPORTED YET
    # elif configs['model'] == 'cae':
    #     print('Concrete AEs not supported yet...')
    #     #TODO: when this works, run over K, do Yuval check of K hyperparameter

    #     # Initialize the feature selector model
    #     K = 10  # Number of features to select
    #     output_function = nn.Linear(K, len(cp_features))  # Update with your desired output function
    #     model = ConcreteAutoencoderFeatureSelector(K, output_function)

    #     #TODO: test if CAE fit works
    #     selector = ConcreteAutoencoderFeatureSelector(hparams)
    #     trainer.fit(model, dataloaders['train'],dataloaders['val'])

    #     #TODO: test if extraction of important features work
    #     selector.compute_probs(trainer.concrete_select)

    #     # Extract the indices of the most important features
    #     indices = selector.get_indices()

    #     # Extract the mask indicating the most important features
    #     mask = selector.get_mask()

    #     # Extract the most important features from the original data
    #     selected_features = mask[:, indices]
    #     #TODO: add support for test

    # Test model
    test_dataloaders = ['train','val','test_ctrl', 'test_treat']


    for data_subset in test_dataloaders:
        dataloader = dataloaders[data_subset]
        # dict_res = trainer.test(model, dataloader)
        # for res in dict_res:
        #     losses[data_subset][res].append(dict_res[res])
        losses[data_subset] = trainer.test(model, dataloader)

    # disable grads + batchnorm + dropout
    torch.set_grad_enabled(False)
    model.eval()
    all_preds = {}
    x_recon_preds = {}
    z_preds = {}
    predict_dataloaders = ['test_ctrl', 'test_treat']
    for subset in list(dataloaders.keys())[2:]:
        dataloader = dataloaders[subset]
        x_recon_preds[subset] =[]
        z_preds[subset] = []
        for batch_idx, batch in enumerate(dataloader):
            x_recon_pred, z_pred = model.predict_step(batch, batch_idx)
            x_recon_preds[subset].append(x_recon_pred.numpy())
            z_preds[subset].append(z_pred.numpy())


    x_recon_ctrl = np.concatenate(x_recon_preds['test_ctrl'])
    z_pred_ctrl = np.concatenate(z_preds['test_ctrl'])

    x_recon_treat = np.concatenate(x_recon_preds['test_treat'])
    z_pred_treat = np.concatenate(z_preds['test_treat'])

    # process and save anomaly detection output    
    # output_dir = os.path.join(configs.general.base_dir, 'anomaly_output', configs.general.dataset,'CellPainting',configs.general.exp_name)
    # os.makedirs(output_dir, exist_ok=True)
    old = False
    if old:
        test_ctrl = data[data['Metadata_set'] == 'test_ctrl'].copy()
        test_treat = data[data['Metadata_set'] == 'test_treat'].copy()
        # test_out = pd.concat([test_ctrl_out,test_treat_out],axis=0)
        # test_treat_out = datasets['test_treat'].copy()

        test_ctrl_out = test_ctrl.copy()
        test_treat_out = test_treat.copy()

        test_ctrl_out.loc[:,features] = x_recon_ctrl
        test_treat_out.loc[:,features] = x_recon_treat
        test_out = pd.concat([test_ctrl_out,test_treat_out],axis=0)
        # test_ctrl_out_normalized = datasets['test_ctrl'].copy()
        # test_treat_out_normalized = datasets['test_treat'].copy()
        # test_ctrl_out_normalized = test_ctrl_out.copy()
        # test_treat_out_normalized = test_treat_out.copy()

        # if 'by_plate' in configs.general.exp_name:
            # test_out_normalized, cols_to_remove = normalize(test_out, features, configs.data.modality,normalize_condition = 'test_ctrl',plate_normalized=1, norm_method = "standardize", remove_non_normal_features = False, clip_outliers=False)
        # else:
        test_out_normalized, cols_to_remove = normalize(test_out, features, configs.data.modality,normalize_condition = 'test_ctrl',plate_normalized=0, norm_method = "standardize", remove_non_normal_features = False, clip_outliers=False)
        test_treat_out_normalized = test_out_normalized.loc[raw_data['Metadata_set'] == 'test_treat', :]
        # save_profiles(test_treat_out_normalized, output_dir, raw_filename)
        # scaler_cp = preprocessing.StandardScaler()
        
        # test_ctrl_out_normalized.loc[:,features] = scaler_cp.fit_transform(test_ctrl_out[features].values.astype('float64'))
        # test_treat_out_normalized.loc[:,features] = scaler_cp.transform(test_treat_out[features].values.astype('float64'))
        # print(test_treat_out_normalized.shape)

        # save_profiles(test_ctrl_out_normalized, os.path.join(output_dir, f'ad_out_ctrl_zscores.csv.gz'))
        if not configs.general.debug_mode:
            pred_filename = f'replicate_level_cp_{configs.data.profile_type}_ae'
            save_profiles(test_treat_out_normalized, output_dir, pred_filename)

        print('saved prediction files')

        meta_features = [col for col in test_ctrl_out.columns if 'Metadata_' in col]
        test_meta_df = test_treat_out[meta_features].reset_index(drop=True)

        # test_ctrl_out_normalized = test_ctrl_out.copy()
        # test_treat_out_normalized = test_treat_out.copy()
        scaler_cp = preprocessing.StandardScaler()
        scaler_cp.fit(z_pred_ctrl.astype('float64'))

        # test_z_out_normalized = test_treat_out.copy()[meta_features]
        
        # test_treat_out_normalized.loc[:,features] = scaler_cp.transform(test_treat_out[features].values.astype('float64'))
        pred_filename = f'replicate_level_cp_{configs.data.profile_type}_ae_embeddings'
        test_z_out_normalized = pd.DataFrame(scaler_cp.transform(z_pred_treat.astype('float64'))).reset_index(drop=True)
        test_z_out_normalized = pd.concat([test_meta_df,test_z_out_normalized],axis=1)
        if not configs.general.debug_mode:
            save_profiles(test_z_out_normalized, output_dir, pred_filename)

        # test_treat_out_normalized.to_csv(os.path.join(configs.general.data_dir, 'anomaly_output', configs.data.dataset,configs.data.profile_type, f'ad_out_ctrl_zscores.csv'),compression='gzip')
        # test_treat_out_normalized.to_csv(
            # os.path.join(configs.general.data_dir, 'anomaly_output', configs.data.dataset,configs.data.profile_type,configs.general.exp_name, f'ad_out_treated_zscores.csv'),compression='gzip')


        # process and save anomaly detection output differences
        test_ctrl_out_diff = test_ctrl_out.copy()
        test_treat_diff =test_treat_out.copy()

        test_ctrl_out_diff.loc[:,features] = x_recon_ctrl - test_ctrl_out.loc[:,features]
        test_treat_diff.loc[:,features] = x_recon_treat - test_treat_out.loc[:,features]

        test_out_diff = pd.concat([test_ctrl_out_diff,test_treat_diff],axis=0)

        # test_ctrl_out_normalized_diff = test_ctrl_out.copy()
        # test_treat_out_normalized_diff = test_treat_out.copy()

        # test_ctrl_out_normalized_diff.loc[:,features] = scaler_cp.fit_transform(test_ctrl_out_diff[features].values.astype('float64'))
        # test_treat_out_normalized_diff.loc[:,features] = scaler_cp.transform(test_treat_diff[features].values.astype('float64'))
        
        
        # if 'by_plate' in configs.general.exp_name:
            # test_diff_out_normalized, cols_to_remove = normalize(test_out_diff, features, configs.data.modality,normalize_condition = 'test_ctrl',plate_normalized=1, norm_method = "standardize", remove_non_normal_features = False, clip_outliers=False)
        # else:
        test_diff_out_normalized, cols_to_remove = normalize(test_out_diff, features, configs.data.modality,normalize_condition = 'test_ctrl',plate_normalized=0, norm_method = "standardize", remove_non_normal_features = False, clip_outliers=False)
        test_treat_out_normalized = test_diff_out_normalized.loc[test_diff_out_normalized['Metadata_set'] == 'test_treat', :]
        # test_ctrl_out_normalized_diff.to_csv(os.path.join(configs.general.data_dir, 'anomaly_output', configs.data.dataset,configs.data.profile_type, f'ad_out_diff_ctrl_zscores.csv'),compression='gzip')
        # test_treat_out_normalized_diff.to_csv(
            # os.path.join(configs.general.data_dir, 'anomaly_output', configs.data.dataset,configs.data.profile_type,configs.general.exp_name, f'ad_out_diff_treated_zscores.csv'),compression='gzip')

        # save_profiles(test_ctrl_out_normalized_diff, os.path.join(output_dir, f'ad_out_diff_ctrl_zscores.csv.gz'))
        diff_filename = f'replicate_level_cp_{configs.data.profile_type}_ae_diff'
        if not configs.general.debug_mode:
            save_profiles(test_treat_out_normalized_diff, output_dir, diff_filename)
    else:

        save_treatments(data, x_recon_ctrl,x_recon_treat, configs.general.output_exp_dir,  f'replicate_level_{configs.data.modality_str}_{configs.data.profile_type}_ae', configs, features)
        # save_treatments(data, z_pred_ctrl,z_pred_treat, configs.general.output_exp_dir,  f'replicate_level_{configs.data.modality_str}_{configs.data.profile_type}_ae_embeddings', configs, features, embeddings=True)

        save_treatments(data, x_recon_ctrl - data[data['Metadata_set'] == 'test_ctrl'][features],x_recon_treat - data[data['Metadata_set'] == 'test_treat'][features], configs.general.output_exp_dir,  f'replicate_level_{configs.data.modality_str}_{configs.data.profile_type}_ae_diff', configs, features)


    print('saved diff files')
    metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")
    del metrics["step"]
    metrics.set_index("epoch", inplace=True)
    print(metrics.dropna(axis=1, how="all").head())
    sns.relplot(data=metrics, kind="line")
    plt.ylim(0, 1)
    plt.show()
    plt.savefig(os.path.join(configs.general.output_exp_dir, f'training_loss.png'))
    plt.close()

    return model, losses

def save_treatments(data, preds_ctrl,preds_treat, output_dir, filename, configs, features,embeddings=False):

    # test_ctrl_out_normalized = datasets['test_ctrl'].copy()
    # test_treat_out_normalized = datasets['test_treat'].copy()
    # test_ctrl_out_normalized = test_ctrl_out.copy()
    # test_treat_out_normalized = test_treat_out.copy()
    test_ctrl = data[data['Metadata_set'] == 'test_ctrl']
    test_treat = data[data['Metadata_set'] == 'test_treat']

    if embeddings:
        meta_features = [col for col in test_treat.columns if 'Metadata_' in col]
        test_treat_meta_df = test_treat[meta_features].reset_index(drop=True)
        
        scaler = preprocessing.StandardScaler()
        scaler.fit(preds_ctrl.astype('float64'))
        test_treat_z_out_normalized = pd.DataFrame(scaler.transform(preds_treat.astype('float64'))).reset_index(drop=True)
        test_out_normalized = pd.concat([test_treat_meta_df,test_treat_z_out_normalized],axis=1)
    else:

        test_ctrl.loc[:,features] = preds_ctrl
        test_treat.loc[:,features] = preds_treat
        test_out = pd.concat([test_ctrl,test_treat],axis=0)
        test_out_normalized, _ = normalize(test_out,features, configs.data.modality, normalize_condition = 'test_ctrl',plate_normalized=0, norm_method = "standardize", remove_non_normal_features = False, clip_outliers=False)
        # test_treat_out_normalized = test_out_normalized.loc[test_out_normalized['Metadata_set'] == 'test_treat', :]

    if not configs.general.debug_mode:
        # pred_filename = f'replicate_level_cp_{configs.data.profile_type}_ae'
        save_profiles(test_out_normalized, output_dir, filename)

if __name__ == '__main__':

    seed_everything(42)
    import yaml
    base_dir = os.getcwd()
    # Read YAML file
    with open(f"{base_dir}/AnomalyDetectionScreening/ads/configs.yaml", 'r') as stream:
        configs = yaml.safe_load(stream)
    print(f'start training {configs["exp_name"]}...')
    
    # configs = set_configs()
    
    # os.makedirs(os.path.join(configs.general.data_dir, 'anomaly_output'), exist_ok=True)
    # os.makedirs(os.path.join(configs.general.data_dir, 'anomaly_output',configs.general.dataset), exist_ok=True)
    # os.makedirs(os.path.join(configs.general.data_dir, 'anomaly_output', configs.general.dataset, 'CellPainting'), exist_ok=True)

    tune_ldims = False
    if tune_ldims:
        l_dims = [32]
        all_res = []
        pcc_res =[]
        for l_dim in l_dims:
            configs.model.latent_size = l_dim
            model, res = train_autoencoder(configs)
            all_res.append(res)
            pcc_res.append(res['val'][0]['pcc'])

        #TODO: test plot latent effect
        plot_latent_effect(pcc_res, l_dims)

        best_ldim_ind = np.argmax(all_res)

        configs.model.latent_size = l_dims[best_ldim_ind]

    model, res = train_autoencoder(configs)
    res



    # model