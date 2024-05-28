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

# from interpret_layer.shap_anomalies import run_anomaly_shap
from model_layer.AEModel import AutoencoderModel

from data_layer.data_utils import load_data, pre_process, to_dataloaders,normalize,get_features
from utils.readProfiles import save_profiles
import os
from typing import Dict, Union
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import optuna


def main(configs):
    data , __ = load_data(configs.general.base_dir,configs.general.dataset,configs.data.profile_type, modality=configs.data.modality)
    data_preprocess,features =  pre_process(data,configs,configs.data.data_reps)
    dataloaders = to_dataloaders(data_preprocess,configs.model.batch_size,features)
    model = train_autoencoder(dataloaders, features, configs)

    test_dataloaders = list(dataloaders.keys())[2:]
    preds = test_autoencoder(model, dataloaders)
    # preds = test_autoencoder(model, dataloaders, features, configs)
    
    diffs_ctrl = preds['test_ctrl'] - data_preprocess[data_preprocess['Metadata_set'] == 'test_ctrl'][features].values
    diffs_treat = preds['test_treat'] -  data_preprocess[data_preprocess['Metadata_set'] == 'test_treat'][features].values 

    diffs_ctrl_abs = np.abs(preds['test_ctrl'] - data_preprocess[data_preprocess['Metadata_set'] == 'test_ctrl'][features].values)
    diffs_treat_abs = np.abs(preds['test_treat'] -  data_preprocess[data_preprocess['Metadata_set'] == 'test_treat'][features].values)

    diffs_ctrl_power = np.power(preds['test_ctrl'] - data_preprocess[data_preprocess['Metadata_set'] == 'test_ctrl'][features].values,2)
    diffs_treat_power = np.power(preds['test_treat'] -  data_preprocess[data_preprocess['Metadata_set'] == 'test_treat'][features].values,2)

    
    preds_normalized=post_process_anomaly_and_save(data_preprocess, preds['test_ctrl'],preds['test_treat'], configs.general.output_exp_dir,  f'replicate_level_{configs.data.modality_str}_{configs.data.profile_type}_preds', configs, features)
    # z_preds_normalized = save_treatments(data, z_preds['test_ctrl'],z_preds['test_treat'], configs.general.output_exp_dir,  f'replicate_level_{configs.data.modality_str}_{configs.data.profile_type}_ae_embeddings', configs, features, embeddings=True)
    diffs_normalized= post_process_anomaly_and_save(data_preprocess, diffs_ctrl,diffs_treat, 
        configs.general.output_exp_dir,  f'replicate_level_{configs.data.modality_str}_{configs.data.profile_type}_ae_diff', configs, features)
    # diffs_normalized_abs= save_treatments(data_preprocess, diffs_ctrl_abs,diffs_treat_abs,
        # configs.general.output_exp_dir,  f'replicate_level_{configs.data.modality_str}_{configs.data.profile_type}_ae_diff_abs', configs, features)

    # diffs_normalized_power= save_treatments(data_preprocess, diffs_ctrl_power,diffs_treat_power,
        # configs.general.output_exp_dir,  f'replicate_level_{configs.data.modality_str}_{configs.data.profile_type}_ae_diff_power', configs, features)    
            
        

def train_autoencoder(dataloaders, features, configs,losses = {}):

    # features = get_features(data, configs.data.modality)

    callbacks = set_trainer_callbacks(configs)


    if configs.model.tune_hyperparams:
        configs.general.logger.info('Tuning hyperparams...')
        configs.general.logger.info(f'latent dim size is {configs.model.latent_dim}')
        hparams = tune_hyperparams(dataloaders, features, configs)
    else:
        l1_latent_lambda = configs.model.l1_latent_lambda if configs.model.l1_latent_lambda else 0
        l2_lambda = configs.model.l2_lambda if configs.model.l2_lambda else 0.007
        hparams = {'input_size': len(features),
                'latent_size': configs.model.latent_dim,
                'l2_lambda': configs.model.l2_lambda,
                'l1_latent_lambda': configs.model.l1_latent_lambda,
                'lr': configs.model.lr,
                'dropout': configs.model.dropout,
                'batch_size': configs.model.batch_size,
                'deep_decoder': configs.model.deep_decoder,
                'model_type': configs.model.model_type,
                'encoder_type': configs.model.encoder_type
            }
    
    configs.general.logger.info('Model Parameters:')
    for key, value in hparams.items():
        configs.general.logger.info(f'    {key}: {value}')

    model = AutoencoderModel(hparams)

    # Move the model to the desired device (e.g., GPU if available)
    
    trainer = pl.Trainer(
        # logger=logger,
        default_root_dir=configs.model.ckpt_dir,
        callbacks=callbacks,
        max_epochs=configs.model.max_epochs,
        accelerator="auto",
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    trainer.fit(model, dataloaders['train'],dataloaders['val'])

    # Test model
    test_dataloaders = ['train','val','test_ctrl', 'test_treat']

    # print('saved diff files')
    # metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")
    # del metrics["step"]
    # metrics.set_index("epoch", inplace=True)
    # print(metrics.dropna(axis=1, how="all").head())
    # sns.relplot(data=metrics, kind="line")
    # plt.ylim(0, 1)
    # plt.show()
    # plt.savefig(os.path.join(configs.general.output_exp_dir, f'training_loss.png'))
    # plt.close()

    return model

def test_autoencoder(model, dataloaders, test_dataloaders= ['test_ctrl', 'test_treat']):
    # disable grads + batchnorm + dropout
    
    # torch.set_grad_enabled(False)
    model.eval()
    preds = {}
    x_recon_preds = {}
    z_preds = {}
    indices = {}
    # test_dataloaders = ['test_ctrl', 'test_treat']
    for subset in test_dataloaders:
        dataloader = dataloaders[subset]
        x_recon_preds[subset] =[]
        z_preds[subset] = []
        indices[subset] = []
        for batch_idx, batch in enumerate(dataloader):
            # records, cpds, plates,batch_indices = batch
            # x_recon_pred, z_pred = model.predict_step(batch, batch_idx)
            batch_device = batch.to(model.device)
            # x_recon_pred, z_pred = model(batch)
            x_recon_pred = model.predict(batch_device)
            # indices[subset].append(batch_indices)
            # x_recon_pred = model.predict_step(batch)
            x_recon_preds[subset].append(x_recon_pred.cpu().detach().numpy())
            # z_preds[subset].append(z_pred.cpu().numpy())

    for subset in test_dataloaders:
        preds[subset] = np.concatenate(x_recon_preds[subset])
        # indices[subset] = np.concatenate(indices[subset])
        # z_preds[subset] = np.concatenate(z_preds[subset])

    # x_explain_tensor = torch.tensor(x_explain[self.features].values, dtype=torch.float32).to(self.autoencoder.device)

    return preds

def test_autoencoder2(model, X, features, configs):
    
    x_meta = X.loc[:,~X.columns.isin(features)]
    x_tensor = torch.tensor(X[features].values, dtype=torch.float32).to(model.device)
    preds = model.predict(x_tensor).cpu().detach().numpy()

    X_preds = X.copy()
    X_preds.loc[:,features] = preds
    df_preds = pd.DataFrame(preds, columns=features)
    preds_with_meta = pd.concat([x_meta.reset_index(drop=True),df_preds],axis=1)
    # preds_with_meta = pd.concat([x_meta.reset_index(drop=True),pd.DataFrame(preds)],axis=1)

    assert np.sum(preds_with_meta[features].values - X_preds[features].values).sum() == 0

    return X_preds


# Define a PyTorch Lightning model evaluation function for Optuna
def objective(trial, dataloaders,features,hidden_size=None,deep_decoder=False, encoder_type = 'default',model_type='AE',max_epochs=100,tune_l2=None,tune_l1=None, l2_lambda=0, l1_latent_lambda=0):
    # Define hyperparameters to tune    

    if hidden_size is None:
        hidden_size = trial.suggest_categorical('hidden_size', [16, 32, 64, 128, 256])
    if tune_l2:
        # l2_lambda = trial.suggest_float('l2_lambda', 1e-7, 0.01)
        l2_lambda = trial.suggest_categorical('l2_lambda', [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1])
    if tune_l1:
        # l1_latent_lambda = trial.suggest_float('l1_latent_lambda', 1e-7, 0.01)
        l1_latent_lambda = trial.suggest_categorical('l1_latent_lambda', [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1])
    dropout = trial.suggest_float('dropout', 0, 0.15)
    # output_size = 10
    # batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    # hidden_size = trial.suggest_int('hidden_size', 32, 256, log=True)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3)
    # l2_lambda = trial.suggest_float('l2_lambda', 1e-7, 5e-2)
    # l1_latent_reg = trial.suggest_float('l1_latent_reg', 1e-7, 5e-2)
        # Initialize model

    hparams = {
        'input_size': len(features),
        'latent_size': hidden_size,
        'l2_lambda': l2_lambda,
        'l1_latent_lambda': l1_latent_lambda,
        'lr': learning_rate,
        'dropout': dropout,
        # 'batch_size': batch_size,
        'deep_decoder': deep_decoder,
        'model_type': model_type,
        'encoder_type': encoder_type
    }
    
    # Initialize the PyTorch Lightning model
    model = AutoencoderModel(hparams)


    # Create a PyTorch Lightning trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[EarlyStopping(monitor='val_loss')],
        enable_checkpointing=False,
        enable_progress_bar = False,
        enable_model_summary = False,
        logger=False  # Disable the logger to reduce overhead
    )

    # Perform training and validation
    trainer.fit(model, dataloaders['train'],dataloaders['val'])

    # Retrieve the best validation loss from the trainer
    best_val_loss = trainer.callback_metrics['val_loss'].item()

    # Return the best validation loss as the objective to minimize
    return best_val_loss

def set_trainer_callbacks(configs):

    checkpoint_callback = ModelCheckpoint(
        dirpath=configs.model.ckpt_dir,
        filename='autoencoder-{epoch:02d}-{val_loss:.2f}',
        save_top_k=configs.model.save_top_k,
        monitor='val_loss',
        mode='min'
    )

    # Set up logger and checkpoint callbacks
    # logger = TensorBoardLogger(save_dir=configs.model.tb_logs_dir, name=configs.general.exp_name)
    # logger = CSVLogger(save_dir=configs.model.tb_logs_dir),

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=70,
        mode="min"
    )


    progressbar_callback = TQDMProgressBar(refresh_rate=150)
    return [checkpoint_callback, early_stop_callback, progressbar_callback]
    # Train model

def tune_hyperparams(dataloaders, features, configs):

    # Optuna study
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, dataloaders=dataloaders, features =features,
                                            hidden_size = configs.model.latent_dim,
                                            deep_decoder=configs.model.deep_decoder,
                                            encoder_type=configs.model.encoder_type, 
                                            model_type=configs.model.model_type,
                                            max_epochs=configs.model.max_epochs_in_trial,
                                            tune_l2=configs.model.tune_l2,
                                            tune_l1=configs.model.tune_l1,
                                            l2_lambda=configs.model.l2_lambda,
                                            l1_latent_lambda=configs.model.l1_latent_lambda), 
                                            n_trials=configs.model.n_tuning_trials
                                            ) 

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
        'l2_lambda': configs.model.l2_lambda,
        'l1_latent_lambda': configs.model.l1_latent_lambda,
        'lr': trial.params['learning_rate'],
        'dropout': trial.params['dropout'],
        'batch_size': configs.model.batch_size,
        'deep_decoder': configs.model.deep_decoder,
        'encoder_type': configs.model.encoder_type
    }

    if configs.model.tune_l1:
        hparams['l1_latent_lambda'] = trial.params['l1_latent_lambda']
    if configs.model.tune_l2:
        hparams['l2_lambda'] = trial.params['l2_lambda']


    return hparams


def post_process_anomaly_and_save(data, preds_ctrl,preds_treat, output_dir, filename, configs, features,embeddings=False,normalize_reps = True):

    test_ctrl = data[data['Metadata_set'] == 'test_ctrl']
    test_treat = data[data['Metadata_set'] == 'test_treat']

    if normalize_reps:
            
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
            # test_out_normalized = test_out.copy()
            # no plate normalization after training
            test_out_normalized = normalize(test_out,features, configs.data.modality, normalize_condition = 'test_ctrl',plate_normalized=0, norm_method = "standardize", clip_outliers=False)
    else:
        test_ctrl.loc[:,features] = preds_ctrl
        test_treat.loc[:,features] = preds_treat
        test_out = pd.concat([test_ctrl,test_treat],axis=0)
        test_out_normalized = test_out.copy()
        
    if not configs.general.debug_mode:
        # pred_filename = f'replicate_level_cp_{configs.data.profile_type}_ae'
        save_profiles(test_out_normalized, output_dir, filename)
    return test_out_normalized


def load_checkpoint(checkpoint_dir):
    checkpoint_name = load_most_advanced_checkpoint(checkpoint_dir)
    if not checkpoint_name:
        print("No checkpoint files found in the directory.")
        return None
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    model = AutoencoderModel.load_from_checkpoint(checkpoint_path)
    
    return model

def load_most_advanced_checkpoint(checkpoint_dir):

    if not os.path.exists(checkpoint_dir):
        print("The checkpoint directory does not exist.")
        return None
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
    
    if not checkpoint_files:
        print("No checkpoint files found in the directory.")
        return None
    
    # Extract epoch numbers from filenames
    epochs = [int(f.split('=')[-1].split('.')[0]) for f in checkpoint_files]
    
    # Find the index of the checkpoint file with the highest epoch number
    most_advanced_idx = epochs.index(max(epochs))
    
    # Load the most advanced checkpoint
    most_advanced_checkpoint = os.path.join(checkpoint_dir, checkpoint_files[most_advanced_idx])
    
    return most_advanced_checkpoint
