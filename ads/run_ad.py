import pandas as pd
import numpy as np
import torch
from sklearn import preprocessing
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from sklearn.model_selection import GroupShuffleSplit
# from pytorch_lightning.tuner import random_search
from Model import TabularDataset, Autoencoder, AutoencoderModel
import sys

sys.path.insert(0, '../../2022_Haghighi_NatureMethods/utils/')
print(sys.path)
from readProfiles import *
from pred_models import *
from data_utils import load_data, split_train_test
import os
from typing import Dict, Union
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.cli import LightningCLI
from Model import TabularDataset, Autoencoder, AutoencoderModel


def train_autoencoder(config: Dict[str, Union[str, float, int]]) -> pl.LightningModule:
    # Read data
    cp, cp_features = load_data(config['data_dir'], config['dataset'], config['profile_type'])

    # devide to train, val, test_mocks, and test_treated
    datasets = split_train_test(cp,config,cp_features)

    # construct dataset
    dataset_modules = {}
    for key in datasets.keys():
        dataset_modules[key] = TabularDataset(datasets[key][cp_features])

    # construct dataloaders
    dataloaders = {}
    for key in datasets.keys():
        if key == 'train':
            dataloaders[key] = DataLoader(dataset_modules[key], config['batch_size'], shuffle=True)
        else:
            dataloaders[key] = DataLoader(dataset_modules[key], config['batch_size'])

    # Initialize model
    hparams = {'input_size': len(cp_features),
        'latent_size': config['latent_dim'],
        'l2_lambda': config['l2_lambda'],
        'lr': config['lr'],
    }
    # (input_size=len(cp_features), latent_size = latent_size, lr = lr, l2_lambda = l2_lambda)
    model = AutoencoderModel(hparams)

    # Set up logger and checkpoint callbacks
    logger = TensorBoardLogger(save_dir=config['tb_logs_dir'], name=config['model_name'])

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=20,
        mode="min"
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=config['ckpt_dir'],
        filename='autoencoder-{epoch:02d}-{val_loss:.2f}',
        save_top_k=config['save_top_k'],
        monitor='val_loss',
        mode='min'
    )
    progressbar_callback = TQDMProgressBar(refresh_rate=150)
    # Train model
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback,early_stop_callback,progressbar_callback],
        max_epochs=config['max_epochs'],
        gpus=config['gpus'],
        auto_select_gpus=True,
        # progress_bar_refresh_rate=50,
        precision=16 if config['use_16bit'] else 32,
        deterministic=True,
        fast_dev_run=config['fast_dev_run'],
        auto_lr_find=config['auto_lr_find']
    )
    trainer.fit(model, dataloaders['train'],dataloaders['val'])

    # Test model
    test_dataloaders = ['train','val','test_ctrl', 'test_treat']


    losses = {}
    for data_subset in test_dataloaders:
        dataloader = dataloaders[data_subset]
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
    test_ctrl_out = datasets['test_ctrl'].copy()
    test_treat_out = datasets['test_treated'].copy()

    test_ctrl_out[cp_features] = x_recon_ctrl
    test_treat_out[cp_features] = x_recon_treat

    os.makedirs(os.path.join(config['data_dir'], 'anomaly_output'), exist_ok=True)
    os.makedirs(os.path.join(config['data_dir'], 'anomaly_output',config['dataset']), exist_ok=True)
    test_ctrl_out.to_csv(os.path.join(config['data_dir'], 'anomaly_output', config['dataset'],f'ad_out_ctrl.csv'))
    test_treat_out.to_csv(os.path.join(config['data_dir'], 'anomaly_output', config['dataset'],f'ad_out_treated.csv'))

    #TODO: add z_pred_saving and normalizing, keeping required indices.
    # z_pred_subset.to_csv(os.path.join(config['data_dir'], 'anomaly_output', config['dataset'],f'embedding_test_ctrl.csv'))

    test_ctrl_out_normalized = test_data_mocks.copy()
    test_treat_out_normalized = test_data_treated.copy()

    scaler_cp = preprocessing.StandardScaler()
    test_ctrl_out_normalized[cp_features] = scaler_cp.fit_transform(x_recon_ctrl)
    test_treat_out_normalized[cp_features] = scaler_cp.transform(x_recon_treat)

    test_treat_out_normalized.to_csv(os.path.join(config['data_dir'], 'anomaly_output', config['dataset'], f'ad_out_ctrl_normalized.csv'))
    test_treat_out_normalized.to_csv(
        os.path.join(config['data_dir'], 'anomaly_output', config['dataset'], f'ad_out_treated_normalized.csv'))

    return model, losses
    # # Evaluate the dataloaders, will get the metrics from validation_step in the model
    # loaders_metrics = trainer.validate(model, dataloaders[predict_dataloaders])
    # for i, (idx, metrics) in enumerate(zip(loaders_idx, loaders_metrics)):
    #     mse = metrics[f'val_loss_epoch/dataloader_idx_{i}']
    #     pcc = metrics[f'val_pcc_epoch/dataloader_idx_{i}']
    #     res.append([*idx, mse, pcc])
    #
    # # Export results
    # res_df = pd.DataFrame(res, columns=['Dataset', 'Plate', 'Subset', 'MSE', 'PCC'])
    # if split_loaders is None:
    #     save_results(res_df, args)
    # else:
    #     save_results(res_df, args, f'results_{split_loaders}.csv')
    # # net = Net.load_from_checkpoint(PATH)
    # # net.freeze()
    # # out = net(x)
    # return model, res

if __name__ == '__main__':

    seed_everything(42)
    import yaml
    import io

    # Read YAML file
    with open("config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)
    print('start training...')

    model, res = train_autoencoder(config)
    # cli = LightningCLI(train_autoencoder)
                       # model_class=AutoencoderModel,
                       # save_config_overwrite=True,
                       # log_save_interval=100)
    model
#
# if __name__ == "__main__":
#
#
#     procProf_dir = '/sise/assafzar-group/assafzar/genesAndMorph'
#     # dataset type: CDRP, CDRP-bio, LINCS, LUAD, TAORF
#     dataset = 'CDRP-bio'
#
#     # CP Profile Type options: 'augmented' , 'normalized', 'normalized_variable_selected'
#     profileType = 'normalized_variable_selected'
#
#     ################################################
#     # filtering to compounds which have high replicates for both GE and CP datasets
#     # highRepOverlapEnabled=0
#     # 'highRepUnion','highRepOverlap'
#     # filter_perts = ''
#     # repCorrFilePath = '../results/RepCor/RepCorrDF.xlsx'
#     #
#     # filter_repCorr_params = [filter_perts, repCorrFilePath]
#     #
#     # ################################################
#     # pertColName = 'PERT'
#     #
#     # if dataset == 'TAORF':
#     #     filter_perts = ''
#     # else:
#     #     filter_perts = 'highRepOverlap'
#     #
#     # if filter_perts:
#     #     f = 'filt'
#     # else:
#     #     f = ''
#
#     seed_everything(42)
#
#     batch_size = 64
#     latent_size = 16
#     lr = 0.001
#     max_epochs = 100
#     val_check_interval = 0.5
#     l2_lambda = 0.01
#
#
#     cp, cp_features = load_data(procProf_dir, dataset, profileType)
#
#     mocks = cp[cp['Metadata_ASSAY_WELL_ROLE'] == 'mock']
#
#     train_data, val_data = train_test_split(cp, test_size=0.4)
#     val_data, test_data_mocks = train_test_split(val_data, test_size=0.5)
#     test_data_treated = cp[cp['Metadata_ASSAY_WELL_ROLE'] != 'mock']
#
#     # scale to training set
#     scaler_cp = preprocessing.StandardScaler()
#     train_data[cp_features] = scaler_cp.fit_transform(train_data[cp_features].values.astype('float64'))
#     val_data[cp_features] = scaler_cp.transform(val_data[cp_features].values.astype('float64'))
#     test_data_mocks[cp_features] = scaler_cp.transform(test_data_mocks[cp_features].values.astype('float64'))
#     test_data_treated[cp_features] = scaler_cp.transform(test_data_treated[cp_features].values.astype('float64'))
#
#     # construct dataset
#     train_dataset = TabularDataset(train_data[cp_features])
#     val_dataset = TabularDataset(val_data[cp_features])
#     test_dataset_mocks = TabularDataset(test_data_mocks[cp_features])
#     test_dataset_treated = TabularDataset(test_data_treated[cp_features])
#
#     # construct dataloaders
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size)
#     test_loader_mocks = DataLoader(test_dataset_mocks, batch_size=batch_size)
#     test_loader_treated = DataLoader(test_dataset_treated, batch_size=batch_size)
#
#     # def train_autoencoder(data_path, batch_size, latent_size, lr, max_epochs, val_check_interval, l2_lambda):
#     # data = pd.read_csv(data_path)
#
#     # train_data, val_data = train_test_split(cp_scaled, test_size=0.2)
#     # train_dataset = TabularDataset(train_data[cp_features])
#     # val_dataset = TabularDataset(val_data[cp_features])
#     # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     # val_loader = DataLoader(val_dataset, batch_size=batch_size)
#
#     model = AutoencoderModel(input_size=len(cp_features), latent_size=latent_size, lr=lr, l2_lambda=l2_lambda)
#
#     early_stop_callback = EarlyStopping(
#         monitor="val_loss",
#         patience=5,
#         mode="min"
#     )
#
#     checkpoint_callback = ModelCheckpoint(
#         monitor="val_loss",
#         filename="best_autoencoder",
#         save_top_k=1,
#         mode="min",
#     )
#
#     logger = TensorBoardLogger("tb_logs", name="autoencoder")
#
#     trainer = Trainer(
#         logger=logger,
#         callbacks=[early_stop_callback, checkpoint_callback],
#         max_epochs=max_epochs,
#         val_check_interval=val_check_interval,
#         gpus=torch.cuda.device_count(),
#         precision=16,
#         deterministic=True,
#         benchmark=True,
#         auto_lr_find=True
#     )
#
#     trainer.fit(model, train_loader, val_loader)
#     # trainer.fit()
#     # result = trainer.tuner.random_search(
#     #     model,
#     #     train_loader,
#     #     val_loader,
#     #     num_trials=10,
#     #     timeout=600,
#     #     objective="val_loss",
#     #     direction="minimize",
#     # )
#     trainer.test(dataloaders=[val_loader,test_loader_mocks,test_loader_treated])
#     # return result.best_model, result.best_hyperparameters