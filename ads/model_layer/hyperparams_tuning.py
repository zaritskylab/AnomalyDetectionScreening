import optuna
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from ads.model_layer.AEModel import AutoencoderModel

def tune_hyperparams(dataloaders, features, configs):

    # Optuna study
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, dataloaders=dataloaders, features =features,
                                            hidden_size = configs.model.latent_dim,
                                            deep_decoder=configs.model.deep_decoder,
                                            encoder_type=configs.model.encoder_type, 
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

# Define a PyTorch Lightning model evaluation function for Optuna
def objective(trial, dataloaders,features,hidden_size=None,deep_decoder=False, encoder_type = 'default',max_epochs=100,tune_l2=None,tune_l1=None, l2_lambda=0, l1_latent_lambda=0):
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
