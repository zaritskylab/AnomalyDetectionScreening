import pandas as pd
import numpy as np
import torch
from pytorch_lightning.cli import ReduceLROnPlateau
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.core.decorators import auto_move_data
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.utilities import AttributeDict
from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning.core.lightning import LightningModule
from torchmetrics.functional import pearson_corrcoef,r2_score
# from pytorch_lightning.metrics.functional import mse


class TabularDataset(Dataset):
  def __init__(self, data):
    self.data = data.to_numpy().astype(np.float32)

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    return self.data[idx]


class Autoencoder(nn.Module):
  def __init__(self, input_size, latent_size):
    super(Autoencoder, self).__init__()
    self.encoder = nn.Sequential(
      nn.Linear(input_size, 32),
      # nn.ReLU(),
      # nn.Linear(64, 32),
      nn.ReLU(),
      nn.Linear(32, latent_size)
    )
    self.decoder = nn.Sequential(
      nn.Linear(latent_size, input_size),
      # nn.ReLU(),
      # nn.Linear(32, 64),
      # nn.ReLU(),
      # nn.Linear(64, input_size)
    )

  def forward(self, x):
    z = self.encoder(x)
    x_recon = self.decoder(z)
    return x_recon, z


class AutoencoderModel(LightningModule):
  def __init__(self, hparams):
    super().__init__()
    self.save_hyperparameters(hparams)

    self.autoencoder = Autoencoder(
      input_size=self.hparams.input_size,
      latent_size=self.hparams.latent_size
    )

    self.l2_loss = nn.MSELoss(reduction='mean')

  def forward(self, x):
    x_recon, z = self.autoencoder(x)
    return x_recon, z

  def training_step(self, x, batch_idx):
    # x = batch
    x_recon, z = self.autoencoder(x)

    recon_loss = self.l2_loss(x_recon, x)
    l2_reg = self.hparams.l2_lambda * self.l2_loss(x_recon,torch.zeros_like(x_recon))
    loss = recon_loss + l2_reg

    # self.log('train_loss', recon_loss, prog_bar=False)
    self.log_dict({"train_loss": recon_loss, "train_l2_loss": l2_reg})
    return loss

  def validation_step(self, x, batch_idx):
    # x = batch
    x_recon, z = self.autoencoder(x)

    recon_loss = self.l2_loss(x_recon, x)
    l2_reg = self.hparams.l2_lambda * self.l2_loss(x_recon,torch.zeros_like(x_recon))
    loss = recon_loss + l2_reg

    # self.log('val_loss', recon_loss, prog_bar=True)
    self.log_dict({"val_loss": recon_loss, "val_l2_loss": l2_reg})

    return loss

  def test_step(self, x, batch_idx):
    # x, y = batch
    x_recon, z = self.autoencoder(x)

    recon_loss = self.l2_loss(x_recon, x)

    l2_reg = self.hparams.l2_lambda * self.l2_loss(x_recon,torch.zeros_like(x_recon))
    pcc = pearson_corrcoef(x_recon.reshape(-1), x.reshape(-1))
    r2 = r2_score(x_recon.reshape(-1), x.reshape(-1))

    loss = recon_loss + l2_reg

    # self.log('test_loss', recon_loss)
    # self.log_dict({"test_loss": recon_loss, "test_l2_loss": l2_reg})

    return {'mse_test': recon_loss, 'l2_reg':l2_reg, 'x_recon': x_recon, 'z': z, 'pcc_test':pcc, 'r2_score':r2}

  def test_epoch_end(self, outputs):
    mse_mean = torch.stack([x['mse_test'] for x in outputs]).mean()
    pcc_mean = torch.stack([x['pcc_test'] for x in outputs]).mean()
    r2_mean = torch.stack([x['r2_score'] for x in outputs]).mean()

    x_recon = torch.cat([x['x_recon'] for x in outputs], dim=0)
    z = torch.cat([x['z'] for x in outputs], dim=0)

    # self.logger.experiment.add_image('Reconstructed Images', x_recon, self.current_epoch)
    # self.logger.experiment.add_embedding(z, metadata=None, global_step=self.current_epoch)

    self.log('mse', mse_mean)
    self.log('pcc', pcc_mean)
    self.log('r2', r2_mean)

    return {'test_loss': mse_mean, 'x_recon': x_recon, 'z': z}


  def predict_step(self, x, batch_idx):

    x_recon, z = self.autoencoder(x)

    # recon_loss = self.l2_loss(x_recon, x)
    # l2_reg = self.hparams.l2_lambda * self.l2_loss(x_recon,torch.zeros_like(x_recon))
    # loss = recon_loss + l2_reg
    return x_recon, z
    # pred = torch.vstack([self.dropout(self.model(batch)).unsqueeze(0) for _ in range(self.mc_iteration)]).mean(dim=0)

  # def predict_epoch_end(self, outputs):
  #   test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
  #   x_recon = torch.cat([x['x_recon'] for x in outputs], dim=0)
  #   z = torch.cat([x['z'] for x in outputs], dim=0)

    # self.logger.experiment.add_image('Reconstructed Images', x_recon, self.current_epoch)
    # self.logger.experiment.add_embedding(z, metadata=None, global_step=self.current_epoch)
    #
    # self.log('test_loss', test_loss_mean)
    #
    # return {'test_loss': test_loss_mean, 'x_recon': x_recon, 'z': z}

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    return {
        "optimizer": optimizer,
        "lr_scheduler": {
          "scheduler": ReduceLROnPlateau(optimizer,...),
          "monitor": "val_loss",
          "frequency": 10
          # If "monitor" references validation metrics, then "frequency" should be set to a
          # multiple of "trainer.check_val_every_n_epoch".
        },
      }


