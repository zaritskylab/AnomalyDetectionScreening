import pandas as pd
import numpy as np
import torch
from pytorch_lightning.cli import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR
from torch import nn
from torch.utils.data import DataLoader, random_split, Dataset
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
# import lightning.pytorch as pl
from torchmetrics.functional import pearson_corrcoef,r2_score



# import lightning as L
# from sklearn.metrics import accuracy_score
# import torch
# import torch.nn.functional as F
# from torch import nn, optim
# from torchmetrics import Accuracy
# # from sklearn.metrics import accuracy_score
# # from lightning_quant.core.metrics import regularization
# from pytorch_lightning.metrics.functional import mse


import lightning as L
from sklearn.metrics import accuracy_score
import torch
import torch.nn.functional as F
from torch import nn, optim
from torchmetrics import Accuracy
# from sklearn.metrics import accuracy_score
# from lightning_quant.core.metrics import regularization

# Step 2: Create a LightningDataModule
class TabularDataModule(pl.LightningDataModule):
    def __init__(self, X_train, y_train, X_test, y_test, batch_size=64):
        super().__init__()
        self.data = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.batch_size = batch_size

    def train_dataloader(self):
        train_dataset = TensorDataset(self.X_train, self.y_train)
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        test_dataset = TensorDataset(self.X_test, self.y_test)
        return DataLoader(test_dataset, batch_size=self.batch_size)

class TabularDataset(Dataset):
  def __init__(self, x_train, y_train):
        self.data = x_train
        self.y_train = y_train
        # self.X_test = X_test
        # self.y_test = y_test
        # self.batch_size = batch_size

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    X = self.data[idx]
    y = self.y_train[idx]
    return X,y


class MLP(nn.Module):
    def __init__(self, in_features, num_classes, hidden_size=200):
        super(MLP, self).__init__()
        self.input_size = in_features
        self.hidden_size  = hidden_size
        self.output_size = num_classes
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_size, int(self.hidden_size/4))
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(int(self.hidden_size/4), self.output_size)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        output = self.softmax(output)
        return output


class ElasticNetMLP(L.LightningModule):
    """Logistic Regression with L1 and L2 Regularization"""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        bias: bool = False,
        lr: float = 0.001,
        l1_strength: float = 0.5,
        l2_strength: float = 0.5,
        epochs = 200,
        hidden_size = 200,
        optimizer="Adam",
        accuracy_task: str = "multiclass",
        dtype="float32",
    ):
        super().__init__()

        self.lr = lr
        self.l1_strength = l1_strength
        self.l2_strength = l2_strength
        self.accuracy_task = accuracy_task
        self.num_classes = num_classes
        self.epochs = epochs
        self._dtype = getattr(torch, dtype)
        self.optimizer = getattr(optim, optimizer)
        self.model = MLP(
            in_features=in_features,
            hidden_size=hidden_size,
            num_classes=num_classes,
            # bias=bias,
            # dtype=self._dtype,
        )
        self.save_hyperparameters()

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def training_step(self, batch):
        return self.common_step(batch, "training")

    def test_step(self, batch, *args):
        self.common_step(batch, "test")

    def validation_step(self, batch, *args):
        self.common_step(batch, "val")

    def common_step(self, batch, stage):
        x, y = batch
        x = x.to(self._dtype)
        y = y.to(torch.long)
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        # loss = regularization(
        #     self.model,
        #     criterion,
        #     self.l1_strength,
        #     self.l2_strength,
        # )

        if stage == "training":
            self.log(f"{stage}_loss", loss)
            return loss
        if stage in ["val", "test"]:
            acc = accuracy(
                y_hat.argmax(dim=-1),
                y,
                task=self.accuracy_task,
                num_classes=self.num_classes,
            )
            self.log(f"{stage}_acc", acc)
            self.log(f"{stage}_loss", loss)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat = self(x)
        y_hat = y_hat.argmax(dim=-1)
        return y_hat

    def configure_optimizers(self):
        optimizer = self.optimizer(
            self.parameters(),
            lr=self.lr,
        )
        return {"optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": StepLR(optimizer, step_size=50, gamma=0.5),
        },}

        # return optimizer
    def fit(self, x_train, y_train):
            model_tr = L.Trainer(
            # logger=logger,
            # callbacks=[checkpoint_callback,early_stop_callback,progressbar_callback],
            max_epochs=self.epochs,
            accelerator="auto",
            # progress_bar_refresh_rate=50,

            # precision=16 if config['use_16bit'] else 32

            # deterministic=True,
            # fast_dev_run=config['fast_dev_run']
            )
            # print(x_train.shape)
            dataLoader = get_data_loaders(x_train, y_train, batch_size=64, shuffle=True)    
            model_tr.fit(self, dataLoader)

    def score(self, x_test, y_test):
        self.eval()
        pred = self.predict(x_test)
        # dataloader = get_data_loaders(x_test, y_test, batch_size=32, shuffle=False)
        # score = accuracy_score(x_test, y_test)
        # acc_metric = accuracy(task=self.accuracy_task, num_classes=self.num_classes, top_k=1)
        # torchmetrics_accuracy = Accuracy(task='multiclass',                                           
                                    #  num_classes=self.num_classes)
        score = accuracy_score(pred, y_test)
        # score = acc_metric(pred, y_test)

        return score

    def predict_proba(self, x):
        self.eval()
        with torch.no_grad():
            x = torch.tensor(x, dtype=torch.float32)
            y_hat = self(x)
            y_hat = y_hat.numpy()
        return y_hat

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            x = torch.tensor(x, dtype=torch.float32)
            y_hat = self(x)
            y_hat = y_hat.argmax(dim=-1)
            y_hat = y_hat.numpy()
        return y_hat
    def classes_(self):
        return self.classes

            
def get_data_loaders(x_train, y_train, batch_size, shuffle):
    train = TabularDataset(x_train, y_train)
    dataloader = DataLoader(train, batch_size=batch_size, shuffle=shuffle)
    # dataset = TabularDataModule(x_train, y_train, x_test, y_test, batch_size=batch_size)
    # dataLoader = dataModule.train_dataloader()
    return dataloader

# def Dataset()

