from torch.utils.data import Dataset
import numpy as np

class TabularDataset(Dataset):
  def __init__(self, data,features):

    self.data = data[features].to_numpy().astype(np.float32)

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):

    return self.data[idx]
