from torch.utils.data import Dataset
import numpy as np

class TabularDataset(Dataset):
  def __init__(self, data,features):
    # self.features = features

    # self.data = data
    # self.cpd_col = cpd_col
    # self.dose_col = dose_col
    # self.role_col = role_col
    # self.plate_col = plate_col
    # self.mock_val = mock_val
    # self.modality = modality
    # self.dataset = data
    self.data = data[features].to_numpy().astype(np.float32)

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):

    # record = self.data.iloc[idx,:]
    # index = self.data.index[idx]
    # data = record[self.features].to_numpy().astype(np.float32)
    
    # return data, record[self.cpd_col], record[self.plate_col],index
    return self.data[idx]
