import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning import LightningModule, Trainer


class ConcreteSelect(nn.Module):
    def __init__(self, output_dim, start_temp=10.0, min_temp=0.1, alpha=0.99999):
        super(ConcreteSelect, self).__init__()

        self.output_dim = output_dim
        self.start_temp = start_temp
        self.min_temp = min_temp
        self.alpha = alpha

        self.temp = nn.Parameter(torch.Tensor([start_temp]), requires_grad=False)
        self.logits = nn.Parameter(torch.Tensor(output_dim, input_shape[1]))

    def forward(self, x):
        uniform = torch.empty_like(self.logits).uniform_(torch.finfo(torch.float32).eps, 1.0)
        gumbel = -torch.log(-torch.log(uniform))
        temp = torch.max(self.temp * self.alpha, torch.tensor(self.min_temp))
        noisy_logits = (self.logits + gumbel) / temp
        samples = F.softmax(noisy_logits, dim=-1)

        discrete_logits = F.one_hot(torch.argmax(self.logits), self.logits.shape[1])

        selections = samples if self.training else discrete_logits
        y = torch.bmm(x.unsqueeze(1), selections.unsqueeze(2))
        y = y.view(-1, self.output_dim)

        return y

class ConcreteAutoencoderFeatureSelector(LightningModule):
    def __init__(self, K, output_function, num_epochs=300, batch_size=None, learning_rate=0.001, start_temp=10.0, min_temp=0.1, tryout_limit=5):
        super(ConcreteAutoencoderFeatureSelector, self).__init__()
        self.K = K
        self.output_function = output_function
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.start_temp = start_temp
        self.min_temp = min_temp
        self.tryout_limit = tryout_limit
        self.concrete_select = None
        self.probabilities = None
        self.indices = None

    def forward(self, x):
        selected_features = self.concrete_select(x)
        outputs = self.output_function(selected_features)
        return outputs

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = F.mse_loss(outputs, y)
        self.log('train_loss', loss)
        return loss

    def compute_probs(self, concrete_select):
        # self.probabilities = concrete_select.logits.softmax(dim=-1).detach().cpu().numpy()
        self.probabilities = self.concrete_select.logits.softmax(dim=-1).detach().cpu().numpy()
        # self.indices = self.concrete_select.logits.argmax(dim=-1).detach().cpu().numpy()
        self.indices = self.concrete_select.logits.argmax(dim=-1).detach().cpu().numpy()

    def get_indices(self):
        return self.indices

    def get_mask(self):
        mask = torch.zeros(self.output_dim)
        mask[self.indices] = 1
        return mask.detach().cpu()