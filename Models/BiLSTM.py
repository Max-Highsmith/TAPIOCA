import pdb
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F

def weighted_mse(output, target):
    alpha = 11
    N     = output.shape[1]
    wmse  = torch.mean(1/N * torch.sum((target-output)**2* (alpha-target)/alpha, dim=1))
    return wmse

class BiLSTMModule(pl.LightningModule):
    def __init__(self,
            input_size,
            hidden_size,
            num_layers,
            dropout,
            bias=True,
            batch_first=True,
            bidirectional=True,
            proj_size=1,
            lr=0.001,
            loss_type="weighted",
            hparams=False):
        super().__init__()
        self.lr = lr
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.bias=bias
        self.batch_first=batch_first
        self.dropout=dropout
        self.bidirectional=bidirectional
        self.proj_size=proj_size
        self.hparams = hparams
        self.loss_func = self._get_loss_type(loss_type)
        self.lstm      = torch.nn.LSTM(input_size=input_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                bias=bias,
                                batch_first=batch_first,
                                dropout=dropout,
                                bidirectional=bidirectional,
                                proj_size=proj_size)
        self.save_hyperparameters()

    def _get_loss_type(self, loss_type):
        if loss_type == "weighted":
            return weighted_mse
        if loss_type == "mse":
            return F.mse_loss

    def forward(self, src):
        output, _ = self.lstm(src)
        output    = torch.sum(output, axis=2)
        output    = torch.unsqueeze(output, dim=2)
        return output

    def training_step(self, batch, batch_idx):
        feature, label = batch
        feature        = feature.float()
        label          = label.float()
        output         = self.forward(feature)
        loss           = self.loss_func(output, label)
        self.log("train weighted mse loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        feature, label = batch
        feature        = feature.float()
        label          = label.float()
        output         = self.forward(feature)
        loss           = self.loss_func(output, label)
        self.log("val weighted mse loss", loss)
        return loss

    def test_step(self, batch, batch_indx):
        print("TODO")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
