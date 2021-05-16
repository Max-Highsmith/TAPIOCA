import pytorch_lightning as pl
import numpy as npa
import torch
from torch.nn import functional as F


class LinearRegressionModule(pl.LightningModule):
    def __init__(self,
            inputSize,
            outputSize,
            lr=.001,
            hasRidge=False,
            hasLasso=False,
            hparams=False):
        print("init")
        super().__init__()
        self.inputSize  = inputSize
        self.outputSize = outputSize
        self.linear     = torch.nn.Linear(inputSize, outputSize)
        self.lr         = lr
        self.hasRidge   = hasRidge
        self.hasLasso   = hasLasso
        self.hparams    = hparams
        self.save_hyperparameters()
    
    def forward(self, src):
        out = self.linear(src)
        return out

    def training_step(self, batch, batch_idx):
        feature, label = batch
        feature        = feature.float()
        label          = label.float()
        output         = self.forward(feature)
        loss           = F.mse_loss(output, label)
        self.log("train weighted mse loss", loss)
        if self.hasLasso == True:
           l1_reg  = torch.tensor(0).cuda().float()
           for param in self.linear.parameters():
               l1_reg += torch.norm(param,1)**2

           loss = loss + l1_reg
        if self.hasRidge == True:
            l2_reg = torch.tensor(0).cuda().float()
            for param in self.linear.parameters():
                l2_reg += torch.norm(param, 2)**2
            loss = loss+ l2_reg
        return loss

    def validation_step(self, batch, batch_idx):
        feature, label = batch
        feature        = feature.float()
        label          = label.float()
        output         = self.forward(feature)
        loss           = F.mse_loss(output, label)
        self.log("val weighted mse loss", loss)
        return loss

    def test_step(self, batch, batch_indx):
        feature, label = batch
        feature        = feature.float()
        output         = self.forward(feature)
        print("TBD")

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)
