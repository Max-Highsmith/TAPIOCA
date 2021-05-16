import matplotlib.pyplot as plt
import pdb
import math
import torch
from torch import nn as nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from torch.nn import functional as F
from IPython.core.debugger import set_trace
import pytorch_lightning as pl

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=11):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        if d_model %2 ==0:
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)[:,:-1]
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        rep_pe = self.pe.repeat(1,x.shape[1],1)
        combo  = torch.cat((x, rep_pe), dim=2)
        return combo
        #combo = x + self.pe[:x.size(0), :]
        #drop_combo = self.dropout(combo)
        #pdb.set_trace()
        #return drop_combo
    
def weighted_mse(output, target):
    alpha = 11
    N     = output.shape[1]
    wmse  = torch.mean(1/N * torch.sum((target-output)**2* (alpha-target)/alpha, dim=1))
    return wmse

class  TransformerModule(pl.LightningModule):
    def __init__(self,
                ntoken,
                ninp,
                nhead,
                nhid,
                nlayers,
                dropout=0.5,
                optimi="Adam",
                lr=.01,
                loss_type="weighted",
                hparams=False):
        print("init")
        super().__init__()
        
        self.ntoken    = ntoken
        self.ninp      = int(ninp)
        self.nhead     = nhead
        self.nhid      = nhid
        self.nlayers   = nlayers
        self.dropout   = dropout
        self.optimi    = optimi
        self.lr        = lr
        self.loss_type = loss_type
        self.hparams   = hparams
        #new
        #self.embedding = nn.Linear(29,40)
        #new
        #self.pos_enc   = PositionalEncoding(ninp, max_len=11)
        POS_ENC        = 49-ninp
        self.pos_enc   = PositionalEncoding(POS_ENC, max_len=11)
        encoder_layer  = TransformerEncoderLayer(POS_ENC+ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layer, nlayers)
        self.decoder   = nn.Linear(POS_ENC+ninp, ntoken)
        self.src_mask  = None
        self.init_weights()
        self.save_hyperparameters()
    
    def init_weights(self):
        initrange = 0.1
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0,1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask ==1, float(0.0))
        return mask
    
    def forward(self, src, has_mask=True):
        #new
        #src_1  = F.tanh(self.embedding(src))
        src_1   = src
        #new
        BATCH_SIZE = src_1.shape[0]
        SEQ_LEN    = src_1.shape[1]
        EMBED_DIM  = src_1.shape[2]
        src_2        = src_1.permute(1, 0, 2)
        '''
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                mask = mask.float()
                self.src_mask = mask
        else:
            self.src_mask = None
        '''
        self.src_mask = None
        src_3    = src_2 #* math.sqrt(self.ninp)
        src_4    = F.relu(self.pos_enc(src_3))
        output_1 = self.transformer_encoder(src_4, self.src_mask)
        output_2 = self.decoder(output_1)
        output_3 = output_2.permute(1,0,2)
        return output_3
    
    def training_step(self, batch, batch_idx):
        feature, label = batch
        feature        = feature.float()
        label          = label.float()
        output         = self.forward(feature)
        if self.loss_type =="weighted":
            loss           = weighted_mse(output, label)
        if self.loss_type=="mse":
            loss       = F.mse_loss(output, label)
        self.log("train weighted mse loss", loss)
        return loss
    
    def validation_step(self, batch, batch_indx):
        feature, label = batch
        feature        = feature.float()
        label          = label.float()
        output         = self.forward(feature)
        if self.loss_type == "weighted":
            loss           = weighted_mse(output, label)
        if self.loss_type=="mse":
            loss           = F.mse_loss(output, label)
        self.log("val weighted mse loss", loss)
        return loss

    def test_step(self, batch, batch_indx):
        feature, label = batch
        feature        = feature.float()
        output         = self.forward(feature)
        fig, ax = plt.subplots(1)
        ax.plot(label[0,:,0].cpu(), label="label")
        ax.plot(output[0,:,0].cpu(),  label="output")
        plt.legend()
        plt.show()
    
    def configure_optimizers(self):
        #optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        if self.optimi == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        if self.optimi == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        if self.optimi == "RMSprop":
            optimizer = torch.optim.RMSprop(self.parameters(), lr=self.lr)
        return optimizer
        
    
