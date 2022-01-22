import pdb
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image
from IPython.core.debugger import set_trace
from torch import nn as nn
import torch
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from torch.nn import functional as F

class FlyDataModule(LightningDataModule):
    class FlyDataset(Dataset):
        def _condense_features(self,
                features,
                solo_feature,
                exclude_feature):
            if solo_feature == None and exclude_feature == None:
                return features

            if not solo_feature == None:
                return features[:,:,solo_feature:solo_feature+1]

            if not exclude_feature == None:
                return np.delete(features, exclude_feature, axis=2)

        def _get_cell_line_idx(self, strin):
            if strin == "S2":
                return 2
            if strin =="KC":
                return 3
            if strin == "BG":
                return 4
            if strin == "ALL":
                return "ALL"
        
        def _get_label_col(self, label_type, label_val):
            if label_type == "gamma":
                return 7
            if label_type == "insulation":
                if label_val==3:
                    return 8
                if label_val==5:
                    return 9
                if label_val==10:
                    return 10
            if label_type =="directionality":
                if label_val ==3:
                    return 11
                if label_val ==5:
                    return 12
                if label_val ==10:
                    return 13
                
        def __init__(self,
                    cell_line,
                    tvt,
                    data_win_radius,
                    label_type="gamma",
                    label_val=0,
                    solo_feature=None,
                    exclude_feature=None,
                    all_tads=False):
            self.cell_line       = cell_line
            self.data_win_radius = data_win_radius
            self.tvt             = tvt
            self.label_type      = label_type
            self.label_val       = label_val
            self.exclude_feature = exclude_feature
            self.solo_feature    = solo_feature
            self.all_tads        = all_tads
            self.feature_vecs = []
            self.label_vecs   = []
        
            FEATURE_STRING = "Data/Drosophilla/clean_features.csv"
            LABEL_STRING   = "Data/Drosophilla/clean_labels.csv"
            
            cell_line_idx = self._get_cell_line_idx(cell_line)
        
            features       = np.loadtxt(FEATURE_STRING,
                                   delimiter=',',
                                   dtype=str)
        
            labels         = np.loadtxt(LABEL_STRING,
                                   delimiter=',',
                                   dtype=str)
            
            self.feature_head = features[0]
            self.label_head   = labels[0]
                        
            features          = features[1:]
            labels            = labels[1:]
        
            if cell_line_idx != "ALL":
                features          = features[features[:, cell_line_idx].astype(int)==1]
                labels            = labels[labels[:, cell_line_idx].astype(int)==1]

            if cell_line_idx == "ALL":
                print("fix all")
                '''
                features_s2  = features[features[:, 2].astype(int)==1]
                features_kc  = features[features[:, 3].astype(int)==1]
                features_bg  = features[features[:, 4].astype(int)==1]
                labels_s2    = features[features[:, 2].astype(int)==1]
                labels_kc    = features[features[:, 3].astype(int)==1]
                labels_bg    = features[features[:, 4].astype(int)==1]
                tra_fe       = []
                tra_la       = []
                val_fe       = []
                val_la       = []
                tes_fe       = []
                tes_la       = []
                for fe, lab in zip([features_s2, features_kc, features_bg],
                                    [labels_s2, labels_kc, labels_bg]):
                    cutoff1=int(.7*fe.shape[0])
                    cutoff2=int(.85*fe.shape[0])
                    tra_fe.append(fe[:cutoff1])
                    tra_la.append(la[:cutoff1])
                    val_fe.append(fe[cutoff1:cutoff2])
                    val_la.append(la[cutoff1:cutoff2])
                    tes_fe.append(fe[cutoff2:])
                    tes_la.append(la[cutoff2:])
                features = np.stack([tra_fe, val_fe, tes_fe])
                labels   = np.stack([tra_la, val_la, tes_la])
                '''
                    
                


            features          = features[:,7:].astype(float)            
            
            if all_tads == False:
                label_idx         = self._get_label_col(self.label_type, 
                                                    self.label_val)
                self.labels       = labels[:, label_idx].astype(float)
            else:
                self.labels       = np.concatenate((labels[:,7:8], labels[:,10:11], labels[:,13:14]), axis=1)
                self.labels       = self.labels.astype(float)


            self.features = preprocessing.scale(features,
                                           axis=0,
                                           with_mean=True,
                                           with_std=True)
            
            
            feature_vecs = []
            label_vecs   = []
            
            for i in range(0, self.features.shape[0] - self.data_win_radius):
                start = i - self.data_win_radius
                end   = i + self.data_win_radius + 1
                if start < 0:
                    continue
                feature_vec = self.features[start:end]
                label_vec   = self.labels[start:end]
                feature_vecs.append(feature_vec)
                label_vecs.append(label_vec)
                
            self.feature_vecs = np.array(feature_vecs)

            if all_tads == False:
                self.label_vecs   = np.expand_dims(np.array(label_vecs), axis=2)
            else:
                self.label_vecs   = np.array(label_vecs)

            self.feature_vecs = self._condense_features(self.feature_vecs,
                                                        solo_feature,
                                                        exclude_feature)
            
            if self.tvt == 'mini':
                self.feature_vecs = self.feature_vecs[0:200]
                self.label_vecs   = self.label_vecs[0:200]
                
            if self.tvt == 'train':
                cutoff=int(.7*self.feature_vecs.shape[0])
                self.feature_vecs = self.feature_vecs[:cutoff]
                self.label_vecs   = self.label_vecs[:cutoff]
                
            if self.tvt == 'val':
                cutoff1=int(.7*self.feature_vecs.shape[0])
                cutoff2=int(.85*self.feature_vecs.shape[0])
                self.feature_vecs = self.feature_vecs[cutoff1:cutoff2]
                self.label_vecs   = self.label_vecs[cutoff1:cutoff2]
                
            if self.tvt == 'test':
                cutoff=int(.85*self.feature_vecs.shape[0])
                self.feature_vecs = self.feature_vecs[cutoff:]
                self.label_vecs   = self.label_vecs[cutoff:] 
        
        def __len__(self):
            return self.feature_vecs.shape[0]
        
        def __getitem__(self, idx):
            return self.feature_vecs[idx], self.label_vecs[idx]
    
    def __init__(self, 
                 cell_line,
                 data_win_radius,
                 batch_size,
                 label_type="gamma",
                 label_val=0,
                 exclude_feature=None,
                 solo_feature=None,
                 all_tads=False):
        super().__init__()
        self.batch_size      = batch_size
        self.cell_line       = cell_line
        self.data_win_radius = data_win_radius
        self.label_type      = label_type
        self.label_val       = label_val
        self.exclude_feature = exclude_feature
        self.solo_feature    = solo_feature
        self.all_tads        = all_tads
    
    def setup(self):
        self.train = self.FlyDataset(cell_line=self.cell_line,
                        tvt="train",
                        data_win_radius=self.data_win_radius,
                        label_type=self.label_type,
                        label_val=self.label_val,
                        exclude_feature=self.exclude_feature,
                        solo_feature=self.solo_feature,
                        all_tads=self.all_tads)
        
        self.val   = self.FlyDataset(cell_line=self.cell_line,
                        tvt="val",
                        data_win_radius=self.data_win_radius,
                        label_type=self.label_type,
                        label_val=self.label_val,
                        exclude_feature=self.exclude_feature,
                        solo_feature=self.solo_feature,
                        all_tads=self.all_tads)
        
        self.test  = self.FlyDataset(cell_line=self.cell_line,
                        tvt="test",
                        data_win_radius=self.data_win_radius,
                        label_type=self.label_type,
                        label_val=self.label_val,
                        exclude_feature=self.exclude_feature,
                        solo_feature=self.solo_feature,
                        all_tads=self.all_tads)
                                
        print("Everything set")
        
    def train_dataloader(self):
        return DataLoader(self.train, 
                          batch_size=self.batch_size,
                         num_workers=8)
    def val_dataloader(self):
        return DataLoader(self.val,
                          batch_size=self.batch_size,
                         num_workers=8)
    def test_dataloader(self):
        return DataLoader(self.test,
                        batch_size=self.batch_size,
                         num_workers=8)
