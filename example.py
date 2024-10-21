from dataset import WelQrateDataset
from models.gnn2d.GCN import GCN_Model 
import torch
from train import train
import configparser
from torch.optim import AdamW


AID1843_dataset_2d = WelQrateDataset(dataset_name = 'AID435034', root =f'../dataset_test', mol_repr ='2dmol')
# AID1843_dataset_3d = WelQrateDataset(dataset_name = 'AID435034', root =f'../dataset_test', mol_repr ='3dmol')

# print(AID1843_dataset_2d[0])
# print(AID1843_dataset_3d[0])

# split_dict = AID1843_dataset_2d.get_idx_split(split_scheme ='random_cv2')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = configparser.ConfigParser()
config.read('./config/GCN.cfg')


model = GCN_Model(in_channels= 28, 
            hidden_channels=32,
            num_layers=3,
            one_hot = False).to(device)


train(model, AID1843_dataset_2d, config, device)
