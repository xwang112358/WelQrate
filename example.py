from welqrate.dataset import WelQrateDataset
from welqrate.models.gnn2d.GCN import GCN_Model 
import torch
from welqrate.train import train
import configparser



AID1798_dataset_2d = WelQrateDataset(dataset_name = 'AID1798', root =f'./datasets', mol_repr ='2dmol')
# AID1843_dataset_3d = WelQrateDataset(dataset_name = 'AID435034', root =f'../dataset_test', mol_repr ='3dmol')

# print(AID1843_dataset_2d[0])
# print(AID1843_dataset_3d[0])

# split_dict = AID1798_dataset_2d.get_idx_split(split_scheme ='random_cv1')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = configparser.ConfigParser()
config.read('./config/example.cfg')

hidden_channels = int(config['MODEL']['hidden_channels'])
num_layers = int(config['MODEL']['num_layers'])

model = GCN_Model(hidden_channels=hidden_channels, 
                  num_layers=num_layers).to(device)

train(model, AID1798_dataset_2d, config, device)
