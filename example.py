from welqrate.dataset import WelQrateDataset
from welqrate.train import train
from welqrate.models.gnn2d.GCN import GCN_Model
from welqrate.loader import get_train_loader, get_valid_loader, get_test_loader
import yaml 
import torch

dataset_name = 'AID1798'
split_scheme = 'random_cv1'
AID1798_2d = WelQrateDataset(dataset_name=dataset_name, root='./datasets', mol_repr='2d_graph',
                             source='inchi')
print(AID1798_2d[0])
split_dict = AID1798_2d.get_idx_split(split_scheme)

train_loader = get_train_loader(AID1798_2d[split_dict['train']], batch_size=128, num_workers=0, seed=1)
valid_loader = get_valid_loader(AID1798_2d[split_dict['valid']], batch_size=128, num_workers=0)
test_loader = get_test_loader(AID1798_2d[split_dict['test']], batch_size=128, num_workers=0)

print(len(train_loader))

config = {}
# default train config
for config_file in ['./config/train.yaml', './config/gcn.yaml']:
    with open(config_file) as file:
        config.update(yaml.safe_load(file))

# initialize model
hidden_channels = config['model']['hidden_channels']
num_layers = config['model']['num_layers']
model = GCN_Model(hidden_channels = hidden_channels, 
                  num_layers = num_layers)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

train(model = model,
      train_loader = train_loader,
      valid_loader = valid_loader,
      test_loader = test_loader,
      config = config,
      device = device,
      save_path = f'./results/{dataset_name}/{split_scheme}/gcn'
      )
