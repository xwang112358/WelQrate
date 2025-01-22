# WelQrate
Official Implementation of WelQrate Benchmark


## Installation
We provide the recommended environment, which were used for benchmarking. Users can also build their own environment based on their own needs.
```
conda create -n welqrate python=3.9s
```

```
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```

```
pip install -r requirements.txt
```

```
pip install -e .
```


## Load the Dataset
Users can download and preprocess the dataset by calling `WelQrateDataset` class. Available datasets include `AID1798`, `AID435008`, `AID435034`, `AID1843`, `AID2258`, `AID463087`, `AID488997`, `AID2689`, and `AID485290`. Please refer to our [website](https://www.welqrate.org/) for more details. Besides, users can choose between 2D and 3D molecular representations by setting `mol_repr` to `2dmol` or `3dmol`.

```python
from welqrate.dataset import WelQrateDataset
# Load the 2D dataset
AID1798_dataset_2d = WelQrateDataset(dataset_name = 'AID1798', root =f'./datasets', mol_repr ='2dmol')

# Load the 3D dataset 
AID1843_dataset_3d = WelQrateDataset(dataset_name = 'AID1843', root =f'./datasets', mol_repr ='3dmol')

# Load a split dictionary
split_dict = AID1798_dataset_2d.get_idx_split(split_scheme ='random_cv1') # or 'scaffold_seed1; we provide 1-5 for both random_cv and scaffold_seed

```

## Train a model


```python
from welqrate.train import train
from welqrate.models.gnn2d.GCN import GCN_Model
import configparser
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = configparser.ConfigParser()
config.read('./config/example.cfg')

# Load the model configuration
hidden_channels = int(config['MODEL']['hidden_channels'])
num_layers = int(config['MODEL']['num_layers'])
model = GCN_Model(hidden_channels=hidden_channels, num_layers=num_layers).to(device)

# Train the model
train(model, AID1798_dataset_2d, config, device)
```
