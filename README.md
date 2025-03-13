# WelQrate: Defining the Gold Standard in Small Molecule Drug Discovery


## Installation
We provide the recommended environment, which were used for benchmarking in the original paper. Users can also build their own environment based on their own needs.
```
conda create -n welqrate python=3.9
```

```
pip install welqrate
```


## Load the Dataset
Users can download and preprocess the datasets by calling `WelQrateDataset` class. Available datasets include AID1798, AID435008, AID435034, AID1843, AID2258, AID463087, AID488997, AID2689, and AID485290. Please refer to our [website](https://www.welqrate.org/) for more details. Besides, users can choose between 2D and 3D molecular representations by setting `mol_repr` to `2d_graph` or `3d_graph`.

```python
from welqrate.dataset import WelQrateDataset
# Load the 2D dataset
AID1798_dataset_2d = WelQrateDataset(dataset_name = 'AID1798', root =f'./datasets', mol_repr ='2d_graph')

# Load the 3D dataset 
AID1843_dataset_3d = WelQrateDataset(dataset_name = 'AID1843', root =f'./datasets', mol_repr ='3d_graph')

# Load a split dictionary
split_dict = AID1798_dataset_2d.get_idx_split(split_scheme ='random_cv1') # or 'scaffold_seed1; we provide 1-5 for both random_cv and scaffold_seed

```

## Train a model
We can store hyperparameters related to model, training scheme, and dataset in a configuration file. Users can refer to configuration files in `./config/` for different models. Then, we can config the model and start training by calling `train` function. 

```python
dataset_name = 'AID1798'
split_scheme = 'random_cv1'
AID1798_2d = WelQrateDataset(dataset_name=dataset_name, root='./datasets', mol_repr='2d_graph',
                             source='inchi')
split_dict = AID1798_2d.get_idx_split(split_scheme)

train_loader = get_train_loader(AID1798_2d[split_dict['train']], batch_size=128, num_workers=0, seed=1)
valid_loader = get_valid_loader(AID1798_2d[split_dict['valid']], batch_size=128, num_workers=0)
test_loader = get_test_loader(AID1798_2d[split_dict['test']], batch_size=128, num_workers=0)


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

```

## Citation
If you find our work helpful, please cite our paper:


```       
@article{dong2024welqrate,
  title={WelQrate: Defining the Gold Standard in Small Molecule Drug Discovery Benchmarking},
  author={Yunchao, Liu and Dong, Ha and Wang, Xin and Moretti, Rocco and Wang, Yu and Su, Zhaoqian and Gu, Jiawei and Bodenheimer, Bobby and Weaver, Charles David and Meiler, Jens and Derr, Tyler and others},
  journal={arXiv preprint arXiv:2411.09820},
  year={2024}
}

```