# WelQrate
Official Implementation of WelQrate Benchmark


## Installation
We provide the recommended environment, which were used for benchmarking. Users can also build their own environment based on their own needs.
```
conda create -n welqrate python=3.10
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

```python
from welqrate.dataset import WelQrateDataset
# Load the 2D dataset
AID1843_dataset_2d = WelQrateDataset(dataset_name = 'AID488997', root =f'./datasets', mol_repr ='2dmol')

# Load the 3D dataset with 3D coordinates
AID1843_dataset_3d = WelQrateDataset(dataset_name = 'AID435034', root =f'./datasets', mol_repr ='3dmol')

```

