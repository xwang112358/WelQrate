import torch 
from dataset import WelQrateDataset
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from mol_utils.scaffold_split import (mol_to_smiles, generate_scaffolds, PoorScaffoldSplitter, generate_scaffolds_list)


name = 'AID1798'
split_scheme = 'AID1798_2d_random_cv1'

dataset =  WelQrateDataset(name, '../poor_dataset', mol_repr='3dmol')
print(dataset[0])
# split = dataset.get_idx_split(split_scheme)

# train_smiles = dataset[split['train']].smiles
# valid_smiles = dataset[split['valid']].smiles
# test_smiles = dataset[split['test']].smiles

# train_scaffolds = generate_scaffolds_list(train_smiles)  
# valid_scaffolds = generate_scaffolds_list(valid_smiles)  
# test_scaffolds = generate_scaffolds_list(test_smiles)  


# count = 0
# for i in range(len(test_scaffolds)):
#     if test_scaffolds[i] in train_scaffolds:
#         count += 1
        
# print(count/len(test_scaffolds)*100)


