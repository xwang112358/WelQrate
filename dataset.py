from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Dataset, Data
import torch
import os
import os.path as osp
from tqdm import tqdm
from rdkit import Chem, RDLogger
from argparse import ArgumentParser
import pandas as pd
from mol_utils.preprocess import (smiles2graph, inchi2graph, sdffile2mol_conformer,
                                  mol_conformer2graph3d)
from glob import glob
import numpy as np
import random
import zipfile
import subprocess


class WelQrateDataset(InMemoryDataset):
    def __init__(self, dataset_name, root ='dataset', mol_repr ='2dmol', task_type='classification'):
        if task_type not in ['classification', 'regression']:
            raise ValueError("task_type must be either 'classification' or 'regression'")
        
        self.name = dataset_name
        self.root = osp.join(root, dataset_name)
        self.task_type = task_type
        self.mol_repr = mol_repr
        
        print(f'dataset stored in {self.root}')
        
        if self.task_type == 'regression':
            raise ValueError('Regression task is not supported yet')

        # if self.name not in ['AID2689', 'AID488997', 'AID435008'] and self.task_type == 'regression':
        #     raise ValueError(f'{self.name} is not a regression dataset')
        
        super(WelQrateDataset, self).__init__(self.root)
        
        self.data, self.slices = torch.load(self.processed_paths[0])

        y = self.y.squeeze()
        self.num_active = len([x for x in y if x == 1])
        self.num_inactive = len([x for x in y if x == 0])

        print(f"Dataset {self.name} loaded.")
        print(f"Number of active molecules: {self.num_active}")
        print(f"Number of inactive molecules: {self.num_inactive}")

    @property
    def dataset_info(self):
        return {
            "AID1798": {
                "raw_url": "https://vanderbilt.box.com/shared/static/itigbntjvjp6zzw38mn93sjti5ra7ypw.zip",
                "file_type": "zip",
                "raw_files": ['AID1798_actives.csv', 'AID1798_inactives.csv', 'AID1798_actives.sdf', 'AID1798_inactives.sdf'],
                "split_url": "https://vanderbilt.box.com/shared/static/68m9qigxd7kt0xtta3chrx270grd1l4p.zip",
                "split_file_type": "zip"
            },
            "AID2258": {
                "url": None,  # Placeholder for future URL
                "file_type": "unknown",  # Update when known
                "raw_files": ['AID2258_actives.csv', 'AID2258_inactives.csv']  # Assumed file names
            },
            "AID2689": {
                "url": None,  # Placeholder for future URL
                "file_type": "unknown",  # Update when known
                "raw_files": ['AID2689_actives.csv', 'AID2689_inactives.csv']  # Assumed file names
            },
            "AID435008": {
                "url": None,  # Placeholder for future URL
                "file_type": "unknown",  # Update when known
                "raw_files": ['AID435008_actives.csv', 'AID435008_inactives.csv']  # Assumed file names
            },
            "AID435034": {
                "url": None,  # Placeholder for future URL
                "file_type": "unknown",  # Update when known
                "raw_files": ['AID435034_actives.csv', 'AID435034_inactives.csv']  # Assumed file names
            },
            "AID463087": {
                "url": None,  # Placeholder for future URL
                "file_type": "unknown",  # Update when known
                "raw_files": ['AID463087_actives.csv', 'AID463087_inactives.csv']  # Assumed file names
            },
            "AID485290": {
                "url": None,  # Placeholder for future URL
                "file_type": "unknown",  # Update when known
                "raw_files": ['AID485290_actives.csv', 'AID485290_inactives.csv']  # Assumed file names
            },
            "AID488997": {
                "url": None,  # Placeholder for future URL
                "file_type": "unknown",  # Update when known
                "raw_files": ['AID488997_actives.csv', 'AID488997_inactives.csv']  # Assumed file names
            }
        }

    @property
    def raw_file_names(self):
        # List files in the raw directory that match the dataset pattern
        pattern_csv = os.path.join(self.raw_dir, f'raw_{self.name}_*.csv')
        pattern_sdf = os.path.join(self.raw_dir, f'raw_{self.name}_*.sdf')
        files_csv = glob(pattern_csv)
        files_sdf = glob(pattern_sdf)
        return [os.path.basename(f) for f in files_csv + files_sdf]
    
    # need to change inchi/smiles to 2dmol
    @property
    def processed_file_names(self):
        return [f'processed_{self.mol_repr}_{self.name}.pt'] # example: processed_2dmol_AID1798.pt, processed_3dmol_AID1798.pt
    
    @property
    def processed_dir(self):
        return osp.join(self.root, 'processed', self.mol_repr) # example: processed/2dmol

    def download(self):
        os.makedirs(osp.join(self.root, 'raw'), exist_ok=True)
        os.makedirs(osp.join(self.root, 'split'), exist_ok=True)
        
        dataset_info = self.dataset_info.get(self.name)
        if not dataset_info:
            print(f"Download information for {self.name} is not available.")
            return

        # # Download and extract raw files
        # if not all(osp.exists(osp.join(self.root, 'raw', f)) for f in dataset_info['raw_files']):
        #     print(f"Downloading {self.name} raw files...")
        #     raw_zip_path = osp.join(self.root, 'raw', f'{self.name}.zip')
        #     subprocess.run(['curl', '-L', dataset_info['raw_url'], '--output', raw_zip_path], check=True)
            
        #     print("Extracting raw files...")
        #     subprocess.run(['unzip', '-j', raw_zip_path, '-d', osp.join(self.root, 'raw')], check=True)
            
        #     print("Removing raw zip file...")
        #     os.remove(raw_zip_path)

        # Download and extract split files
        # if not os.listdir(osp.join(self.root, 'split')):  # Check if split directory is empty
        #     print(f"Downloading {self.name} split files...")
        #     split_zip_path = osp.join(self.root, 'split', f'{self.name}_split.zip')
        #     subprocess.run(['curl', '-L', dataset_info['split_url'], '--output', split_zip_path], check=True)
            
        #     print("Extracting split files...")
        #     subprocess.run(['unzip', split_zip_path, '-d', osp.join(self.root, 'split')], check=True)
            
        #     print("Removing split zip file...")
        #     os.remove(split_zip_path)


    def process(self):

        print(f'molecule representation:{self.mol_repr}.')
        if self.mol_repr == '2dmol':   # combine smiles and inchi to 2dmol later
            self.file_type = '.csv'
        elif self.mol_repr == '3dmol':
            self.file_type = '.sdf'
        print(f'processing dataset {self.name}')
        RDLogger.DisableLog('rdApp.*')
        
        data_list = []
        # invalid_id_list = []
        mol_id = 0
        
        for file_name, label in [(f'{self.name}_actives{self.file_type}', 1),
                                 (f'{self.name}_inactives{self.file_type}', 0)]:

            source_path = os.path.join(self.root, 'raw', file_name)
            print(f'loaded raw file from {source_path}')
            
            if self.mol_repr == '2dmol':
                inchi_list = pd.read_csv(source_path, sep=',')['InChI'].tolist()
                cid_list = pd.read_csv(source_path, sep=',')['CID'].tolist() 
                smiles_list = pd.read_csv(source_path, sep=',')['SMILES'].tolist()
                # if self.task_type == 'regression':
                #     activity_value_list = pd.read_csv(source_path, sep=',')['activity_value'].tolist()
                
                # extract the smiles/inchi column
                for i, mol in tqdm(enumerate(inchi_list), total = len(inchi_list)):
                    pyg_data = inchi2graph(mol)

                    # if pyg_data.valid is False:
                    #     invalid_id_list.append([mol_id, mol])
                    #     print('skip 1 invalid mol')
                    #     continue
                    
                    pyg_data.y = torch.tensor([label], dtype=torch.int) 
                    # if self.task_type == 'regression':
                    #     pyg_data.activity_value = torch.tensor([activity_value_list[i]], dtype=torch.float)
                    pyg_data.pubchem_cid = torch.tensor([int(cid_list[i])], dtype=torch.int)
                    pyg_data.mol_id = torch.tensor([mol_id], dtype=torch.int)  # index of the molecule in the dataset
                    pyg_data.smiles = smiles_list[i]
                    data_list.append(pyg_data)
                    mol_id += 1
                    
            elif self.mol_repr == '3dmol':
                
                df = pd.read_csv(f'{source_path[:-4]}.csv')
                smiles_dict = df.set_index('CID')['SMILES'].to_dict()
                inchi_dict = df.set_index('CID')['InChI'].to_dict()
                # if self.task_type == 'regression':  
                #     activity_dict = df.set_index('CID')['activity_value'].to_dict()

                mol_conformer_list, cid_list = sdffile2mol_conformer(source_path)
                # get the smiles/inchi from the csv file with CID
                
                for i, (mol, conformer) in tqdm(enumerate(mol_conformer_list), total=len(mol_conformer_list)):
                    pyg_data = mol_conformer2graph3d(mol, conformer)
                    # if pyg_data.valid is False: # need to implement valid
                    #     invalid_id_list.append([mol_id, mol])
                    #     print('skip 1 invalid mol')
                    #     continue
                    # need to extract the smiles/inchi from the csv file with CID
                    # directly converting mol to smiles/inchi encounters some bugs
                    pyg_data.pubchem_cid = torch.tensor([int(cid_list[i])], dtype=torch.int)
                    pyg_data.y = torch.tensor([label], dtype=torch.int)
                    pyg_data.smiles = smiles_dict[int(cid_list[i])]
                    pyg_data.inchi = inchi_dict[int(cid_list[i])]
     
                    # if self.task_type == 'regression':
                    #     pyg_data.activity_value = torch.tensor([activity_dict[int(cid_list[i])]], dtype=torch.float)
                    pyg_data.mol_id = torch.tensor([mol_id], dtype=torch.int)
                    data_list.append(pyg_data)
                    mol_id += 1
        
        # save invalid_id_list
        # pd.DataFrame(invalid_id_list).to_csv(
        #     os.path.join(self.processed_dir, f'{self.name}-{self.mol_repr}-invalid_id.csv')
        #     , header=None, index=None)

        # if len(invalid_id_list) > 0:
        #     print(f'number of invalid molecules: {len(invalid_id_list)}, check the invalid_id_list.csv')
        #     raise ValueError('invalid molecules found')
        
        data, slices = self.collate(data_list)
        processed_file_path = os.path.join(self.processed_dir, f'processed_{self.mol_repr}_{self.name}.pt')
        torch.save((data, slices), processed_file_path)

        
    def get_idx_split(self, split_type = 'random', num = 1):
        path = osp.join(self.root, 'split')
        try: 
            if split_type == 'random':
                print(f'loading random split cv {num}')
                split_scheme = f'{self.name}_{self.mol_repr[:2]}_random_cv{num}'
                split_dict = torch.load(osp.join(path, 'random', f'{split_scheme}.pt'))
            elif split_type =='scaffold':
                print(f'loading scaffold split seed {num}')
                split_scheme = f'{self.name}_{self.mol_repr[:2]}_scaffold_seed{num}'
                split_dict = torch.load(osp.join(path, 'scaffold', f'{split_scheme}.pt'))

        except Exception as e:
            print(f'split file not found. Error msg: {e}')

        # count the number of active molecules in the split
        y = self.y.squeeze()
        num_active_train = len([x for x in y[split_dict['train']] if x == 1])
        print(f'training set: {num_active_train} active molecules and {len(split_dict["train"]) - num_active_train} inactive molecules')
        num_active_valid = len([x for x in y[split_dict['valid']] if x == 1])
        print(f'validation set: {num_active_valid} active molecules and {len(split_dict["valid"]) - num_active_valid} inactive molecules')
        num_active_test = len([x for x in y[split_dict['test']] if x == 1])
        print(f'test set: {num_active_test} active molecules and {len(split_dict["test"]) - num_active_test} inactive molecules')



        # try:
        #     invalid_id_list = pd.read_csv(os.path.join(self.processed_dir, 
        #                                                f'{self.name}-{self.mol_repr}-invalid_id.csv')).values.tolist()
        #     if len(invalid_id_list) == 0:
        #         print(f'invalid_id_list is empty')
        #     else:
        #         for id, label in invalid_id_list:
        #             print(f'checking invalid id {id}')
        #             if label == 1:
        #                 print('====warning: a positive label is removed====')
        #             if id in split_dict['train']:
        #                 split_dict['train'].remove(id)
        #                 print(f'found in train and removed')
        #             if id in split_dict['test']:
        #                 split_dict['test'].remove(id)
        #                 print(f'found in test and removed')           
            
        #     print(f'using {split_scheme} split')
        #     split_dict = torch.load(f'data_split/{self.name}_{split_scheme}.pt')
        # except Exception as e:
        #     print(f'Cannot open invalid mol file: {e}')
        
        return split_dict



