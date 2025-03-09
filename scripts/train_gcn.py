from welqrate.dataset import WelQrateDataset
from welqrate.models.gnn2d.GCN import GCN_Model 
import torch
from welqrate.train import train
import yaml
import itertools
import copy
import os
import pandas as pd
import csv
from datetime import datetime
import argparse
import optuna
import sys
from tqdm import tqdm
from welqrate.loader import get_train_loader, get_valid_loader, get_test_loader
from welqrate.experiment_utils.seed_run import run_with_seed, save_summary_statistics


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='AID1798', required=True)
parser.add_argument('--split', type=str, default='random_cv1', required=True)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--early_stop', type=int, default=30)
parser.add_argument('--n_trials', type=int, default=4)
parser.add_argument('--n_jobs', type=int, default=4)
args = parser.parse_args()

# Load base config
base_config = {}
with open('./config/train.yaml') as file:
    train_config = yaml.safe_load(file)
    base_config.update(train_config)
with open('./config/gcn.yaml') as file:
    model_config = yaml.safe_load(file)
    base_config.update(model_config)

# Setup dataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = WelQrateDataset(dataset_name=args.dataset, root='./datasets', mol_repr='2d_graph')

# Get dataset name and split scheme from args
dataset_name = args.dataset
split_scheme = args.split

# Create results directory and CSV file with headers
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = 'results'
csv_file = f'results/gcn_finetuning_{dataset_name}_{split_scheme}_{timestamp}.csv'
os.makedirs(results_dir, exist_ok=True)

# Get dataset splits
split_dict = dataset.get_idx_split(split_scheme)

batch_size = base_config['train']['batch_size']
num_workers = base_config['general']['num_workers']
seed = base_config['general']['seed']

# Create data loaders
train_loader = get_train_loader(dataset[split_dict['train']], batch_size, num_workers, seed)
valid_loader = get_valid_loader(dataset[split_dict['valid']], batch_size, num_workers)
test_loader = get_test_loader(dataset[split_dict['test']], batch_size, num_workers)

# create csv for finetuning results
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'hidden_channels', 'num_layers', 'peak_lr',
        'test_logAUC', 'test_EF', 'test_DCG', 'test_BEDROC',
        'test_EF500', 'test_EF1000', 'test_DCG500', 'test_DCG1000'
    ])

def objective(trial):
    # Define hyperparameter search space
    hidden_channels = trial.suggest_categorical('hidden_channels', [32, 64, 128])
    num_layers = trial.suggest_int('num_layers', 2, 4)
    peak_lr = trial.suggest_float('peak_lr', 1e-4, 1e-2, log=True)
    trial_dir = f"{results_dir}/{dataset_name}/{split_scheme}/gcn/trial{trial.number}"
    os.makedirs(trial_dir, exist_ok=True)

    try:
        # Update config
        config = copy.deepcopy(base_config)
        config['model']['hidden_channels'] = hidden_channels
        config['model']['num_layers'] = num_layers
        config['train']['peak_lr'] = peak_lr
        config['data']['split_scheme'] = args.split
        config['train']['num_epochs'] = args.num_epochs
        config['train']['early_stop'] = args.early_stop

        # Initialize model with current params
        model = GCN_Model(
            in_channels=28,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
        ).to(device)
        
        print(f"\nTrial {trial.number}")
        print(f"Hidden channels: {hidden_channels}")
        print(f"Number of layers: {num_layers}")
        print(f"Peak learning rate: {peak_lr}")
        
        # Train model and get metrics
        test_logAUC, test_EF100, test_DCG100, test_BEDROC, test_EF500, test_EF1000, test_DCG500, test_DCG1000 = train(
            model=model, 
            train_loader=train_loader,
            valid_loader=valid_loader,
            test_loader=test_loader,
            config=config, 
            device=device,
            save_path=trial_dir,
        )
        
        # Save results to CSV
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([hidden_channels, num_layers, peak_lr, 
                            test_logAUC, test_EF100, test_DCG100, test_BEDROC,
                            test_EF500, test_EF1000, test_DCG500, test_DCG1000])
        
        return test_BEDROC
        
    except Exception as e:
        print(f"Error occurred in trial {trial.number}")
        print(f"Error message: {str(e)}")
        # Save error info to CSV
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([hidden_channels, num_layers, peak_lr, 
                            float('-inf'), float('-inf'), float('-inf'), float('-inf'),
                            float('-inf'), float('-inf'), float('-inf'), float('-inf')])
        return float('-inf')  

# Create study object and optimize
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=4, n_jobs=4,
               timeout=16200)  # Adjust n_trials as needed

# Get best parameters
best_params = study.best_params
best_value = study.best_value

print("\nBest parameters found:")
print(f"Hidden channels: {best_params['hidden_channels']}")
print(f"Number of layers: {best_params['num_layers']}")
print(f"Peak learning rate: {best_params['peak_lr']}")
print(f"Best test BEDROC: {best_value:.4f}")

# Initialize model with best params outside the seed loop
best_model = GCN_Model(
    in_channels=28,
    hidden_channels=int(best_params['hidden_channels']),
    num_layers=int(best_params['num_layers']),
).to(device)

# Run with different seeds using best parameters
seeds = [1, 2, 3]
seed_results = []

for seed in seeds:
    result = run_with_seed(
        seed=seed, 
        model=best_model, 
        best_params=best_params, 
        base_config=base_config, 
        args=args,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        device=device, 
        results_dir=results_dir, 
        csv_file=csv_file
    )
    if result:
        seed_results.append(result)

# Calculate and save summary statistics
save_summary_statistics(seed_results, results_dir, dataset_name, split_scheme, timestamp)







