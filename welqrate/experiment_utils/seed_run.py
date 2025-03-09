import pandas as pd
import os
import csv
import copy
import torch
from welqrate.train import train


def save_summary_statistics(seed_results, results_dir, dataset_name, split_scheme, timestamp):
    """
    Calculate and save summary statistics across multiple seeds.
    
    Args:
        seed_results: List of dictionaries containing results from different seeds
        results_dir: Directory to save results
        dataset_name: Name of the dataset
        split_scheme: Split scheme used
        timestamp: Timestamp for file naming
    """
    if not seed_results:
        print("No valid seed results to summarize")
        return
        
    seed_df = pd.DataFrame(seed_results)
    print("\nResults across seeds:")
    # Print summary statistics
    print(f"Mean test logAUC: {seed_df['test_logAUC'].mean():.4f} ± {seed_df['test_logAUC'].std():.4f}")
    print(f"Mean test EF100: {seed_df['test_EF100'].mean():.4f} ± {seed_df['test_EF100'].std():.4f}")
    print(f"Mean test DCG100: {seed_df['test_DCG100'].mean():.4f} ± {seed_df['test_DCG100'].std():.4f}")
    print(f"Mean test BEDROC: {seed_df['test_BEDROC'].mean():.4f} ± {seed_df['test_BEDROC'].std():.4f}")
    print(f"Mean test EF500: {seed_df['test_EF500'].mean():.4f} ± {seed_df['test_EF500'].std():.4f}")
    print(f"Mean test EF1000: {seed_df['test_EF1000'].mean():.4f} ± {seed_df['test_EF1000'].std():.4f}")
    print(f"Mean test DCG500: {seed_df['test_DCG500'].mean():.4f} ± {seed_df['test_DCG500'].std():.4f}")
    print(f"Mean test DCG1000: {seed_df['test_DCG1000'].mean():.4f} ± {seed_df['test_DCG1000'].std():.4f}")

    # Save summary statistics to CSV
    summary_stats = {
        'Metric': ['logAUC', 'EF100', 'DCG100', 'BEDROC', 'EF500', 'EF1000', 'DCG500', 'DCG1000'],
        'Mean': [
            seed_df['test_logAUC'].mean(),
            seed_df['test_EF100'].mean(),
            seed_df['test_DCG100'].mean(),
            seed_df['test_BEDROC'].mean(),
            seed_df['test_EF500'].mean(),
            seed_df['test_EF1000'].mean(),
            seed_df['test_DCG500'].mean(),
            seed_df['test_DCG1000'].mean()
        ],
        'Std': [
            seed_df['test_logAUC'].std(),
            seed_df['test_EF100'].std(),
            seed_df['test_DCG100'].std(),
            seed_df['test_BEDROC'].std(),
            seed_df['test_EF500'].std(),
            seed_df['test_EF1000'].std(),
            seed_df['test_DCG500'].std(),
            seed_df['test_DCG1000'].std()
        ]
    }
    summary_df = pd.DataFrame(summary_stats)
    summary_csv = f'{results_dir}/gcn_summary_stats_{dataset_name}_{split_scheme}_{timestamp}.csv'
    summary_df.to_csv(summary_csv, index=False)


def run_with_seed(seed, model, best_params, base_config, args, train_loader, valid_loader, test_loader, device, results_dir, csv_file):
    print(f"\nRunning with seed {seed}")
    config = copy.deepcopy(base_config)
    config['general']['seed'] = seed
    config['train']['peak_lr'] = best_params['peak_lr']
    config['data']['split_scheme'] = args.split
    config['model']['hidden_channels'] = best_params['hidden_channels']
    config['model']['num_layers'] = best_params['num_layers']
    config['train']['num_epochs'] = args.num_epochs

    final_results_dir = f'{results_dir}/{args.dataset}/{args.split}/gcn/seed{seed}'
    os.makedirs(final_results_dir, exist_ok=True)
    
    try:
        # Train model and get metrics
        test_logAUC, test_EF100, test_DCG100, test_BEDROC, test_EF500, test_EF1000, test_DCG500, test_DCG1000 = train(
            model=model, 
            train_loader=train_loader,
            valid_loader=valid_loader,
            test_loader=test_loader,
            config=config, 
            device=device,
            save_path=final_results_dir,
        )
        
        result = {
            'seed': seed,
            'test_logAUC': test_logAUC,
            'test_EF100': test_EF100, 
            'test_DCG100': test_DCG100,
            'test_BEDROC': test_BEDROC,
            'test_EF500': test_EF500,
            'test_EF1000': test_EF1000,
            'test_DCG500': test_DCG500,
            'test_DCG1000': test_DCG1000
        }

        # Save seed results to CSV
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                best_params['hidden_channels'],
                best_params['num_layers'],
                best_params['peak_lr'],
                test_logAUC,
                test_EF100, 
                test_DCG100,
                test_BEDROC,
                test_EF500,
                test_EF1000,
                test_DCG500,
                test_DCG1000,
                seed
            ])
        
        return result

    except Exception as e:
        print(f"Error occurred with seed {seed}")
        print(f"Error message: {str(e)}")
        return None


