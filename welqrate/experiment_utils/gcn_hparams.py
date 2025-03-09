



def gcn_objective(trial):
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
        
        # Initialize model with current params
        model = GCN_Model(
            in_channels=12,
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