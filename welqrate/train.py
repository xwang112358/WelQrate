from tqdm import tqdm
import numpy as np
import torch
import random
import os
from welqrate.loader import get_train_loader, get_test_loader, get_valid_loader
from welqrate.scheduler import get_scheduler, get_lr
from welqrate.test import get_test_metrics
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
import yaml


def get_train_loss(model, loader, optimizer, scheduler, device, loss_fn):
    
    model.train()
    loss_list = []

    for i, batch in enumerate(tqdm(loader, miniters=100)):
        batch.to(device)
        y_pred = model(batch)
        
        loss= loss_fn(y_pred.view(-1), batch.y.view(-1).float())
            
        loss_list.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    loss = np.mean(loss_list)
    return loss


def train(model, train_loader, valid_loader, test_loader, config, device, train_eval=False, save_path=None):

    # load train info
    batch_size = int(config['train']['batch_size'])
    num_epochs = int(config['train']['num_epochs'])
    num_workers = int(config['general']['num_workers'])
    seed = int(config['general']['seed'])
    weight_decay = float(config['train']['weight_decay'])
    early_stopping_limit = int(config['train']['early_stop'])
    split_scheme = config['data']['split_scheme']
    dataset_name = config['data']['dataset_name']
    model_name = config['model']['model_name']
    
    loss_fn = BCEWithLogitsLoss()
    

    # load optimizer and scheduler
    optimizer = AdamW(model.parameters(), weight_decay=weight_decay)
    scheduler = get_scheduler(optimizer, config, train_loader)
    
    print('\n' + '=' * 10 + f"Training {model} on {dataset_name}'s {split_scheme} split" '\n' + '=' * 10 )
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    base_path = save_path
    model_save_path = os.path.join(base_path, f'{model_name}.pt')
    log_save_path = os.path.join(base_path, f'train.log')
    metrics_save_path = os.path.join(base_path, f'test_results.txt')
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    # save config
    with open(os.path.join(base_path, f'config.yaml'), 'w') as file:
        yaml.dump(config, file)

    
    best_epoch = 0
    best_valid_BEDROC = -1
    early_stopping_counter = 0
    print(f'Training with early stopping limit of {early_stopping_limit} epochs')
    
    with open(log_save_path, 'w+') as out_file:
        for epoch in range(num_epochs):
            
            train_loss = get_train_loss(model, train_loader, optimizer, scheduler, device, loss_fn)
            
            lr = get_lr(optimizer)
            
            if train_eval:
                train_logAUC, train_EF, train_DCG, train_BEDROC = get_test_metrics(model, train_loader, device)
                print(f'current_epoch={epoch} train_loss={train_loss:.4f} lr={lr}')
                print(f'train_logAUC={train_logAUC:.4f} train_EF={train_EF:.4f} train_DCG={train_DCG:.4f} train_BEDROC={train_BEDROC:.4f}')
                out_file.write(f'Epoch:{epoch}\tloss={train_loss}\tlogAUC={train_logAUC}\tEF={train_EF}\tDCG={train_DCG}\tBEDROC={train_BEDROC}\tlr={lr}\t\n')
            
            else:
                print(f'current_epoch={epoch} train_loss={train_loss:.4f} lr={lr}')
                out_file.write(f'Epoch:{epoch}\tloss={train_loss}\tlr={lr}\t\n')
                
            valid_logAUC, valid_EF, valid_DCG, valid_BEDROC = get_test_metrics(model, valid_loader, device, type='valid', save_per_molecule_pred=True, save_path=base_path)  
            print(f'valid_logAUC={valid_logAUC:.4f} valid_EF={valid_EF:.4f} valid_DCG={valid_DCG:.4f} valid_BEDROC={valid_BEDROC:.4f}')
            out_file.write(f'Epoch:{epoch}\tlogAUC={valid_logAUC}\tEF={valid_EF}\tDCG={valid_DCG}\tBEDROC={valid_BEDROC}\t\n')  
            
            if valid_BEDROC > best_valid_BEDROC:
                best_valid_BEDROC = valid_BEDROC
                best_epoch = epoch
                torch.save({'model': model.state_dict(),
                            'epoch': epoch}, model_save_path)
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            if early_stopping_counter >= early_stopping_limit:
                print(f'Early stopping at epoch {epoch}')
                break
        print(f'Training finished')
        print(f'Best epoch: {best_epoch} with valid BEDROC: {best_valid_BEDROC:.4f}')
        
    # testing the model
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path)['model'])
        print(f'Best Model loeaded from {model_save_path}')
    else:
        raise Exception(f'Model not found at {model_save_path}')
   
    print('Testing ...')
    test_logAUC, test_EF100, test_DCG100, test_BEDROC, test_EF500, test_EF1000, test_DCG500, test_DCG1000 = get_test_metrics(model, test_loader, device, 
                                                                   save_per_molecule_pred=True,
                                                                   save_path=base_path, extra_metrics=True)
    
    print(f'{model_name} at epoch {best_epoch} test logAUC: {test_logAUC:.4f} test EF: {test_EF100:.4f} test DCG: {test_DCG100:.4f} test BEDROC: {test_BEDROC:.4f}')
    with open(metrics_save_path, 'w+') as result_file:
        result_file.write(f'logAUC={test_logAUC}\tEF100={test_EF100}\tDCG100={test_DCG100}\tBEDROC={test_BEDROC}\tEF500={test_EF500}\tEF1000={test_EF1000}\tDCG500={test_DCG500}\tDCG1000={test_DCG1000}\n')
    
    return test_logAUC, test_EF100, test_DCG100, test_BEDROC, test_EF500, test_EF1000, test_DCG500, test_DCG1000
    

