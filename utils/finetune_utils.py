import os       # For interacting with the operating system (e.g., changing directories, running shell commands)
import shutil   # For high-level file operations (e.g., copying directories)
import errno    # For handling specific error codes
import os.path as osp



def gitclone():
    back_to_experiment_folder()
    os.system(f'git clone git@github.com:xwang38438/{git_repo_name}.git')
    os.chdir(f'{git_repo_name}')
    os.system(f'git checkout {branch}')
    
    
def gitupdate():
    back_to_experiment_folder()
    os.chdir(f'{git_repo_name}')
    print(f'git pull')
    os.system('git pull')
    print(f'git checkout {branch}')
    os.system(f'git checkout {branch}')
    print(f'git pull')
    os.system('git pull')

def run_command(exp_id, args):
    write_config(args)
    os.system(f'python main.py --config config/{config_file_name} --no_train_eval')
    
    
def copyanything(src, dst):
    '''
    does not overwrite
    return True if created a new one
    return False if folder exist
    '''
    if os.path.exists(dst):
        # shutil.rmtree(dst)
        print(f'{dst} exists and hence do not copy from the template')
        return False
    else:
        try:
            shutil.copytree(src, dst)
        except OSError as exc:  # python >2.5
            if exc.errno in (errno.ENOTDIR, errno.EINVAL):
                shutil.copy(src, dst)
            else:
                raise
        return True
    
def overwrite_dir(src, dst):
    '''
    copy and overwrite
    '''
    # If dst exits, remove it first
    if os.path.exists(dst):
        shutil.rmtree(dst)
        print(f'{dst} exists and overwritten')
    try:
        shutil.copytree(src, dst)
    except OSError as exc:  # python >2.5
        if exc.errno in (errno.ENOTDIR, errno.EINVAL):
            shutil.copy(src, dst)
        else:
            raise

def run(exp_id, *args):
    # Unified Folder name: exp_id, task_name, gnn_type_values , dataset_name, split_scheme, one_hot_features
    # Step 1 Fine-tune Experiment: task_name = FineTune
    # Step 2 All-Split Experiment: task_name = AllSplit

    exp_folder_name = f'exp{exp_id}_{task_name}_{args[5]}'
    print(f'=====running {exp_folder_name}')

    back_to_experiment_folder()
    newly_created = copyanything(git_repo_name, exp_folder_name)
    back_to_experiment_folder()
    os.chdir(f'{exp_folder_name}')

    # Task
    if not osp.exists('result/gcn/test_result.txt'):  # If no results
        if not newly_created:  # If the folder has no result and not newly created (i.e., incomplete run), copy the current template folder and rerun
            back_to_experiment_folder()
            overwrite_dir(git_repo_name, exp_folder_name)
            back_to_experiment_folder()
            os.chdir(f'{exp_folder_name}')

        back_to_experiment_folder()
        os.chdir(f'{exp_folder_name}')
        run_command(exp_id, args)
        # time.sleep(3)
        print(f'----{exp_folder_name} finishes')
    else:
        print(f'----{exp_folder_name} was done previously')

def attach_exp_id(input_tuple, tuple_id):
    # Add experiment id in front of the input hyperparam tuple
    record = [tuple_id]
    record.extend(list(input_tuple))
    return record

