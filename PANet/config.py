import os
import re
import glob
import itertools

import sacred
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds

sacred.SETTINGS['CONFIG']['READ_ONLY_CONFIG'] = False
sacred.SETTINGS.CAPTURE_MODE = 'no'

ex = Experiment('PANet')
ex.captured_out_filter = apply_backspaces_and_linefeeds

source_folders = ['.', './dataloaders', './models', './util']
sources_to_save = list(itertools.chain.from_iterable(
    [glob.glob(f'{folder}/*.py') for folder in source_folders]))
for source_file in sources_to_save:
    ex.add_source_file(source_file)

@ex.config
def cfg():
    """Default configurations"""
    input_size = (256, 256)  # Pritaikykite pagal poreikį
    seed = 1234
    cuda_visable = '0'
    gpu_id = 0
    mode = 'train'  # 'train' arba 'test'

    if mode == 'train':
        resume_path = '/content/drive/MyDrive/Colab Notebooks/Kursinis/PANet/runs/PANet_Custom_sets_0_3way_5shot_[train]/2/snapshots/5000.pth'
        dataset = 'Custom'  # 'VOC', 'COCO' arba 'Custom'
        n_steps = 5000
        label_sets = 0
        batch_size = 8
        lr_milestones = [1000]
        align_loss_scaler = 1.0
        ignore_label = 255
        print_interval = 100
        save_pred_every = 1000

        model = {
            'align': True,
        }
        task = {
            'n_ways': 3,         # 3 klasės
            'n_shots': 2,        # 2 palaikymo pavyzdžiai per klasę
            'n_queries': 1,      # 1 užklausos pavyzdys per klasę
            'name': 'brains',    # Užduoties pavadinimas
            'fold': 0,           # Fold numeris
            'max_iters': 1000,
        }
        optim = {
            'lr': 1e-2,
            'momentum': 0.9,
            'weight_decay': 0.0005,
        }

    elif mode == 'test':
        notrain = False
        snapshot = '/content/drive/MyDrive/Colab Notebooks/Kursinis/PANet/runs/PANet_Custom_sets_0_3way_5shot_[train]/2/snapshots/5000.pth'
        n_runs = 1
        n_steps = 1000
        batch_size = 1
        scribble_dilation = 0
        bbox = False
        scribble = False

        # Nustatyti dataset konfigūraciją iš snapshot string
        if 'VOC' in snapshot:
            dataset = 'VOC'
        elif 'COCO' in snapshot:
            dataset = 'COCO'
        elif 'Custom' in snapshot:
            dataset = 'Custom'
        else:
            raise ValueError('Netinkamas snapshot pavadinimas!')

        # Nustatyti modelio konfigūraciją iš snapshot string
        model = {}
        for key in ['align',]:
            model[key] = key in snapshot

        # Nustatyti label_sets iš snapshot string
        label_sets = int(re.search(r"_sets_(\d+)", snapshot).group(1))

        # Nustatyti task konfigūraciją iš snapshot string
        task = {
            'n_ways': int(re.search(r"(\d+)way", snapshot).group(1)),
            'n_shots': int(re.search(r"(\d+)shot", snapshot).group(1)),
            'n_queries': 1,
        }

    else:
        raise ValueError('Netinkama konfigūracija "mode" reikšmei!')

    exp_str = '_'.join(
        [dataset,]
        + [key for key, value in model.items() if value]
        + [f'sets_{label_sets}', f'{task["n_ways"]}way_{task["n_shots"]}shot_[{mode}]']
    )

    path = {
        'log_dir': './runs',
        'init_path': './pretrained_model/vgg16-397923af.pth',
        'VOC':{
            'data_dir': '../../data/Pascal/VOCdevkit/VOC2012/',
            'data_split': 'trainaug',
        },
        'COCO':{
            'data_dir': '../../data/COCO/',
            'data_split': 'train',
        },
        'Custom':{
            'data_dir': '/content/drive/MyDrive/Colab Notebooks/Kursinis/PANet/dataset',
            'data_split': 'Testing',
        }
    }

@ex.config_hook
def add_observer(config, command_name, logger):
    """Pridėti observer"""
    exp_name = f'{ex.path}_{config["exp_str"]}'
    if config['mode'] == 'test':
        if config.get('notrain', False):
            exp_name += '_notrain'
        if config.get('scribble', False):
            exp_name += '_scribble'
        if config.get('bbox', False):
            exp_name += '_bbox'
    observer = FileStorageObserver.create(os.path.join(config['path']['log_dir'], exp_name))
    ex.observers.append(observer)
    return config
