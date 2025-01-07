"""
Training Script
"""
import os
import shutil

import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import torch.backends.cudnn as cudnn
from torchvision.transforms import Compose

# --- Models & Dataloaders ---
from models.fewshot import FewShotSeg
from dataloaders.customized import custom_fewshot
from dataloaders.transforms import RandomMirror, Resize, ToTensorNormalize

# --- Utility ---
from util.utils import set_seed, CLASS_LABELS
from config import ex


@ex.automain
def main(_run, _config, _log):
    if _run.observers:
        os.makedirs(f'{_run.observers[0].dir}/snapshots', exist_ok=True)
        for source_file, _ in _run.experiment_info['sources']:
            os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                        exist_ok=True)
            _run.observers[0].save_file(source_file, f'source/{source_file}')
        shutil.rmtree(f'{_run.observers[0].basedir}/_sources')

    # ---------------------
    # 0) Basic setup
    # ---------------------
    set_seed(_config['seed'])
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.cuda.set_device(device=_config['gpu_id'])
    torch.set_num_threads(1)

    _log.info('###### Create model ######')
    model = FewShotSeg(pretrained_path=_config['path']['init_path'], cfg=_config['model'])
    model = nn.DataParallel(model.cuda(), device_ids=[_config['gpu_id']])
    model.train()

    # ---------------------
    # [Optional] Resume from previous checkpoint if provided
    # ---------------------
    resume_path = _config['resume_path']
    if os.path.isfile(resume_path):
        print(f'###### Loading checkpoint: {resume_path} ######')
        state_dict = torch.load(resume_path, map_location='cpu')
        model.load_state_dict(state_dict)  # or partial load if needed
        # If you also saved the optimizer, load that here too.
    else:
        print(f'Resume path does not exist: {resume_path}')

    print('###### Load data ######')
    data_name = _config['dataset']
    if data_name == 'Custom':
        make_data = custom_fewshot
    else:
        raise ValueError('Wrong config for dataset!')

    labels = CLASS_LABELS[data_name][_config['label_sets']]

    transforms = Compose([Resize(size=_config['input_size']),
                          RandomMirror()])

    dataset = make_data(
        base_dir=_config['path'][data_name]['data_dir'],
        split=_config['path'][data_name]['data_split'],
        transforms=transforms,
        to_tensor=ToTensorNormalize(),
        labels=labels,
        max_iters=_config['n_steps'] * _config['batch_size'],
        n_ways=_config['task']['n_ways'],
        n_shots=_config['task']['n_shots'],
        n_queries=_config['task']['n_queries']
    )

    trainloader = DataLoader(
        dataset,
        batch_size=_config['batch_size'],
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        drop_last=True
    )

    _log.info('###### Set optimizer ######')
    optimizer = torch.optim.SGD(model.parameters(), **_config['optim'])
    scheduler = MultiStepLR(optimizer, milestones=_config['lr_milestones'], gamma=0.1)
    criterion = nn.CrossEntropyLoss(ignore_index=_config['ignore_label'])

    i_iter = 0
    log_loss = {'loss': 0, 'align_loss': 0}
    _log.info('###### Training ######')
    for i_iter, sample_batched in enumerate(trainloader):
        # Prepare input
        support_images = [[shot.cuda() for shot in way]
                          for way in sample_batched['support_images']]
        support_fg_mask = [[shot['fg_mask'].float().cuda() for shot in way]
                           for way in sample_batched['support_mask']]
        support_bg_mask = [[shot['bg_mask'].float().cuda() for shot in way]
                           for way in sample_batched['support_mask']]

        query_images = [q_img.cuda() for q_img in sample_batched['query_images']]
        query_labels = torch.cat(
            [lbl.long().cuda() for lbl in sample_batched['query_labels']], dim=0
        )

        # Forward + Backward
        optimizer.zero_grad()
        query_pred, align_loss = model(support_images, support_fg_mask, support_bg_mask, query_images)
        query_loss = criterion(query_pred, query_labels)
        loss = query_loss + align_loss * _config['align_loss_scaler']
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Log losses
        query_loss_val = query_loss.detach().cpu().item()
        align_loss_val = align_loss.detach().cpu().item() if align_loss != 0 else 0
        _run.log_scalar('loss', query_loss_val)
        _run.log_scalar('align_loss', align_loss_val)
        log_loss['loss'] += query_loss_val
        log_loss['align_loss'] += align_loss_val

        # Print interval
        if (i_iter + 1) % _config['print_interval'] == 0:
            avg_loss = log_loss['loss'] / (i_iter + 1)
            avg_align_loss = log_loss['align_loss'] / (i_iter + 1)
            print(f'step {i_iter+1}: loss: {avg_loss:.4f}, align_loss: {avg_align_loss:.4f}')

        # Save snapshots
        if (i_iter + 1) % _config['save_pred_every'] == 0:
            _log.info('###### Taking snapshot ######')
            save_path = os.path.join(f'{_run.observers[0].dir}/snapshots', f'{i_iter + 1}.pth')
            torch.save(model.state_dict(), save_path)

    _log.info('###### Saving final model ######')
    save_path = os.path.join(f'{_run.observers[0].dir}/snapshots', f'{i_iter + 1}.pth')
    torch.save(model.state_dict(), save_path)
