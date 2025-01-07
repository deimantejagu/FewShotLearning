"""
Evaluation Script
"""
import os
import shutil

import tqdm
import numpy as np
import torch
import torch.optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torchvision.transforms import Compose

from models.fewshot import FewShotSeg
from dataloaders.customized import custom_fewshot
from dataloaders.transforms import ToTensorNormalize
from dataloaders.transforms import Resize, DilateScribble
from util.metric import Metric
from util.utils import set_seed, CLASS_LABELS, get_bbox
from config import ex


@ex.automain
def main(_run, _config, _log):
    for source_file, _ in _run.experiment_info['sources']:
        os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                    exist_ok=True)
        _run.observers[0].save_file(source_file, f'source/{source_file}')
    shutil.rmtree(f'{_run.observers[0].basedir}/_sources')

    set_seed(_config['seed'])
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.cuda.set_device(device=_config['gpu_id'])
    torch.set_num_threads(1)

    _log.info('###### Create model ######')
    model = FewShotSeg(pretrained_path=_config['path']['init_path'], cfg=_config['model'])
    model = nn.DataParallel(model.cuda(), device_ids=[_config['gpu_id'], ])
    if not _config['notrain']:
        model.load_state_dict(torch.load(_config['snapshot'], map_location='cpu'))
    model.eval()

    _log.info('###### Prepare data ######')
    data_name = _config['dataset']
    if data_name == 'Custom':
        make_data = custom_fewshot
        max_label = 3
    else:
        raise ValueError('Wrong config for dataset!')

    # For your custom classes, you might replace these with actual IDs or names
    labels = [1, 2, 3]  

    transforms = [Resize(size=_config['input_size'])]
    if _config['scribble_dilation'] > 0:
        transforms.append(DilateScribble(size=_config['scribble_dilation']))
    transforms = Compose(transforms)

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
    testloader = DataLoader(dataset, batch_size=_config['batch_size'], shuffle=False,
                            num_workers=1, pin_memory=True, drop_last=False)

    # -------------------------
    # Metric calculation helpers
    # -------------------------
    def calculate_iou(pred, target, num_classes):
        """ Return IoU for each foreground class index [1..num_classes]. """
        ious = []
        for cls in range(1, num_classes + 1):  # skip cls=0 (background)
            pred_inds = (pred == cls)
            target_inds = (target == cls)
            intersection = (pred_inds & target_inds).sum()
            union = (pred_inds | target_inds).sum()
            iou = intersection / (union + 1e-8)
            ious.append(iou)
        return ious  # length == num_classes

    def calculate_dice(pred, target, num_classes):
        """ Return Dice for each foreground class index [1..num_classes]. """
        dice_scores = []
        for cls in range(1, num_classes + 1):
            pred_inds = (pred == cls)
            target_inds = (target == cls)
            intersection = (pred_inds & target_inds).sum()
            dice = (2.0 * intersection) / (pred_inds.sum() + target_inds.sum() + 1e-8)
            dice_scores.append(dice)
        return dice_scores

    def calculate_sensitivity(pred, target, num_classes):
        """ Return sensitivity (recall) for each class index [1..num_classes]. """
        sensitivities = []
        for cls in range(1, num_classes + 1):
            pred_inds = (pred == cls)
            target_inds = (target == cls)
            true_positive = (pred_inds & target_inds).sum()
            false_negative = (~pred_inds & target_inds).sum()
            sens = true_positive / (true_positive + false_negative + 1e-8)
            sensitivities.append(sens)
        return sensitivities

    def calculate_specificity(pred, target, num_classes):
        """ Return specificity for each class index [1..num_classes]. """
        specificities = []
        for cls in range(1, num_classes + 1):
            pred_inds = (pred == cls)
            target_inds = (target == cls)
            true_negative = (~pred_inds & ~target_inds).sum()
            false_positive = (pred_inds & ~target_inds).sum()
            spec = true_negative / (true_negative + false_positive + 1e-8)
            specificities.append(spec)
        return specificities

    # We store each metric per class in a list-of-lists, index = class-1
    all_ious = [[] for _ in range(max_label)]          # for classes 1..max_label
    all_dices = [[] for _ in range(max_label)]
    all_sensitivities = [[] for _ in range(max_label)]
    all_specificities = [[] for _ in range(max_label)]

    _log.info('###### Testing begins ######')
    with torch.no_grad():
        for run in range(_config['n_runs']):
            _log.info(f'### Run {run + 1} ###')
            set_seed(_config['seed'] + run)

            for sample_batched in tqdm.tqdm(testloader):
                # --------------------------------
                # 1) Prepare data from batch
                # --------------------------------
                # Each element in 'support_mask' is typically a nested list of dicts,
                # if you are using the standard approach. Adjust as needed.
                support_images = [[shot.cuda() for shot in way]
                                  for way in sample_batched['support_images']]
                support_fg_mask = [[shot['fg_mask'].float().cuda() for shot in way]
                                  for way in sample_batched['support_mask']]
                support_bg_mask = [[shot['bg_mask'].float().cuda() for shot in way]
                                  for way in sample_batched['support_mask']]
                query_images = [query_image.cuda() for query_image in sample_batched['query_images']]
                query_labels = torch.stack(sample_batched['query_labels'], dim=0).cuda()  # shape [B, H, W] if stacked

                # --------------------------------
                # 2) Forward Pass
                # --------------------------------
                query_pred, _ = model(support_images, support_fg_mask, support_bg_mask, query_images)

                # --------------------------------
                # 3) Convert to numpy for metric calculation
                # --------------------------------
                pred_labels = query_pred.argmax(dim=1).cpu().numpy()
                true_labels = query_labels.cpu().numpy()

                # --------------------------------
                # 4) Metric Calculations, per sample in batch
                # --------------------------------
                for b_idx in range(pred_labels.shape[0]):
                    pred_b = pred_labels[b_idx]
                    true_b = true_labels[b_idx]

                    ious_b = calculate_iou(pred_b, true_b, max_label)
                    dices_b = calculate_dice(pred_b, true_b, max_label)
                    sens_b = calculate_sensitivity(pred_b, true_b, max_label)
                    spec_b = calculate_specificity(pred_b, true_b, max_label)

                    # ious_b, dices_b, etc. each is length == max_label (for classes 1..max_label)
                    # Save each metric to the appropriate list
                    for cls_idx in range(max_label):
                        all_ious[cls_idx].append(ious_b[cls_idx])
                        all_dices[cls_idx].append(dices_b[cls_idx])
                        all_sensitivities[cls_idx].append(sens_b[cls_idx])
                        all_specificities[cls_idx].append(spec_b[cls_idx])

    # -------------------------
    # Final metric logging
    # -------------------------
    # We'll compute the mean for each class (1..max_label) and also a macro-average
    per_class_iou = []
    per_class_dice = []
    per_class_sens = []
    per_class_spec = []

    for cls in range(max_label):
        mean_iou_cls = np.mean(all_ious[cls]) if len(all_ious[cls]) > 0 else 0
        mean_dice_cls = np.mean(all_dices[cls]) if len(all_dices[cls]) > 0 else 0
        mean_sens_cls = np.mean(all_sensitivities[cls]) if len(all_sensitivities[cls]) > 0 else 0
        mean_spec_cls = np.mean(all_specificities[cls]) if len(all_specificities[cls]) > 0 else 0

        print(f"Class {cls}: IoU={mean_iou_cls:.4f}, Dice={mean_dice_cls:.4f}, "
                  f"Sens={mean_sens_cls:.4f}, Spec={mean_spec_cls:.4f}")

        # If you want the macro-average to exclude class 0,
        # only append if cls>0
        if cls > 0:
            per_class_iou.append(mean_iou_cls)
            per_class_dice.append(mean_dice_cls)
            per_class_sens.append(mean_sens_cls)
            per_class_spec.append(mean_spec_cls)

    # Macro-average across the foreground classes
    mean_iou = np.mean(per_class_iou) if len(per_class_iou) > 0 else 0
    mean_dice = np.mean(per_class_dice) if len(per_class_dice) > 0 else 0
    mean_sensitivity = np.mean(per_class_sens) if len(per_class_sens) > 0 else 0
    mean_specificity = np.mean(per_class_spec) if len(per_class_spec) > 0 else 0

    print(f"### Final Macro-Averages (Excl. BG) ###")
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Mean Dice: {mean_dice:.4f}")
    print(f"Mean Sensitivity: {mean_sensitivity:.4f}")
    print(f"Mean Specificity: {mean_specificity:.4f}")
