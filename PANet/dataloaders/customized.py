import os
import random

import torch
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CustomDataset(Dataset):
    """
    Custom dataset to load images and corresponding masks (single-class or multi-class).
    """
    def __init__(self, mode, dataset_root, task, fold, resize_to=None):
        self.mode = mode
        self.dataset_root = dataset_root
        self.task = task
        self.fold = str(fold)
        self.resize_to = resize_to

        # Example folder structure:
        #  dataset_root / task / fold / "Training" / "images"
        #                                    ...     / "masks"
        #  or "Testing" if mode is test/val
        self.root = os.path.join(self.dataset_root, self.task, self.fold)
        sub_dir = "Training" if 'train' in self.mode.lower() else "Testing"
        self.images_dir = os.path.join(self.root, sub_dir, "images")
        self.masks_dir  = os.path.join(self.root, sub_dir, "masks")

        self.imgs = self._make_dataset()

    def _make_dataset(self):
        """
        Create a dataset by pairing images and masks.
        """
        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"Images directory does not exist: {self.images_dir}")
        if not os.path.exists(self.masks_dir):
            raise FileNotFoundError(f"Masks directory does not exist: {self.masks_dir}")

        img_files = sorted(os.listdir(self.images_dir))
        mask_files = sorted(os.listdir(self.masks_dir))

        if len(img_files) != len(mask_files):
            raise ValueError("Number of images and masks do not match.")

        data_list = []
        for img_file, mask_file in zip(img_files, mask_files):
            img_path = os.path.join(self.images_dir, img_file)
            mask_path = os.path.join(self.masks_dir, mask_file)
            if os.path.exists(img_path) and os.path.exists(mask_path):
                data_list.append((img_path, mask_path))
        return data_list

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path, mask_path = self.imgs[idx]

        # --- Load image in RGB --- #
        image = Image.open(img_path).convert('RGB')
        # --- Load mask in L (8-bit, single-channel) --- #
        mask = Image.open(mask_path).convert('L')

        # --- Optional resizing --- #
        if self.resize_to:
            image = image.resize(self.resize_to, Image.BILINEAR)
            mask  = mask.resize(self.resize_to, Image.NEAREST)

        # --- Convert to torch Tensor --- #
        # image: [3, H, W], mask: [H, W]
        image = transforms.ToTensor()(image)
        mask  = torch.tensor(np.array(mask), dtype=torch.long)

        # For debugging, you can print unique classes:
        # unique_classes = torch.unique(mask).tolist()
        # print(f"Mask {mask_path} unique classes: {unique_classes}")

        return image, mask


class PairedDataset(Dataset):
    """
    PairedDataset for few-shot segmentation (PANet style),
    returning an 'episode' of data containing support & query examples
    across n_ways classes.
    """
    def __init__(self, dataset, n_ways, n_shots, n_queries, max_iters):
        """
        dataset   : an instance of CustomDataset (or similar)
        n_ways    : how many classes per episode
        n_shots   : number of support samples per class
        n_queries : number of query samples per class
        max_iters : how many episodes to iterate over
        """
        self.dataset   = dataset
        self.n_ways    = n_ways
        self.n_shots   = n_shots
        self.n_queries = n_queries
        self.max_iters = max_iters

        # Build a {class_label: [list_of_sample_indices]} mapping
        self.class_to_indices = self._map_classes()

    def _map_classes(self):
        """
        For each sample in 'dataset', read the mask, find the unique
        class IDs, and store the index in a dictionary.
        """
        class_to_indices = {}
        # We iterate over the entire dataset, which returns (image, mask)
        for idx in range(len(self.dataset)):
            _, mask = self.dataset[idx]
            unique_classes = torch.unique(mask).tolist()
            for cls in unique_classes:
                # 255 is often "ignore" in segmentation
                if cls == 255:
                    continue
                if cls not in class_to_indices:
                    class_to_indices[cls] = []
                class_to_indices[cls].append(idx)
        return class_to_indices

    def __len__(self):
        return self.max_iters

    def __getitem__(self, idx):
        available_classes = [
            cls for cls, indices in self.class_to_indices.items()
            if len(indices) >= (self.n_shots + self.n_queries)
        ]
        if len(available_classes) < self.n_ways:
            raise ValueError("Not enough classes...")

        sampled_classes = random.sample(available_classes, self.n_ways)

        support_images = [[] for _ in range(self.n_ways)]
        support_mask   = [[] for _ in range(self.n_ways)]  # singular key: 'support_mask'
        query_images   = [[] for _ in range(self.n_ways)]
        query_labels   = [[] for _ in range(self.n_ways)]  # rename 'query_masks' -> 'query_labels'

        for way_idx, cls in enumerate(sampled_classes):
            indices_for_cls = self.class_to_indices[cls]
            sampled_indices = random.sample(indices_for_cls, self.n_shots + self.n_queries)

            support_indices = sampled_indices[:self.n_shots]
            query_indices   = sampled_indices[self.n_shots:]

            # Gather support
            for s_idx in support_indices:
                img, raw_mask = self.dataset[s_idx]
                # Convert raw_mask -> dict with fg/bg
                mask_dict = {
                    'fg_mask': (raw_mask == cls).long(),
                    'bg_mask': ((raw_mask != cls) & (raw_mask != 255)).long()
                }
                support_images[way_idx].append(img)
                support_mask[way_idx].append(mask_dict)

            # Gather query
            for q_idx in query_indices:
                img, raw_mask = self.dataset[q_idx]
                # For queries, your training script expects `'query_labels'` 
                # to be a list of 2D label Tensors (not fg/bg dict).
                # If that’s the case, just rename it to "label" or "raw_mask" 
                # to be consistent with the script’s logic:
                query_images[way_idx].append(img)
                query_labels[way_idx].append(raw_mask)
                #   or, if you want to unify naming, call it query_labels

        # Flatten query_images + query_labels from shape [n_ways][n_queries] 
        # into a single list (because the training script does a flat iteration):
        flat_query_images = []
        flat_query_labels = []
        for way_idx in range(self.n_ways):
            flat_query_images.extend(query_images[way_idx])   # list of image Tensors
            flat_query_labels.extend(query_labels[way_idx])   # list of mask Tensors

        return {
            'support_images': support_images,   # nested list [n_ways][n_shots]
            'support_mask':  support_mask,      # nested list of dicts [way][shot]['fg_mask','bg_mask']
            'query_images':  flat_query_images, # flat list
            'query_labels':  flat_query_labels  # flat list
        }

# Import your classes
def custom_fewshot(
    base_dir,
    split,
    transforms,
    to_tensor,
    labels,
    max_iters,
    n_ways,
    n_shots,
    n_queries=1
):
    """
    Build a few-shot dataset from your custom images & masks,
    returning a PairedDataset that yields episodes for PANet training.

    The arguments match your training script exactly:
      base_dir, split, transforms, to_tensor, labels, max_iters, n_ways, n_shots, n_queries
    """

    # 1) Determine "mode" for CustomDataset from 'split' 
    #    e.g. if split contains 'train', set mode='train'. Otherwise 'test'.
    if 'train' in split.lower():
        mode = 'train'
    else:
        mode = 'test'

    # 2) Optionally pick a default task/fold. If you want to read them from config,
    #    add them to your function signature. For now, let's assume "task='DefaultTask', fold='0'".
    dataset_root = base_dir      # same as 'dataset_root' in your CustomDataset 
    task         = 'brains' # or read from config
    fold         = '0'           # or read from config

    # 3) Create your "base" dataset
    #    Notice we ignore 'transforms' and 'to_tensor' 
    #    because your CustomDataset already does resizing & ToTensor internally.
    base_dataset = CustomDataset(
        mode=mode,
        dataset_root=dataset_root,
        task=task,
        fold=fold,
        resize_to=None  # or (256,256), or any size if you want
    )

    # If you do want to apply additional transforms like RandomMirror or normalization,
    # you'd do them manually in __getitem__, or adapt the dataset to use them.
    # For a minimal fix, we skip that to avoid argument mismatch.

    # 4) Wrap it in a PairedDataset to produce few-shot episodes
    fewshot_dataset = PairedDataset(
        dataset=base_dataset,
        n_ways=n_ways,
        n_shots=n_shots,
        n_queries=n_queries,
        max_iters=max_iters
    )

    return fewshot_dataset
