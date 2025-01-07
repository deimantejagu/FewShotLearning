import os
import random
from PIL import Image
from torch.utils.data import Dataset
import torch

class FewShotSegmentationDataset(Dataset):
    def __init__(self, root_dir, split='Training', ways=3, shots=1, queries=1, transform=None):
        """
        Args:
            root_dir (str): Path to the dataset directory.
            split (str): 'Training' or 'Testing'.
            ways (int): Number of classes per episode.
            shots (int): Number of support images per class.
            queries (int): Number of query images per class.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.split = split
        self.ways = ways
        self.shots = shots
        self.queries = queries
        self.transform = transform

        # List all classes
        self.classes = [d for d in os.listdir(os.path.join(root_dir, split, 'images')) 
                       if os.path.isdir(os.path.join(root_dir, split, 'images', d))]
        
        # Create a mapping from class to image files
        self.class_to_images = {cls: os.listdir(os.path.join(root_dir, split, 'images', cls)) 
                                for cls in self.classes}
        
    def __len__(self):
        # Define as per your requirement; for example, number of episodes
        return 1000  # Arbitrary large number
    
    def __getitem__(self, idx):
        # Sample W classes
        sampled_classes = random.sample(self.classes, self.ways)
        
        support_images = []
        support_masks = []
        query_images = []
        query_masks = []
        
        for cls in sampled_classes:
            images = self.class_to_images[cls]
            selected_images = random.sample(images, self.shots + self.queries)
            support_imgs = selected_images[:self.shots]
            query_imgs = selected_images[self.shots:]
            
            for img in support_imgs:
                img_path = os.path.join(self.root_dir, self.split, 'images', cls, img)
                mask_path = os.path.join(self.root_dir, self.split, 'masks', cls, img.replace('.jpg', '.png'))  # Adjust extension if needed
                support_images.append(Image.open(img_path).convert('RGB'))
                support_masks.append(Image.open(mask_path).convert('L'))  # Assuming masks are grayscale
                
            for img in query_imgs:
                img_path = os.path.join(self.root_dir, self.split, 'images', cls, img)
                mask_path = os.path.join(self.root_dir, self.split, 'masks', cls, img.replace('.jpg', '.png'))  # Adjust extension if needed
                query_images.append(Image.open(img_path).convert('RGB'))
                query_masks.append(Image.open(mask_path).convert('L'))  # Assuming masks are grayscale
        
        if self.transform:
            support_images = [self.transform(img) for img in support_images]
            support_masks = [self.transform(mask) for mask in support_masks]
            query_images = [self.transform(img) for img in query_images]
            query_masks = [self.transform(mask) for mask in query_masks]
        
        # Organize data as per model's forward method
        # supp_imgs: way x shot x [B x 3 x H x W], assuming B=1
        supp_imgs = []
        supp_fore_masks = []
        supp_back_masks = []
        for w in range(self.ways):
            supp_imgs.append([support_images[w * self.shots + s].unsqueeze(0) for s in range(self.shots)])
            supp_fore_masks.append([support_masks[w * self.shots + s].unsqueeze(0) for s in range(self.shots)])
            supp_back_masks.append([1 - support_masks[w * self.shots + s].unsqueeze(0) for s in range(self.shots)])
        
        # qry_imgs: N x [B x 3 x H x W]
        qry_imgs = [query_images[w * self.queries + q].unsqueeze(0) for w in range(self.ways) for q in range(self.queries)]
        
        return supp_imgs, supp_fore_masks, supp_back_masks, qry_imgs
