import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import pandas as pd
import numpy as np
from torchvision import transforms
from torchvision.transforms import functional as F
import random
import logging
from multiprocessing import Pool
from tqdm import tqdm

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def denormalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    img_np = tensor.cpu().numpy().transpose(1, 2, 0)
    img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
    return img_np

def unique_mask_values(mask_file): # 获取mask中所有的类别
    mask = np.asarray(Image.open(mask_file))
    return np.unique(mask)

class OxfordPetDataset(Dataset):
    def __init__(self, image_dir, mask_dir, file_txt, img_size=256, train=True, dataset_analyse=False):
        # self.images = images
        # self.masks = masks
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.train = train
        self.file_dict = self._load_file_dict(file_txt)
        self.image_transform, self.mask_transform = self._get_transforms()
        if dataset_analyse:
            self.dataset_analyse()

    def __len__(self):
        return len(self.file_dict)
    
    def _load_file_dict(self, file_txt):
        df = pd.read_csv(
            file_txt,
            sep=" ",
            names=["image_name", "class_id", "species", "breed_id"],
            dtype={"image_name": str, "class_id": int, "species": int, "breed_id": int},
        )
        return df.to_dict(orient="records")

    def _get_transforms(self):
        image_base = [
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        mask_base = [
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
        ]
        image_transform = transforms.Compose(image_base)
        mask_transform = transforms.Compose(mask_base)
        return image_transform, mask_transform
    
    def _apply_sync_transforms(self, image, mask):  # Train时使用几何变换
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            image, scale=(0.8, 1.0), ratio=(0.9, 1.1)
        )
        image = F.crop(image, i, j, h, w)
        mask = F.crop(mask, i, j, h, w)
        if random.random() > 0.5:
            image = F.hflip(image)
            mask = F.hflip(mask)
        return image, mask
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, f"{self.file_dict[idx]['image_name']}.jpg")
        mask_path = os.path.join(self.mask_dir, f"{self.file_dict[idx]['image_name']}.png")
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.train:
            image, mask = self._apply_sync_transforms(image, mask)  # 确保image和mask做相同的几何变换
        image = self.image_transform(image)
        mask = self.mask_transform(mask)
        mask = torch.squeeze(mask)
        mask = torch.where((mask == 1/255), 1.0, 0.0)
        mask = mask.to(torch.float32)
        return image, mask
    
    
    
    def dataset_analyse(self):
        # logging.info(f'Creating dataset with {self.__len__()} examples')
        # logging.info('Scanning mask files to determine unique values')
        print('Creating dataset with {} examples'.format(self.__len__()))
        print('Scanning mask files to determine unique values')
        # mask_files = [join(self.mask_dir, img_name) for img_name in self.img_names]
        mask_files = [
            os.path.join(self.mask_dir, f"{item['image_name']}.png")
            for item in self.file_dict
        ]
        
        with Pool() as p: # 多进程读取mask文件
            unique_values = list(tqdm(
                p.imap(unique_mask_values, mask_files),
                total=len(mask_files)
            ))
        
        unique = np.unique(np.concatenate(unique_values))
        # self.mask_values = self.map[unique].tolist()
        self.mask_values = unique.tolist()  # [1, 2, 3]

        # logging.info(f'Unique mask values: {self.mask_values}')
        print('Unique mask values: {}'.format(self.mask_values))
