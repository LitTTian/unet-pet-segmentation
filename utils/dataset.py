import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import pandas as pd
import numpy as np
from torchvision import transforms
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode
from .tools import tensor2image
import random
import logging
from multiprocessing import Pool
from tqdm import tqdm

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def denormalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD, returnImage=False, device='cpu'):
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    # img_np = tensor.cpu().numpy().transpose(1, 2, 0)
    # img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
    if returnImage:
        return tensor2image(tensor)
    return tensor.to(device)


def unique_mask_values(mask_file): # 获取mask中所有的类别
    mask = np.asarray(Image.open(mask_file))
    return np.unique(mask)

class OxfordPetDataset(Dataset):
    def __init__(self, image_dir, mask_dir, file_txt, img_size=(256, 256), resized_size=320,
    train=True, dataset_analyse=False, n_samples=0):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.resized_size = resized_size
        self.train = train
        self.file_dict = self._load_file_dict(file_txt, n_samples)
        if dataset_analyse:
            self.dataset_analyse()

    def __len__(self):
        return len(self.file_dict)
    
    def _load_file_dict(self, file_txt, n_samples=0):
        df = pd.read_csv(
            file_txt,
            sep=" ",
            names=["image_name", "class_id", "species", "breed_id"],
            dtype={"image_name": str, "class_id": int, "species": int, "breed_id": int},
        )
        if n_samples > 0:
            df = df.sample(n=n_samples, random_state=42).reset_index(drop=True)
        return df.to_dict(orient="records")
    
    def _apply_sync_transforms(self, image, mask):  # Train时使用几何变换
        # 无拉伸Resize + RandomCrop + RandomHorizontalFlip
        # print("before resize:", image.shape, mask.shape)
        image = F.resize( image, size=self.resized_size, interpolation=InterpolationMode.BILINEAR, antialias=True)
        mask = F.resize( mask, size=self.resized_size, interpolation=InterpolationMode.NEAREST, antialias=False)
        # print("after resize:", image.shape, mask.shape)
        if self.train:
            h, w = self.img_size
            i, j = random.randint(0, image.shape[1] - h), random.randint(0, image.shape[2] - w)
            image = F.crop(image, i, j, h, w)
            mask = F.crop(mask, i, j, h, w)
            if random.random() > 0.5:
                image = F.hflip(image)
                mask = F.hflip(mask)
        else:  # 测试或者验证时使用中心裁剪
            # 保证图像是16的倍数
            if image.shape[1] != mask.shape[1] or image.shape[2] != mask.shape[2]:
                # print("val尺寸不一致", image.shape, mask.shape)
                raise ValueError("Image and mask must have the same dimensions for cropping.")
            _, h_resized, w_resized = image.shape
            h_final = (h_resized // 16) * 16
            w_final = (w_resized // 16) * 16
            i = (h_resized - h_final) // 2
            j = (w_resized - w_final) // 2
            image = F.crop(image, i, j, h_final, w_final)
            mask = F.crop(mask, i, j, h_final, w_final)
        # print("mode: ", self.train, "after crop:", image.shape, mask.shape)
        image = F.normalize(image, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        return image, mask
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, f"{self.file_dict[idx]['image_name']}.jpg")
        mask_path = os.path.join(self.mask_dir, f"{self.file_dict[idx]['image_name']}.png")
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image, mask = self._apply_sync_transforms(F.to_tensor(image), F.to_tensor(mask))
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
