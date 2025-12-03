import logging
import torch
from torch.utils.data import DataLoader, random_split
from utils.dataset import OxfordPetDataset
from model.unet import UNet
from train import train_one_epoch, validate
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('logs/train.log'), logging.StreamHandler()]
)
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-4
IMG_SIZE = 256
IMAGE_DIR = 'C:\\Users\\Alvis\\Study\\datasets\\image\\Oxford-IIIT_Pet_Dataset\\images\\'
MASK_DIR = 'C:\\Users\\Alvis\\Study\\datasets\\image\\Oxford-IIIT_Pet_Dataset\\annotations\\trimaps\\'
TRAIN_VAL_TXT = 'C:\\Users\\Alvis\\Study\\datasets\\image\\Oxford-IIIT_Pet_Dataset\\annotations\\trainval.txt'
TRAIN_VAL_SPLIT = 0.8
CHECKPOINT_DIR = './checkpoints/'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs('logs', exist_ok=True)

if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f'Using device: {device}')
    traain_val_dataset = OxfordPetDataset(
        image_dir=IMAGE_DIR,
        mask_dir=MASK_DIR,
        file_txt=TRAIN_VAL_TXT,
        img_size=IMG_SIZE,
        train=True,
        # n_samples=20,  # DEBUG: 测试使用较小的数据集
    )
    train_size = int(TRAIN_VAL_SPLIT * len(traain_val_dataset))
    val_size = len(traain_val_dataset) - train_size
    train_dataset, val_dataset = random_split(
        traain_val_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    val_dataset.train = False
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # 初始化模型、损失函数和优化器
    logging.info('Initializing model, loss function, and optimizer...')
    model = UNet(n_channels=3, n_classes=1).to(device)
    criterion = nn.BCEWithLogitsLoss()  # 包含Sigmoid的二分类损失
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 训练和验证循环
    logging.info("Starting training...")
    best_val_iou = 0.0
    worst_val_iou = 1.0
    train_losses = []
    val_losses = []
    val_ious = []

    for epoch in range(EPOCHS):
        logging.info(f'Epoch {epoch+1}/{EPOCHS}')
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_iou = validate(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_ious.append(val_iou)
        logging.info(
            f"Epoch {epoch+1} - Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f}"
        )
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f'unet_best.pth')
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(f'Saved best model to {checkpoint_path} with IoU: {best_val_iou:.4f}')
        if val_iou < worst_val_iou:
            worst_val_iou = val_iou
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f'unet_worst.pth')
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(f'Saved worst model to {checkpoint_path} with IoU: {worst_val_iou:.4f}')
    logging.info("Training complete.")
    logging.info(f'Best Val IoU: {best_val_iou:.4f}, Worst Val IoU: {worst_val_iou:.4f}')
    # 保存训练和验证损失曲线
    loss_df = pd.DataFrame({
        'train_loss': train_losses,
        'val_loss': val_losses,
        'val_iou': val_ious
    })
    loss_df.to_csv(os.path.join(CHECKPOINT_DIR, 'loss_iou_curve.csv'), index=False)
    