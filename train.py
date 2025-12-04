import logging
import torch
from torch.utils.data import DataLoader, random_split
from utils.dataset import OxfordPetDataset
from model.unet import UNet
from model.transunet import TransUnet
from model.transunetP import TransUnetP
from trainer import train_one_epoch, validate
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import pandas as pd
import os
import random
import numpy as np
import wandb
from datetime import datetime
import warnings
import argparse  # 新增：命令行参数解析
warnings.filterwarnings('ignore')

# ===================== 基础配置 =====================
WANDB_PROJECT = "oxford-pet-segmentation-lr1e-4bs16"
WANDB_ENTITY = None
WANDB_NAME = None

# 创建唯一的运行ID
RUN_ID = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# ===================== 日志配置 =====================
def setup_logging(log_dir='logs'):
    """配置日志系统"""
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # 文件处理器
    file_handler = logging.FileHandler(os.path.join(log_dir, f'train_{RUN_ID}.log'))
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    
    # 配置根日志器
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# ===================== 随机种子设置 =====================
def set_seed(seed=42):
    """设置所有随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ===================== 命令行参数解析 =====================
def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Oxford Pet Segmentation Training (UNet/TransUNet)')
    
    # 基础训练参数
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size (default: 16)')
    parser.add_argument('--lr', '--learning-rate', type=float, default=1e-4, dest='learning_rate',
                        help='Initial learning rate (default: 1e-4)')
    parser.add_argument('--img-size', type=int, nargs=2, default=(256, 256), help='Input image size (default: 256 256)')
    parser.add_argument('--train-val-split', type=float, default=0.8,
                        help='Train/validation split ratio (default: 0.8)')
    parser.add_argument('--epochs', type=int, default=200, help='Total training epochs (default: 200)')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                        help='Learning rate warmup epochs (default: 5)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--num-workers', type=int, default=1,
                        help='Number of data loader workers (default: 1)')
    
    # 模型选择与参数
    parser.add_argument('--model', type=str, default='unet', choices=['unet', 'transunet', 'transunetp'],
                        help='Model type (unet/transunet/transunetp, default: unet)')
    parser.add_argument('--transunet-num-heads', type=int, default=12,
                        help='TransUNet number of attention heads (default: 12)')
    parser.add_argument('--transunet-num-layers', type=int, default=12,
                        help='TransUNet number of transformer layers (default: 12)')
    parser.add_argument('--transunet-mlp-dim', type=int, default=3072,
                        help='TransUNet MLP dimension (default: 3072)')
    parser.add_argument('--transunet-dropout-rate', type=float, default=0.1,
                        help='TransUNet dropout rate (default: 0.1)')
    parser.add_argument('--transunet-embed-dim', type=int, default=768,
                        help='TransUNet embedding dimension (default: 768)')
    
    # 优化器与损失函数
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'AdamW'],
                        help='Optimizer type (default: Adam)')
    parser.add_argument('--loss-function', type=str, default='BCEWithLogitsLoss',
                        help='Loss function (default: BCEWithLogitsLoss)')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='Weight decay (L2 regularization, default: 1e-5)')
    
    # 训练策略
    parser.add_argument('--patience', type=int, default=999,
                        help='Early stopping patience (default: 20)')
    parser.add_argument('--gradient-clip-norm', type=float, default=1.0,
                        help='Gradient clip norm (0 to disable, default: 1.0)')
    parser.add_argument('--checkpoint-save-freq', type=int, default=1,
                        help='Checkpoint save frequency (epochs, default: 1)')
    parser.add_argument('--save-all-checkpoints', action='store_true', default=True,
                        help='Save all checkpoints (default: True)')
    parser.add_argument('--no-save-all-checkpoints', action='store_false', dest='save_all_checkpoints',
                        help='Disable saving all checkpoints')
    
    # 模式开关
    parser.add_argument('--run-on-test-set', action='store_true', default=False,
                        help='Evaluate on test set during training (default: False)')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Debug mode (small epochs/batch size, default: False)')
    parser.add_argument('--wandb-name', type=str, default=None,
                        help='WandB run name (default: auto-generated)')
    
    # 路径配置（可选，覆盖默认路径）
    parser.add_argument('--image-dir', type=str, default='C:\\Users\\Alvis\\Study\\datasets\\image\\Oxford-IIIT_Pet_Dataset\\images\\',
                        help='Path to images directory')
    parser.add_argument('--mask-dir', type=str, default='C:\\Users\\Alvis\\Study\\datasets\\image\\Oxford-IIIT_Pet_Dataset\\annotations\\trimaps\\',
                        help='Path to masks directory')
    parser.add_argument('--train-val-txt', type=str, default='C:\\Users\\Alvis\\Study\\datasets\\image\\Oxford-IIIT_Pet_Dataset\\annotations\\trainval.txt',
                        help='Path to trainval.txt')
    parser.add_argument('--test-txt', type=str, default='C:\\Users\\Alvis\\Study\\datasets\\image\\Oxford-IIIT_Pet_Dataset\\annotations\\test.txt',
                        help='Path to test.txt')
    
    return parser.parse_args()

# ===================== 主训练函数 =====================
def main(args):
    # 构建配置字典（兼容原有逻辑）
    CONFIG = {
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "img_size": args.img_size,
        "train_val_split": args.train_val_split,
        "epochs": args.epochs,
        "warmup_epochs": args.warmup_epochs,
        "model": args.model,
        "optimizer": args.optimizer,
        "loss_function": args.loss_function,
        "transunet_num_heads": args.transunet_num_heads,
        "transunet_num_layers": args.transunet_num_layers,
        "transunet_mlp_dim": args.transunet_mlp_dim,
        "transunet_dropout_rate": args.transunet_dropout_rate,
        "transunet_embed_dim": args.transunet_embed_dim,
        "save_all_checkpoints": args.save_all_checkpoints,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "run_on_test_set": args.run_on_test_set,
        "debug": args.debug,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "gradient_clip_norm": args.gradient_clip_norm,
        "checkpoint_save_freq": args.checkpoint_save_freq,
    }
    
    # 初始化日志
    logger = setup_logging()
    
    # 打印命令行参数
    logger.info("="*50)
    logger.info("Training Configuration (Command Line Args)")
    logger.info("="*50)
    for key, value in sorted(CONFIG.items()):
        logger.info(f"{key:25} : {value}")
    logger.info("="*50)
    
    # 设置随机种子
    set_seed(CONFIG['seed'])
    
    # 初始化wandb
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=args.wandb_name or f"{CONFIG['model']}_{RUN_ID}",
        config=CONFIG,
        mode="disabled" if CONFIG['debug'] else "online",
        save_code=True
    )
    
    # 动态生成检查点目录
    CHECKPOINT_DIRS = {
        'unet': f'./checkpoints/unet_{RUN_ID}/',
        'transunet': f'./checkpoints/transunet_{RUN_ID}/',
        'transunetp': f'./checkpoints/transunetp_{RUN_ID}/',
    }
    CHECKPOINT_DIR = CHECKPOINT_DIRS[CONFIG['model']]
    
    # 创建目录
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # DEBUG模式调整
    if CONFIG['debug']:
        CONFIG['epochs'] = 2
        CONFIG['batch_size'] = 8  # 减小batch size避免内存不足
        CONFIG['warmup_epochs'] = 1
        logger.warning("DEBUG mode enabled - reduced epochs/batch size!")
    
    # 设置设备
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f'Using CUDA device: {torch.cuda.get_device_name(0)}')
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info('Using MPS device (Apple Silicon)')
    else:
        device = torch.device("cpu")
        logger.info('Using CPU (warning: training will be slow)')
    
    wandb.config.update({"device": str(device)})
    
    # ===================== 数据集加载 =====================
    logger.info("Loading datasets...")
    train_val_dataset = OxfordPetDataset(
        image_dir=args.image_dir,       # 使用命令行传入的路径
        mask_dir=args.mask_dir,
        file_txt=args.train_val_txt,
        img_size=CONFIG['img_size'],
        train=True,
        n_samples=20 if CONFIG['debug'] else 0,
    )
    
    # 数据集分割
    train_size = int(CONFIG['train_val_split'] * len(train_val_dataset))
    val_size = len(train_val_dataset) - train_size
    logger.info(f"Dataset split: train={train_size}, val={val_size}")
    
    train_dataset, val_dataset = random_split(
        train_val_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(CONFIG['seed'])
    )
    
    # 设置验证集为非训练模式
    val_dataset.dataset.train = False
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        generator=torch.Generator().manual_seed(CONFIG['seed']),  # NT: 确保每个epoch数据顺序相同
        num_workers=CONFIG['num_workers'],
        pin_memory=True,
        drop_last=True  # 避免最后一个批次样本数不足
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=True
    )
    
    # 测试集加载（如果需要）
    test_loader = None
    if CONFIG['run_on_test_set']:
        test_dataset = OxfordPetDataset(
            image_dir=args.image_dir,
            mask_dir=args.mask_dir,
            file_txt=args.test_txt,
            img_size=CONFIG['img_size'],
            train=False,
            n_samples=20 if CONFIG['debug'] else 0,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=CONFIG['batch_size'],
            shuffle=False,
            num_workers=CONFIG['num_workers'],
            pin_memory=True
        )
    
    # ===================== 模型初始化 =====================
    logger.info(f"Initializing {CONFIG['model']} model...")
    try:
        if CONFIG["model"] == "unet":
            model = UNet(n_channels=3, n_classes=1).to(device)
        elif CONFIG["model"] == "transunet":
            model = TransUnet(
                img_size=CONFIG["img_size"],
                n_channels=3,
                n_classes=1,
                num_heads=CONFIG["transunet_num_heads"],
                num_layers=CONFIG["transunet_num_layers"],
                mlp_dim=CONFIG["transunet_mlp_dim"],
                dropout_rate=CONFIG["transunet_dropout_rate"],
                embed_dim=CONFIG["transunet_embed_dim"],
            ).to(device)
        elif CONFIG["model"] == "transunetp":
            model = TransUnetP(
                img_size=CONFIG["img_size"],
                n_channels=3,
                n_classes=1,
                num_heads=CONFIG["transunet_num_heads"],
                num_layers=CONFIG["transunet_num_layers"],
                mlp_dim=CONFIG["transunet_mlp_dim"],
                dropout_rate=CONFIG["transunet_dropout_rate"],
                embed_dim=CONFIG["transunet_embed_dim"],
            ).to(device)
        
        # 打印模型参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: total={total_params/1e6:.2f}M, trainable={trainable_params/1e6:.2f}M")
        wandb.config.update({
            "total_params": total_params,
            "trainable_params": trainable_params
        })
        
    except Exception as e:
        logger.error(f"Model initialization failed: {e}")
        raise
    
    # ===================== 损失函数和优化器 =====================
    # 损失函数
    if CONFIG["loss_function"] == "BCEWithLogitsLoss":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()  # 默认
    
    # 优化器
    if CONFIG["optimizer"] == "Adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=CONFIG['learning_rate'],
            weight_decay=CONFIG['weight_decay']
        )
    elif CONFIG["optimizer"] == "AdamW":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=CONFIG['learning_rate'],
            weight_decay=CONFIG['weight_decay']
        )
    else:
        optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    
    # 学习率调度器（带预热）
    if CONFIG['warmup_epochs'] > 0:
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=CONFIG['warmup_epochs']
        )
        main_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=CONFIG['epochs'] - CONFIG['warmup_epochs'],
            eta_min=1e-6
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[CONFIG['warmup_epochs']]
        )
    else:
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=CONFIG['epochs'],
            eta_min=1e-6
        )
    
    # wandb监视模型
    wandb.watch(model, criterion=criterion, log="all", log_freq=10)
    
    # ===================== 训练循环 =====================
    logger.info("Starting training loop...")
    
    # 初始化跟踪变量
    best_val_iou = 0.0
    worst_val_iou = float('inf')
    train_losses = []
    val_losses = []
    val_ious = []
    val_dices = []
    test_losses = []
    test_ious = []
    test_dices = []
    no_improvement_count = 0
    
    for epoch in range(CONFIG['epochs']):
        epoch_start_time = datetime.now()
        logger.info(f"\nEpoch {epoch+1}/{CONFIG['epochs']}")
        
        # 训练阶段
        model.train()
        train_loss = train_one_epoch(
            model, 
            train_loader, 
            criterion, 
            optimizer, 
            device,
            gradient_clip_norm=CONFIG['gradient_clip_norm']  # 传递梯度裁剪参数
        )
        
        # 验证阶段
        model.eval()
        with torch.no_grad():
            val_loss, val_iou, val_dice = validate(model, val_loader, criterion, device)
        
        # 更新列表
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_ious.append(val_iou)
        val_dices.append(val_dice)
        
        # 测试集评估（如果需要）
        test_loss = None
        test_iou = None
        test_dice = None
        if CONFIG['run_on_test_set'] and test_loader:
            with torch.no_grad():
                test_loss, test_iou, test_dice = validate(model, test_loader, criterion, device)
                test_losses.append(test_loss)
                test_ious.append(test_iou)
                test_dices.append(test_dice)
        
        # 学习率调度
        scheduler.step()
        
        # 计算耗时
        epoch_duration = datetime.now() - epoch_start_time
        
        # 日志记录
        log_msg = (
            f"Epoch {epoch+1} completed in {epoch_duration.total_seconds():.2f}s | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val IoU: {val_iou:.4f} | "
            f"Val Dice: {val_dice:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )
        if test_loss is not None:
            log_msg += f" | Test Loss: {test_loss:.4f} | Test IoU: {test_iou:.4f} | Test Dice: {test_dice:.4f}"
        
        logger.info(log_msg)
        
        # wandb日志
        log_data = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_iou': val_iou,
            'val_dice': val_dice,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'epoch_duration': epoch_duration.total_seconds()
        }
        if test_loss is not None:
            log_data.update({
                'test_loss': test_loss,
                'test_iou': test_iou,
                'test_dice': test_dice
            })
        wandb.log(log_data)
        
        # ===================== 更新CSV日志 =====================
        loss_df = pd.DataFrame({
            'epoch': list(range(1, len(train_losses)+1)),
            'train_loss': train_losses,
            'val_loss': val_losses,
            'val_iou': val_ious,
            'val_dice': val_dices,
            'learning_rate': [optimizer.param_groups[0]['lr']] * len(train_losses)
        })
        
        if test_losses:
            loss_df['test_loss'] = test_losses
            loss_df['test_iou'] = test_ious
            loss_df['test_dice'] = test_dices
        
        loss_df_path = os.path.join(CHECKPOINT_DIR, 'loss_iou_curve.csv')
        loss_df.to_csv(loss_df_path, index=False)
        wandb.save(loss_df_path)

        # ===================== 模型保存 =====================
        # 保存所有检查点（按频率）
        if CONFIG['save_all_checkpoints'] and (epoch + 1) % CONFIG['checkpoint_save_freq'] == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f'epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_iou': val_iou,
                'val_dice': val_dice,
                'best_val_iou': best_val_iou,
            }, checkpoint_path)
            logger.info(f'Saved checkpoint to {checkpoint_path}')
        
        # 保存最佳模型
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            no_improvement_count = 0  # 重置早停计数器
            checkpoint_path = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': val_iou,
                'config': CONFIG
            }, checkpoint_path)
            logger.info(f'Updated best model (IoU: {best_val_iou:.4f})')
        
        # 保存最差模型
        if val_iou < worst_val_iou:
            worst_val_iou = val_iou
            checkpoint_path = os.path.join(CHECKPOINT_DIR, 'worst_model.pth')
            torch.save(model.state_dict(), checkpoint_path)
        
        # ===================== 早停检查 =====================
        no_improvement_count += 1
        if no_improvement_count >= CONFIG['patience'] and not CONFIG['debug']:
            logger.info(f"Early stopping triggered after {epoch+1} epochs (no improvement for {CONFIG['patience']} epochs)")
            break
        

    
    # ===================== 训练结束 =====================
    logger.info("\nTraining completed!")
    logger.info(f'Best Val IoU: {best_val_iou:.4f}')
    logger.info(f'Worst Val IoU: {worst_val_iou:.4f}')
    logger.info(f'Final Train Loss: {train_losses[-1]:.4f}')
    logger.info(f'Final Val Loss: {val_losses[-1]:.4f}')
    
    # 更新wandb总结
    wandb.summary.update({
        "best_val_iou": best_val_iou,
        "best_val_dice": max(val_dices) if val_dices else None,
        "worst_val_iou": worst_val_iou,
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
        "total_epochs_trained": len(train_losses),
        "run_id": RUN_ID
    })
    
    if test_losses:
        wandb.summary.update({
            "final_test_loss": test_losses[-1],
            "final_test_iou": test_ious[-1],
            "final_test_dice": test_dices[-1]
        })
    
    # 保存最终模型
    final_checkpoint_path = os.path.join(CHECKPOINT_DIR, 'final_model.pth')
    torch.save({
        'epoch': len(train_losses),
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_ious': val_ious,
        'val_dices': val_dices,
        'best_val_iou': best_val_iou,
        'config': CONFIG
    }, final_checkpoint_path)
    
    logger.info(f'Saved final model to {final_checkpoint_path}')
    
    # 结束wandb
    wandb.finish()
    logger.info("All processes completed successfully!")

# ===================== 入口函数 =====================
if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()
    
    try:
        main(args)
    except KeyboardInterrupt:
        logging.error("Training interrupted by user")
        wandb.finish(exit_code=1)
    except Exception as e:
        logging.error(f"Training failed with error: {e}", exc_info=True)
        wandb.finish(exit_code=1)
        raise