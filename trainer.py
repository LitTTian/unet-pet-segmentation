import torch
from tqdm import tqdm

EPS = 1e-6

def calculate_iou(preds, targets, threshold=0.5):
    preds = torch.sigmoid(preds)  # HL: 因为BCEWithLogitsLoss内部集成了Sigmoid，这里也要加上
    preds = (preds > threshold).float()
    if preds.dim() == 4:
        preds = preds.squeeze(1)  # (B, 1, H, W) -> (B, H, W)
    targets = targets.float()
    intersection = (preds * targets).sum(dim=(1, 2))
    union = (preds + targets).sum(dim=(1, 2)) - intersection
    iou = (intersection + EPS) / (union + EPS)
    iou = torch.nan_to_num(iou, nan=0.0, posinf=0.0, neginf=0.0)
    return iou.mean().item()

def calculate_dice(preds, targets, threshold=0.5):
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    if preds.dim() == 4:
        preds = preds.squeeze(1)
    targets = targets.float()
    intersection = (preds * targets).sum(dim=(1, 2))
    dice = (2.0 * intersection + EPS) / (preds.sum(dim=(1, 2)) + targets.sum(dim=(1, 2)) + EPS)
    dice = torch.nan_to_num(dice, nan=0.0, posinf=0.0, neginf=0.0)
    return dice.mean().item()

def train_one_epoch(model, loader, criterion, optimizer, device, gradient_clip_norm=None):
    model.train()
    total_loss = 0.0
    total_samples = 0
    pbar = tqdm(loader, desc="Training", leave=False)
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清零
        outputs = model(images)
        if outputs.dim() == 4 and outputs.size(1) == 1:
            outputs = outputs.squeeze(1)
        loss = criterion(outputs, masks.float())
        loss.backward()
        if gradient_clip_norm is not None and gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        pbar.set_postfix({
            'batch_loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss/total_samples:.4f}'
        })
    avg_loss = total_loss / max(total_samples, 1)
    pbar.close()
    
    return avg_loss

def validate(model, loader, criterion, device, threshold=0.5):
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    total_samples = 0
    pbar = tqdm(loader, desc="Validation", leave=False)
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            outputs = model(images)
            outputs_for_metrics = outputs.clone()
            if outputs.dim() == 4 and outputs.size(1) == 1:
                outputs = outputs.squeeze(1)
            loss = criterion(outputs, masks.float())
            iou = calculate_iou(outputs_for_metrics, masks, threshold=threshold)
            dice = calculate_dice(outputs_for_metrics, masks, threshold=threshold)
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_iou += iou * batch_size
            total_dice += dice * batch_size
            total_samples += batch_size
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'iou': f'{iou:.4f}',
                'dice': f'{dice:.4f}',
                'avg_loss': f'{total_loss/total_samples:.4f}'
            })
    
    avg_loss = total_loss / max(total_samples, 1)
    avg_iou = total_iou / max(total_samples, 1)
    avg_dice = total_dice / max(total_samples, 1)
    pbar.close()
    return avg_loss, avg_iou, avg_dice