import torch
from tqdm import tqdm

EPS = 1e-6

def calculate_iou(preds, targets, threshold=0.5):
    preds = torch.sigmoid(preds)  # HL: 因为BCEWithLogitsLoss内部集成了Sigmoid，这里也要加上
    preds = (preds > threshold).float()
    preds = preds.squeeze(1)  # (B, 1, H, W) -> (B, H, W)
    intersection = (preds * targets).sum(dim=(1, 2))
    union = (preds + targets).sum(dim=(1, 2)) - intersection
    iou = (intersection + EPS) / (union + EPS)
    return iou.mean().item()

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc="Training")
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs.squeeze(1), masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    avg_loss = total_loss / len(loader)
    return avg_loss

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    pbar = tqdm(loader, desc="Validation")
    with torch.no_grad():
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs.squeeze(1), masks)
            total_loss += loss.item()

            iou = calculate_iou(outputs, masks)
            total_iou += iou

            pbar.set_postfix({'loss': loss.item(), 'iou': iou})
    avg_loss = total_loss / len(loader)
    avg_iou = total_iou / len(loader)
    return avg_loss, avg_iou
