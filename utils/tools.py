import torch
import numpy as np
from PIL import Image

def tensor2numpy(tensor):  # 返回numpy数组，
    image_np = tensor.cpu().numpy()
    # 灰度图/rgb图
    if len(image_np.shape) == 2:
        pass
    else:
        image_np = tensor.cpu().numpy().transpose(1, 2, 0)
    image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
    return image_np

def tensor2image(tensor):  # 返回numpy数组，
    image_np = tensor2numpy(tensor)
    image = Image.fromarray(image_np)
    return image

def predict_image_mask(model, image, returnImage=False, device='cpu'):
    input_tensor = torch.unsqueeze(image, 0)  # 添加批次维度
    input_tensor = input_tensor.to(device)
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        output = torch.sigmoid(output)
        output = (output > 0.5).float()
        output_mask = output.squeeze(0).squeeze(0)  # 去掉批次维度和通道维度
    print(output_mask.shape)
    if returnImage:
        return tensor2image(output_mask)
    else:
        return output_mask