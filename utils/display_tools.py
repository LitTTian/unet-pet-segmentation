from PIL import Image
from IPython.display import display
from io import BytesIO
import graphviz
import matplotlib.pyplot as plt
import numpy as np


def get_title_from_titles(i, j, titles=[]):
    if isinstance(titles, dict):
        return titles.get((i, j), None)
    elif isinstance(titles, list):
        if not isinstance(titles[0], list):  # 第一个元素不是列表
            return titles[j % len(titles)]
        if i < len(titles) and j < len(titles[i]):
            return titles[i][j]
    return None

def get_img(img):
    if isinstance(img, str):
        img = Image.open(img)
    elif isinstance(img, bytes):
        img = Image.open(BytesIO(img))
    elif isinstance(img, graphviz.Digraph) or isinstance(img, graphviz.Graph):
        img = Image.open(BytesIO(img.pipe(format='png')))
    elif isinstance(img, np.ndarray):
        pass
    elif not isinstance(img, Image.Image):
        raise TypeError("Unsupported image type: {}".format(type(img)))
    
    # img = img.convert('RGB')
    return np.array(img)

def display_matplotlib_grid(images, 
                            titles=None, 
                            fontsize=20, 
                            title_loc='center', 
                            row_height=4, 
                            col_width=6, 
                            aspect='equal'
                            ):
    """
    用matplotlib展示图像网格
    inputs:
        images: [
                [image1, image2, ...],
                [image1, image2, ...],
                ...
        ] # list of lists of images
        titles: {
            (row_index, col_index): title,
            ...
        } # optional dictionary of titles for each image
    outputs:
        Displays a grid of images using matplotlib.
    """
    # conver 1d list to 2d list if necessary
    if isinstance(images, list) and all(isinstance(row, list) for row in images):
        pass  # already a 2D list
    elif isinstance(images, list):
        images = [images]
    else:
        raise ValueError("images should be a list of lists or a 1D list of images")
    if isinstance(titles, list) and not all(isinstance(row, list) for row in titles):
        titles = [titles]
    num_rows = len(images)
    num_cols = max(len(row) for row in images)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * col_width, num_rows * row_height))
    for i, row in enumerate(images):
        for j, img in enumerate(row):
            if num_rows == 1 and num_cols == 1:
                ax = axes
            elif num_rows == 1:
                ax = axes[j]
            elif num_cols == 1:
                ax = axes[i]
            else:
                ax = axes[i, j]
            # 统一处理 axes 的维度，保证是 2D numpy 数组
            img = get_img(img)
            ax.imshow(img, cmap='gray', aspect=aspect)
            ax.axis('off')
            title = get_title_from_titles(i, j, titles)
            if title is not None:
                ax.set_title(title, fontsize=fontsize, loc=title_loc)
    # Hide unused subplots if some rows are shorter
    for i in range(num_rows):
        for j in range(len(images[i]), num_cols):
            if num_rows == 1:
                axes[j].axis('off')
            elif num_cols == 1:
                axes[i].axis('off')
            else:
                axes[i, j].axis('off')
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def display_images_horizontal(*imgs, images=[], padding=10, bg_color='white'):
    """
    横向拼接并显示多张图片（PIL.Image对象）

    参数:
        images (List[Image.Image]): 要拼接的图片列表
        padding (int): 每张图之间的间距（像素）
        bg_color (str): 背景颜色，默认白色
    """
    if imgs:
        images = list(imgs)
    image_files = []
    for image in images:
        if isinstance(image, str):
            image_files.append(Image.open(image))
        elif isinstance(image, bytes):
            image_files.append(Image.open(BytesIO(image)))
        elif isinstance(image, (graphviz.Digraph, graphviz.Graph)):
            image_files.append(Image.open(BytesIO(image.pipe(format='png'))))
        else:
            raise TypeError("Unsupported image type: {}".format(type(image)))
    images = image_files
    widths = [img.width for img in images]
    heights = [img.height for img in images]
    
    total_width = sum(widths) + padding * (len(images) - 1)
    max_height = max(heights)
    
    combined = Image.new('RGB', (total_width, max_height), color=bg_color)
    
    x_offset = 0
    for img in images:
        combined.paste(img, (x_offset, 0))
        x_offset += img.width + padding
    
    display(combined)