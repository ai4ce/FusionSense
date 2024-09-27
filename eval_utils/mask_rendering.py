import cv2
import os
import numpy as np

def mask_images(image_path, mask_path, save_path):
    """
    读取所有图片和mask图片，把图片mask掉再保存
    """
    for img_name in os.listdir(image_path):
        if not img_name.endswith(".jpg"):
            continue
        image = cv2.imread(os.path.join(image_path, img_name))
        mask_name = img_name.replace(".jpg", ".png")
        mask = cv2.imread(os.path.join(mask_path, mask_name), cv2.IMREAD_GRAYSCALE)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        mask = mask > 0
        image[~mask, :] = (255, 255, 255)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        cv2.imwrite(os.path.join(save_path, img_name), image)

image_path = "eval/black_bunny/9view/test/rgb"
mask_path = "datasets/black_bunny/masks"
save_path = "eval/black_bunny/9view/masked_rendering"
mask_images(image_path, mask_path, save_path)