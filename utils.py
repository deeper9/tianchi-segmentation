import numpy as np
import random
import os
import torch
import matplotlib.pyplot as plt
import cv2
import pandas as pd

# 掩码标签转换为rle格式
def rle_encode(im):
    '''
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = im.flatten(order = 'F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# rle格式转换为可训练掩码
def rle_decode(mask_rle, shape=(512, 512)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')

#设置随机种子
def set_seeds(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#展示图像，原图像可以为路径
def show_image(img,mask, is_path=True):
    if is_path:
        img = cv2.imread(img)
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(img)

    plt.subplot(1, 2, 2)
    plt.imshow(mask)

    plt.show()

# 读取训练集的图像
# img_path = "./data/test_a/"
# df = pd.read_csv("./data/test_a_submit.csv",sep="\t", names=["names","masks"])
# name = df["names"].iloc[6]
# mask = df["masks"].iloc[6]
# mask_rle = rle_decode(mask,shape=(512,512))
# print(mask_rle.shape, type(mask_rle))
# show_image(img_path + name, mask_rle)

# 使用GPU训练
def try_gpu(i=0):
    if torch.cuda.device_count() >= i+1:
        return torch.device(f"cuda:{i}")
    return torch.device("cpu")