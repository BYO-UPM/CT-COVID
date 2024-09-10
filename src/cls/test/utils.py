import matplotlib.pyplot as plt
import numpy as np
import random

def plot_image_and_histogram(image, label, img_path):
    fig, axs = plt.subplots(2)
    axs[0].imshow(image.permute(1, 2, 0))
    axs[0].set_title(f'Label: {label}, Path: {img_path}')
    axs[1].hist(image.flatten(), bins=256)
    plt.show()

#Plot batch without using the make_grid function to allow for more flexibility
# 
def plot_batch(batched_img_tensor, batched_label_tensor, batched_img_nm, batched_dataset_nm, batch_size=16):
    fig, axs = plt.subplots(4, 4, figsize=(20, 20))
    for i in range(4):
        for j in range(4):
            idx = i + j
            img = batched_img_tensor[idx]
            label = batched_label_tensor[idx]
            img_nm = batched_img_nm[idx]
            dataset_nm = batched_dataset_nm[idx]
            axs[i, j].imshow(img.permute(1, 2, 0))
            axs[i, j].set_title(f'Label: {label}, Path: {img_nm}, Dataset: {dataset_nm}')
    plt.show()


