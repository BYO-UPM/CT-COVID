import os
import configparser
import pandas as pd
import numpy as np
import pydicom
from pydicom import read_file
from pydicom.pixel_data_handlers.util import apply_voi_lut, apply_windowing, apply_modality_lut
import nibabel as nib
import matplotlib.pyplot as plt
from skimage import measure
import cv2
import plotly.express as px
import shutil
import zipfile
from tqdm import tqdm

SCHEMA_METADATA_DF = {
    "dataset": "string",
    "label" : "string",
    "scan_index": "int",
    "slice_index":"int",
    "ct_slice_path": "string",
    "infection_mask_path": "string",
    "lung_mask_path": "string",
    "height": "int",
    "width":"int",
    "has_infection": "bool",
    "has_lung_mask": "bool"
}

def read_config(config_path="./config.ini"):
    config = configparser.ConfigParser()
    config.read(config_path)
    return config


# Create a function named custom_window that takes in an image and a window center, width and window type between linear and sigmoid and returns the windowed image from a dicom
def custom_window(img, window_center, window_width, window_type):
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    window_img = img.copy()
    window_img[window_img < img_min] = img_min
    window_img[window_img > img_max] = img_max
    if window_type == "linear":
        window_img = (window_img - img_min) / (img_max - img_min)
    elif window_type == "sigmoid":
        window_img = 1 / (1 + np.exp((window_img - window_center) / window_width))
    return window_img


def read_dicom(path, voi_lut=True, modality_lut=False, view_index=0, fix_monochrome=True, window=False, clip_values=None, rescale=True, force=False):
    # Original from: https://www.kaggle.com/raddar/convert-dicom-to-np-array-the-correct-way
    dicom = read_file(path, force=force)
    if force:
        dicom.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    data = dicom.pixel_array
    if modality_lut:
        data = apply_modality_lut(data, dicom)
    if voi_lut:
        data = apply_voi_lut(data, dicom, view_index)
    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data # Inverting the image
    if clip_values:
        data = np.clip(data, clip_values[0], clip_values[1])
    
    # Normalize the image array
    if rescale:
        data = data - np.min(data)
        data = data / np.max(data)
        data = (data * 255).astype(np.uint8)    
    return data


# General utils
def unzip_and_remove_all(folder_path: str):
    for filename in tqdm(os.listdir(folder_path), total = len(os.listdir(folder_path))):
        if filename.endswith(".zip"):
            # Create a new directory path for the contents of the zip
            new_folder_path = os.path.join(folder_path, os.path.splitext(filename)[0])
            
            # Create the new directory if it doesn't exist
            if not os.path.exists(new_folder_path):
                os.makedirs(new_folder_path)
            try:
            # Extract the zip file into the new directory
                with zipfile.ZipFile(os.path.join(folder_path, filename), 'r') as zip_ref:
                    zip_ref.extractall(new_folder_path)
            except:
                print(f"Error extracting {filename}")
            
            # Remove the zip file
            os.remove(os.path.join(folder_path, filename))


def copy_image_to_folder(image_path, folder_path):
    shutil.copy(image_path, folder_path)

def get_num_files(path):
    return len(os.listdir(path))

def get_all_files(directory, extension):
    files = []
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(extension):
                files.append(os.path.join(dirpath, filename))
    return files

def copy_and_resize_img(img_path, target_folder, target_size=(512,512)):
    os.makedirs(target_folder, exist_ok=True)
    img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_array = cv2.resize(img_array, target_size)
    file_name = os.path.basename(img_path)
    target_path = os.path.join(target_folder, file_name)
    assert cv2.imwrite(target_path, img_array, [cv2.IMWRITE_PNG_COMPRESSION, 0])

# Image utils
def enhance_ct(ct, equalize=False, clahe=None):
    if isinstance(ct, str):
        ct_array = cv2.imread(ct, cv2.IMREAD_GRAYSCALE)
    else:
        ct_array = ct
    if equalize:
        ct_array = cv2.equalizeHist(ct_array)
    if clahe:
        clahe = cv2.createCLAHE(clipLimit=clahe[0], tileGridSize=(clahe[1],clahe[1]))
        ct_array = clahe.apply(ct_array)
    return ct_array

def enhance_ct_udf(row, target_folder = None, equalize=True, clahe=True, target_size=(512,512)):
    ct_path = row["ct_slice_path"]
    file_name = os.path.basename(ct_path)
    target_path = os.path.join(target_folder, file_name)    
    enhanced_ct = enhance_ct(ct_path, equalize=equalize, clahe=clahe)
    if target_path:
        assert cv2.imwrite(target_path, enhanced_ct, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    if target_size:
        assert target_size == enhanced_ct.shape



# Nii utils

def read_nii(path, rotate=False):
    ct_scan = nib.load(path)
    array = ct_scan.get_fdata()
    if rotate:
        array = np.rot90(np.array(array))
    return array

def normalize(array, clip=False):
    if clip:
        array = np.clip(array, -1000, 1000)
    array_min = np.min(array)
    array_max = np.max(array)
    return (array - array_min) / (array_max - array_min)

def to_uint8(array, norm=False, range_255=False):
    if norm:
        array = normalize(array)
    if range_255:
        return np.uint8(255 * array)
    else :
        return np.uint8(array)
    
def to_rgb(array, norm=False, range_255=False, clip=False):
    if norm:
        array = normalize(array, clip=clip)
    if range_255:
        return np.uint8(255 * np.stack((array,)*3, axis=-1))
    else :
        return np.uint8(np.stack((array,)*3, axis=-1))
    
def has_true_label(array):
    return np.max(array) > 0

def get_hist_as_df(array):
    array_list = array.flatten()
    array_hist_df = pd.DataFrame(array_list, columns=['values'])
    array_hist_df = array_hist_df.groupby(["values"])["values"].agg("count").to_frame(name='count').reset_index()
    return array_hist_df


# Visualization utils
def plot_ct_slice(ct_array, lung_mask=None, infection_mask=None,figsize=(10,10)):
    fig = plt.figure(figsize=figsize)
    plt.imshow(ct_array, cmap="gray")
    if lung_mask is not None:
        plt.imshow(lung_mask, cmap="Blues", alpha=0.5)
    if infection_mask is not None:
        plt.imshow(infection_mask, cmap="Reds", alpha=0.5)
    plt.show()

def plot_ct_slice_with_hist(ct_array, lung_mask=None, infection_mask= None, figsize=(10,10)):
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3)
    ax_0_0 = fig.add_subplot(gs[0, 0])
    ax_0_0.imshow(ct_array, cmap="gray")
    ax_0_0.set_title("CT slice")
    ax_1_0 = fig.add_subplot(gs[1, 0])
    ax_1_0.hist(ct_array.flatten(), bins=100, color="blue", alpha=0.5)
    if lung_mask is not None:
        ax_0_1 = fig.add_subplot(gs[0, 1])
        ax_0_1.set_title("Lung Mask")
        ax_0_1.imshow(ct_array, cmap="gray")
        ax_1_1 = fig.add_subplot(gs[1, 1])
        ax_0_1.imshow(lung_mask, cmap="Blues", alpha=0.5)
        ax_1_1.hist(lung_mask.flatten(),bins = range(int(np.max(lung_mask.flatten()))+2))
    if infection_mask is not None:
        ax_0_2 = fig.add_subplot(gs[0, 2])
        ax_0_2.set_title("Infection Mask")
        ax_0_2.imshow(ct_array, cmap="gray")
        ax_0_2.imshow(infection_mask, cmap="Reds", alpha=0.5)
        ax_1_2 = fig.add_subplot(gs[1, 2])
        ax_1_2.hist(infection_mask.flatten(), bins=range(int(np.max(infection_mask.flatten()))+2))
    plt.show()


def plot_full_ct_scan(ct_array,init_index=None, end_index=None,infection_mask = None, lung_mask= None, figsize=(10,10)):
    if init_index:
        ct_array = ct_array[:,:,init_index:]
    if end_index:
        ct_array = ct_array[:,:,:end_index]
    fig = px.imshow(ct_array, animation_frame=2, binary_string=True, labels=dict(animation_frame="slice"))
    if lung_mask is not None:
        fig.add_trace(px.imshow(lung_mask, binary_string=True).data[0], opacity=0.5)
    if infection_mask is not None:
        fig.add_trace(px.imshow(infection_mask, binary_string=True).data[0], opacity=0.5)
    fig.show()


def read_ct_from_df(df, idx, load_lung_mask=False, load_infection_mask=False, rotate=False):
    ct_array = cv2.imread(df.loc[idx, 'ct_scan_path'], cv2.IMREAD_UNCHANGED)
    if load_lung_mask:
        lung_mask = cv2.imread(df.loc[idx, 'lung_mask_path'], cv2.IMREAD_UNCHANGED)
    if load_infection_mask:
        infection_mask = cv2.imread(df.loc[idx, 'infection_mask_path'], cv2.IMREAD_UNCHANGED)
    return ct_array, lung_mask, infection_mask



    

