import sys ,os, importlib
import pandas as pd
import numpy as np
import configparser
import matplotlib.pyplot as plt
import cv2
import zipfile
from tqdm import tqdm
import glob
import json

sys.path.append('/media/my_ftp/TFTs/amoure/TFM_MUIT')
import data_processing.data_processing_functions as DP
import utils as Utils
importlib.reload(Utils)
importlib.reload(DP)

config = configparser.ConfigParser()
config.read("config.ini")
pd.set_option('display.max_colwidth', None)


def main():
    CNCB_COVID_CT_PATH = config['PATHS']['CNCB_COVID_CT_PATH']
    METADATA_PATH = os.path.join(CNCB_COVID_CT_PATH, 'metadata.csv')
    LESION_SLICES_PATH = os.path.join(CNCB_COVID_CT_PATH, 'lesions_slices.csv')
    SEGMENTED_CNCB = os.path.join(CNCB_COVID_CT_PATH, 'ct_lesion_seg')
    unzip_file_names_path = os.path.join(CNCB_COVID_CT_PATH, 'unzip_filenames.csv')
    unzip_file_names_df = pd.read_csv(unzip_file_names_path)
    unzip_file_names_df["folder"] = unzip_file_names_df["zip_file"].apply(lambda x: x.split(".")[0])
    unzip_file_names_df["folder_id"] = unzip_file_names_df["folder"].apply(lambda x: x.split("-")[1].split(".")[0])
    
    seg_meta_df = DP.read_cnbc_folder_structure(SEGMENTED_CNCB)
    scan_seg_meta_df = seg_meta_df[seg_meta_df["mask"].notna()]
    covid_unzip_file_names_df = unzip_file_names_df[unzip_file_names_df['label'] == 'NCP'].copy()
    covid_unzip_file_names_df["scan_folders"] = CNCB_COVID_CT_PATH + "/" + "COVID19-" + covid_unzip_file_names_df["folder_id"] + "/NCP/" + covid_unzip_file_names_df["patient_id"].astype(str) + "/" + covid_unzip_file_names_df["scan_id"].astype(str)
    # Load all images from scan_seg_meta_df["image"] into memory
    images_dict = {image_path: cv2.imread(image_path) for image_path in tqdm(scan_seg_meta_df["image"], total=len(scan_seg_meta_df))}
    matches_dict = {}
    # Iterate over the folders in covid_unzip_file_names_df["scan_folders"]
    for covid_image_folder in tqdm(covid_unzip_file_names_df["scan_folders"], total=len(covid_unzip_file_names_df)):
        # For each folder, iterate over the images in the folder
        for covid_image_path in glob.glob(covid_image_folder + "/*.png"):
            covid_image = cv2.imread(covid_image_path)
            # For each image in the folder, compare it with all images in the dictionary
            for image_path, image in images_dict.items():
                if np.array_equal(image, covid_image):
                    matches_dict[image_path] = covid_image_path
                    break
    print(matches_dict)
    with open('/media/my_ftp/TFTs/amoure/TFM_MUIT/Notebooks/EDA/matches_dict.json', 'w') as fp:
        json.dump(matches_dict, fp)


if __name__ == "__main__":
    main()