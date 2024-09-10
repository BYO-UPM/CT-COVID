import pandas as pd
import os
from pathlib import Path
import pydicom
from shutil import copy2
import numpy as np
import glob
from tqdm import tqdm
from functools import lru_cache

def remove_duplicates(lst):
    return list(dict.fromkeys(lst))

def generate_rgb_mask(mask, color_dict=None):
    """
    Generate RGB mask from mask
    Args:
        mask (np.array): mask
        color_dict (dict): dictionary with colors for each class
    Returns:
        np.array: RGB mask
    """
    if color_dict is None:
        color_dict = {
            0: [0, 0, 0],
            1: [0, 255, 0],
            2: [0, 0, 255],
            3: [255, 255, 0],
            4: [255, 0, 0],
            5: [0, 255, 255],
            6: [255, 255, 255],
            7: [255, 0, 255],
            8: [192, 192, 192],
            9: [128, 128, 128],
            10: [128, 0, 0],
            11: [128, 128, 0],
            12: [0, 128, 0],
            13: [128, 0, 128],
            14: [0, 128, 128],
            15: [0, 0, 128],
        }
    rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3))
    for i in range(1, mask.max() + 1):
        rgb_mask[mask == i] = color_dict[i]
    return rgb_mask.astype(np.uint8)

def generate_semantic_mask(mask, color_dict=None):
    """
    Generate semantic mask from mask
    Args:
        mask (np.array): mask
        color_dict (dict): dictionary with colors for each class
    Returns:
        np.array: semantic mask
    """
    if color_dict is None:
        raise Exception("Color dict is not defined")

    mapped_mask = np.vectorize(lambda x: color_dict.get(x,x))(mask)
    return mapped_mask


############################################
# COVID_SEG_1 Processing functions
############################################
def generate_covid_seg_1_infection_mask(mask):
    color_dict = {
        0: 0,
        1: 1,
        2: 1,
        3: 1,
    }
    return generate_semantic_mask(mask, color_dict)

def generate_covid_seg_1_lung_mask(mask, rgb=True):
    """
    Covid_seg_1 dataset has left and right mask for their datasets, therefore this is needed.
    """
    if rgb:
        color_dict = {
            0: 0,
            1: 255,
            2: 255,
        }
    else:
        color_dict = {
            0: 0,
            1: 1,
            2: 1,
        }
        
    return generate_semantic_mask(mask, color_dict)

############################################
# COVID_SEG_2 Processing functions
############################################

def read_covid_seg_2_folder_structure(path):
    df_list = []
    for i in range(1,10):
        data = {
            "id":i,
            "image": os.path.join(path, "rp_im", str(i) + ".nii"),
            "infection_mask": os.path.join(path, "rp_msk", str(i) + ".nii"),
            "lung_mask": os.path.join(path, "rp_lung_msk", str(i) + ".nii")
            }
        df_list.append(data)

    return pd.DataFrame(df_list)

def generate_covid_seg_2_lung_mask(mask, rgb=True):

    if rgb:
        color_dict = {
            0: 0,
            1: 255,
        }
    else:
        color_dict = {
            0: 0,
            1: 1,
        }
    return generate_semantic_mask(mask, color_dict)


############################################
# COVID_CT_JunMa Processing functions
############################################
def read_covid_ct_junma_folder_structure(path):
    df_list = []
    for i, file in enumerate(os.listdir(os.path.join(path, "COVID-19-CT-Seg_20cases"))):
        if file.endswith(".nii"):
            data = {
                "id": i + 1,
                "image":os.path.join(path, "COVID-19-CT-Seg_20cases", file),
                "infection_mask":os.path.join(path, "Infection_Mask", file),
                "lung_mask":os.path.join(path, "Lung_Mask", file)
                }
            df_list.append(data)
    return pd.DataFrame(df_list)

def generate_covid_ct_junma_lung_mask(mask, rgb=True):
    if rgb:
        color_dict = {
            0: 0,
            1: 255,
            2: 255
        }
    else:
        color_dict = {
            0: 0,
            1: 1,
            2: 1
        }
    return generate_semantic_mask(mask, color_dict)


############################################
# MIDRC-RICORD-1A Processing functions
############################################

# def directory_preprocessing(raw_images_path, processed_images_path):
#     images_path = Path(raw_images_path)
#     filenames = list(images_path.glob('**/*.dcm'))
#     info = []
#     for f in filenames:
#         d = pydicom.dcmread(str(f),stop_before_pixels=True)
#         info.append({'fn':str(f), 
#         'StudyInstanceUID':d.StudyInstanceUID,
#         'SeriesInstanceUID':d.SeriesInstanceUID, 
#         'SOPInstanceUID':d.SOPInstanceUID, 
#         'description':d.SeriesDescription if 'SeriesDescription' in d else "", 
#         'name':d.SequenceName if 'SequenceName' in d else "",
#         'Modality':d.Modality if 'Modality' in d else "",
#         'ContrastAgent':d.ContrastBolusAgent if 'ContrastBolusAgent' in d else "",
#         'ScanOptions':d.ScanOptions if 'ScanOptions' in d else "",
#         'WW':d.WindowWidth if 'WindowWidth' in d else "",
#         'WC':d.WindowCenter if 'WindowCenter' in d else "",
#         'ImageType' :d.ImageType if 'ImageType' in d else "",
#         'PixelSpacing' :d.PixelSpacing if 'PixelSpacing' in d else "",
#         'SliceThickness':d.SliceThickness if 'SliceThickness' in d else "",
#         'PhotometricInterpretation':d.PhotometricInterpretation if 'PhotometricInterpretation' in d else ""
#                   })
#     df = pd.DataFrame(info)
#     #creating list of StudyInstanceUID
#     StudyInstanceUID_items = []
#     for i in range(0,len(df)):
#         StudyInstanceUID_items.append(df.iloc[i].StudyInstanceUID)
#     StudyInstanceUID_reduced = remove_duplicates(StudyInstanceUID_items)
#     info = []
#     os.makedirs(processed_images_path, exist_ok=True)
#     for i in range(0, len(StudyInstanceUID_reduced)):
#         StudyInstanceUID_dir = '{}'.format(StudyInstanceUID_reduced[i])
#         StudyInstanceUID_path = os.path.join(processed_images_path, StudyInstanceUID_dir)
#         os.mkdir(StudyInstanceUID_path)
#         SeriesInstanceUID_items = []
#         for k in range(0,len(df)):
#             if StudyInstanceUID_reduced[i] == df.iloc[k].StudyInstanceUID:
#                 SeriesInstanceUID_items.append(df.iloc[k].SeriesInstanceUID)
#         SeriesInstanceUID_reduced = remove_duplicates(SeriesInstanceUID_items)
#         for j in range(0,len(SeriesInstanceUID_reduced)):
#             SeriesInstanceUID_dir = '{}'.format(SeriesInstanceUID_reduced[j])
#             SeriesInstanceUID_path = os.path.join(StudyInstanceUID_path, SeriesInstanceUID_dir)
#             os.mkdir(SeriesInstanceUID_path)
#             #print(SeriesInstanceUID_path)
#             for b in range(0,len(df)):
#                 if SeriesInstanceUID_reduced[j] == df.iloc[b].SeriesInstanceUID:
#                     copy2(df.iloc[b].fn,'{}/{}/{}/{}.dcm'.format(processed_images_path,StudyInstanceUID_reduced[i],SeriesInstanceUID_reduced[j],df.iloc[b].SOPInstanceUID))
#     return df

# def get_image_data_from_dicom(series, series_instance_uid, drop_dupl_slices):
#     series = expand_volumetric(series)
#     series = drop_duplicated_instances(series)

#     if drop_dupl_slices:
#         _original_num_slices = len(series)
#         series = drop_duplicated_slices(series)
#         if len(series) < _original_num_slices:
#             warnings.warn(f'Dropped duplicated slices for series {series_instance_uid}.')

#     series = order_series(series)

#     image = stack_images(series, -1).astype(np.int16)
#     pixel_spacing = get_pixel_spacing(series).tolist()
#     slice_locations = get_slice_locations(series)
    
#     sop_uids = [str(get_tag(i, 'SOPInstanceUID')) for i in series]

#     return image, pixel_spacing, slice_locations, sop_uids


# def create_midrc_dataset(src, dst, joined_path = None):
#     src = Path(src)
#     dst = Path(dst)
#     os.makedirs(dst / 'images', exist_ok=True)
#     os.makedirs(dst / 'masks', exist_ok=True)
    
#     meta = []
#     if joined_path is not None:
#         joined = pd.read_csv(joined_path)
#     else:
#         joined = join_tree(src / "manifest-1608266677008" / 'MIDRC-RICORD-1A', verbose=2, ignore_extensions=[".csv"])
#         joined.to_csv(dst / 'joined.csv', index=False)
#     annotations = mdai.common_utils.json_to_dataframe(
#         src / 'MIDRC-RICORD-1a_annotations_labelgroup_all_2020-Dec-8.json')['annotations']
    
#     for series_uid, rows in tqdm(list(joined.groupby('SeriesInstanceUID'))):
#         files = {str(src / "manifest-1608266677008" / 'MIDRC-RICORD-1A' / row.PathToFolder / row.FileName) for _, row in rows.iterrows()}
#         series = list(map(pydicom.dcmread, files)) # list of dicom data (array + metadata)
        
#         try:
#             # image is 3D
#             # pixel_spacing, slice_locations and sop_uids are obtained from dicom metadata
#             #
#             image, pixel_spacing, slice_locations, sop_uids = get_image_data_from_dicom(series, series_uid, True)
#         except Exception as e:
#             print(f'Preparing ct {series_uid} failed with {e.__class__.__name__}: {str(e)}.')
#             continue

#         # calculate spacing. It seems spacing is not constant for all images
#         diffs, counts = np.unique(np.round(np.diff(slice_locations), decimals=5), return_counts=True)
#         spacing = np.float32([pixel_spacing[0], pixel_spacing[1], -diffs[np.argsort(counts)[-1]]])
        
#         # masks
#         mask = np.zeros(image.shape, dtype=bool)
#         for label, _rows in annotations[(annotations.SeriesInstanceUID == series_uid) &
#                                         (annotations.scope == 'INSTANCE')].groupby('labelName'):
            
#             if label in ['Infectious opacity', 'Infectious TIB/micronodules']:
#                 new_mask = np.zeros(image.shape, dtype=bool)
#                 for _, row in _rows.iterrows():
#                     slice_index = sop_uids.index(row['SOPInstanceUID'])
#                     if row['data'] is None:
#                         warnings.warn(
#                             f'{label} annotations for series {series_uid} contains None for slice {slice_index}.')
#                         continue
#                     ys, xs = np.array(row['data']['vertices']).T[::-1]
#                     new_mask[(*polygon(ys, xs, image.shape[:2]), slice_index)] = True

#                 if new_mask is not None:
#                     mask |= new_mask
                    
#         if mask.sum() == 0:
#             print(f'CT {series_uid} has empty mask, skipping it')
#             continue
        
#         # update metadata
#         meta.append({'ID': series_uid, 'CT': f'images/{series_uid}.npy.gz', 'mask': f'masks/{series_uid}.npy.gz',
#                      'x': spacing[0], 'y': spacing[1], 'z': spacing[2]})

#         save(image, dst / 'images' / f'{series_uid}.npy', compression=0)
#         save(mask, dst / 'masks' / f'{series_uid}.npy', compression=0)
#         # We cannot make cv2 imwrite because these are 3D images
#         # cv2.imwrite(str(dst / 'images' / f'{series_uid}.png'), image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
#         # cv2.imwrite(str(dst / 'masks' / f'{series_uid}.png'), mask, [cv2.IMWRITE_PNG_COMPRESSION, 0])

#     meta = pd.DataFrame.from_records(meta, index='ID')
#     meta.to_csv(dst / 'meta.csv', index_label='ID')

############################################
# MOSMED Processing functions
############################################

def read_mosmed_folder_structure(mosmed_path):
    meta_df = pd.read_excel(os.path.join(mosmed_path, "COVID19_1110/dataset_registry.xlsx"))
    meta_df["study_file"] = meta_df["study_file"].apply(lambda x: mosmed_path  + "/COVID19_1110" +x) 
    meta_df["mask_file"] = meta_df["mask_file"].apply(lambda x: mosmed_path + "/COVID19_1110"+ x if pd.notna(x) else "")
    return meta_df

#############################################
# CNCB Processing functions
#############################################

def read_cnbc_folder_structure(cnbc_path):
    cncb_images_parent_path = os.path.join(cnbc_path, "image")
    cncb_masks_parent_path = os.path.join(cnbc_path, "mask")
    images_list = []
    masks_list = []
    # glob all images .jpg
    for ct_scan in os.listdir(cncb_images_parent_path):
        if ct_scan != ".DS_Store":
            slices = os.listdir(os.path.join(cncb_images_parent_path, ct_scan))
            for slice in slices:
                data = {
                    "scan_index": int(ct_scan),
                    "slice_index": int(slice.split(".")[0]),
                    "image_path": os.path.join(cncb_images_parent_path, ct_scan, slice),
                    "n_slices": len(slices),
                }
                images_list.append(data)
            
    for mask in os.listdir(cncb_masks_parent_path):
        if mask != ".DS_Store":
            masks = os.listdir(os.path.join(cncb_masks_parent_path, mask))
            for slice in masks:
                data = {
                    "scan_index": int(mask),
                    "slice_index": int(slice.split(".")[0]),
                    "mask_path": os.path.join(cncb_masks_parent_path, mask, slice),
                    "n_masks": len(masks),
                }
                masks_list.append(data)
    images_df = pd.DataFrame(images_list)
    masks_df = pd.DataFrame(masks_list)
    df = pd.merge(images_df, masks_df, on=["scan_index", "slice_index"], how="left")
    return df

def generate_cnbc_infection_mask(mask, rgb=False):
    color_dict = {
        0: 0,
        1: 0, # We ignore the lung-field class because it refers to the lung only
        2: 1,
        3: 1,
    }
    return generate_semantic_mask(mask, color_dict)


#############################################
# CLEANED CNCB Processing functions
#############################################
label_dic = {
    "CP": "common-pneumonia",
    "NCP": "covid",
    "Normal": "normal",
}
@lru_cache(maxsize=1)
def read_cleaned_cncb_folder_structure(root_dir):
    images_dir = os.path.join(root_dir, "dataset_cleaned")
    masks_dir = os.path.join(root_dir, "dataset_seg")
    images_metadata_list = []
    masks_metadata_list = []
    
    for categ in tqdm(os.listdir(images_dir), desc="Reading cleaned cncb images...", total=len(os.listdir(images_dir))):
        categ_path = os.path.join(images_dir, categ)
        for case in os.listdir(categ_path): # covid, common-pneumonia, normal
            case_path = os.path.join(categ_path, case)
            for scan in os.listdir(case_path):
                scan_path = os.path.join(case_path, scan)
                n_slices = len(os.listdir(scan_path))
                for slice in os.listdir(scan_path):
                    if slice.endswith(".png"):
                        slice_path = os.path.join(scan_path, slice)
                        ct_slice_index = slice.split(".")[0]
                        images_metadata_list.append([label_dic[categ], int(case), int(scan), int(ct_slice_index), int(n_slices), slice_path])

    images_metadata_df = pd.DataFrame(images_metadata_list, columns=["label", "patient_index", "scan_index", "slice_index", "n_slices", "ct_slice_path"])
    
    for categ in tqdm(os.listdir(masks_dir), desc="Reading cleaned cncb masks...", total=len(os.listdir(masks_dir))):
        categ_path = os.path.join(masks_dir, categ)
        for case in os.listdir(categ_path):
            case_path = os.path.join(categ_path, case)
            for scan in os.listdir(case_path):
                scan_path = os.path.join(case_path, scan)
                n_slices = len(os.listdir(scan_path))
                for slice in os.listdir(scan_path):
                    if slice.endswith(".png"):
                        lung_mask_path = os.path.join(scan_path, slice)
                        ct_slice_index = slice.split(".")[0]
                        masks_metadata_list.append([label_dic[categ], int(case), int(scan), int(ct_slice_index), int(n_slices), lung_mask_path])
    
    mask_metadata_df = pd.DataFrame(masks_metadata_list, columns=["label", "patient_index", "scan_index", "slice_index", "n_slices", "lung_mask_path"])
    print(f"Len images metadata: {len(images_metadata_df)}")
    print(f"Len masks metadata: {len(mask_metadata_df)}")
    df = pd.merge(images_metadata_df, mask_metadata_df, on=["label", "patient_index", "scan_index", "slice_index", "n_slices"], how="left")
    df["dataset"] = "CNCB"
    return df


#############################################
# COVID-CT-MD Processing functions
#############################################
def read_covid_ct_md_folder_structure(root_dir):

    # Define the list of folders to iterate through
    folders = ['Normal Cases', 'Cap Cases', 'COVID-19 Cases']

    # Initialize an empty list to store the data
    data = []

    # Iterate through each folder
    for folder in folders:
        folder_path = os.path.join(root_dir, folder)
        
        # Iterate through each patient folder
        for patient_folder in os.listdir(folder_path):
            if patient_folder.startswith('.'):
                continue
            #remove all non numerical characters
            patient_id = ''.join(filter(str.isdigit, patient_folder))
            
            # Iterate through each .dcm file
            for file_name in os.listdir(os.path.join(folder_path, patient_folder)):
                if file_name.endswith('.dcm'):
                    slice_number = file_name.split('.dcm')[0].split('IM0')[1]
                    file_path = os.path.join(folder_path, patient_folder, file_name)
                    
                    # Determine the label based on the folder name
                    if folder == 'Normal Cases':
                        label = 'normal'
                    elif folder == 'Cap Cases':
                        label = 'common-pneumonia'
                    else:
                        label = 'covid'
                    
                    # Append the data to the list
                    data.append({'scan_index': patient_id, 'label': label, 'slice_index': slice_number, 'ct_slice_path': file_path})

    df = pd.DataFrame(data)
    return df
