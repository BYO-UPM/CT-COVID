import os
import sys
import pandas as pd
import configparser
import utils.utils as Utils
import utils.bbox_utils as BboxUtils
from tqdm import tqdm
tqdm.pandas()
config = configparser.ConfigParser()
config.read("config.ini")

PROCESSED_DATA_PATH = config["PATHS"]["PROCESSED_DATA_PATH"]
bounding_boxes_col_names = ["nms_bboxes_25", "merged_hdbscan_bboxes_e0_01", "merged_hdbscan_bboxes_e0_1_threshold_10"]
path_coco_annons = os.path.join(PROCESSED_DATA_PATH, "object_detection", "coco_annotations")

def main():
    processed_metadata_df = pd.read_pickle(os.path.join(PROCESSED_DATA_PATH, "object_detection", "metadata_with_bboxes.pkl"))
    print(f"Size full dataset : {processed_metadata_df.shape}")
    ## New: Leave COVID_CT_JUNMA out as an independant testing dataset
    processed_metadata_df = processed_metadata_df[processed_metadata_df["dataset"]!='COVID_CT_JunMa']
    print(f"Size filtered dataset : {processed_metadata_df.shape}")
    print(processed_metadata_df.groupby(["split"]).size())
    path_yolo_annos = os.path.join(PROCESSED_DATA_PATH, "object_detection", "yolo_annotations")

    # Copy normal images
    print("Copying normal images")
    # path_yolo_train_images = os.path.join(path_yolo_annos,"normal_images", "train")
    # path_yolo_val_images = os.path.join(path_yolo_annos, "normal_images", "val")
    # os.makedirs(path_yolo_train_images, exist_ok=True)
    # os.makedirs(path_yolo_val_images, exist_ok=True)

    # (processed_metadata_df[processed_metadata_df["split"]=="train"]["ct_slice_path"]
    # .progress_apply(
    #     lambda path:
    #     Utils.copy_image_to_folder(path, os.path.join(path_yolo_train_images, path.split("/")[-1])))
    # )

    # (processed_metadata_df[processed_metadata_df["split"]=="test"]["ct_slice_path"]
    # .progress_apply(
    #     lambda path: Utils.copy_image_to_folder(path, os.path.join(path_yolo_val_images, path.split("/")[-1])))
    #     )
    
    # Copy enhanced images
    print("Copying enhanced images")
    path_yolo_train_images = os.path.join(path_yolo_annos, "enhanced_images", "train")
    path_yolo_val_images = os.path.join(path_yolo_annos, "enhanced_images", "val")
    os.makedirs(path_yolo_train_images, exist_ok=True)
    os.makedirs(path_yolo_val_images, exist_ok=True)

    # We read from the enhanced folder (with hist eq and clahe) and copy to the yolo folder
    (processed_metadata_df[processed_metadata_df["split"]=="train"]["ct_slice_path"]
    .progress_apply(
        lambda path:
        Utils.copy_image_to_folder(path.replace("rgb_images", "enhanced"), os.path.join(path_yolo_train_images, path.split("/")[-1])))
    )

    (processed_metadata_df[processed_metadata_df["split"]=="test"]["ct_slice_path"]
    .progress_apply(
        lambda path: Utils.copy_image_to_folder(path.replace("rgb_images", "enhanced"), os.path.join(path_yolo_val_images, path.split("/")[-1])))
    )

    # Generate yolo annotations
    print("Generating yolo annotations")
    for bbox_col_name in bounding_boxes_col_names:
        path_bbox_col_name = os.path.join(path_coco_annons, bbox_col_name)
        coco_train_annos_path = os.path.join(path_bbox_col_name, "coco_train.json")
        coco_val_annos_path = os.path.join(path_bbox_col_name, "coco_val.json")
        path_yolo_train_annos = os.path.join(path_yolo_annos,f"labels_{bbox_col_name}", "train")
        path_yolo_val_annos = os.path.join(path_yolo_annos, f"labels_{bbox_col_name}","val",)
        # create all needed directories
        os.makedirs(path_yolo_train_annos, exist_ok=True)
        os.makedirs(path_yolo_val_annos, exist_ok=True)
        print(path_yolo_train_annos)
        print(path_yolo_val_annos)
        BboxUtils.convert_coco_to_yolo(coco_train_annos_path, path_yolo_train_annos, False)
        BboxUtils.convert_coco_to_yolo(coco_val_annos_path, path_yolo_val_annos, False)
    
if __name__ == "__main__":
    main()