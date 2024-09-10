from functools import lru_cache
import os
import numpy as np
import cv2
import pandas as pd
from typing import List
import numpy as np
from skimage.measure import label, regionprops
import json
from sklearn.cluster import OPTICS
from tqdm.notebook import tqdm
import utils as Utils
import hdbscan
from functools import lru_cache
import matplotlib.pyplot as plt
import plotly.express as px

@lru_cache(maxsize=None)
def read_mask(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    return mask

def get_bbox_from_row(box):
    x1,y1,x2,y2 = box['x_min'],box['y_min'], box['x_max'] , box['y_max']
    return (int(x1),int(y1),int(x2),int(y2))

def scale_bbox(original_size, bboxes, target_size):
    # Get scaling factor
    scale_x = original_size[0]/target_size[0]
    scale_y = original_size/target_size[1]
    
    scaled_bboxes = []
    for bbox in bboxes:
        x = int(np.round(bbox[0]/scale_y, 4))
        y = int(np.round(bbox[1]/scale_x, 4))
        x1 = int(np.round(bbox[2]/scale_y, 4))
        y1= int(np.round(bbox[3]/scale_x, 4))

        scaled_bboxes.append([x, y, x1, y1]) # xmin, ymin, xmax, ymax
        
    return scaled_bboxes

def scale_image_and_bbox_from_row(row, target_size):
    image = cv2.imread(row['ct_slice_path'], cv2.IMREAD_UNCHANGED)
    image = cv2.resize(image, target_size)
    bboxes = scale_bbox(image.shape, row['bboxes'], target_size)
    return image, bboxes

def get_bbox_from_segmentation(mask, oriented=False, use_width_and_height=False):
    if oriented:
        rect = cv2.minAreaRect(mask)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        return box
    else:
        x, y, w, h = cv2.boundingRect(mask)
        return [x, y, x+w, y+h]


def get_bbox_from_seg_row(row, column_name, oriented=False):
    mask = cv2.imread(row[column_name], cv2.IMREAD_UNCHANGED)
    return get_bbox_from_segmentation(mask, oriented)
    
def read_txt_bounding_boxes(txt_file):
    with open(txt_file, 'r') as f:
        lines = f.readlines()
        rows = []
        for line in lines:
            line = line.strip().split(' ')
            rows.append([line[0], int(line[1]),
                           int(float(line[2])), int(float(line[3])),int(float(line[4])), int(float(line[5]))])
        return rows

def txt_bbox_to_df(txt_file):
    rows = read_txt_bounding_boxes(txt_file)
    df = pd.DataFrame(rows, columns=['image_id', "label", 'x_min', 'y_min', 'x_max', 'y_max'])
    return df

def get_bboxes_from_seg_row(row, column_name="infection_mask_path"):
    mask = cv2.imread(row[column_name],cv2.IMREAD_UNCHANGED)

    # Label the islands (connected components) in the binary mask
    labels = label(mask)

    # Get the properties of each labeled region
    props = regionprops(labels)

    # Get the bounding box of each island

    return [[prop.bbox[1], prop.bbox[0], prop.bbox[3], prop.bbox[2]] for prop in props]


def non_max_suppression(boxes, overlapThresh):
    if boxes is not None and len(boxes) > 0:
        boxes = np.array(boxes)
        if boxes.ndim == 1:
            boxes = boxes[np.newaxis, :]
        pick = []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(area)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]
            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

        return boxes[pick].tolist()
    else:
        return boxes

def merge_overlapping_boxes(bboxes, overlapThresh):
    if bboxes is not None:
        def iou(box1, box2):
            x1, y1, x2, y2 = box1
            x1_, y1_, x2_, y2_ = box2

            xi1, yi1, xi2, yi2 = max(x1, x1_), max(y1, y1_), min(x2, x2_), min(y2, y2_)
            inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

            box1_area = (x2 - x1) * (y2 - y1)
            box2_area = (x2_ - x1_) * (y2_ - y1_)

            union_area = box1_area + box2_area - inter_area

            return inter_area / union_area

        def merge_boxes(box1, box2):
            x1, y1, x2, y2 = box1
            x1_, y1_, x2_, y2_ = box2

            return [min(x1, x1_), min(y1, y1_), max(x2, x2_), max(y2, y2_)]

        boxes = np.array(bboxes)
        merged = True
        while merged:
            merged = False
            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    if iou(boxes[i], boxes[j]) > overlapThresh:
                        boxes[i] = merge_boxes(boxes[i], boxes[j])
                        boxes = np.delete(boxes, j, axis=0)
                        merged = True
                        break
                if merged:
                    break
        return boxes.tolist()
    else:
        return bboxes

def hdbscan_bboxes(boxes, min_cluster_size=3, min_samples=None, cluster_selection_epsilon=0.5, use_sizes=False, n_boxes_threshold=None):
    if boxes is None or len(boxes) < min_cluster_size:
        return boxes
    
    # Skip clustering if there are less than n_boxes_thres boxes
    if n_boxes_threshold is not None and len(boxes) < n_boxes_threshold:
        return boxes
    
    boxes = np.array(boxes)
    min_coord = np.min(boxes)
    max_coord = np.max(boxes)
    # Normalize the bounding box coordinates to the range [0,1]
    boxes_normalized = (boxes - min_coord) / (max_coord - min_coord)
    # Calculate the centers and optionally the sizes of the normalized bounding boxes
    if use_sizes:
        data = np.array([[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2, box[2] - box[0], box[3] - box[1]] for box in boxes_normalized])
    else:
        data = np.array([[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2] for box in boxes_normalized])

    # Use HDBSCAN to cluster the data
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, cluster_selection_epsilon=cluster_selection_epsilon)
    labels = clusterer.fit_predict(data)

    # For each cluster, find the bounding box that encloses all bounding boxes in the cluster
    merged_boxes = []
    for label in set(labels):
        cluster_boxes = boxes_normalized[labels == label]
        if label == -1:
            merged_boxes.extend(cluster_boxes.tolist())
        elif len(cluster_boxes) >= min_cluster_size:
            x1 = min(cluster_boxes[:, 0])
            y1 = min(cluster_boxes[:, 1])
            x2 = max(cluster_boxes[:, 2])
            y2 = max(cluster_boxes[:, 3])
            merged_boxes.append([x1, y1, x2, y2])

    # Denormalize the merged bounding boxes to the original scale
    merged_boxes = (np.array(merged_boxes) * (max_coord - min_coord)) + min_coord

    return merged_boxes.tolist()

def optics_bboxes(boxes, min_samples=2, xi=0.05, min_cluster_size=None, use_sizes=False):
    if boxes is None or len(boxes) < min_samples:
        return boxes
    # Convert boxes to numpy array for easier manipulation
    boxes = np.array(boxes)

    # Calculate the min and max of the bounding box coordinates for normalization
    min_coord = np.min(boxes)
    max_coord = np.max(boxes)

    # Normalize the bounding box coordinates to the range [0,1]
    boxes_normalized = (boxes - min_coord) / (max_coord - min_coord)

    # Calculate the centers and optionally the sizes of the normalized bounding boxes
    if use_sizes:
        data = np.array([[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2, box[2] - box[0], box[3] - box[1]] for box in boxes_normalized])
    else:
        data = np.array([[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2] for box in boxes_normalized])

    # Use OPTICS to cluster the data
    clusterer = OPTICS(min_samples=min_samples, xi=xi, min_cluster_size=min_cluster_size)
    labels = clusterer.fit_predict(data)

    # If no clusters are found, return the original boxes
    if len(set(labels)) == 1 and -1 in labels:
        return boxes.tolist()

    # For each cluster, find the bounding box that encloses all bounding boxes in the cluster
    merged_boxes = []
    for label in set(labels):
        if label == -1:
            continue  # Ignore noise (outliers)
        cluster_boxes = boxes_normalized[labels == label]
        x1 = min(cluster_boxes[:, 0])
        y1 = min(cluster_boxes[:, 1])
        x2 = max(cluster_boxes[:, 2])
        y2 = max(cluster_boxes[:, 3])
        merged_boxes.append([x1, y1, x2, y2])

    # Denormalize the merged bounding boxes to the original scale
    merged_boxes = (np.array(merged_boxes) * (max_coord - min_coord)) + min_coord

    return merged_boxes.tolist()


def calculate_hdbscan_bboxes_and_iou(df, min_samples, min_cluster_size, epsilons, post_nms_iou_threshold=0.25, source_bbox_col_name=None, use_sizes=False, column_sufix=None, n_boxes_threshold=None, skip_iou_calculation=False):
    iou_scores = []
    mask_uncovered_list = []
    for epsilon in epsilons:
        epsilon_nm = str(epsilon).replace(".", "_")
        column_name = f"merged_hdbscan_bboxes_e{epsilon_nm}"
        if column_sufix is not None:
            column_name += f"_{column_sufix}"
        iou_column_name = f"{column_name}_iou"
        mask_uncovered_column_name = f"{column_name}_mask_uncovered"

        df[column_name] = (
            df[source_bbox_col_name]
            .progress_apply(hdbscan_bboxes, min_cluster_size=min_cluster_size, min_samples=min_samples, cluster_selection_epsilon=epsilon, use_sizes=use_sizes, n_boxes_threshold=n_boxes_threshold)
            .progress_apply(merge_overlapping_boxes, overlapThresh=1e-9)
            .progress_apply(non_max_suppression, overlapThresh=post_nms_iou_threshold)
        )
        if not skip_iou_calculation:
            df[iou_column_name] = df.progress_apply(lambda row: udf_iou_custom_loss(row, "infection_mask_path", column_name), axis=1)
            df[mask_uncovered_column_name] = df.progress_apply(lambda row: udf_mask_uncovered(row, "infection_mask_path", column_name), axis=1)

            iou_score = df[iou_column_name].sum() / df[df["has_infection"]==True].shape[0]
            mask_uncovered = df[mask_uncovered_column_name].sum() / df[df["has_infection"]==True].shape[0]
            print(f"{iou_column_name}: ", iou_score)
            print(f"{mask_uncovered_column_name}: ", mask_uncovered)
            iou_scores.append(iou_score)
            mask_uncovered_list.append(mask_uncovered)
    
    if not skip_iou_calculation:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        ax1.plot(epsilons, iou_scores, marker='o')
        ax1.set_title('IOU scores per epsilon value')
        ax1.set_xlabel('Epsilon value')
        ax1.set_ylabel('IOU score')
        ax2.plot(epsilons, mask_uncovered_list, marker='o')
        ax2.set_title('Mask uncovered per epsilon value')
        ax2.set_xlabel('Epsilon value')
        ax2.set_ylabel('Mask uncovered')
    return df

def calculate_optics_bboxes_and_iou(df, xis, min_samples=2):
    iou_scores = []
    for xi in xis:
        column_name = f"merged_optics_bboxes_xi{xi}"
        iou_column_name = f"{column_name}_iou"

        df[column_name] = (
            df["nms_bboxes_50%"]
            .progress_apply(optics_bboxes, min_samples=min_samples, xi=xi)
            .progress_apply(merge_overlapping_boxes, overlapThresh=1e-9)
            .progress_apply(non_max_suppression, overlapThresh=0.25)
        )

        df[iou_column_name] = df.apply(lambda row: udf_iou_custom_loss(row, "infection_mask_path", column_name), axis=1)

        iou_score = df[iou_column_name].sum() / df[df["has_infection"]==True].shape[0]
        print(f"{iou_column_name}: ", iou_score)
        iou_scores.append(iou_score)

    # Plot IOU scores per xi value
    plt.plot(xis, iou_scores, marker='o')
    plt.title('IOU scores per xi value')
    plt.xlabel('Xi value')
    plt.ylabel('IOU score')
    plt.show()
    return df

def create_mask_from_bbox(image_shape, bbox, offset=0):
    """
    Create a binary mask from a bounding box.

    Parameters:
    image_shape: tuple, shape of the output mask (height, width)
    bbox: tuple, bounding box coordinates (x1, y1, x2, y2)
    offset: int, offset value to increase the size of the bounding box

    Returns:
    mask: 2D numpy array of shape image_shape
    """
    mask = np.zeros(image_shape)
    x1, y1, x2, y2 = bbox
    # Apply offset to the bounding box
    x1 -= offset
    y1 -= offset
    x2 += offset
    y2 += offset
    # cast to int to avoid errors with numpy slicing
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    mask[y1:y2, x1:x2] = 1
    return mask


def iou_custom_loss(mask_path, bboxes_list):
    if bboxes_list is None or len(bboxes_list) == 0:
        return 0
    
    total_iou = 0

    mask = read_mask(mask_path)

    if not isinstance(bboxes_list[0], list):
        bboxes_list = [bboxes_list]

    for bbox in bboxes_list:
        bbox_mask = create_mask_from_bbox(mask.shape, bbox)
        # Apply the bounding box mask to the original mask
        masked_mask = mask * (bbox_mask > 0)
        intersection = np.sum((bbox_mask > 0) & (masked_mask > 0))
        union = np.sum((bbox_mask > 0) | (masked_mask > 0))
        if union == 0:
            continue  # Avoid division by zero
        iou = intersection / union
        total_iou += iou

    return total_iou/len(bboxes_list)

def mask_uncovered(mask_path, bboxes_list, bbox_offset=0):
    if bboxes_list is None or len(bboxes_list) == 0:
        return 1
    original_mask = read_mask(mask_path)
    original_mask[original_mask > 0] = 1
    total_mask_area = np.sum(original_mask)
    if not isinstance(bboxes_list[0], list):
        bboxes_list = [bboxes_list]
    mask_AND_bbox = np.zeros_like(original_mask)
    for bbox in bboxes_list:
        bbox_mask = create_mask_from_bbox(original_mask.shape, bbox, bbox_offset)
        mask_AND_bbox = np.logical_or(mask_AND_bbox, np.logical_and(bbox_mask, original_mask))
    uncovered_mask = original_mask - mask_AND_bbox
    return np.sum(uncovered_mask)/ total_mask_area


def udf_mask_uncovered(row, mask_path_col_name, bboxes_col_name, bbox_offset=0):
    if row["has_infection"] == False:
        return 0
    return mask_uncovered(row[mask_path_col_name], row[bboxes_col_name], bbox_offset)

def udf_iou_custom_loss(row, mask_path_col_name, bboxes_col_name):
    if row["has_infection"] == False:
        return 0
    return iou_custom_loss(row[mask_path_col_name], row[bboxes_col_name])


@lru_cache(maxsize=None)
def generate_kaggle_dataset(path):
    bbox_df_train = txt_bbox_to_df(path + "/train_COVIDx_CT-3A.txt")
    bbox_df_val = txt_bbox_to_df(path + "/val_COVIDx_CT-3A.txt")
    bbox_df_test = txt_bbox_to_df(path + "/test_COVIDx_CT-3A.txt")
    bbox_df = pd.concat([bbox_df_train, bbox_df_val, bbox_df_test])
    file_names = os.listdir(path + "/3A_images")
    file_paths_df = pd.DataFrame(file_names, columns=['image_id'])
    file_paths_df["image_path"] = path + "/3A_images/" + file_paths_df["image_id"] 
    file_paths_df = file_paths_df.merge(bbox_df, on="image_id", how="left")
    file_paths_df["database_id"] = file_paths_df["image_id"].apply(lambda x: x.split("-")[0])
    return file_paths_df

def draw_image_and_bboxes(image, boxes, color, label="Label", label_size=0.5):
    output = image.copy()

    for box in boxes:
        text_width, text_height = cv2.getTextSize(label.upper(), cv2.FONT_HERSHEY_SIMPLEX, label_size, 1)[0]
        output = cv2.rectangle(output, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
        cv2.putText(output, label.upper(), (int(box[0]), int(box[1])-5), cv2.FONT_HERSHEY_SIMPLEX, label_size, (255, 0, 0), 4, cv2.LINE_AA)

    return output

def plot_bboxes_histogram(df, column="original_bbox", figsize=(8, 6)):
    df["n_bboxes"] = df[column].apply(lambda x: len(x) if x is not None else 0)
    fig = px.histogram(df, x="n_bboxes", nbins=int(df["n_bboxes"].max() + 1), title="Number of bounding boxes per image")
    fig.update_xaxes(title_text='Number of bounding boxes')
    fig.update_yaxes(title_text='Number of images')
    fig.update_layout(width=figsize[0], height=figsize[1])
    fig.show()


def create_coco_json(df, bboxes_col_name, has_bbox_col_name, output_file, relative_path = True):
    data_dict = {}
    data_dict['images'] = []
    data_dict['annotations'] = []
    data_dict['categories'] = []

    # Get unique labels
    labels = df['label'].unique()

    # Create categories
    for i, label in enumerate(labels):
        category = {}
        category['id'] = i + 1
        category['name'] = label
        category['supercategory'] = 'none'
        data_dict['categories'].append(category)

    df = df[df[has_bbox_col_name] == True].copy()
    # Create images and annotations
    for i, row in df.iterrows():
        image = {}
        image['id'] = i + 1
        if relative_path:
            image['file_name'] = row['ct_slice_path'].split('/')[-1]
        else:
            image['file_name'] = row['ct_slice_path']
        image['height'] = int(row['height'])
        image['width'] = int(row['width'])
        data_dict['images'].append(image)

        for bbox in row[bboxes_col_name]:
            annotation = {}
            annotation['id'] = len(data_dict['annotations']) + 1
            annotation['image_id'] = i + 1
            # The COCO box format is [top left x, top left y, width, height]
            bbox_coco = [int(bbox[0]), int(bbox[1]), int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])]
            annotation['bbox'] = bbox_coco
            annotation['category_id'] = int(np.where(labels == row['label'])[0][0] + 1)
            annotation['area'] = int(bbox_coco[2] * bbox_coco[3])
            annotation['iscrowd'] = 0
            data_dict['annotations'].append(annotation)

    # Write JSON file
    if output_file is not None:
        with open(output_file, 'w') as json_file:
            json.dump(data_dict, json_file, indent=4)
    return data_dict

def create_unlabeled_coco(source_coco_json_path, images_dir, output_file):
    with open(source_coco_json_path, 'r') as f:
        data = json.load(f)
    categories = data['categories']
    unlabeled_images = Utils.get_all_files(images_dir, extensions=[".png"])
    data_dict = {}
    data_dict['images'] = []
    data_dict['annotations'] = []
    data_dict['categories'] = categories
    for i, image_path in enumerate(unlabeled_images):
        image = {}
        image['id'] = i + 1
        image['file_name'] = image_path.split('/')[-1]
        image['height'] = 512
        image['width'] = 512
        data_dict['images'].append(image)
    # Write JSON file
    if output_file is not None:
        with open(output_file, 'w') as json_file:
            json.dump(data_dict, json_file, indent=4)
    return data_dict

def convert_coco_to_yolo(labels_dir, save_dir, print_file_names=False):
    # Load the COCO format json file
    with open(labels_dir, 'r') as f:
        data = json.load(f)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Iterate over annotations
    for ann in tqdm(data['annotations']):
        # Get image details
        image_id = ann['image_id']
        image_data = next((img for img in data['images'] if img['id'] == image_id), None)
        if not image_data:
            continue

        # Calculate YOLO format bounding box
        x_center = ann['bbox'][0] + ann['bbox'][2] / 2
        y_center = ann['bbox'][1] + ann['bbox'][3] / 2
        width = ann['bbox'][2]
        height = ann['bbox'][3]

        # Normalize bounding box by image size
        x_center /= image_data['width']
        y_center /= image_data['height']
        width /= image_data['width']
        height /= image_data['height']

        # Write to file
        file_name = os.path.join(save_dir, f"{image_data['file_name'].split('.')[0].split('/')[-1]}.txt")
        if print_file_names:
            print(file_name)
        # It needs to be a new file because we are appending all possible annotations for the same image
        # as COCO annotations are 1 entry per bounding box
        with open(file_name, 'a') as f:
            f.write(f"{ann['category_id']} {x_center} {y_center} {width} {height}\n")


def plot_object_detection_df(df, sample_size, *columns, figsize=None, samplimg_method=None, print_overlap_percentage=False):
    df_to_plot = df[df["has_infection"] == True].copy()
    df_to_plot["n_original_bboxes"] = df_to_plot["original_bbox"].apply(lambda x: len(x) if x is not None else 0)
    df_to_plot = df_to_plot.sort_values(by="n_original_bboxes", ascending=False)
    if samplimg_method =='random':
        df_to_plot = df_to_plot.sample(sample_size)
    if figsize is None:
        figsize = (np.ceil(sample_size/2)*10, len(columns)//2*10)
    fig = plt.figure(figsize=figsize)
    rows = len(columns) + 1
    for i in range(1, sample_size+1):
        row = df_to_plot.iloc[i-1]
        img = cv2.imread(row["ct_slice_path"])
        infection_mask = cv2.imread(row["infection_mask_path"], cv2.IMREAD_UNCHANGED)
        file_name = row["ct_slice_path"].split("/")[-1]

        fig.add_subplot(rows, sample_size, i)
        plt.imshow(img, cmap="gray")
        plt.imshow(infection_mask, alpha=0.5, cmap="jet")
        plt.title(file_name, fontsize=7)

        for j, column in enumerate(columns, start=1):
            bboxes = row[column]
            image_with_bboxes = draw_image_and_bboxes(img, bboxes, (255,0,0), "covid")

            fig.add_subplot(rows, sample_size, j*sample_size + i)
            bbox_img_title = f"{column}: {len(bboxes)} boxes "
            if print_overlap_percentage:
                uncovered_percentage = row[f"%uncovered_mask_{column}"]
                bbox_img_title = bbox_img_title + "{:.4f}".format((1-uncovered_percentage)*100) + "%"
            plt.title(bbox_img_title, fontsize=7)
            plt.imshow(image_with_bboxes)

    plt.show()