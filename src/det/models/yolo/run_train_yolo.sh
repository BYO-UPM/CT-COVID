#!/bin/bash

source /media/my_ftp/TFTs/amoure/TFM_MUIT/tfm-muit-venv/bin/activate

python /media/my_ftp/TFTs/amoure/TFM_MUIT/src/det/models/yolo/train_yolo.py \
--model-size yolov8l \
--train-mode sweep \
--results-folder /media/my_ftp/TFTs/amoure/results/ \
--labels-name labels_merged_hdbscan_bboxes_e0_1_threshold_10 \
--images-name enhanced_images
