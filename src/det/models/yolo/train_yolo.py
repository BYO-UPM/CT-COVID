import wandb
from ultralytics import YOLO
import re
import os
from pathlib import Path
import argparse
import shutil
import math
import datetime

WANDB_API_KEY = "66cd51a1dbd6025bc7240caae7c91c254022f0e1"
YOLO_PRETRAINED_PATHS = {
    "yolov8m": "pretrained_models/weights/yolov8m.pt",
    "yolov8l": "pretrained_models/weights/yolov8l.pt",
    "yolov8x" : "pretrained_models/weights/yolov8x.pt",
}
YOLO_IMAGE_PATHS = {
    "normal": "models/yolo/yolo_normal.yaml",
    "enhanced": "models/yolo/yolo_enhanced.yaml",
}

def train_yolo(config=None):
    project_path = get_n_levels_up(3)
    with wandb.init(config=config) as run:
        config = wandb.config
        print(f"WANDB CONFIGURATION FOR THIS RUN : {config}")
        yolo_pretrained_path = os.path.join(project_path, YOLO_PRETRAINED_PATHS[args.model_size])
        print(f"YOLO PRETRAINED PATH : {yolo_pretrained_path}")
        model = YOLO(yolo_pretrained_path) 
        exp_name = generate_exp_name(config)
        #exp_name = parse_name(exp_name)
        data_yaml_path = os.path.join(project_path, "models", "yolo", "yolo_data.yaml")
        train_results = model.train(
            data=data_yaml_path,
            epochs=150,
            name=run.name,
            imgsz=640,
            device=0,
            batch=config.batch,
            lr0=config.lr0,
            lrf=config.lr0 / 10,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            cache=False,
            patience=10,
            save=True,
            save_period=-1,
            workers=6,
            project=args.results_folder,
            exist_ok=False,
            pretrained=True,
            optimizer="SGD",
            verbose=False,
            seed=0,
            deterministic=True,
            single_cls=True,
            rect=False,
            cos_lr=False,
            close_mosaic=10,
            resume=False,
            amp=True,
            fraction=1.0,
            profile=False,
            freeze=None,
            warmup_epochs=3.0,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            box=7.5,
            cls=0.5,
            dfl=1.5,
            pose=12.0,
            kobj=2.0,
            label_smoothing=0.0,
            nbs=64,
            dropout=0.0,
            )
        val_results = model.val()  # evaluate model performance on the validation set

def parse_args():
    parser = argparse.ArgumentParser(description='Yolo Sweep')
    parser.add_argument("--model-size", default="yolov8l", help="Choose betweeen yolov8m, yolov8l, yolov8x")
    parser.add_argument("--train-mode", default="single", type=str, help="Train type: single or sweep")
    parser.add_argument("--data-folder", default="/media/my_ftp/BasesDeDatos_Torax_RX_CT/COVID19_CT/processed/object_detection/yolo_annotations",type=str, help='Path to the folder with labels')
    parser.add_argument("--results-folder", default = "/media/my_ftp/TFTs/amoure/results")
    parser.add_argument("--labels-name", default="labels_nms_25",type=str, help='Path to the folder with labels')
    parser.add_argument("--images-name", default="images",type=str, help='Path to the folder with labels')
    args = parser.parse_args()
    return args

def main():
    global args
    args = parse_args()
    training_mode = args.train_mode
    print(f"TRAINING TYPE : {training_mode}")
    today = str(datetime.date.today())
    try:
        bash_rename_dir_to_labels(args.data_folder, args.labels_name)
        bash_raname_dir_to_images(args.data_folder, args.images_name)
        wandb.login(key=WANDB_API_KEY)
        if training_mode == "single":
            pass
            # wandb.login(key=WANDB_API_KEY)
            # config = {
            #     "lr0": 0.01,
            #     "lrf": 0.01,
            #     "momentum": 0.9,
            #     "batch": 16,
            #     "device": 0
            # }
            # train_yolo(config)

        elif training_mode == "sweep":
            sweep_config = {
            "cli_args": vars(args),
            "name": f"{today}_{args.labels_name}_sweep",
            'method': 'bayes',
            "metric": {
                "name":"metrics/mAP50(B)",
                "goal":"maximize"
            },
            "parameters": {
                "lr0": {
                    "distribution": "log_uniform",
                    "min": math.log(0.00001),
                    "max": math.log(0.01)
                },
                "momentum": {
                    "values": [0.7, 0.99]
                },
                "batch": {
                    "values": [16, 32]
                    
                },
                "weight_decay": {
                    "distribution": "log_uniform",
                    "min": math.log(0.00005),
                    "max": math.log(0.001)
                }

            }
            }
            sweep_id = wandb.sweep(sweep_config, project="covid-ct-detection")
            wandb.agent(sweep_id, train_yolo)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        bash_rename_to_original_name(args.data_folder, args.labels_name)
        bash_rename_images_to_original_name(args.data_folder, args.images_name)


def parse_name(name):
    parsed_name = re.sub(r'\W+', '-', name)
    return parsed_name

def generate_exp_name(config):
    lr0_rounded = round(config.lr0, 5)
    exp_name = f"yolov8l_lr0_{lr0_rounded}_mom_{config.momentum}_batch_{config.batch}_mosaic"
    return exp_name

def get_n_levels_up(n):
    path = os.path.abspath(__file__)
    for _ in range(n):
        path = os.path.dirname(path)
    return path

def bash_rename_dir_to_labels(yolo_annotations_folder, labels_name):
    source_folder = os.path.join(yolo_annotations_folder, labels_name)
    target_folder = os.path.join(yolo_annotations_folder, "labels/")
    shutil.move(source_folder, target_folder)
    print(f"Renamed {source_folder} to {target_folder}")

def bash_rename_to_original_name(yolo_annotations_folder, labels_name):
    source_folder = os.path.join(yolo_annotations_folder, "labels/")
    target_folder = os.path.join(yolo_annotations_folder, labels_name)
    shutil.move(source_folder, target_folder)
    print(f"Renamed {source_folder} to {target_folder}")

def bash_raname_dir_to_images(yolo_annotations_folder, images_name):
    source_folder = os.path.join(yolo_annotations_folder, images_name)
    target_folder = os.path.join(yolo_annotations_folder, "images/")
    shutil.move(source_folder, target_folder)
    print(f"Renamed {source_folder} to {target_folder}")

def bash_rename_images_to_original_name(yolo_annotations_folder, images_name):
    source_folder = os.path.join(yolo_annotations_folder, "images/")
    target_folder = os.path.join(yolo_annotations_folder, images_name)
    shutil.move(source_folder, target_folder)
    print(f"Renamed {source_folder} to {target_folder}")

if __name__ == '__main__':
    main()