import argparse
import wandb
from ultralytics import YOLO
from wandb.integration.ultralytics import add_wandb_callback


WANDB_API_KEY = "66cd51a1dbd6025bc7240caae7c91c254022f0e1"
YOLO_PRETRAINED_PATHS = {
    "yolov8m": "/media/my_ftp/TFTs/amoure/TFM_MUIT/pretrained_models/weights/yolov8m.pt",
    "yolov8l": "/media/my_ftp/TFTs/amoure/TFM_MUIT/pretrained_models/weights/yolov8l.pt",
    "yolov8x" : "/media/my_ftp/TFTs/amoure/TFM_MUIT/pretrained_models/weights/yolov8x.pt",
}


def parse_args():
    parser = argparse.ArgumentParser(description='Train a YOLOv5 model')
    # Add arguments
    parser.add_argument('--model', type=str, default=YOLO_PRETRAINED_PATHS['yolov8l'], help='path to model file')
    parser.add_argument('--data', type=str, default="/media/my_ftp/TFTs/amoure/TFM_MUIT/src/models/covid_yolo.yaml", help='path to data file')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--patience', type=int, default=50, help='epochs to wait for no observable improvement for early stopping')
    parser.add_argument('--batch', type=int, default=16, help='number of images per batch (-1 for AutoBatch)')
    parser.add_argument('--imgsz', type=int, default=640, help='size of input images as integer')
    parser.add_argument('--save', type=bool, default=True, help='save train checkpoints and predict results')
    parser.add_argument('--save_period', type=int, default=-1, help='save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--cache', type=str, default=False, help='True/ram, disk or False. Use cache for data loading')
    parser.add_argument('--device', type=str, default="[0, 1]", help='device to run on')
    parser.add_argument('--workers', type=int, default=8, help='number of worker threads for data loading')
    parser.add_argument('--project', type=str, default=None, help='project name')
    parser.add_argument('--name', type=str, default=None, help='experiment name')
    parser.add_argument('--exist_ok', type=bool, default=False, help='whether to overwrite existing experiment')
    parser.add_argument('--pretrained', type=bool, default=True, help='whether to use a pretrained model')
    parser.add_argument('--optimizer', type=str, default='auto', help='optimizer to use')
    parser.add_argument('--verbose', type=bool, default=False, help='whether to print verbose output')
    parser.add_argument('--seed', type=int, default=0, help='random seed for reproducibility')
    parser.add_argument('--deterministic', type=bool, default=True, help='whether to enable deterministic mode')
    parser.add_argument('--single_cls', type=bool, default=False, help='train multi-class data as single-class')
    parser.add_argument('--rect', type=bool, default=False, help='rectangular training with each batch collated for minimum padding')
    parser.add_argument('--cos_lr', type=bool, default=False, help='use cosine learning rate scheduler')
    parser.add_argument('--close_mosaic', type=int, default=10, help='disable mosaic augmentation for final epochs')
    parser.add_argument('--resume', type=bool, default=False, help='resume training from last checkpoint')
    parser.add_argument('--amp', type=bool, default=True, help='Automatic Mixed Precision (AMP) training')
    parser.add_argument('--fraction', type=float, default=1.0, help='dataset fraction to train on')
    parser.add_argument('--profile', type=bool, default=False, help='profile ONNX and TensorRT speeds during training for loggers')
    parser.add_argument('--freeze', type=int, nargs='+', default=None, help='freeze first n layers or list of layer indices during training')
    parser.add_argument('--lr0', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--lrf', type=float, default=0.01, help='final learning rate')
    parser.add_argument('--momentum', type=float, default=0.937, help='SGD momentum/Adam beta1')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='optimizer weight decay')
    parser.add_argument('--warmup_epochs', type=float, default=3.0, help='warmup epochs')
    parser.add_argument('--warmup_momentum', type=float, default=0.8, help='warmup initial momentum')
    parser.add_argument('--warmup_bias_lr', type=float, default=0.1, help='warmup initial bias lr')
    parser.add_argument('--box', type=float, default=7.5, help='box loss gain')
    parser.add_argument('--cls', type=float, default=0.5, help='cls loss gain')
    parser.add_argument('--dfl', type=float, default=1.5, help='dfl loss gain')
    parser.add_argument('--pose', type=float, default=12.0, help='pose loss gain')
    parser.add_argument('--kobj', type=float, default=2.0, help='keypoint obj loss gain')
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='label smoothing')
    parser.add_argument('--nbs', type=int, default=64, help='nominal batch size')
    parser.add_argument('--overlap_mask', type=bool, default=True, help='masks should overlap during training')
    parser.add_argument('--mask_ratio', type=int, default=4, help='mask downsample ratio')
    parser.add_argument('--dropout', type=float, default=0.0, help='use dropout regularization')
    parser.add_argument('--val', type=bool, default=True, help='validate/test during training')
    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    wandb.login(key=WANDB_API_KEY)
    wandb.init(project="covid-ct-detection", name=args.name, config=args, job_type="baseline")
    # read from the --model argument
    model = YOLO(args.model)
    add_wandb_callback(model, enable_model_checkpointing=True)
    # Train the model
    model.train(vars(args))
    # Validate the model
    model.val()

if __name__ == '__main__':
    main()
    


