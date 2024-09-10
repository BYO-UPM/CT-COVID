import argparse
import random
import wandb
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from models import get_model
from engine import train_model, validate_model, save_model
from dataset import ClassificationCovidDataset
from metrics import Metrics
 
WANDB_API_KEY = "66cd51a1dbd6025bc7240caae7c91c254022f0e1"

def main():
    args = parse_args()
    set_seeds(args.seed)

    model = get_model(args.model, args.bayesian_fg)
    model.to(device=args.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5, verbose=True)
    
    # Data
    metadata_df = pd.read_pickle(args.data_df_path)
    train_dataset = ClassificationCovidDataset(
        metadata_df,
        mode="train",
        k_fold=None,
        n_classes=3,
        img_size=(args.img_size,args.img_size),
        preprocess=None,
        transform=None,
        limit=args.batch_size*10
        )
    
    val_dataset = ClassificationCovidDataset(
        metadata_df,
        mode="val",
        k_fold=None,
        n_classes=3,
        img_size=(args.img_size,args.img_size),
        preprocess=None,
        transform=None,
        limit=args.batch_size*10
        )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    print("Created data loaders")
    print("Train dataset size: ", len(train_dataset))
    print("Validation dataset size: ", len(val_dataset))

    if args.use_wandb:
        WandbHelper(args).initialize_wandb(args)

    train_metrics = Metrics(args.use_wandb, n_classes=3)
    val_metrics = Metrics(args.use_wandb, n_classes=3)

    # Training and validation
    best_metric = 0
    for epoch in range(args.epochs):
        train_model(epoch, model, train_loader, optimizer, scheduler, train_metrics, args)
        validate_model(epoch, model, val_loader, val_metrics, args)
        best_metric, is_best = save_model(model, optimizer, args, val_metrics, epoch, best_metric, target_metric="accuracy", goal="maximize")
        scheduler.step(val_metrics.get_epoch_metrics(epoch, "val")["loss"])
    
    train_metrics.log_final_metrics(epoch, "train")
    val_metrics.log_final_metrics(epoch, "val")

    if args.use_wandb:
        wandb.finish()
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int, help="Seed")
    parser.add_argument("--data-df-path", default="/media/my_ftp/BasesDeDatos_Torax_RX_CT/COVID19_CT/processed/classification/classification_metadata.pkl", type=str, help="Dataframe path")
    parser.add_argument("--batch-size", default=16, type=int, help="Batch size")
    parser.add_argument("--epochs", default=50, type=int, help="Epochs")
    parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="Momentum")
    parser.add_argument("--weight-decay", default=1e-4, type=float, help="Weight decay")
    parser.add_argument("--num-workers", default=6, type=int, help="Number of workers")
    parser.add_argument("--img-size", default=256, type=int, help="Image size")
    parser.add_argument("--model", default="densenet121", type=str, help="Model")
    parser.add_argument("--bayesian-fg", default=False, type=bool, help="Bayesian FG")
    parser.add_argument("--pretrained", default=False, type=bool, help="Pretrained")
    parser.add_argument("--save", default="src/cls/results", type=str, help="Save path")
    parser.add_argument("--use-wandb", default=False, type=bool, help="Wandb")
    parser.add_argument("--wandb-project", default="tests", type=str, help="Wandb project")
    parser.add_argument("--wandb-run-name", default="test", type=str, help="Wandb run name")
    parser.add_argument("--save-pacience", default=5, type=int, help="Number of epochs to wait before saving model")
    parser.add_argument("--device", default="cuda:0", type=str, help="Device")
    parser.add_argument("--cuda", default=True, type=bool, help="Name")
    args = parser.parse_args()
    return args

def set_seeds(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

class WandbHelper:
    def __init__(self, args):
        if args.use_wandb:
            wandb.login(WANDB_API_KEY)
            self.config = args
    def initialize_wandb(self, args):
        wandb.init(project="covid-ct-detection", name=args.name, config=args, job_type="baseline")
        return True

if __name__ == "__main__":
    main()
