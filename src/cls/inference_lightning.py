import argparse
import torch
import random
import numpy as np
import pandas as pd
import lightning as L

from models import get_model
from models_lightning import LitBayesianModel
from data_lightning import ClassificationCovidDataModule


def main():
    args = parse_args()
    data_module = ClassificationCovidDataModule(
        metadata_path=args.data_df_path,
        k_fold=None,
        n_classes=3,
        img_size=(args.img_size,args.img_size),
        preprocess=None,
        train_transforms=None,
        num_workers=2,
        batch_size=16,
        segment_ct=False,
    )
    checkpoint_dir = args.checkpoint_dir
    checkpoint_dir = "/media/my_ftp/TFTs/amoure/results/cls/test/lightning_logs/version_0/checkpoints/densenet121best-epoch=05-val_loss=1.00.ckpt"
    pytorch_model = get_model(args.model, 3, bayesian_fg=True)
    loaded_lightning_model = LitBayesianModel.load_from_checkpoint(checkpoint_dir, model = pytorch_model, n_classes=3, optimizer="sgd", lr=1e-4, momentum=0.9, weight_decay=1e-4, bayesian_fg=True)
    tester = L.Trainer(accelerator="cpu", limit_test_batches=5, default_root_dir=args.save)
    test_results = tester.test(loaded_lightning_model, data_module)
    print(test_results)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-df-path", default="/media/my_ftp/BasesDeDatos_Torax_RX_CT/COVID19_CT/processed/classification/classification_metadata.pkl", type=str, help="Dataframe path")
    parser.add_argument("--num-workers", default=6, type=int, help="Number of workers")
    parser.add_argument("--img-size", default=256, type=int, help="Image size")
    parser.add_argument("--model", default="densenet121", type=str, help="Model")
    parser.add_argument("--checkpoint_dir", default="/media/my_ftp/TFTs/amoure/results/cls/test/lightning_logs/version_0/checkpoints", type=str, help="Checkpoint directory")
    parser.add_argument("--device", default="1", type=str, help="Device")
    parser.add_argument("--cuda", default=False, type=bool, help="Name")
    parser.add_argument("--save", default="/media/my_ftp/TFTs/amoure/results/cls/test/", type=str, help="Save path")
    args = parser.parse_args()
    return args




if __name__ == "__main__":
    main()