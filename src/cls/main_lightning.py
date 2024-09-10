import argparse
import random
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.loggers import WandbLogger
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary
from data_lightning import ClassificationCovidDataModule
from models_lightning import get_lightning_model

WANDB_API_KEY = "66cd51a1dbd6025bc7240caae7c91c254022f0e1"

def train_test_model(args):
    data_module = ClassificationCovidDataModule(
    metadata_path=args.data_df_path,
    k_fold=None,
    n_classes=3,
    img_size=(args.img_size,args.img_size),
    preprocess=None,
    train_transforms=None,
    num_workers=args.num_workers,
    batch_size=args.batch_size,
    segment_ct=False,
    )
    
    callbacks = [
    ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min", filename=args.model+ "best-{epoch:02d}-{val_loss:.2f}"),
    EarlyStopping(monitor="val_loss", patience=args.save_pacience, mode="min", verbose=True),
    ModelSummary(max_depth=3)]

    # Method to be called by wandb sweep
    model = get_lightning_model(
    model_nm=args.model,
    n_classes=3,
    optimizer=args.optimizer,
    lr=args.lr,
    momentum=args.momentum,
    weight_decay=args.weight_decay,
    batch_size=args.batch_size,
    n_mc_samples=args.n_mc_samples,
    bayesian_fg=args.bayesian_fg,
    )

    wandb_logger = WandbLogger(
        project=args.wandb_project,
        name=args.wandb_run_name,
        log_model=False 
        )

    trainer = L.Trainer(
        max_epochs=args.epochs,
        deterministic=False,
        callbacks=callbacks,
        accelerator="cuda" if args.cuda else "cpu",
        devices= args.device if args.cuda else 1, 
        default_root_dir=args.save,
        limit_train_batches=2,
        limit_val_batches=2,
        limit_test_batches=2,
        logger=wandb_logger
    )

    # trainer.fit(model, data_module)
    data_module.setup()
    trainer.test(model, ckpt_path=args.save, dataloaders=data_module.val_dataloader())

def main():
    args = parse_args()
    # set_seeds(args.seed)
    train_test_model(args)
    # def train_model_sweep():
    #     # Method to be called by wandb sweep
    #     wandb.init(project=args.wandb_project)
    #     config= wandb.config
    #     wandb_logger = WandbLogger()

    #     model = get_lightning_model(
    #     model_nm=args.model,
    #     n_classes=3,
    #     optimizer=args.optimizer,
    #     lr=config.lr,
    #     momentum=config.momentum,
    #     weight_decay=config.weight_decay
    #     )

    #     wandb_logger.watch(model, log=True)

    #     trainer = L.Trainer(
    #         max_epochs=args.epochs,
    #         deterministic=True,
    #         callbacks=callbacks,
    #         accelerator="cuda" if args.cuda else "cpu",
    #         devices=args.device,
    #         default_root_dir=args.save,
    #         limit_train_batches=10,
    #         limit_val_batches=5,
    #         logger=wandb_logger
    #     )

    #     trainer.fit(model, data_module)
    #     train_acc = trainer.validate(dataloaders=data_module.train_dataloader())[0]["val_acc"] # Explicitly calling the train dataloader to avoid the validation one
    #     val_acc = trainer.validate(datamodule=data_module)[0]["val_acc"]
    #     print("Train accuracy:", train_acc)
    #     print("Validation accuracy:", val_acc)

    # today = datetime.datetime.now().strftime("%Y-%m-%d")
    # sweep_config = {
    #     "method": "grid",
    #     "name": f"{today}-test_cls_sweep",
    #     "metric": {"goal": "maximize", "name": "val_acc"},
    #     "parameters": {
    #         "lr": {"min": 1e-6, "max": 1e-2},
    #         "momentum": {"min": 0.0, "max": 1.0},
    #         "weight_decay": {"min": 1e-6, "max": 1e-2}
    #     }
    # }
    # sweep_id = wandb.sweep(sweep_config, project=args.wandb_project)
    # wandb.agent(sweep_id, function=train_model_sweep, count=5)
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",
                         default=42, type=int, help="Seed")
    parser.add_argument("--data-df-path",
                         default="/media/my_ftp/BasesDeDatos_Torax_RX_CT/COVID19_CT/processed/classification/classification_metadata.pkl", type=str, help="Dataframe path")
    parser.add_argument("--batch-size",
                         default=16, type=int, help="Batch size")
    parser.add_argument("--n_mc_samples",
                         default=20, type=int, help="Number of Monte Carlo samples")
    parser.add_argument("--epochs",
                         default=30, type=int, help="Epochs")
    parser.add_argument("--optimizer",
                         default="sgd", type=str, help="Optimizer")
    parser.add_argument("--lr",
                         default=1e-4, type=float, help="Learning rate")
    parser.add_argument("--momentum",
                         default=0.9, type=float, help="Momentum")
    parser.add_argument("--weight-decay",
                         default=1e-4, type=float, help="Weight decay")
    parser.add_argument("--num-workers",
                         default=6, type=int, help="Number of workers")
    parser.add_argument("--img-size",
                         default=256, type=int, help="Image size")
    parser.add_argument("--model",
                         default="densenet121", type=str, help="Model")
    parser.add_argument("--bayesian-fg",
                         default=True, type=bool, help="Bayesian FG")
    parser.add_argument("--pretrained",
                         default=False, type=bool, help="Pretrained")
    parser.add_argument("--save",
                         default="/media/my_ftp/TFTs/amoure/results/cls/test/overfit_batch", type=str, help="Save path")
    parser.add_argument("--use-wandb",
                         default=False, type=bool, help="Wandb")
    parser.add_argument("--wandb-project",
                         default="cls-tests", type=str, help="Wandb project")
    parser.add_argument("--wandb-run-name",
                         default="test-bayesian", type=str, help="Wandb run name")
    parser.add_argument("--save-pacience",
                         default=5, type=int, help="Number of epochs to wait before saving model")
    parser.add_argument("--device",
                         default="1", type=str, help="Device")
    parser.add_argument("--cuda",
                         default=False, type=bool, help="Name")
    args = parser.parse_args()
    return args

def set_seeds(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    main()
