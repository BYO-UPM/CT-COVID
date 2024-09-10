import lightning as L
import pandas as pd
from torch.utils.data import DataLoader
from dataset import ClassificationCovidDataset


class ClassificationCovidDataModule(L.LightningDataModule):
    def __init__(self, metadata_path, k_fold=None, n_classes=3, img_size=(512,512), preprocess=None, train_transforms=None, num_workers=4, batch_size=32, segment_ct=False, debug=False):
        super().__init__()
        self.metadata_df = pd.read_pickle(metadata_path).copy()
        self.k_fold = k_fold
        self.n_classes = n_classes
        self.img_size = img_size
        self.preprocess = preprocess
        self.transform = train_transforms
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.segment_ct = segment_ct
        self.debug = debug

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if self.k_fold is None:
            self.train_dataset = ClassificationCovidDataset(
                self.metadata_df,
                mode="train",
                k_fold=None,
                n_classes=self.n_classes,
                img_size=self.img_size,
                preprocess=self.preprocess,
                transform=self.transform,
                debug=self.debug,
                )
            
            self.val_dataset = ClassificationCovidDataset(
                self.metadata_df,
                mode="val",
                k_fold=None,
                n_classes=self.n_classes,
                img_size=self.img_size,
                preprocess=self.preprocess,
                transform=None,
                debug=self.debug,
                )
        else:
            self.train_dataset = ClassificationCovidDataset(
                self.metadata_df,
                mode="k_fold_train",
                k_fold=self.k_fold,
                n_classes=self.n_classes,
                img_size=self.img_size,
                preprocess=self.preprocess,
                transform=self.transform,
                )
            
            self.val_dataset = ClassificationCovidDataset(
                self.metadata_df,
                mode="k_fold_val",
                k_fold=self.k_fold,
                n_classes=self.n_classes,
                img_size=self.img_size,
                preprocess=self.preprocess,
                transform=self.transform,
                )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
            )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
            )


