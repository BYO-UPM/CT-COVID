import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
from PIL import Image


CLASS_TO_LABEL_DICT = {
    "normal": 0,
    "common-pneumonia": 1,
    "covid": 2
}

CLASS_TO_LABEL_WITH_INFECTION_DICT = {
    "normal":0,
    "common-pneumonia":1,
    "covid-with-infection":2,
    "covid-no-infection":3
}

class ClassificationCovidDataset(Dataset):
    def __init__(self, metadata_df, mode, k_fold=None, n_classes=3, img_size=(512,512), preprocess=None, transform=None, limit=None, debug=False):
        self.metadata_df = metadata_df.copy()
        self.mode = mode
        self.n_classes = n_classes
        self.img_size = img_size
        self.preprocess = preprocess
        self.transform = transform
        self.debug = debug
        self.limit = limit
        if mode == "train":
            if self.limit is not None:
                self.metadata_df = self.metadata_df[self.metadata_df["split"] == "train"].sample(self.limit).reset_index(drop=True).copy()
            else:
                self.metadata_df = self.metadata_df[self.metadata_df["split"] == "train"].reset_index(drop=True).copy()
        
        elif mode == "val":
            if self.limit is not None:
                self.metadata_df = self.metadata_df[self.metadata_df["split"] == "test"].sample(self.limit).reset_index(drop=True).copy()
            else:
                self.metadata_df = self.metadata_df[self.metadata_df["split"] == "test"].reset_index(drop=True).copy()
        
        self.paths, self.labels, self.datasets, self.has_infection = self.metadata_df["ct_slice_path"].tolist(), self.metadata_df["label"].tolist(), self.metadata_df["dataset"].tolist(), self.metadata_df["has_infection"].tolist()
        print("Mode:", mode, "Samples:", len(self.paths), "Limit:", self.limit, "Classes:", self.n_classes, "debug_fg:", self.debug)


    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # Img reading and processing
        img_path = self.paths[idx]
        dataset = self.datasets[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        #print(img_path, " ", img.shape)
        img = cv2.resize(img, self.img_size)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = Image.fromarray(img, mode="RGB") # Convert to PIL image
        if self.preprocess is not None:
            img = self.preprocess(img)
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        default_tensor_preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])

        img_t = default_tensor_preprocess(img)

        # Label processing
        label = self.labels[idx]
        if self.n_classes == 3:
            encoded_label = CLASS_TO_LABEL_DICT[label]
        else:
            if label == "covid":
                if self.has_infection[idx]:
                    encoded_label = CLASS_TO_LABEL_WITH_INFECTION_DICT["covid-with-infection"]
                else:
                    encoded_label = CLASS_TO_LABEL_WITH_INFECTION_DICT["covid-no-infection"]
            else:
                encoded_label = CLASS_TO_LABEL_WITH_INFECTION_DICT[label]
        encoded_label_t = torch.tensor(encoded_label, dtype=torch.long)
        if self.debug:
            return  img_t, encoded_label_t, transforms.Compose([transforms.ToTensor()])(img), img_path, dataset, self.has_infection[idx]
        else:
            return img_t, encoded_label_t

        