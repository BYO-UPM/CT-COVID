from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GroupShuffleSplit
import random
import numpy as np
import torch
import os
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

WANDB_API_KEY = "66cd51a1dbd6025bc7240caae7c91c254022f0e1"

def set_general_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def split_dataframe(df, test_size=0.2, random_state=42):
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    train_df['split'] = 'train'
    test_df['split'] = 'test'
    return pd.concat([train_df, test_df])


def stratified_group_train_test_split_df(df, random_state=42, stratify_columns=None, group_columns=None):
    if stratify_columns is None:
        stratify_columns = ['has_infection']
    if group_columns is None:
        group_columns = 'scan_id'

    # Create a new column that is a combination of the stratify columns
    df['stratify'] = df[stratify_columns].apply(lambda x: '_'.join(x.astype(str)), axis=1)
    df["group"] = df[group_columns].apply(lambda x: '_'.join(x.astype(str)), axis=1)

    split = StratifiedGroupKFold(n_splits=5, random_state=random_state)
    for train_index, test_index in split.split(df, df['stratify'], df["group"]):
        df[f'split'] = 'train'  # Assign all splits to 'train'
        df.loc[test_index, f'split'] = 'test'  # Assign the current split to 'test'

    # Drop the temporary stratify column
    df = df.drop(columns='stratify')
    df = df.drop(columns='group')

    return df

def stratified_group_kfold_split_dataframe(df, n_splits=5, random_state=42, stratify_columns=None, group_columns=None):
    if stratify_columns is None:
        stratify_columns = ['has_infection']
    if group_columns is None:
        group_columns = 'scan_id'

    # Create a new column that is a combination of the stratify columns
    df['stratify'] = df[stratify_columns].apply(lambda x: '_'.join(x.astype(str)), axis=1)
    df["group"] = df[group_columns].apply(lambda x: '_'.join(x.astype(str)), axis=1)

    split = StratifiedGroupKFold(n_splits=n_splits, random_state=random_state)
    for i, (train_index, test_index) in enumerate(split.split(df, df['stratify'], df["group"])):
        df[f'split_{i}'] = 'test'
        df.loc[train_index, f'split_{i}'] = 'train'

    # Drop the temporary stratify column
    df = df.drop(columns='stratify')
    df = df.drop(columns='group')

    return df

# Create a 5-fold split of the data by creating a 5 new columns in the dataframe that 
# either have the value train or test for each row depending on the fold

def kfold_split_dataframe(df, n_splits=5, random_state=42, stratify_columns=None):
    if stratify_columns is None:
        stratify_columns = ['has_infection']

    # Create a new column that is a combination of the stratify columns
    df['stratify'] = df[stratify_columns].apply(lambda x: '_'.join(x.astype(str)), axis=1)

    split = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for i, (train_index, test_index) in enumerate(split.split(df, df['stratify'])):
        df[f'split_{i}'] = 'test'
        df.loc[train_index, f'split_{i}'] = 'train'

    # Drop the temporary stratify column
    df = df.drop(columns='stratify')

    return df
