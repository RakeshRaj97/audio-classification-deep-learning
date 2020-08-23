# train.py
import numpy as np
import pandas as pd
import librosa
import random
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold
import math
from collections import OrderedDict

from PIL import Image
import albumentations
from pydub import AudioSegment

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pretrainedmodels

from joblib import Parallel, delayed
from tqdm.autonotebook import tqdm

import warnings
warnings.filterwarnings('ignore')

from dataloader import BirdDataset
from engine import Engine
from model import ResNet18
from arguments import args

def run(fold_index):
    train = pd.read_csv("/fred/oz138/test/data/train_folds.csv")
    test = pd.read_csv("/fred/oz138/test/data/test.csv")
    submission = pd.read_csv("/fred/oz138/test/data/sample_submission.csv")

    # label encoding
    train["ebird_label"] = LabelEncoder().fit_transform(train['ebird_code'])

    train_df = train[~train.kfold.isin([fold_index])]
    train_dataset = BirdDataset(df=train_df)

    valid_df = train[train.kfold.isin([fold_index])]
    valid_dataset = BirdDataset(df=valid_df, valid=True)

    device = "cuda"
    MX = ResNet18(pretrained=True)
    model = MX.to(device)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False
    )

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr,
                                  betas=args.betas,
                                  eps=args.eps,
                                  weight_decay=args.wd
                                  )

    best_acc = 0

    for epoch in range(args.epochs):

        train_loss = Engine.train_fn(train_loader, model, optimizer, device, epoch)

        valid_loss, valid_acc = Engine.valid_fn(valid_loader, model, device, epoch)

        print(f"Fold {fold_index} ** Epoch {epoch + 1} **==>** Accuracy = {valid_acc}")

        if valid_acc > best_acc:
            torch.save(model.state_dict(), os.path.join(args.MODEL_PATH, f"fold_{fold_index}.bin"))
            best_acc = valid_acc

if __name__ == "__main__":
    # run(0)
    # run(1)
    run(2)
    
