import os
from typing import Optional, Union, Dict, Any, Callable

from pathlib import Path
import numpy as np

from pytorch_accelerated import Trainer
from pytorch_accelerated.callbacks import SaveBestModelCallback, get_default_callbacks
from pytorch_accelerated.schedulers.cosine_scheduler import CosineLrScheduler

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def get_train_transforms(min_crop):
    return A.Compose(
        [
            A.LongestMaxSize(max_size=min_crop),
            A.PadIfNeeded(min_height=min_crop, min_width=min_crop, border_mode=0),
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Affine(scale=(-0.7, 1.3), p=0.2),
            A.Normalize(),
            ToTensorV2(),
        ]
    )


def get_eval_transforms(min_crop):

    return A.Compose(
        [
            A.LongestMaxSize(max_size=min_crop),
            A.PadIfNeeded(min_height=min_crop, min_width=min_crop, border_mode=0),
            A.Resize(224, 224),
            A.Normalize(),
            ToTensorV2(),
        ]
    )


def train_classification_model(
    model: torch.nn.Module,
    train_ds: Dataset,
    eval_ds: Dataset,
    num_epochs: int = 20,
    batch_size: int = 64,
    outputs_path: Union[str, Path] = "./outputs",
    collate_fn: Callable = None,
    tags: Optional[Dict[str, Any]] = None,
) -> Trainer:

    # Use mixed precision to try and make things faster
    os.environ["mixed_precision"] = "fp16"

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    trainer = Trainer(
        model,
        loss_func=loss_func,
        optimizer=optimizer,
        callbacks=[
            *get_default_callbacks(),
            SaveBestModelCallback(save_path=Path(outputs_path) / "best_model.pt"),
        ],
    )

    trainer.train(
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        num_epochs=num_epochs,
        per_device_batch_size=batch_size,
        collate_fn=collate_fn,
        train_dataloader_kwargs={"pin_memory": False, "num_workers": 5},
        create_scheduler_fn=CosineLrScheduler.create_scheduler_fn(
            num_warmup_epochs=5, num_cooldown_epochs=3
        ),
    )

    return trainer


def get_predictions(
    model_backbone: torch.nn.Module,
    ds: Dataset,
    collate_fn: Callable,
    batch_size: int = 64,
    device: torch.device = torch.device("cuda"),
):

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=6,
    )

    preds_features = None

    for idx, batch in enumerate(tqdm(dl)):
        images, labels = batch
        images = images.to(device)
        with torch.no_grad():
            batch_output = model_backbone(images)

        batch_features = batch_output.cpu().numpy()
        if preds_features is None:
            preds_features = batch_features
        else:
            preds_features = np.concatenate((preds_features, batch_features), axis=0)

    return preds_features
