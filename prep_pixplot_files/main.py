import copy
from dataset import MetadataDataset, collate_fn
from func_to_script import script

from typing import Callable
from pathlib import Path
import numpy as np
import pandas as pd

from timm import create_model
import torch


from train import (
    get_train_transforms,
    get_eval_transforms,
    train_classification_model,
    get_predictions,
)


def train_model(
    meta_df: pd.DataFrame,
    data_fldr: str,
    img_fldr: str,
    num_classes: int,
    num_epochs: int = 20,
    min_size: int = 60,
    valid_frac: float = 0.1,
    collate_fn: Callable = None,
):

    model = create_model("resnetrs50", pretrained=True, num_classes=num_classes)

    # train/validation split
    valid_row_indexes = meta_df.sample(frac=valid_frac).index
    meta_df["is_valid"] = False
    meta_df.loc[valid_row_indexes, "is_valid"] = True

    train_ds = MetadataDataset(
        meta_df.query("is_valid == False"), img_fldr, get_train_transforms(min_size)
    )

    eval_ds = MetadataDataset(
        meta_df.query("is_valid == True"), img_fldr, get_eval_transforms(min_size)
    )

    trainer = train_classification_model(
        model,
        train_ds=train_ds,
        eval_ds=eval_ds,
        num_epochs=num_epochs,
        outputs_path=data_fldr,
        collate_fn=collate_fn,
    )

    return trainer.model


@script
def main(
    data_fldr: str,
    img_fldr: str,
    use_imagenet: bool = False,
    num_epochs: int = 20,
    min_size: int = 60,
    device: str = "cuda",
):
    """
    Creates a numpy array containing a 2048 element vector for each image in the folder specified by img_fldr.
    It trains a classification model on the images using the labels in the metadata.csv file.
    It then outputs the vectors from the classification model backbone.
    The output file contains a numpy array consisting of: [num_images, 2048]

    :param data_fldr: The location of the metadata.csv file and the location to save the output file
    :param img_fldr: The location of the images
    :param use_imagenet: Don't train the model, just use the imagenet weights - usually leads to worse results
    :param num_epochs: The number of times to train the dataset
    :param min_size: The minimum size of image
    :param device: The torch device to use ['cpu'|'cuda']
    """

    data_fldr = Path(data_fldr)
    img_fldr = Path(img_fldr)

    device = torch.device(device)

    meta_df = pd.read_csv(data_fldr / "metadata.csv")

    num_classes = meta_df.class_id.unique().shape[0]

    # check if need to update class_ids to ensure class_id >=0 && < num_classes as model training requires it
    if meta_df["class_id"].max() > num_classes:
        existing_classes = sorted(meta_df.class_id.unique())
        new_mapping = {k: idx for idx, k in enumerate(existing_classes)}
        meta_df["class_id"] = meta_df.apply(lambda r: new_mapping[r.class_id], axis=1)

    if use_imagenet:
        model = create_model("resnetrs50", pretrained=True)
    else:
        model = train_model(
            meta_df,
            data_fldr,
            img_fldr,
            num_classes,
            num_epochs,
            min_size,
            collate_fn=collate_fn,
        )

    # strip off the classification head (for a resnet50)
    # as we just want the backbone outputs
    model.to(device=device)
    emb_model = copy.deepcopy(model)
    emb_model.fc = torch.nn.Identity(2048)
    emb_model = emb_model.to(device)

    # whole dataset with eval
    preds_ds = MetadataDataset(meta_df, img_fldr, get_eval_transforms(min_size))

    preds = get_predictions(emb_model, preds_ds, collate_fn, device=device)

    np.save(data_fldr / "image_vectors.npy", preds)


if __name__ == "__main__":
    main()
