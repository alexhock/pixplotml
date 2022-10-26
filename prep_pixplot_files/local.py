from pathlib import Path

from main import main


def local():

    # defaulting to the coco quickstart example.
    # Unzip coco_trained.zip before running this.
    data_fldr = Path("../data/outputs_trained")
    img_fldr = data_fldr / "images"

    main(
        data_fldr=data_fldr,
        img_fldr=img_fldr,
        use_imagenet=False,
        num_epochs=20,
        min_size=60,
    )


if __name__ == "__main__":
    local()
