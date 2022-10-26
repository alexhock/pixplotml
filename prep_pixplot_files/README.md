# Fine-tune the pixplot visualization for your image data

Pixplot uses UMAP to cluster the images using a vector representation of each image. To fine-tune Pixplot we create this image vector file by training a classification ML model on your data. The code uses [Pytorch-Accelerated](https://github.com/Chris-hughes10/pytorch-accelerated) to easily and simply train this classification model. 

Note: As we're training an ML image model we need to use a machine with a GPU and docker enabled with the NVIDIA extensions. 

The inputs:

    metadata.csv - one line for each image to visualize. Minimally must contain filename, category columns
    Image files - one for each line of the metadata.csv file.

The output:

    image_vectors.npy

## Process

To train the model on your images and create the image_vectors.npy file, first create the python environment and run the following command:

```bash
conda create -n prep_pixplot python=3.9
conda activate prep_pixplot
pip install -r ./environment/requirements.txt
```

Then run the code to create the image_vectors.npy file (using the quickstart coco files as an example - unzip the data/outputs_trained.zip first):

```bash
 python main.py --data_fldr ../data/outputs_trained --img_fldr ../data/outputs_trained/images
 ```

The image_vectors.npy file will be saved to the location specified in the --data_fldr command line switch.

At this point you can just run the pixplot server pointing at the --data_fldr folder.

## Using Docker

Build the docker image (one-time only):

```bash
cd prep_pixplot_files
docker build -t prep_pixplot:1.0 ./environment
```

If you have a GPU then we can create image vectors fine-tuned to your data:

```bash
docker run --gpus all --ipc=host -v `pwd`/../data/outputs_trained:/data -v `pwd`:/src prep_pixplot:1.0 python /src/main.py --data_fldr /data --img_fldr /data/images
```

Then follow the instructions to start Pixplot in the quickstart section of the main [README file](../README.md)

If you don't have a GPU then you should just use the imagenet model without finetuning to your data, as it will take a very long time to train a model, but just using the imagenet model will work fine on CPU:

```bash
docker run --ipc=host -v `pwd`/../data/outputs_trained:/data -v `pwd`:/src prep_pixplot:1.0 python /src/main.py --data_fldr /data --img_fldr /data/images --use_imagenet True --device cpu
```

### Optional command line switches

There are some optional command line parameters that you can use if necessary:

    1. use_imagenet -- if we wish we can just use the pre-trained imagenet model and not train on your data. This will lead to worse results but is quicker.
    2. num_epochs -- defaults to 20
    3. min_size -- the size to resize the images to when training the model. The process will also pad the images. This must be <224 but defaults to 60 pixels.
    4. device - the torch device to use. If you don't have a GPU then set use_imagenet to True and set the device='cpu'.
