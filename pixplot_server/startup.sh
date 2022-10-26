#!/bin/bash
FLDR=$1
PORT=$2
METADATA_FILE=$3
IMAGES_PATH=$4

echo $(date)

echo "Folder $FLDR  Port: $PORT"


# run the pre-processing if the initialized file does not exist.
if [ ! -d "./output" ]
then
    echo "output folder does not exist - running preprocessing"
    echo "$(ls)"

    # run the pre-processing
    python pixplot/pixplot.py --seed=4 --metadata="$FLDR/$METADATA_FILE" --images="$FLDR/$IMAGES_PATH" --image_vectors="$FLDR/image_vectors.npy"
    if [ $? -eq 0 ]
    then
        echo "Ran pixplot pre-processing successfully"
        echo "$(ls)"
    else
        exit 1
    fi
fi

echo $(date)

echo "Starting web server"

python -m http.server $PORT