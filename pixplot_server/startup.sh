#!/bin/bash
FLDR=$1
PORT=$2
METADATA_FILE=$3
IMAGES_PATH=$4
CLUSTERS_PROVIDED=$5

echo $(date)

echo "Folder $FLDR  Port: $PORT"
echo "$(ls)"
echo "$(pwd)"

# run the pre-processing if the initialized file does not exist.
if [ ! -d "./output" ]
then
    echo "output folder does not exist - running preprocessing"
    echo "$(ls)"

    # if CLUSTERS_PROVIDED equals True then run pixplot with the clusters provided
    if [ "$CLUSTERS_PROVIDED" = "True" ]
    then
        echo "Running pixplot with clusters provided"
        python pixplot/pixplot.py --seed=4 --metadata="$FLDR/$METADATA_FILE" --images="$FLDR/$IMAGES_PATH" --clusters_provided="$CLUSTERS_PROVIDED"
    else
        echo "Running pixplot with image_vectors.npy"
        python pixplot/pixplot.py --seed=4 --metadata="$FLDR/$METADATA_FILE" --images="$FLDR/$IMAGES_PATH" --clusters_provided="$CLUSTERS_PROVIDED" --image_vectors="$FLDR/image_vectors.npy"
    fi

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

# python -m http.server $PORT
# Run the Nginx server
/usr/sbin/nginx -g 'daemon off;'