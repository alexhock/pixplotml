from __future__ import division

import argparse
import datetime
import os
import sys
from pathlib import Path

import warnings
from distutils.dir_util import copy_tree
from os.path import dirname, exists, join, realpath

import glob2
from itertools import product

from urllib.request import urlopen
from joblib import parallel, delayed

from settings import config

warnings.filterwarnings("ignore")

##
# Unconditional imports
##


def timestamp():
    """Return a string for printing the current time"""
    return str(datetime.datetime.now()) + ":"


##
# Image processing imports
##

if "--copy_web_only" not in sys.argv:
    import copy
    import csv
    import gzip
    import itertools
    import json
    import math
    import multiprocessing
    import operator
    import pickle
    import random
    from collections import defaultdict

    import numpy as np

    np.random.seed(42)
    from dateutil.parser import parse as parse_date
    from iiif_downloader import Manifest
    from PIL import Image as pil_image
    from PIL import ImageFile
    from pointgrid import align_points_to_grid
    from scipy.spatial.distance import cdist
    from scipy.stats import kde
    from sklearn.preprocessing import minmax_scale
    from sklearn.cluster import KMeans
    from tqdm import tqdm

    from urllib.parse import unquote

    cluster_method = "kmeans"

    ##
    # Optional install imports
    ##

    if config["hdbscan_available"]:
        try:
            from hdbscan import HDBSCAN

            cluster_method = "hdbscan"
        except:
            print(timestamp(), "HDBSCAN not available; using sklearn KMeans")

    cuml_ready = False
    from umap import UMAP, AlignedUMAP

    # handle truncated images in PIL (managed by Pillow)
    ImageFile.LOAD_TRUNCATED_IMAGES = True


def array_to_img(x, data_format="channels_last", scale=True, dtype="float32"):
    """Converts a 3D Numpy array to a PIL Image instance.

    # Arguments
        x: Input Numpy array.
        data_format: Image data format, either "channels_first" or "channels_last".
            Default: "channels_last".
        scale: Whether to rescale the image such that minimum and maximum values
            are 0 and 255 respectively.
            Default: True.
        dtype: Dtype to use.
            Default: "float32".

    # Returns
        A PIL Image instance.

    # Raises
        ImportError: if PIL is not available.
        ValueError: if invalid `x` or `data_format` is passed.
    """
    if pil_image is None:
        raise ImportError(
            "Could not import PIL.Image. " "The use of `array_to_img` requires PIL."
        )
    x = np.asarray(x, dtype=dtype)
    if x.ndim != 3:
        raise ValueError(
            "Expected image array to have rank 3 (single image). "
            "Got array with shape: %s" % (x.shape,)
        )

    if data_format not in {"channels_first", "channels_last"}:
        raise ValueError("Invalid data_format: %s" % data_format)

    # Original Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but target PIL image has format (width, height, channel)
    if data_format == "channels_first":
        x = x.transpose(1, 2, 0)
    if scale:
        x = x - np.min(x)
        x_max = np.max(x)
        if x_max != 0:
            x /= x_max
        x *= 255
    if x.shape[2] == 4:
        # RGBA
        return pil_image.fromarray(x.astype("uint8"), "RGBA")
    elif x.shape[2] == 3:
        # RGB
        return pil_image.fromarray(x.astype("uint8"), "RGB")
    elif x.shape[2] == 1:
        # grayscale
        if np.max(x) > 255:
            # 32-bit signed integer grayscale image. PIL mode "I"
            return pil_image.fromarray(x[:, :, 0].astype("int32"), "I")
        return pil_image.fromarray(x[:, :, 0].astype("uint8"), "L")
    else:
        raise ValueError("Unsupported channel number: %s" % (x.shape[2],))


def save_img(
    path, x, data_format="channels_last", file_format=None, scale=True, **kwargs
):
    """Saves an image stored as a Numpy array to a path or file object.

    # Arguments
        path: Path or file object.
        x: Numpy array.
        data_format: Image data format,
            either "channels_first" or "channels_last".
        file_format: Optional file format override. If omitted, the
            format to use is determined from the filename extension.
            If a file object was used instead of a filename, this
            parameter should always be used.
        scale: Whether to rescale image values to be within `[0, 255]`.
        **kwargs: Additional keyword arguments passed to `PIL.Image.save()`.
    """
    img = array_to_img(x, data_format=data_format, scale=scale)
    if img.mode == "RGBA" and (file_format == "jpg" or file_format == "jpeg"):
        warnings.warn(
            "The JPG format does not support " "RGBA images, converting to RGB."
        )
        img = img.convert("RGB")
    img.save(path, format=file_format, **kwargs)


##
# Entry
##


def process_images(**kwargs):
    """Main method for processing user images and metadata"""
    kwargs = preprocess_kwargs(**kwargs)
    copy_web_assets(**kwargs)
    np.random.seed(kwargs["seed"])

    kwargs["out_dir"] = join(kwargs["out_dir"], "data")

    (
        kwargs["image_paths"],
        kwargs["loaded_images"],
        kwargs["metadata"],
        kwargs["vecs"],
    ) = load_and_filter_images(**kwargs)

    # write filtered metadata to files - one json for each image
    write_metadata(**kwargs)

    # create atlas image file
    kwargs["num_images"] = len(kwargs["image_paths"])
    kwargs["atlas_dir"] = get_atlas_data(**kwargs)

    get_manifest(**kwargs)

    # write the resized images and their thumbnail to disk
    write_images(**kwargs)
    print(timestamp(), "Done!")


def preprocess_kwargs(**kwargs):
    """Preprocess incoming key word arguments"""
    for i in ["n_neighbors", "min_dist"]:
        if not isinstance(kwargs[i], list):
            kwargs[i] = [kwargs[i]]
    return kwargs


def copy_web_assets(**kwargs):
    """Copy the /web directory from the pixplot source to the users cwd"""
    src = join(dirname(realpath(__file__)), "web")
    dest = join(os.getcwd(), kwargs["out_dir"])
    copy_tree(src, dest)
    # write version numbers into output
    for i in ["index.html", os.path.join("assets", "js", "tsne.js")]:
        path = join(dest, i)
        print("Reading path:", path)
        with open(path, "r") as f:
            f = f.read().replace("VERSION_NUMBER", get_version())
            with open(path, "w") as out:
                out.write(f)
    if kwargs["copy_web_only"]:
        print(timestamp(), "Done!")
        sys.exit()


##
# Images
##


def load_image_file(metadata, image_root_path):

    image_vector = metadata["image_vector"]

    filename = metadata["filename"]

    if "blob_path" in metadata:
        file_location = metadata["blob_path"]
    else:
        file_location = filename

    img = Image(
        image_root_path,
        file_location,
        filename,
        **{"metadata": metadata, "vec": image_vector}
    )

    return img


def load_input_files(**kwargs):
    """
    Load the metadata csv, image vectors and images - order in the metadata and the vectors must be the same
    """

    # load the metadata csv file
    all_metadata = get_metadata_list(**kwargs)

    if kwargs["image_vectors"]:
        vecs = np.load(kwargs["image_vectors"])

    all_images = []

    image_paths = kwargs["images"]
    image_root_path = Path(image_paths).parent

    # add image vectors to metadata
    for idx, metadata in enumerate(all_metadata):

        if kwargs["image_vectors"]:
            vec = vecs[idx, :]
        else:
            vec = None

        metadata["idx"] = idx
        metadata["image_vector"] = vec

    # load images in parallel using metadata
    all_images = parallel.Parallel(n_jobs=3)(
        delayed(load_image_file)(metadata, image_root_path) for metadata in all_metadata
    )

    return all_images


def load_and_filter_images(**kwargs):
    """Main method for filtering images given user metadata (if provided)"""

    print(timestamp(), "Loading images and filtering...")

    all_loaded_images = load_input_files(**kwargs)

    # validate loaded images, remove any that are in error
    num_valid = 0
    for i in all_loaded_images:
        # get image height and width
        w, h = i.original.size
        # remove images with 0 height or width when resized to lod height
        if (h == 0) or (w == 0):
            print(
                timestamp(),
                "Skipping {} because it contains 0 height or width".format(i.path),
            )
            continue

        # remove images that have 0 height or width when resized
        try:
            resized = i.resize_to_max(kwargs["lod_cell_height"])

        except ValueError:
            print(
                timestamp(),
                "Skipping {} because it contains 0 height or width when resized".format(
                    i.path
                ),
            )
            continue
        except OSError:
            print(
                timestamp(),
                "Skipping {} because it could not be resized".format(i.path),
            )
            continue
        # remove images that are too wide for the atlas
        if (w / h) > (kwargs["atlas_size"] / kwargs["cell_size"]):
            print(
                timestamp(),
                "Skipping {} because its dimensions are oblong".format(i.path),
            )
            continue

        i.metadata["valid"] = True
        i.valid = True
        num_valid += 1

    # if there are no remaining images, throw an error
    if num_valid == 0:
        raise Exception("No images were found! Please check your input image glob.")

    # remove the invalid images from the metadata
    filtered_metadata = []
    filtered_images = []
    filtered_vecs = []
    image_names = []
    for _, img in enumerate(all_loaded_images):
        if img.valid:
            md = img.metadata
            image_vector = img.vec
            clean_fn = clean_filename(md["filename"], **kwargs)

            filtered_metadata.append(md)
            filtered_images.append(img)
            filtered_vecs.append(image_vector)
            image_names.append(clean_fn)

    # save the metadata
    kwargs["metadata"] = filtered_metadata
    # write_metadata(**kwargs)

    return [image_names, filtered_images, filtered_metadata, filtered_vecs]


def clean_filename(s, **kwargs):
    """Given a string that points to a filename, return a clean filename"""
    s = unquote(os.path.basename(s))
    invalid_chars = '<>:;,"/\\|?*[]'
    for i in invalid_chars:
        s = s.replace(i, "")
    return s


def get_image_paths(**kwargs):
    """Called once to provide a list of image paths--handles IIIF manifest input"""
    # handle case where --images points to iiif manifest
    image_paths = None
    if not kwargs["images"]:
        print("\nError: please provide an images argument, e.g.:")
        print('pixplot --images "cat_pictures/*.jpg"\n')
        sys.exit()
    # handle list of IIIF image inputs
    if os.path.exists(kwargs["images"]):
        with open(kwargs["images"]) as f:
            f = [i.strip() for i in f.read().split("\n") if i.strip()]
            for i in f:
                if i.startswith("http"):
                    try:
                        Manifest(url=i).save_images(limit=1)
                    except:
                        print(timestamp(), "Could not download url " + i)
            image_paths = sorted(
                glob2.glob(os.path.join("iiif-downloads", "images", "*"))
            )
    # handle case where images flag points to a glob of images
    if not image_paths:
        image_paths = sorted(glob2.glob(kwargs["images"]))
    # handle case user provided no images
    if not image_paths:
        print("\nError: No input images were found. Please check your --images glob\n")
        sys.exit()
    # optionally shuffle the image_paths
    if kwargs.get("shuffle", False):
        print(timestamp(), "Shuffling input images")
        random.Random(kwargs["seed"]).shuffle(image_paths)
    # optionally limit the number of images in image_paths
    if kwargs.get("max_images", False):
        image_paths = image_paths[: kwargs["max_images"]]
    return image_paths


def stream_images(**kwargs):

    return kwargs["loaded_images"]


##
# Metadata
##


def get_metadata_list(**kwargs):
    """Return a list of objects with image metadata"""
    if not kwargs.get("metadata", False):
        return []
    # handle csv metadata
    l = []
    if kwargs["metadata"].endswith(".csv"):
        with open(kwargs["metadata"]) as f:
            reader = csv.reader(f)
            headers = [i.lower() for i in next(reader)]
            for i in reader:
                l.append(
                    {
                        headers[j]: i[j] if len(i) > j and i[j] else ""
                        for j, _ in enumerate(headers)
                    }
                )
    # handle json metadata
    else:
        for i in glob2.glob(kwargs["metadata"]):
            with open(i) as f:
                l.append(json.load(f))
    # if the user provided a category but not a tag, use the category as the tag
    for i in l:
        if i.get("category", False) and not i.get("tags", False):
            i.update({"tags": i["category"]})
    return l


def write_metadata(metadata, **kwargs):
    """Write list `metadata` of objects to disk"""
    if not metadata:
        return

    print(timestamp(), "Writing metadata...")

    # create the metadata folders
    out_dir = join(kwargs["out_dir"], "metadata")
    for i in ["filters", "options", "file", "labels"]:
        out_path = join(out_dir, i)
        if not exists(out_path):
            os.makedirs(out_path)

    # create the lists of images with each tag
    d = defaultdict(list)
    for i in metadata:
        filename = clean_filename(i["filename"])
        i["tags"] = [j.strip() for j in i.get("tags", "").split("|")]
        i["image_vector"] = None
        for j in i["tags"]:
            d["__".join(j.split())].append(filename)
        write_json(os.path.join(out_dir, "file", filename + ".json"), i, **kwargs)

    # create the filters.json file
    write_json(
        os.path.join(out_dir, "filters", "filters.json"),
        [
            {
                "filter_name": "select",
                "filter_values": list(d.keys()),
            }
        ],
        **kwargs
    )

    # create the options for the category dropdown
    for i in d:
        write_json(os.path.join(out_dir, "options", i + ".json"), d[i], **kwargs)
    # create the map from date to images with that date (if dates present)
    date_d = defaultdict(list)
    for i in metadata:
        date = i.get("year", "")
        if date:
            date_d[date].append(clean_filename(i["filename"]))
    # find the min and max dates to show on the date slider
    dates = np.array([int(i.strip()) for i in date_d if is_number(i)])
    domain = {"min": float("inf"), "max": -float("inf")}
    mean = np.mean(dates)
    std = np.std(dates)
    for i in dates:
        # update the date domain with all non-outlier dates
        if abs(mean - i) < (std * 4):
            domain["min"] = int(min(i, domain["min"]))
            domain["max"] = int(max(i, domain["max"]))
    # write the dates json
    if len(date_d) > 1:
        write_json(
            os.path.join(out_dir, "dates.json"),
            {
                "domain": domain,
                "dates": date_d,
            },
            **kwargs
        )

    # create filename label mapping file
    filenames_label = {md["filename"]: md["category"] for md in metadata}
    out_path = os.path.join(out_dir, "labels", "filenames_label.json")
    write_json(out_path, filenames_label, **kwargs)

    # id_to_label.json
    unique_labels = {md["category"] for md in metadata}
    unique_labels = sorted(unique_labels)
    id_to_label = {idx: l for idx, l in enumerate(unique_labels)}

    # Always add a remove label at the end so a user can flag an image for removal
    id_to_label[max(id_to_label) + 1] = "Remove"

    out_path = os.path.join(out_dir, "labels", "id_to_label.json")
    write_json(out_path, id_to_label, **kwargs)

    base_colors = []
    colors = [
        "#e6194b",
        "#3cb44b",
        "#ffe119",
        "#4363d8",
        "#f58231",
        "#911eb4",
        "#46f0f0",
        "#f032e6",
        "#bcf60c",
        "#fabebe",
        "#008080",
        "#e6beff",
        "#9a6324",
        "#fffac8",
        "#800000",
        "#aaffc3",
        "#808000",
        "#ffd8b1",
        "#000075",
        "#808080",
        "#ffffff",
        "#000000",
    ]
    for c in colors:
        c = c.lstrip("#")
        rgb = [int(c[i : i + 2], 16) / 255 for i in (0, 2, 4)]
        base_colors.append(rgb)

    colors = [c for c in product([0, 128 / 255, 192 / 255, 255 / 255], repeat=3)]
    random.shuffle(colors)  # set seed in commandline args for repeatability
    colors = base_colors + colors
    id_to_color = {idx: c for idx, c in enumerate(colors)}
    # id_to_color = {idx: colors[idx] for idx in id_to_label.keys()}
    out_path = os.path.join(out_dir, "labels", "id_to_color.json")
    write_json(out_path, id_to_color, **kwargs)


def is_number(s):
    """Return a boolean indicating if a string is a number"""
    try:
        int(s)
        return True
    except:
        return False


##
# Main
##


def get_manifest(**kwargs):
    """Create and return the base object for the manifest output file"""
    # load the atlas data
    atlas_data = json.load(open(join(kwargs["atlas_dir"], "atlas_positions.json")))
    # store each cell's size and atlas position
    atlas_ids = set([i["idx"] for i in atlas_data])
    sizes = [[] for _ in atlas_ids]
    pos = [[] for _ in atlas_ids]
    for idx, i in enumerate(atlas_data):
        sizes[i["idx"]].append([i["w"], i["h"]])
        pos[i["idx"]].append([i["x"], i["y"]])
    # obtain the paths to each layout's JSON positions
    layouts = get_layouts(**kwargs)
    # create a heightmap for the umap layout
    if "umap" in layouts and layouts["umap"]:
        get_heightmap(layouts["umap"]["variants"][0]["layout"], "umap", **kwargs)
    # specify point size scalars
    point_sizes = {}
    point_sizes["min"] = 0
    point_sizes["grid"] = 1 / math.ceil(len(kwargs["image_paths"]) ** (1 / 2))
    point_sizes["max"] = point_sizes["grid"] * 1.2
    point_sizes["scatter"] = point_sizes["grid"] * 0.2
    point_sizes["initial"] = point_sizes["scatter"]
    point_sizes["categorical"] = point_sizes["grid"] * 0.6
    point_sizes["geographic"] = point_sizes["grid"] * 0.025
    # fetch the date distribution data for point sizing
    if "date" in layouts and layouts["date"]:
        date_layout = read_json(layouts["date"]["labels"], **kwargs)
        point_sizes["date"] = 1 / (
            (date_layout["cols"] + 1) * len(date_layout["labels"])
        )
    # create manifest json
    manifest = {
        "version": get_version(),
        "plot_id": kwargs["plot_id"],
        "output_directory": os.path.split(kwargs["out_dir"])[0],
        "layouts": layouts,
        "initial_layout": "umap",
        "point_sizes": point_sizes,
        "imagelist": get_path("imagelists", "imagelist", **kwargs),
        "atlas_dir": kwargs["atlas_dir"],
        "metadata": True if kwargs["metadata"] else False,
        "default_hotspots": get_hotspots(layouts=layouts, **kwargs),
        "custom_hotspots": get_path(
            "hotspots", "user_hotspots", add_hash=False, **kwargs
        ),
        "gzipped": kwargs["gzip"],
        "config": {
            "sizes": {
                "atlas": kwargs["atlas_size"],
                "cell": kwargs["cell_size"],
                "lod": kwargs["lod_cell_height"],
            },
        },
        "creation_date": datetime.datetime.today().strftime("%d-%B-%Y-%H:%M:%S"),
    }
    # write the manifest without gzipping
    no_gzip_kwargs = {
        "out_dir": kwargs["out_dir"],
        "gzip": False,
        "plot_id": kwargs["plot_id"],
    }
    path = get_path("manifests", "manifest", **no_gzip_kwargs)
    write_json(path, manifest, **no_gzip_kwargs)
    path = get_path(None, "manifest", add_hash=False, **no_gzip_kwargs)
    write_json(path, manifest, **no_gzip_kwargs)
    # create images json
    imagelist = {
        "cell_sizes": sizes,
        "images": [clean_filename(i) for i in kwargs["image_paths"]],
        "atlas": {
            "count": len(atlas_ids),
            "positions": pos,
        },
    }
    write_json(manifest["imagelist"], imagelist, **kwargs)


##
# Atlases
##


def get_atlas_data(**kwargs):
    """
    Generate and save to disk all atlases to be used for this visualization
    If square, center each cell in an nxn square, else use uniform height
    """
    # if the atlas files already exist, load from cache
    out_dir = os.path.join(kwargs["out_dir"], "atlases", kwargs["plot_id"])
    if (
        os.path.exists(out_dir)
        and kwargs["use_cache"]
        and not kwargs.get("shuffle", False)
    ):
        print(timestamp(), "Loading saved atlas data")
        return out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # else create the atlas images and store the positions of cells in atlases
    print(timestamp(), "Creating atlas files")
    n = 0  # number of atlases
    x = 0  # x pos in atlas
    y = 0  # y pos in atlas
    positions = []  # l[cell_idx] = atlas data
    atlas = np.zeros((kwargs["atlas_size"], kwargs["atlas_size"], 3))
    for idx, i in enumerate(stream_images(**kwargs)):
        cell_data = i.resize_to_height(kwargs["cell_size"])
        _, v, _ = cell_data.shape
        appendable = False
        if (x + v) <= kwargs["atlas_size"]:
            appendable = True
        elif (y + (2 * kwargs["cell_size"])) <= kwargs["atlas_size"]:
            y += kwargs["cell_size"]
            x = 0
            appendable = True
        if not appendable:
            save_atlas(atlas, out_dir, n)
            n += 1
            atlas = np.zeros((kwargs["atlas_size"], kwargs["atlas_size"], 3))
            x = 0
            y = 0
        atlas[y : y + kwargs["cell_size"], x : x + v] = cell_data
        # find the size of the cell in the lod canvas
        lod_data = i.resize_to_max(kwargs["lod_cell_height"])
        h, w, _ = lod_data.shape  # h,w,colors in lod-cell sized image `i`
        positions.append(
            {
                "idx": n,  # atlas idx
                "x": x,  # x offset of cell in atlas
                "y": y,  # y offset of cell in atlas
                "w": w,  # w of cell at lod size
                "h": h,  # h of cell at lod size
            }
        )
        x += v
    save_atlas(atlas, out_dir, n)
    out_path = os.path.join(out_dir, "atlas_positions.json")
    with open(out_path, "w") as out:
        json.dump(positions, out)
    return out_dir


def save_atlas(atlas, out_dir, n):
    """Save an atlas to disk"""
    out_path = join(out_dir, "atlas-{}.jpg".format(n))
    save_img(out_path, atlas)


##
# Layouts
##


def get_layouts(**kwargs):
    """Get the image positions in each projection"""

    if kwargs["clusters_provided"]:
        umap = get_layout_positions(**kwargs)
    else:
        umap = get_umap_layout(**kwargs)

    layouts = {
        "umap": umap,
        "alphabetic": {
            "layout": get_alphabetic_layout(**kwargs),
        },
        "grid": {
            "layout": None,
        },
        "categorical": get_categorical_layout(**kwargs),
        "date": get_date_layout(**kwargs),
        "geographic": get_geographic_layout(**kwargs),
        "custom": get_custom_layout(**kwargs),
    }
    return layouts


def get_layout_positions(**kwargs):
    """ """
    metadata = kwargs["metadata"]

    out_path = get_path("layouts", "umap", **kwargs)

    z = [(m["x"], m["y"]) for m in metadata]

    return {
        "variants": [
            {
                "n_neighbors": kwargs["n_neighbors"][0],
                "min_dist": kwargs["min_dist"][0],
                "layout": write_layout(out_path, z, **kwargs),
                "jittered": get_pointgrid_layout(out_path, "umap", **kwargs),
            }
        ]
    }


def get_umap_layout(**kwargs):
    """Get the x,y positions of images passed through a umap projection"""
    w = kwargs["vecs"]
    # single model umap
    if len(kwargs["n_neighbors"]) == 1 and len(kwargs["min_dist"]) == 1:
        return process_single_layout_umap(w, **kwargs)
    else:
        return process_multi_layout_umap(w, **kwargs)


def process_single_layout_umap(v, **kwargs):
    """Create a single layout UMAP projection"""
    print(timestamp(), "Creating single umap layout")
    model = get_umap_model(**kwargs)
    out_path = get_path("layouts", "umap", **kwargs)
    if cuml_ready:
        z = model.fit(v).embedding_
    else:
        if os.path.exists(out_path) and kwargs["use_cache"]:
            return out_path
        y = []
        if kwargs.get("metadata", False):
            labels = [i.get("label", None) for i in kwargs["metadata"]]
            # if the user provided labels, integerize them
            if any([i for i in labels]):
                d = defaultdict(lambda: len(d))
                for i in labels:
                    if i == None:
                        y.append(-1)
                    else:
                        y.append(d[i])
                y = np.array(y)
        # project the PCA space down to 2d for visualization
        z = model.fit(v, y=y if np.any(y) else None).embedding_
    return {
        "variants": [
            {
                "n_neighbors": kwargs["n_neighbors"][0],
                "min_dist": kwargs["min_dist"][0],
                "layout": write_layout(out_path, z, **kwargs),
                "jittered": get_pointgrid_layout(out_path, "umap", **kwargs),
            }
        ]
    }


def process_multi_layout_umap(v, **kwargs):
    """Create a multi-layout UMAP projection"""
    print(timestamp(), "Creating multi-umap layout")
    params = []
    for n_neighbors, min_dist in itertools.product(
        kwargs["n_neighbors"], kwargs["min_dist"]
    ):
        filename = "umap-n_neighbors_{}-min_dist_{}".format(n_neighbors, min_dist)
        out_path = get_path("layouts", filename, **kwargs)
        params.append(
            {
                "n_neighbors": n_neighbors,
                "min_dist": min_dist,
                "filename": filename,
                "out_path": out_path,
            }
        )
    # map each image's index to itself and create one copy of that map for each layout
    relations_dict = {idx: idx for idx, _ in enumerate(v)}
    # determine the subset of params that have already been computed
    uncomputed_params = [i for i in params if not os.path.exists(i["out_path"])]
    # determine the filepath where this model will be saved
    model_filename = "umap-" + str(abs(hash(kwargs["images"])))
    model_path = get_path("models", model_filename, **kwargs).replace(".json", ".gz")
    out_dir = os.path.join(kwargs["out_dir"], "models")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # load or create the model
    if os.path.exists(model_path):
        model = load_model(model_path)
        for i in uncomputed_params:
            model.update(v, relations_dict.copy())
        # after updating, we can read the results from the end of the updated model
        for idx, i in enumerate(uncomputed_params):
            embedding = z.embeddings_[len(uncomputed_params) - idx]
            write_layout(i["out_path"], embedding, **kwargs)
    else:
        model = AlignedUMAP(
            n_neighbors=[i["n_neighbors"] for i in uncomputed_params],
            min_dist=[i["min_dist"] for i in uncomputed_params],
        )
        # fit the model on the data
        z = model.fit(
            [v for _ in params], relations=[relations_dict for _ in params[1:]]
        )
        for idx, i in enumerate(params):
            write_layout(i["out_path"], z.embeddings_[idx], **kwargs)
        # save the model
        save_model(model, model_path)
    # load the list of layout variants
    l = []
    for i in params:
        l.append(
            {
                "n_neighbors": i["n_neighbors"],
                "min_dist": i["min_dist"],
                "layout": i["out_path"],
                "jittered": get_pointgrid_layout(
                    i["out_path"], i["filename"], **kwargs
                ),
            }
        )
    return {
        "variants": l,
    }


def save_model(model, path):
    try:
        params = model.get_params()
        attributes_names = [
            attr for attr in model.__dir__() if attr not in params and attr[0] != "_"
        ]
        attributes = {key: model.__getattribute__(key) for key in attributes_names}
        attributes["embeddings_"] = list(model.embeddings_)
        for x in ["fit", "fit_transform", "update", "get_params", "set_params"]:
            del attributes[x]
        all_params = {
            "umap_params": params,
            "umap_attributes": {key: value for key, value in attributes.items()},
        }
        pickle.dump(all_params, open(path, "wb"))
    except:
        print(timestamp(), "Could not save model")


def load_model(path):
    params = pickle.load(open(path, "rb"))
    model = AlignedUMAP()
    model.set_params(**params.get("umap_params"))
    for attr, value in params.get("umap_attributes").items():
        model.__setattr__(attr, value)
    model.__setattr__(
        "embeddings_", List(params.get("umap_attributes").get("embeddings_"))
    )


def get_umap_model(**kwargs):
    if cuml_ready:
        return UMAP(
            n_neighbors=kwargs["n_neighbors"][0],
            min_dist=kwargs["min_dist"][0],
            n_components=kwargs["n_components"],
            random_state=kwargs["seed"],
            verbose=5,
        )
    else:
        return UMAP(
            n_neighbors=kwargs["n_neighbors"][0],
            min_dist=kwargs["min_dist"][0],
            n_components=kwargs["n_components"],
            metric=kwargs["metric"],
            random_state=kwargs["seed"],
            transform_seed=kwargs["seed"],
        )


def get_lap_layout(**kwargs):
    print(timestamp(), "Creating linear assignment layout")
    try:
        import lap
    except:
        raise Exception("LAP must be installed to use get_lap_layout")
    out_path = get_path("layouts", "linear-assignment", **kwargs)
    if os.path.exists(out_path) and kwargs["use_cache"]:
        return out_path
    # load the umap layout
    umap = np.array(read_json(kwargs["umap"]["variants"][0]["layout"], **kwargs))
    umap = (umap + 1) / 2  # scale 0:1
    # determine length of each side in square grid
    side = math.ceil(umap.shape[0] ** (1 / 2))
    # create square grid 0:1 in each dimension
    grid_x, grid_y = np.meshgrid(np.linspace(0, 1, side), np.linspace(0, 1, side))
    grid = np.dstack((grid_x, grid_y)).reshape(-1, 2)
    # compute pairwise distance costs
    cost = cdist(grid, umap, "sqeuclidean")
    # increase cost
    cost = cost * (10000000.0 / cost.max())
    # run the linear assignment
    min_cost, row_assignments, col_assignments = lap.lapjv(
        np.copy(cost), extend_cost=True
    )
    # use the assignment vals to determine gridified positions of `arr`
    pos = grid[col_assignments]
    return write_layout(out_path, pos, **kwargs)


def get_alphabetic_layout(**kwargs):
    """Get the x,y positions of images in a grid projection"""
    print(timestamp(), "Creating grid layout")
    out_path = get_path("layouts", "grid", **kwargs)
    if os.path.exists(out_path) and kwargs["use_cache"]:
        return out_path
    paths = kwargs["image_paths"]
    n = math.ceil(len(paths) ** (1 / 2))
    l = []  # positions
    for i, _ in enumerate(paths):
        x = i % n
        y = math.floor(i / n)
        l.append([x, y])
    z = np.array(l)
    return write_layout(out_path, z, **kwargs)


def get_pointgrid_layout(path, label, **kwargs):
    """Gridify the positions in `path` and return the path to this new layout"""
    print(timestamp(), "Creating {} pointgrid".format(label))
    out_path = get_path("layouts", label + "-jittered", **kwargs)
    if os.path.exists(out_path) and kwargs["use_cache"]:
        return out_path
    arr = np.array(read_json(path, **kwargs))
    if arr.shape[-1] != 2:
        print(timestamp(), "Could not create pointgrid layout because data is not 2D")
        return None
    z = align_points_to_grid(arr, fill=0.01)
    return write_layout(out_path, z, **kwargs)


def get_custom_layout(**kwargs):
    out_path = get_path("layouts", "custom", **kwargs)
    if os.path.exists(out_path) and kwargs["use_cache"]:
        return out_path
    if not kwargs.get("metadata"):
        return
    found_coords = False
    coords = []
    for i in stream_images(**kwargs):
        x = i.metadata.get("x")
        y = i.metadata.get("y")
        if x and y:
            found_coords = True
            coords.append([x, y])
        else:
            if found_coords:
                print(
                    timestamp(),
                    "Some images are missing coordinates; skipping custom layout",
                )
    if not found_coords:
        return
    coords = np.array(coords).astype(np.float)
    coords = (minmax_scale(coords) - 0.5) * 2
    print(timestamp(), "Creating custom layout")
    return {
        "layout": write_layout(
            out_path, coords.tolist(), scale=False, round=False, **kwargs
        ),
    }


##
# Date Layout
##


def get_date_layout(cols=3, bin_units="years", **kwargs):
    """
    Get the x,y positions of input images based on their dates
    @param int cols: the number of columns to plot for each bar
    @param str bin_units: the temporal units to use when creating bins
    """
    date_vals = [
        kwargs["metadata"][i].get("year", False) for i in range(len(kwargs["metadata"]))
    ]
    if not kwargs["metadata"] or not any(date_vals):
        return False
    # if the data layouts have been cached, return them
    positions_out_path = get_path("layouts", "timeline", **kwargs)
    labels_out_path = get_path("layouts", "timeline-labels", **kwargs)
    if (
        os.path.exists(positions_out_path)
        and os.path.exists(labels_out_path)
        and kwargs["use_cache"]
    ):
        return {
            "layout": positions_out_path,
            "labels": labels_out_path,
        }
    # date layout is not cached, so fetch dates and process
    print(timestamp(), "Creating date layout with {} columns".format(cols))
    datestrings = [i.metadata.get("year", "no_date") for i in stream_images(**kwargs)]
    dates = [datestring_to_date(i) for i in datestrings]
    rounded_dates = [round_date(i, bin_units) for i in dates]
    # create d[formatted_date] = [indices into datestrings of dates that round to formatted_date]
    d = defaultdict(list)
    for idx, i in enumerate(rounded_dates):
        d[i].append(idx)
    # determine the number of distinct grid positions in the x and y axes
    n_coords_x = (cols + 1) * len(d)
    n_coords_y = 1 + max([len(d[i]) for i in d]) // cols
    if n_coords_y > n_coords_x:
        return get_date_layout(cols=int(cols * 2), **kwargs)
    # create a mesh of grid positions in clip space -1:1 given the time distribution
    grid_x = (np.arange(0, n_coords_x) / (n_coords_x - 1)) * 2
    grid_y = (np.arange(0, n_coords_y) / (n_coords_x - 1)) * 2
    # divide each grid axis by half its max length to center at the origin 0,0
    grid_x = grid_x - np.max(grid_x) / 2.0
    grid_y = grid_y - np.max(grid_y) / 2.0
    # make dates increase from left to right by sorting keys of d
    d_keys = np.array(list(d.keys()))
    seconds = np.array([date_to_seconds(dates[d[i][0]]) for i in d_keys])
    d_keys = d_keys[np.argsort(seconds)]
    # determine which images will fill which units of the grid established above
    coords = np.zeros(
        (len(datestrings), 2)
    )  # 2D array with x, y clip-space coords of each date
    for jdx, j in enumerate(d_keys):
        for kdx, k in enumerate(d[j]):
            x = jdx * (cols + 1) + (kdx % cols)
            y = kdx // cols
            coords[k] = [grid_x[x], grid_y[y]]
    # find the positions of labels
    label_positions = np.array(
        [[grid_x[i * (cols + 1)], grid_y[0]] for i in range(len(d))]
    )
    # move the labels down in the y dimension by a grid unit
    dx = grid_x[1] - grid_x[0]  # size of a single cell
    label_positions[:, 1] = label_positions[:, 1] - dx
    # quantize the label positions and label positions
    image_positions = round_floats(coords)
    label_positions = round_floats(label_positions.tolist())
    # write and return the paths to the date based layout
    return {
        "layout": write_json(positions_out_path, image_positions, **kwargs),
        "labels": write_json(
            labels_out_path,
            {
                "positions": label_positions,
                "labels": d_keys.tolist(),
                "cols": cols,
            },
            **kwargs
        ),
    }


def datestring_to_date(datestring):
    """
    Given a string representing a date return a datetime object
    """
    try:
        return parse_date(
            str(datestring), fuzzy=True, default=datetime.datetime(9999, 1, 1)
        )
    except Exception as exc:
        print(timestamp(), "Could not parse datestring {}".format(datestring))
        return datestring


def date_to_seconds(date):
    """
    Given a datetime object return an integer representation for that datetime
    """
    if isinstance(date, datetime.datetime):
        return (date - datetime.datetime.today()).total_seconds()
    else:
        return -float("inf")


def round_date(date, unit):
    """
    Return `date` truncated to the temporal unit specified in `units`
    """
    if not isinstance(date, datetime.datetime):
        return "no_date"
    formatted = date.strftime("%d %B %Y -- %X")
    if unit in set(["seconds", "minutes", "hours"]):
        date = formatted.split("--")[1].strip()
        if unit == "seconds":
            date = date
        elif unit == "minutes":
            date = ":".join(d.split(":")[:-1]) + ":00"
        elif unit == "hours":
            date = date.split(":")[0] + ":00:00"
    elif unit in set(["days", "months", "years", "decades", "centuries"]):
        date = formatted.split("--")[0].strip()
        if unit == "days":
            date = date
        elif unit == "months":
            date = " ".join(date.split()[1:])
        elif unit == "years":
            date = date.split()[-1]
        elif unit == "decades":
            date = str(int(date.split()[-1]) // 10) + "0"
        elif unit == "centuries":
            date = str(int(date.split()[-1]) // 100) + "00"
    return date


##
# Metadata Layout
##


def get_categorical_layout(null_category="Other", margin=2, **kwargs):
    """
    Return a numpy array with shape (n_points, 2) with the point
    positions of observations in box regions determined by
    each point's category metadata attribute (if applicable)
    """
    if not kwargs.get("metadata", False):
        return False
    # determine the out path and return from cache if possible
    out_path = get_path("layouts", "categorical", **kwargs)
    labels_out_path = get_path("layouts", "categorical-labels", **kwargs)
    # accumulate d[category] = [indices of points with category]
    categories = [i.get("category", None) for i in kwargs["metadata"]]
    if not any(categories) or len(set(categories) - set([None])) == 1:
        return False
    d = defaultdict(list)
    for idx, i in enumerate(categories):
        d[i].append(idx)
    # store the number of observations in each group
    keys_and_counts = [{"key": i, "count": len(d[i])} for i in d]
    keys_and_counts.sort(key=operator.itemgetter("count"), reverse=True)
    # get the box layout then subdivide into discrete points
    boxes = get_categorical_boxes([i["count"] for i in keys_and_counts], margin=margin)
    points = get_categorical_points(boxes)
    # sort the points into the order of the observations in the metadata
    counts = {i["key"]: 0 for i in keys_and_counts}
    offsets = {i["key"]: 0 for i in keys_and_counts}
    for idx, i in enumerate(keys_and_counts):
        offsets[i["key"]] += sum([j["count"] for j in keys_and_counts[:idx]])
    sorted_points = []
    for idx, i in enumerate(stream_images(**kwargs)):
        category = i.metadata.get("category", null_category)
        sorted_points.append(points[offsets[category] + counts[category]])
        counts[category] += 1
    sorted_points = np.array(sorted_points)
    # add to the sorted points the anchors for the text labels for each group
    text_anchors = np.array([[i.x, i.y - margin / 2] for i in boxes])
    # add the anchors to the points - these will be removed after the points are projected
    sorted_points = np.vstack([sorted_points, text_anchors])
    # scale -1:1 using the largest axis as the scaling metric
    _max = np.max(sorted_points)
    for i in range(2):
        _min = np.min(sorted_points[:, i])
        sorted_points[:, i] -= _min
        sorted_points[:, i] /= _max - _min
        sorted_points[:, i] -= np.max(sorted_points[:, i]) / 2
        sorted_points[:, i] *= 2
    # separate out the sorted points and text positions
    text_anchors = sorted_points[-len(text_anchors) :]
    sorted_points = sorted_points[: -len(text_anchors)]
    z = round_floats(sorted_points.tolist())
    return {
        "layout": write_json(out_path, z, **kwargs),
        "labels": write_json(
            labels_out_path,
            {
                "positions": round_floats(text_anchors.tolist()),
                "labels": [i["key"] for i in keys_and_counts],
            },
            **kwargs
        ),
    }


def get_categorical_boxes(group_counts, margin=2):
    """
    @arg [int] group_counts: counts of the number of images in each
      distinct level within the metadata's caetgories
    @kwarg int margin: space between boxes in the 2D layout
    @returns [Box] an array of Box() objects; one per level in `group_counts`
    """
    group_counts = sorted(group_counts, reverse=True)
    boxes = []
    for i in group_counts:
        w = h = math.ceil(i ** (1 / 2))
        boxes.append(Box(i, w, h, None, None))
    # find the position along x axis where we want to create a break
    wrap = math.floor(sum([i.cells for i in boxes]) ** (1 / 2)) - (2 * margin)
    # find the valid positions on the y axis
    y = margin
    y_spots = []
    for i in boxes:
        if (y + i.h + margin) <= wrap:
            y_spots.append(y)
            y += i.h + margin
        else:
            y_spots.append(y)
            break
    # get a list of lists where sublists contain elements at the same y position
    y_spot_index = 0
    for i in boxes:
        # find the y position
        y = y_spots[y_spot_index]
        # find members with this y position
        row_members = [j.x + j.w for j in boxes if j.y == y]
        # assign the y position
        i.y = y
        y_spot_index = (y_spot_index + 1) % len(y_spots)
        # assign the x position
        i.x = max(row_members) + margin if row_members else margin
    return boxes


def get_categorical_points(arr, unit_size=None):
    """Given an array of Box() objects, return a 2D distribution with shape (n_cells, 2)"""
    points_arr = []
    for i in arr:
        area = i.w * i.h
        per_unit = (area / i.cells) ** (1 / 2)
        x_units = math.ceil(i.w / per_unit)
        y_units = math.ceil(i.h / per_unit)
        if not unit_size:
            unit_size = min(i.w / x_units, i.h / y_units)
        for j in range(i.cells):
            x = j % x_units
            y = j // x_units
            points_arr.append(
                [
                    i.x + x * unit_size,
                    i.y + y * unit_size,
                ]
            )
    return np.array(points_arr)


class Box:
    """Store the width, height, and x, y coords of a box"""

    def __init__(self, *args):
        self.cells = args[0]
        self.w = args[1]
        self.h = args[2]
        self.x = None if len(args) < 4 else args[3]
        self.y = None if len(args) < 5 else args[4]


##
# Geographic Layout
##


def get_geographic_layout(**kwargs):
    """Return a 2D array of image positions corresponding to lat, lng coordinates"""
    out_path = get_path("layouts", "geographic", **kwargs)
    l = []
    coords = False
    for idx, i in enumerate(stream_images(**kwargs)):
        lat = float(i.metadata.get("lat", 0)) / 180
        lng = (
            float(i.metadata.get("lng", 0)) / 180
        )  # the plot draws longitude twice as tall as latitude
        if lat or lng:
            coords = True
        l.append([lng, lat])
    if coords:
        print(timestamp(), "Creating geographic layout")
        if kwargs["geojson"]:
            process_geojson(kwargs["geojson"])
        return {"layout": write_layout(out_path, l, scale=False, **kwargs)}
    elif kwargs["geojson"]:
        print(
            timestamp(),
            "GeoJSON is only processed if you also provide lat/lng coordinates for your images in a metadata file!",
        )
    return None


def process_geojson(geojson_path):
    """Given a GeoJSON filepath, write a minimal JSON output in lat lng coordinates"""
    with open(geojson_path, "r") as f:
        geojson = json.load(f)
    l = []
    for i in geojson:
        if isinstance(i, dict):
            for j in i.get("coordinates", []):
                for k in j:
                    l.append(k)
    with open(
        os.path.join("output", "assets", "json", "geographic-features.json"), "w"
    ) as out:
        json.dump(l, out)


##
# Helpers
##


def get_path(*args, **kwargs):
    """Return the path to a JSON file with conditional gz extension"""
    sub_dir, filename = args
    out_dir = join(kwargs["out_dir"], sub_dir) if sub_dir else kwargs["out_dir"]
    if kwargs.get("add_hash", True):
        filename += "-" + kwargs["plot_id"]
    path = join(out_dir, filename + ".json")
    return path + ".gz" if kwargs.get("gzip", False) else path


def write_layout(path, obj, **kwargs):
    """Write layout json `obj` to disk and return the path to the saved file"""
    if kwargs.get("scale", True) != False:
        obj = (minmax_scale(obj) - 0.5) * 2  # scale -1:1
    if kwargs.get("round", True) != False:
        obj = round_floats(obj)
    if isinstance(obj, np.ndarray):
        obj = obj.tolist()
    return write_json(path, obj, **kwargs)


def round_floats(obj, digits=5):
    """Return 2D array obj with rounded float precision"""
    return [[round(float(j), digits) for j in i] for i in obj]


def write_json(path, obj, **kwargs):
    """Write json object `obj` to disk and return the path to that file"""
    out_dir, filename = os.path.split(path)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if kwargs.get("gzip", False):
        with gzip.GzipFile(path, "w") as out:
            out.write(json.dumps(obj, indent=4).encode(kwargs["encoding"]))
        return path
    else:
        with open(path, "w") as out:
            json.dump(obj, out, indent=4)
        return path


def read_json(path, **kwargs):
    """Read and return the json object written by the current process at `path`"""
    if kwargs.get("gzip", False):
        with gzip.GzipFile(path, "r") as f:
            return json.loads(f.read().decode(kwargs["encoding"]))
    with open(path) as f:
        return json.load(f)


def get_provided_hotspots(**kwargs):

    metadata = kwargs["metadata"]

    d = defaultdict(lambda: defaultdict(list))
    for idx, m in enumerate(metadata):
        cluster_id = m["cluster"]
        if cluster_id != -1:
            d[cluster_id]["images"].append(idx)
            d[cluster_id]["img"] = clean_filename(kwargs["image_paths"][idx])
            d[cluster_id]["layout"] = "inception_vectors"

    return d


def run_clustering(layouts={}, use_high_dimensional_vectors=True, **kwargs):
    print(timestamp(), "Clustering data with {}".format(cluster_method))
    if use_high_dimensional_vectors:
        vecs = kwargs["vecs"]
    else:
        vecs = read_json(layouts["umap"]["variants"][0]["layout"], **kwargs)
    model = get_cluster_model(**kwargs)
    z = model.fit(vecs)

    # create a map from cluster label to image indices in cluster
    d = defaultdict(lambda: defaultdict(list))
    for idx, i in enumerate(z.labels_):
        if i != -1:
            d[i]["images"].append(idx)
            d[i]["img"] = clean_filename(kwargs["image_paths"][idx])
            d[i]["layout"] = "inception_vectors"
    return d


def get_hotspots(layouts={}, use_high_dimensional_vectors=True, **kwargs):
    """Return the stable clusters from the condensed tree of connected components from the density graph"""

    if kwargs["clusters_provided"]:
        d = get_provided_hotspots(**kwargs)
    else:
        d = run_clustering(layouts, use_high_dimensional_vectors, **kwargs)

    # remove massive clusters
    deletable = []
    for i in d:
        # find percent of images in cluster
        image_percent = len(d[i]["images"]) / kwargs["num_images"]
        # determine if image or area percent is too large
        if image_percent > 0.5:
            deletable.append(i)

    for i in deletable:
        del d[i]
    # sort the clusers by size and then label the clusters
    clusters = d.values()
    clusters = sorted(clusters, key=lambda i: len(i["images"]), reverse=True)

    for idx, i in enumerate(clusters):
        i["label"] = "Cluster {}".format(idx + 1)

    # slice off the first `max_clusters`
    clusters = clusters[: kwargs["max_clusters"]]

    # save the hotspots to disk and return the path to the saved json
    print(timestamp(), "Found", len(clusters), "hotspots")

    return write_json(get_path("hotspots", "hotspot", **kwargs), clusters, **kwargs)


def get_cluster_model(**kwargs):
    """Return a model with .fit() method that can be used to cluster input vectors"""
    if cluster_method == "hdbscan":
        config = {
            "core_dist_n_jobs": multiprocessing.cpu_count(),
            "min_cluster_size": kwargs["min_cluster_size"],
            "cluster_selection_epsilon": 0.01,
            "min_samples": 1,
            "approx_min_span_tree": False,
        }
        return HDBSCAN(**config)
    else:
        return KMeans(n_clusters=kwargs["n_clusters"], random_state=kwargs["seed"])


def get_heightmap(path, label, **kwargs):
    """Create a heightmap using the distribution of points stored at `path`"""
    import matplotlib.pyplot as plt

    X = read_json(path, **kwargs)
    if "positions" in X:
        X = X["positions"]
    X = np.array(X)
    if X.shape[-1] != 2:
        print(timestamp(), "Could not create heightmap because data is not 2D")
        return
    # create kernel density estimate of distribution X
    nbins = 200
    x, y = X.T
    xi, yi = np.mgrid[x.min() : x.max() : nbins * 1j, y.min() : y.max() : nbins * 1j]
    zi = kde.gaussian_kde(X.T)(np.vstack([xi.flatten(), yi.flatten()]))
    # create the plot
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    fig.subplots_adjust(0, 0, 1, 1)
    plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading="gouraud", cmap=plt.cm.gray)
    plt.axis("off")
    # save the plot
    out_dir = os.path.join(kwargs["out_dir"], "heightmaps")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_path = os.path.join(out_dir, label + "-heightmap.png")
    plt.savefig(out_path, pad_inches=0)


def convert_and_write_image(i, out_dir, thumbnail_out_dir, lod_cell_height):
    """Save an primary image and thumbnail to disk"""
    filename = clean_filename(i.filename)

    # save resized original for lightbox
    out_path = os.path.join(out_dir, filename)
    if not os.path.exists(out_path):
        resized = i.resize_to_height(600)
        resized = array_to_img(resized)
        save_img(out_path, resized)

    # save thumbnail for lod texture
    out_path = join(thumbnail_out_dir, filename)
    img = array_to_img(i.resize_to_max(lod_cell_height))
    save_img(out_path, img)


def write_images(**kwargs):
    """Write all originals and thumbs to the output dir"""

    print(timestamp(), "Writing images and thumbnails...")

    out_dir = join(kwargs["out_dir"], "originals")
    if not exists(out_dir):
        os.makedirs(out_dir)

    thumbnail_out_dir = join(kwargs["out_dir"], "thumbs")
    if not exists(thumbnail_out_dir):
        os.makedirs(thumbnail_out_dir)

    lod_cell_height = kwargs["lod_cell_height"]

    results = parallel.Parallel(n_jobs=-1)(
        delayed(convert_and_write_image)(
            img, out_dir, thumbnail_out_dir, lod_cell_height
        )
        for img in stream_images(**kwargs)
    )


def get_version():
    """Return the version of pixplot installed"""
    return "1.0.0"  # pkg_resources.get_distribution("pixplot").version


def load_image(root_path, filename):
    """Load image located in the full `image_path`"""

    try:
        if filename.startswith("http"):
            img = pil_image.open(urlopen(filename)).convert("RGB")
        else:
            image_path = os.path.join(root_path, filename)
            img = pil_image.open(image_path).convert("RGB")
    except Exception as e:
        print(timestamp(), "Could not load image from path:", root_path, filename)
        print(e)
        return None

    return img


class Image:
    def __init__(self, *args, **kwargs):
        self.root_path = args[0]
        self.file_location = args[1]
        self.filename = args[2]
        self.original = load_image(self.root_path, self.file_location)
        self.metadata = kwargs["metadata"]
        self.vec = kwargs["vec"]
        self.valid = False

    def resize_to_max(self, n):
        """
        Resize self.original so its longest side has n pixels (maintain proportion)
        """
        w, h = self.original.size
        size = (n, int(n * h / w)) if w > h else (int(n * w / h), n)
        return np.array(self.original.resize(size), dtype=np.float32)

    def resize_to_height(self, height):
        """
        Resize self.original into an image with height h and proportional width
        """
        w, h = self.original.size
        if (w / h * height) < 1:
            resizedwidth = 1
        else:
            resizedwidth = int(w / h * height)
        size = (resizedwidth, height)
        return np.array(self.original.resize(size), dtype=np.float32)

    def resize_to_square(self, n, center=False):
        """
        Resize self.original to an image with nxn pixels (maintain proportion)
        if center, center the colored pixels in the square, else left align
        """
        a = self.resize_to_max(n)
        h, w, c = a.shape
        pad_lr = int((n - w) / 2)  # left right pad
        pad_tb = int((n - h) / 2)  # top bottom pad
        b = np.zeros((n, n, 3))
        if center:
            b[pad_tb : pad_tb + h, pad_lr : pad_lr + w, :] = a
        else:
            b[:h, :w, :] = a
        return b


##
# Entry Point
##


def parse():
    """Read command line args and begin data processing"""
    description = "Create the data required to create a PixPlot viewer"
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--images",
        "-i",
        type=str,
        default=config["images"],
        help="path to a glob of images to process",
        required=False,
    )
    parser.add_argument(
        "--image_vectors",
        type=str,
        default=None,
        help="Path to a numpy array file of feature vectors of the images in metadata.csv. Numpy shape is: [[num_images, size_of_embedding]]. File must be created using np.save",
        required=False,
    )
    parser.add_argument(
        "--metadata",
        "-m",
        type=str,
        default=config["metadata"],
        help="path to a csv or glob of JSON files with image metadata (see readme for format)",
        required=False,
    )
    parser.add_argument(
        "--clusters_provided",
        type=bool,
        default=config["clusters_provided"],
        help="whether the metadata file contains cluster labels in the x and y columns.",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=config["max_images"],
        help="maximum number of images to process from the input glob",
        required=False,
    )
    parser.add_argument(
        "--use_cache",
        type=bool,
        default=config["use_cache"],
        help="given inputs identical to prior inputs, load outputs from cache",
        required=False,
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default=config["encoding"],
        help="the encoding of input metadata",
        required=False,
    )
    parser.add_argument(
        "--min_cluster_size",
        type=int,
        default=config["min_cluster_size"],
        help="the minimum number of images in a cluster",
        required=False,
    )
    parser.add_argument(
        "--max_clusters",
        type=int,
        default=config["max_clusters"],
        help="the maximum number of clusters to return",
        required=False,
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=config["out_dir"],
        help="the directory to which outputs will be saved",
        required=False,
    )
    parser.add_argument(
        "--cell_size",
        type=int,
        default=config["cell_size"],
        help="the size of atlas cells in px",
        required=False,
    )
    parser.add_argument(
        "--n_neighbors",
        nargs="+",
        type=int,
        default=config["n_neighbors"],
        help="the n_neighbors arguments for UMAP",
    )
    parser.add_argument(
        "--min_dist",
        nargs="+",
        type=float,
        default=config["min_dist"],
        help="the min_dist arguments for UMAP",
    )
    parser.add_argument(
        "--n_components",
        type=int,
        default=config["n_components"],
        help="the n_components argument for UMAP",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default=config["metric"],
        help="the metric argument for umap",
    )
    parser.add_argument(
        "--pointgrid_fill",
        type=float,
        default=config["pointgrid_fill"],
        help="float 0:1 that determines sparsity of jittered distributions (lower means more sparse)",
    )
    parser.add_argument(
        "--copy_web_only",
        action="store_true",
        help="update ./output/assets without reprocessing data",
    )
    parser.add_argument(
        "--min_size",
        type=float,
        default=config["min_size"],
        help="min size of cropped images",
    )
    parser.add_argument(
        "--gzip", action="store_true", help="save outputs with gzip compression"
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="shuffle the input images before data processing begins",
    )
    parser.add_argument(
        "--plot_id",
        type=str,
        default=config["plot_id"],
        help="unique id for a plot; useful for resuming processing on a started plot",
    )
    parser.add_argument(
        "--seed", type=int, default=config["seed"], help="seed for random processes"
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=config["n_clusters"],
        help="number of clusters to use when clustering with kmeans",
    )
    parser.add_argument(
        "--geojson",
        type=str,
        default=config["geojson"],
        help="path to a GeoJSON file with shapes to be rendered on a map",
    )
    config.update(vars(parser.parse_args()))
    process_images(**config)


if __name__ == "__main__":
    parse()
