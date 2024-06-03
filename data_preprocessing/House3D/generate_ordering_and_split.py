import numpy as np
import cv2
import torch
from tqdm import tqdm 
import os
import time
import random

# make it so tensorflow cant load on the GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Silence tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf

# add to the top of your code under import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)




tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def process_file(file, output_dir, dataset_type):
    print("")
    print("")
    print("")
    print("")
    print("")
    print("")
    print("================================================================================")
    print("Processing File")
    print(file)
    print("================================================================================")

    # Make sure the output dir exists
    ensure_directory_exists(output_dir)

    # Get the dataset size
    dataset_size = sum(1 for _ in tf.compat.v1.python_io.tf_record_iterator(file))
    print("dataset_size", dataset_size)


    # Generate a random ordering for the dataset
    ordering = [i for i in range(dataset_size)]
    random.shuffle(ordering)

    if(dataset_type == "test"):
        torch.save(ordering, "{}/ordering.pt".format(output_dir))

    else:

        # Find the percentage for training and the rest is validation
        ratio_training = 0.8
        split_idx = int(float(len(ordering)) * ratio_training)

        ordering_train = ordering[0:split_idx]
        ordering_valid = ordering[split_idx:]

        torch.save(ordering_train, "{}/train_ordering.pt".format(output_dir))
        torch.save(ordering_valid, "{}/valid_ordering.pt".format(output_dir))



process_file("../../data/House3D/test.tfrecords", "../../data/House3D/test/", "test")
# process_file("../../data/House3D/valid.tfrecords", "../../data/House3D/valid/")
process_file("../../data/House3D/train.tfrecords", "../../data/House3D/train/", "train")
