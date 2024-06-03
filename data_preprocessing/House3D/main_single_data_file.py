import numpy as np
import cv2
import torch
from tqdm import tqdm 
import os
import time

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


def decode_image(img_str, resize=None):
    """
    Decode image from tfrecord data
    :param img_str: image encoded as a png in a string
    :param resize: tuple width two elements that defines the new size of the image. optional
    :return: image as a numpy array
    """
    nparr = np.fromstring(img_str, np.uint8)
    img_str = cv2.imdecode(nparr, -1)
    if resize is not None:
        img_str = cv2.resize(img_str, resize)
    return img_str


def raw_images_to_array(images):
    """
    Decode and normalize multiple images from tfrecord data
    :param images: list of images encoded as a png in a string
    :return: a numpy array of size (N, 56, 56, channels), normalized for training
    """
    image_list = []
    for image_str in images:
        image = decode_image(image_str, (56, 56))
        # image = scale_observation(np.atleast_3d(image.astype(np.float32)))
        image_list.append(image)

    return np.stack(image_list, axis=0)

def scale_observation(x):
    """
    Normalizes observation input, either an rgb image or a depth image
    :param x: observation input as numpy array, either an rgb image or a depth image
    :return: numpy array, a normalized observation
    """
    if x.ndim == 2 or x.shape[2] == 1:  # depth
        return x * (2.0 / 100.0) - 1.0
    else:  # rgb
        return x * (2.0/255.0) - 1.0


def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def process_file(file, output_dir):
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

    all_sequence_data = dict()

    # Iterate over all the records in the dataset
    gen = tf.compat.v1.python_io.tf_record_iterator(file)    
    for data_i, string_record in tqdm(enumerate(gen), total=dataset_size):
        result = tf.compat.v1.train.Example.FromString(string_record)
        features = result.features.feature

        # convert to a dict
        features = dict(features)
        # print(features.keys())

        # The room ID
        roomid = features['roomID'].bytes_list.value[0].decode()

        # The house ID
        houseid = features['houseID'].bytes_list.value[0].decode()

        # true states
        # (x, y, theta). x,y: pixel coordinates; theta: radians
        # coordinates index the map as a numpy array: map[x, y]
        true_states = features['states'].bytes_list.value[0]
        true_states = np.frombuffer(true_states, np.float32).reshape((-1, 3))

        # Load the odometry
        odometry = features['odometry'].bytes_list.value[0]
        odometry = np.frombuffer(odometry, np.float32).reshape((-1, 3))

        # Make a copy so we can edit the array
        odometry = np.copy(odometry)

        # Add noise!

        #  Too Little Noise
        odometry[:-1, 0:2] += np.random.normal(loc=0, scale=10.0, size=odometry[:-1, 0:2].shape)
        odometry[:-1, 2] += np.random.vonmises(mu=0, kappa=35.0, size=odometry[:-1, 2].shape)

        # More Noise!!
        # odometry[:-1, 0:2] += np.random.normal(loc=0, scale=20.0, size=odometry[:-1, 0:2].shape)
        # odometry[:-1, 2] += np.random.vonmises(mu=0, kappa=20.0, size=odometry[:-1, 2].shape)

        # Even Noise!!
        # odometry[:-1, 0:2] += np.random.normal(loc=0, scale=30.0, size=odometry[:-1, 0:2].shape)
        # odometry[:-1, 2] += np.random.vonmises(mu=0, kappa=5.0, size=odometry[:-1, 2].shape)


        # Create the dictionary of information to save
        save_dict = dict()
        save_dict["roomid"] = roomid
        save_dict["houseid"] = houseid
        save_dict["true_states"] = true_states
        save_dict["odometry"] = odometry

        all_sequence_data[data_i] = save_dict


    torch.save(all_sequence_data, "{}/data.pt".format(output_dir))

process_file("../../data/House3D/test.tfrecords", "../../data/House3D/test/")
process_file("../../data/House3D/valid.tfrecords", "../../data/House3D/valid/")
process_file("../../data/House3D/train.tfrecords", "../../data/House3D/train/")
