import tensorflow as tf
import numpy as np
import cv2
import torch
from tqdm import tqdm 
import os

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

    # Make sure the output dir exists
    ensure_directory_exists(output_dir)

    # Get the dataset size
    dataset_size = sum(1 for _ in tf.compat.v1.python_io.tf_record_iterator(file))

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

        # wall map: 0 for free space, 255 for walls
        map_wall = decode_image(features['map_wall'].bytes_list.value[0])

        # door map: 0 for free space, 255 for doors
        map_door = decode_image(features['map_door'].bytes_list.value[0])

        # roomtype map: binary encoding of 8 possible room categories
        # one state may belong to multiple room categories
        map_roomtype = decode_image(features['map_roomtype'].bytes_list.value[0])

        # roomid map: pixels correspond to unique room ids.
        # for overlapping rooms the higher ids overwrite lower ids
        map_roomid = decode_image(features['map_roomid'].bytes_list.value[0])

        # true states
        # (x, y, theta). x,y: pixel coordinates; theta: radians
        # coordinates index the map as a numpy array: map[x, y]
        true_states = features['states'].bytes_list.value[0]
        true_states = np.frombuffer(true_states, np.float32).reshape((-1, 3))

        # odometry
        # each entry is true_states[i+1]-true_states[i].
        # last row is always [0,0,0]
        # odometry = features['odometry'].bytes_list.value[0]
        # odometry = np.frombuffer(odometry, np.float32).reshape((-1, 3))
        # compute the odometry manually since it is not correct from the file
        odometry = np.zeros_like(true_states)
        odometry[:-1, ...] = true_states[1:, ...] - true_states[:-1, ...]


        # observations are enceded as a list of png images
        rgb = raw_images_to_array(list(features['rgb'].bytes_list.value))
        depth = raw_images_to_array(list(features['depth'].bytes_list.value))


        # Make the general save dir
        save_dir = "{}/sequence_{:07d}".format(output_dir, data_i)
        ensure_directory_exists(save_dir)

        # Make the map save dir
        map_save_dir = "{}/map_files/".format(save_dir)
        ensure_directory_exists(map_save_dir)

        # Make the image save dirs
        rgb_save_dir = "{}/rgb_images/".format(save_dir)
        depth_save_dir = "{}/depth_images/".format(save_dir)
        ensure_directory_exists(rgb_save_dir)
        ensure_directory_exists(depth_save_dir)


        # Create the dictionary of information to save
        save_dict = dict()
        save_dict["roomid"] = roomid
        save_dict["houseid"] = houseid
        save_dict["true_states"] = true_states
        save_dict["odometry"] = odometry
        torch.save(save_dict, "{}/data.pt".format(save_dir))
        all_sequence_data[data_i] = save_dict


        # Save the map artifacts
        cv2.imwrite("{}/map_wall.png".format(map_save_dir), map_wall)
        cv2.imwrite("{}/map_door.png".format(map_save_dir), map_door)
        cv2.imwrite("{}/map_roomtype.png".format(map_save_dir), map_roomtype)
        cv2.imwrite("{}/map_roomid.png".format(map_save_dir), map_roomid)

        # Save the rgb images
        for i in range(rgb.shape[0]):
            rgb_image_save_file = "{}/{:04d}.png".format(rgb_save_dir, i)
            cv2.imwrite(rgb_image_save_file, rgb[i])

        # Save the depth images
        for i in range(depth.shape[0]):
            depth_image_save_file = "{}/{:04d}.png".format(depth_save_dir, i)
            cv2.imwrite(depth_image_save_file, depth[i])

    torch.save(all_sequence_data, "{}/data.pt".format(output_dir))

process_file("../../data/House3D/test.tfrecords", "../../data/House3D/test/")
process_file("../../data/House3D/valid.tfrecords", "../../data/House3D/valid/")
process_file("../../data/House3D/train.tfrecords", "../../data/House3D/train/")
