# Standard Imports
import os
import numpy as np
import math
import PIL
from PIL import Image
from tqdm import tqdm
import cv2

# Pytorch Imports
import torch
import torch.distributions as D

# Project imports
from utils import *
from datasets.base_dataset import *


class Sequence:

    def __init__(self, dataset_base_dir, sequence_number, data, max_sequence_length):

        # Save some of the important info
        self.sequence_number = sequence_number

        # Create the main sequence directory
        self.sequence_dir = "{}/sequence_{:07d}".format(dataset_base_dir, sequence_number)

        # Create the directories for all the sub-parts
        self.map_files_directory = "{}/map_files/".format(self.sequence_dir)
        self.rgb_directory = "{}/rgb_images/".format(self.sequence_dir)
        self.depth_directory = "{}/depth_images/".format(self.sequence_dir)

        # Load and extract the data files
        # data = torch.load("{}/data.pt".format(self.sequence_dir))
        self.room_id = data["roomid"]
        self.house_id = data["houseid"]
        self.states = torch.from_numpy(data["true_states"])
        self.odometry = torch.from_numpy(data["odometry"])
        # self.rgb = data["rgb"]

        # Compute the sequence length
        self.sequence_length = self.states.shape[0]

        # If we have a max sequence length then we should use it
        if(max_sequence_length is not None):
            self.states = self.states[:max_sequence_length, ...]
            self.odometry = self.odometry[:max_sequence_length, ...]

            self.sequence_length = max_sequence_length            

    def get_sequence_length(self):
        return self.sequence_length

    def add_data_to_dict(self, return_dict, data, name, disabled_outputs):
        if(name not in disabled_outputs):
            return_dict[name] = data

    def get_data(self, disabled_outputs):

        # Load the observations
        observations = self.load_observations()

        # Load the map
        world_map, current_size = self.load_world_map()

        # Pack everything up
        return_dict = dict()

        self.add_data_to_dict(return_dict, self.states, "states", disabled_outputs)
        self.add_data_to_dict(return_dict, observations, "observations", disabled_outputs)
        self.add_data_to_dict(return_dict, self.odometry, "actions", disabled_outputs)
        self.add_data_to_dict(return_dict, world_map, "world_map", disabled_outputs)
        self.add_data_to_dict(return_dict, current_size, "world_map_origional_size", disabled_outputs)



        # return_dict["states"] = self.states
        # return_dict["observations"] = observations
        # return_dict["actions"] = self.odometry
        # return_dict["world_map"] = world_map
        # return_dict["world_map_origional_size"] = current_size

        return return_dict        

    def get_states(self):
        return self.states

    def load_observations(self):

        # The default image size for the images on disk. 
        DEFAULT_IMAGE_SIZE = 56

        # Load all the images into 1 big array
        images = np.zeros((self.sequence_length, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, 3))
        for i in range(self.sequence_length):
            
            # Load the image and convert from OpenCV BGR to RGB
            rgb_image_file_path = "{}/{:04d}.png".format(self.rgb_directory, i)
            img = cv2.imread(rgb_image_file_path, cv2.IMREAD_COLOR) 

            # img = self.rgb[i]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Append to the images 
            images[i,...] = img

        # Normalize the image to be between 0 and 1
        images = images.astype("float32") / 255.0

        # Convert to pytorch
        images = torch.from_numpy(images)

        # Permute to make sure the channels are how pytorch expects them 
        images = torch.permute(images, (0, 3, 1, 2))        

        return images
      
    def load_world_map(self):

        # The max size of any of the world maps (found by looking for the map with the max size)
        WORLD_MAP_MAX_SIZE = 3010

        # Load the walls map file
        map_wall = cv2.imread("{}/map_wall.png".format(self.map_files_directory), cv2.IMREAD_GRAYSCALE) 
        # map_door = cv2.imread("{}/map_door.png".format(self.map_files_directory), cv2.IMREAD_GRAYSCALE) 

        # Invert so that free space=255 and walls = 0
        map_wall = 255.0 - map_wall

        # Put it in the size of the large world map
        current_size = map_wall.shape[0]
        full_map = np.zeros((WORLD_MAP_MAX_SIZE, WORLD_MAP_MAX_SIZE))
        full_map[0:current_size, 0:current_size] = map_wall

        # convert to floats
        full_map = full_map.astype("float32")

        # Rescale to 0..2 range. this way zero padding will produce the equivalent of obstacles
        full_map = full_map * (2.0 / 255.0)

        # Convert to pytorch
        world_map = torch.from_numpy(full_map)

        # transpose the map since for some reason the saved png has w for h and h for w
        world_map = torch.permute(world_map, (1, 0))        

        return world_map, current_size


class House3DDataset(BaseDataset):
    def __init__(self, dataset_params, dataset_type):
        super().__init__()
        
        # Save the inputs in case we need them later
        self.dataset_type = dataset_type
        self.dataset_params = dataset_params

        # Extract some parameters
        self.dataset_directory = get_parameter_safely("dataset_directory", dataset_params, "dataset_params")

        # Get the max sequence length param if we have one
        if("max_sequence_length" in dataset_params):
            self.max_sequence_length = dataset_params["max_sequence_length"]
        else:
            self.max_sequence_length = None

        # See if we should use sparse ground truths
        if("sparse_ground_truth_keep_modulo" in dataset_params):
            self.sparse_ground_truth_keep_modulo = dataset_params["sparse_ground_truth_keep_modulo"]
        else:
            self.sparse_ground_truth_keep_modulo = None

        # How much data to use.  This is only applied during training
        if(((self.dataset_type == "training") or (self.dataset_type == "validation")) and ("percent_of_dataset_to_use" in dataset_params)):
            self.percent_of_dataset_to_use = dataset_params["percent_of_dataset_to_use"]
            assert(self.percent_of_dataset_to_use > 0.0)
            assert(self.percent_of_dataset_to_use <= 1.0)

            print("")
            print("")
            print("")
            print("")
            print("WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING ")
            print("WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING ")
            print("WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING ")
            print("WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING ")
            print("")
            print("Not using all of the data! Make sure this is what you want to do!")
            print("")
            print("WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING ")
            print("WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING ")
            print("WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING ")
            print("WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING ")
            print("")
            print("")
            print("")
            print("")
        else:
            self.percent_of_dataset_to_use = None

        # Load the dataset
        self.load_dataset()        

    def __len__(self):
        if(self.percent_of_dataset_to_use is None):
            return len(self.all_sequences)  
        else:
            return int(float(len(self.all_sequences)) * self.percent_of_dataset_to_use)                      

    def __getitem__(self, idx_in):

        # Compute the index based on the ordering
        # idx = self.ordering[idx_in]
        idx = idx_in


        return_dict = self.all_sequences[idx].get_data(self.disabled_outputs)
        return_dict["dataset_index"] = idx


        if(self.sparse_ground_truth_keep_modulo is not None):

            states = self.all_sequences[idx].get_states()
            ground_truth_mask = torch.full(size=(states.shape[0],), fill_value=False)

            for i in range(ground_truth_mask.shape[0]):
                if((i % self.sparse_ground_truth_keep_modulo) == 0):
                    ground_truth_mask[i] = True

            return_dict["ground_truth_mask"] = ground_truth_mask

        return return_dict


    def get_subsequence_length(self):
        return self.all_sequences[0].get_sequence_length()

    def load_dataset(self):

        print("Loading Dataset: {}".format(self.dataset_type))

        # dataset base dir
        if(self.dataset_type == "training"):
            self.dataset_base_dir = "{}/train/".format(self.dataset_directory)
            ordering_filepath = "{}/train_ordering.pt".format(self.dataset_base_dir)
        
        elif(self.dataset_type == "validation"):
            # self.dataset_base_dir = "{}/valid/".format(self.dataset_directory)
            self.dataset_base_dir = "{}/train/".format(self.dataset_directory)
            ordering_filepath = "{}/valid_ordering.pt".format(self.dataset_base_dir)

        elif(self.dataset_type == "evaluation"):
            self.dataset_base_dir = "{}/test/".format(self.dataset_directory)
            ordering_filepath = "{}/ordering.pt".format(self.dataset_base_dir)
        
        else:
            assert(False)

        # Load the data.pt file
        data_filepath = "{}/data.pt".format(self.dataset_base_dir)
        print("Loading:", data_filepath)
        data_map = torch.load(data_filepath)

        # load the ordering
        self.ordering = torch.load(ordering_filepath)

        # Count how many sequences there are
        # self.number_of_sequences = len([name for name in os.(self.dataset_base_dir) if os.path.isdir(os.path.join(self.dataset_base_dir, name))])
        self.number_of_sequences = len(self.ordering)
        
        # Create the sequences
        self.all_sequences = []
        for i in tqdm(range(self.number_of_sequences), leave=False, desc="Loading"):  
            idx = self.ordering[i]
            sequence = Sequence(self.dataset_base_dir, idx, data_map[idx], self.max_sequence_length)
            self.all_sequences.append(sequence)

        # Verify that all the sequences have the same sequence length
        sequence_length = self.all_sequences[0].get_sequence_length()
        for sequence in tqdm(self.all_sequences, leave=False, desc="Verifying"):
            assert(sequence.get_sequence_length() == sequence_length)



    # def get_collate_fn(self):
    #     # None means that there is no collate function to return
    #     return House3DDataset.collate_fn

    # @staticmethod
    # def collate_fn(batch_list):

    #     # create the dictionary to return
    #     return_dict = dict()
    #     return_dict["states"] = []
    #     return_dict["observations"] = []
    #     return_dict["actions"] = []
    #     return_dict["world_map"] = []
    #     return_dict["dataset_index"] = []

    #     # Pack into a large thing
    #     for i in range(len(batch_list)):
    #         for k in return_dict.keys():
    #             return_dict[k].append(batch_list[i][k])

    #     # Stack the correct keys
    #     keys_to_stack = ["states", "observations", "actions", "dataset_index"]
    #     for k in keys_to_stack:
    #         if(torch.is_tensor(return_dict[k][0])):
    #             return_dict[k] = torch.stack(return_dict[k])    
    #         else:
    #             return_dict[k] = torch.FloatTensor(return_dict[k])    

    #     return return_dict