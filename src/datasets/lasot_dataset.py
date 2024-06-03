# Standard Imports
import os
import numpy as np
import math
import cv2
import yaml
import random
from tqdm import tqdm

# Pytorch Imports
import torch

# # Project imports
from utils import *
from datasets.base_dataset import *



class ReferencePatch:
    def __init__(self, directory=None, patch_frame_number=None):
        
        # Save info about the patch
        self.directory = directory
        self.patch_frame_number = patch_frame_number

    def create_save_dict(self):
        save_dict = dict()
        save_dict["directory"] = self.directory
        save_dict["patch_frame_number"] = self.patch_frame_number
        return save_dict

    def load_from_save_dict(self, save_dict):
        self.directory = save_dict["directory"] 
        self.patch_frame_number = save_dict["patch_frame_number"] 

    def get_patch(self):

        return self.load_patch()

    def load_patch(self):

        # Create the image file path.  Note the image names start at 1 not 0 so add 1 to the image name 
        # to get the correct image
        image_filepath = "{}/patches/{:08d}.png".format(self.directory, self.patch_frame_number+1)

        # Load the image
        img = cv2.imread(image_filepath, cv2.IMREAD_COLOR)

        # Convert from OpenCV BGR to standard RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalize the image to be between 0 and 1
        img = img.astype("float32") / 255.0

        # Convert to pytorch
        img = torch.from_numpy(img)

        # Permute to make sure the channels are how pytorch expects them 
        img = torch.permute(img, (2, 0, 1))        

        return img



class Subsequence:
    def __init__(self, directory=None, start_index=None, length=None, groundtruths=None, not_in_frame=None, save_dict=None):

        if(save_dict is not None):
            self.load_from_save_dict(save_dict)
        else:
            # Save the information for this subsequence
            self.directory = directory
            self.start_index = start_index
            self.length = length
            self.groundtruths = groundtruths[start_index:(start_index + length)]
            self.reference_patch = self.extract_reference_patch(directory, start_index, length, not_in_frame, groundtruths)
            self.ground_truth_mask = ~not_in_frame[start_index:(start_index + length)]

    def create_save_dict(self):
        save_dict = dict()
        save_dict["directory"] = self.directory
        save_dict["start_index"] = self.start_index
        save_dict["scale_output_range"] = self.scale_output_range
        save_dict["length"] = self.length
        save_dict["groundtruths"] = self.groundtruths
        save_dict["reference_patch"] = self.reference_patch.create_save_dict()
        save_dict["ground_truth_mask"] = self.ground_truth_mask
        return save_dict

    def load_from_save_dict(self, save_dict):
        self.directory = save_dict["directory"]
        self.start_index = save_dict["start_index"]
        self.length = save_dict["length"]
        self.groundtruths = save_dict["groundtruths"]
        self.scale_output_range = save_dict["scale_output_range"]
        self.ground_truth_mask = save_dict["ground_truth_mask"]

        self.reference_patch = ReferencePatch()
        self.reference_patch.load_from_save_dict(save_dict["reference_patch"])

    def extract_reference_patch(self, directory, start_index, length, not_in_frame, groundtruths):

        start_index = 1

        def patch_is_valid(index):

            # If it is not in the frame at all then it is not valid
            if(not_in_frame[index]):
                return False

            # If the bounding box has 0 width or height then it is also not valid
            if((groundtruths[index, 2] == 0) or (groundtruths[index, 3] == 0)):
                return False

            return True


        # If the first frame is in view then use that as the reference patch
        if(patch_is_valid(start_index)):
            return ReferencePatch(directory, start_index)

        else:
            # The first frame was not valid so find the frame closest that is in view

            # Closest in view looking forward
            closest_frame_in_view_forward = None
            for i in range(start_index, len(not_in_frame)):
                if(patch_is_valid(i)):
                    closest_frame_in_view_forward = i
                    break

            # Closest in view looking backward
            closest_frame_in_view_backward = None
            for i in range(start_index, 0, -1):
                if(patch_is_valid(i)):
                    closest_frame_in_view_backward = i
                    break

            # Make sure we have a frame
            assert((closest_frame_in_view_forward is not None) or (closest_frame_in_view_backward is not None))

            # If we have a none then choose the other one
            if(closest_frame_in_view_forward is None):
                closest_frame = closest_frame_in_view_backward

            elif(closest_frame_in_view_backward is None):
                closest_frame = closest_frame_in_view_forward

            else:
                # Both are not none so choose the closest frame

                # Compute the distances
                delta_forward = closest_frame_in_view_forward - start_index
                delta_backward = start_index - closest_frame_in_view_backward

                # Choose closer, if both equal distance then prefer the one that is forward
                if(delta_forward <=  delta_backward):
                    closest_frame = closest_frame_in_view_forward
                else:
                    closest_frame = closest_frame_in_view_backward

            # Create the reference patch
            return ReferencePatch(directory, closest_frame)

    def get_return_dict(self):
        
        # Get the images
        images = self.load_image_sequence()

        # Get the reference patch
        reference_patch = self.reference_patch.get_patch()

        # Pack and return the dict
        return_dict = {}
        return_dict["states"] = self.groundtruths.float()
        return_dict["observations"] = images.float()
        return_dict["reference_patch"] = reference_patch.float()
        return_dict["ground_truth_mask"] = self.ground_truth_mask
        return return_dict

    def load_image_sequence(self):

        # Load the images 1 by 1
        images = [self.load_image(self.directory, self.start_index + i) for i in range(self.length)]

        # convert to pytorch
        images = torch.stack(images)

        return images

    def load_image(self, directory, frame_index):

        # Create the image file path.  Note the image names start at 1 not 0 so add 1 to the image name 
        # to get the correct image
        image_filepath = "{}/img/{:08d}.png".format(directory, frame_index+1)

        # Load the image
        img = cv2.imread(image_filepath, cv2.IMREAD_COLOR)

        # Convert from OpenCV BGR to standard RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalize the image to be between 0 and 1
        img = img.astype("float32") / 255.0

        # Convert to pytorch
        img = torch.from_numpy(img)

        # Permute to make sure the channels are how pytorch expects them 
        img = torch.permute(img, (2, 0, 1))        

        return img







class LasotDataset(BaseDataset):
    def __init__(self, dataset_params, dataset_type):

        # Save the inputs in case we need them later
        self.dataset_type = dataset_type
        self.dataset_params = dataset_params

        # The scale for the output range
        self.scale_output_range = 20.0

        # Extract the params        
        self.subsequence_length = get_parameter_safely("subsequence_length", dataset_params, "dataset_params")
        dataset_directory = get_parameter_safely("dataset_directory", dataset_params, "dataset_params")

        # Load the dataset splits
        dataset_splits_file = get_parameter_safely("dataset_splits_file", dataset_params, "dataset_params")

        # Extract the dataset sequences for this trainer type
        with open(dataset_splits_file) as file:

            # Load the whole file into a dictionary
            dataset_splits_file_data = yaml.load(file, Loader=yaml.FullLoader)

            # Get the splits
            dataset_splits = get_parameter_safely("dataset_splits", dataset_splits_file_data, "dataset_splits_file_data")

        # Get the specific sequence
        self.dataset_sequences = get_parameter_safely(dataset_type, dataset_splits, "dataset_splits")

        # check if we can load a dataset
        did_load_dataset = self.load_from_save(dataset_params, dataset_type)

        if(not did_load_dataset):

            # Create the sequences
            self.all_subsequences = self.create_sequences(dataset_directory, self.dataset_sequences)

            # Shuffle the lists
            random.shuffle(self.all_subsequences)

            # Save the dataset
            self.save_dataset()

    def save_dataset(self):

        # Nothing to do if we dont have a save location
        if(self.save_location is None):
            return

        save_dict = dict()
        save_dict["dataset_type"] = self.dataset_type
        save_dict["dataset_params"] = self.dataset_params
        save_dict["scale_output_range"] = self.scale_output_range
        save_dict["subsequence_length"] = self.subsequence_length
        save_dict["image_size"] = self.image_size
        save_dict["reference_patch_size"] = self.reference_patch_size
        save_dict["dataset_sequences"] = self.dataset_sequences

        sequences_dict = dict()
        for i in range(len(self.all_subsequences)):
            sequences_dict[i] = self.all_subsequences[i].create_save_dict()
        save_dict["sequences_dict"] = self.sequences_dict

        torch.save(save_dict, self.save_location)


    def load_from_save(self, dataset_params, dataset_type):

        # The dafault save location is none (aka dont save)
        self.save_location = None

        # Check if we have the save params, if not then we cant load from save
        if("dataset_saves" not in dataset_params):
            return False

        # Extract this dataset save location 
        dataset_saves = get_parameter_safely("dataset_saves", dataset_params, "dataset_params")

        # If this dataset type is not in the list of saved datasets then we cannot save or load this specific dataset
        if(dataset_type not in dataset_saves):
            return False

        # Extract the save location
        self.save_location = get_parameter_safely(dataset_type, dataset_saves, "dataset_saves")

        # If the directory does not exist then we want to create it!
        data_dir, _ = os.path.split(self.save_location)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # Check if the file exists, if not then we cant load 
        if(not os.path.exists(self.save_location)):
            return False

        # Load and unpack data
        all_data = torch.load(self.save_location)
        dataset_type = save_dict["dataset_type"]
        dataset_params = save_dict["dataset_params"]
        scale_output_range = save_dict["scale_output_range"]
        subsequence_length = save_dict["subsequence_length"]
        image_size = save_dict["image_size"]
        reference_patch_size = save_dict["reference_patch_size"]
        dataset_sequences = save_dict["dataset_sequences"]

        # Not the same so must rebuild the dataset
        if((dataset_params != self.dataset_params) or (dataset_type != self.dataset_type)):
            return False

        # Load the dataset
        self.all_subsequences = []
        for i in save_dict["sequences_dict"]:
            self.all_subsequences.append(Subsequence(save_dict=save_dict["sequences_dict"][i]))

        # We have successfully loaded the dataset
        return True


    def get_subsequence_length(self):
        return self.subsequence_length
        
    def get_x_range(self):
        return (-10, 10)

    def get_y_range(self):
        return (-10, 10)


    def create_sequences(self, dataset_directory, dataset_sequences):

        problem_sequences = []
        problem_sequences.append(("coin", 13))
        problem_sequences.append(("cup", 19))
        problem_sequences.append(("drone", 16))
        problem_sequences.append(("giraffe", 4))
        problem_sequences.append(("hat", 18))
        problem_sequences.append(("helmet", 8))
        problem_sequences.append(("helmet", 9))
        problem_sequences.append(("kangaroo", 7))
        problem_sequences.append(("lion", 4))
        problem_sequences.append(("person", 6))
        problem_sequences.append(("pool", 13))

        # Construct the full paths for the sequences.  Each full path points to a directory containing 
        # the information for 1 sequence (of varying length)
        all_sequence_directories_full_paths = []
        for sequence_type in dataset_sequences.keys():
            for seq_num in dataset_sequences[sequence_type]:

                # Skip all the sequences that had a problem
                if((sequence_type == problem_sequences[0]) and (seq_num == problem_sequences[1])):
                    continue


                sequence_full_path = "{}/{}/{}-{}/".format(dataset_directory,sequence_type,sequence_type, seq_num)
                all_sequence_directories_full_paths.append(sequence_full_path)
        
        # Figure out the length of the
        all_subsequences = []
        for sequence_directory in tqdm(all_sequence_directories_full_paths):
            all_subsequences.extend(self.split_sequence_into_subsequence(sequence_directory))
        
        return all_subsequences    

    def split_sequence_into_subsequence(self, sequence_directory):
        
        # Use the ground truth file to get the length of the sequence
        groundtruths = torch.load("{}/groundtruths.pt".format(sequence_directory))
        sequence_length = groundtruths.shape[0]

        # Load the not in frame data
        not_in_frame = torch.load("{}/not_in_frame.pt".format(sequence_directory))

        # Find the first frame that doesnt have an occluded
        first_in_view_frame = None
        for i in range(sequence_length):
            if(not not_in_frame[i]):
                first_in_view_frame = i
                break

        # Get the image size once so we can load quickly
        image_size = self.get_image_size(sequence_directory)

        # Split sequences
        sequence_start_indices = [i for i in range(first_in_view_frame, sequence_length, self.subsequence_length)]

        # Create the subsequences
        subsequences = []
        for i in range(len(sequence_start_indices)):

            # Make sure we have a full sequence
            if((sequence_start_indices[i] + self.subsequence_length) > sequence_length):
                continue

            # Check if the sequence is mostly not in view.  If it is then skip it
            not_in_frame_count = 0
            for j in range(self.subsequence_length):
                if(not_in_frame[sequence_start_indices[i] + j]):
                    not_in_frame_count += 1

            # Skip if mostly not in view
            not_in_frame_percentage = float(not_in_frame_count) / float(self.subsequence_length)
            if(not_in_frame_percentage > 0.5):
                continue

            # The subsequence is in view so make it
            subsequences.append(Subsequence(sequence_directory, sequence_start_indices[i], self.subsequence_length, groundtruths, not_in_frame))

        return subsequences

    def get_image_size(self, directory):
        # Get the image size by loading an image
        image_filepath = "{}/img/{:08d}.png".format(directory, 1)
        img = cv2.imread(image_filepath, cv2.IMREAD_COLOR)
        img_width = float(img.shape[1])
        img_height = float(img.shape[0])

        return (img_width, img_height)



    def __len__(self):
        return len(self.all_subsequences)

    def __getitem__(self, idx):

        # Compute the return dictionary
        return_dict = self.all_subsequences[idx].get_return_dict()

        # Add some extras to it
        return_dict["dataset_index"] = idx
        return return_dict

