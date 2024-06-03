# Standard Imports
import os
import numpy as np
import math

# Useful Imports
import PIL
from PIL import Image
from tqdm import tqdm

# Pytorch Imports
import torch
import torch.distributions as D

# Project imports
from utils import *
from datasets.base_dataset import *


class DeepMindMazeDataset(BaseDataset):
    def __init__(self, dataset_params, dataset_type):
        super().__init__()

        # Save the inputs in case we need them later
        self.dataset_type = dataset_type
        self.dataset_params = dataset_params

        # The dafault save location is none (aka dont save)
        self.save_location = None


        # Extract the params        
        self.subsequence_length = get_parameter_safely("subsequence_length", dataset_params, "dataset_params")
        dataset_directory = get_parameter_safely("dataset_directory", dataset_params, "dataset_params")
        self.use_actions = get_parameter_safely("use_actions", dataset_params, "dataset_params")

        # Extract which files to use
        all_filenames = get_parameter_safely("filenames", dataset_params, "dataset_params")
        files_to_use = get_parameter_safely(dataset_type, all_filenames, "all_filenames")

        assert(len(files_to_use) == 1)

        if("01" in files_to_use[0]):
            self.selected_map_id = 0
        elif("02" in files_to_use[0]):
            self.selected_map_id = 1
        elif("03" in files_to_use[0]):
            self.selected_map_id = 2

        # See if we should use sparse ground truths
        if("sparse_ground_truth_keep_modulo" in dataset_params):
            self.sparse_ground_truth_keep_modulo = dataset_params["sparse_ground_truth_keep_modulo"]
        else:
            self.sparse_ground_truth_keep_modulo = None


        # The scale for the output range
        if("scale_output_range" in dataset_params):
            self.scale_output_range = dataset_params["scale_output_range"]
        else:
            self.scale_output_range = 20.0

        # The min max scales for the different maps
        self.scale_min_max_x = dict()
        self.scale_min_max_y = dict()

        # Load it
        did_load = self.load_from_save(dataset_params, dataset_type)

        # Load the data if needed
        if(did_load == False):

            # Load the data from the files and process the data
            self.load_data_from_files(files_to_use, dataset_directory, self.subsequence_length)

            # Save the data
            self.save_dataset()

    def __len__(self):
        return self.states.shape[0]

    def __getitem__(self, idx):

        # Compute the return dictionary
        return_dict = {}
        return_dict["states"] = self.states[idx]
        return_dict["observations"] = self.observations[idx]
        return_dict["map_id"] = self.map_id[idx]
        return_dict["dataset_index"] = idx

        if(self.use_actions):
            return_dict["actions"] = self.actions[idx]

        if(self.sparse_ground_truth_keep_modulo is not None):
            ground_truth_mask = torch.full(size=(self.states[idx].shape[0],), fill_value=False)

            for i in range(ground_truth_mask.shape[0]):
                if((i % self.sparse_ground_truth_keep_modulo) == 0):
                    ground_truth_mask[i] = True

            return_dict["ground_truth_mask"] = ground_truth_mask

        return return_dict


    def scale_data_up(self, data, map_id=None):

        if(map_id is None):
            map_id = self.selected_map_id

        scale_min_x, scale_max_x = self.scale_min_max_x[map_id]
        scale_min_y, scale_max_y = self.scale_min_max_y[map_id]

        output_data = data.clone()

        output_data[...,0] += (self.scale_output_range / 2.0)
        output_data[...,0] /= (self.scale_output_range )
        output_data[...,0] *= (scale_max_x - scale_min_x)
        output_data[...,0] += (scale_min_x)

        output_data[...,1] += (self.scale_output_range / 2.0)
        output_data[...,1] /= (self.scale_output_range )
        output_data[...,1] *= (scale_max_y - scale_min_y)
        output_data[...,1] += (scale_min_y)

        return output_data

    def scale_data_down(self, data, map_id=None):

        if(map_id is None):
            map_id = self.selected_map_id

        scale_min_x, scale_max_x = self.scale_min_max_x[map_id]
        scale_min_y, scale_max_y = self.scale_min_max_y[map_id]

        if(isinstance(data, np.ndarray)):
            output_data = data.copy()
            output_data = output_data.astype("float")
        else:
            output_data = data.clone()

        output_data[...,0] -= float(scale_min_x)
        output_data[...,0] /= float(scale_max_x - scale_min_x)
        output_data[...,0] *= float(self.scale_output_range )
        output_data[...,0] -= float(self.scale_output_range / 2.0)

        output_data[...,1] -= float(scale_min_y)
        output_data[...,1] /= float(scale_max_y - scale_min_y)
        output_data[...,1] *= float(self.scale_output_range )
        output_data[...,1] -= float(self.scale_output_range / 2.0)


        return output_data

    def get_subsequence_length(self):
        return self.subsequence_length
        
    def get_x_range(self, map_id):
        scale_min_x, scale_max_x = self.scale_min_max_x[map_id]
        return (scale_min_x, scale_max_x)

    def get_y_range(self, map_id):
        scale_min_y, scale_max_y = self.scale_min_max_y[map_id]
        return (scale_min_y, scale_max_y)

    def get_x_range_scaled(self, map_id):
        return (-self.scale_output_range / 2.0, self.scale_output_range / 2.0)

    def get_y_range_scaled(self, map_id):
        return (-self.scale_output_range / 2.0, self.scale_output_range / 2.0)

    def get_height_to_width_ratio(self, map_id):
        w = self.get_x_range(map_id)[1] - self.get_x_range(map_id)[0]
        h = self.get_y_range(map_id)[1] - self.get_y_range(map_id)[0]
        return float(h) / float(w)

    def load_data_from_files(self, filenames, dataset_directory, subsequence_length):
        # Make sure we have files to load
        assert(len(filenames) > 0)

        # Load all the files into ram
        all_data = []
        number_of_subsequences = 0
        for filename in tqdm(filenames, leave=False):

            # Load the data splits
            splits = self.load_splits_file(filename, dataset_directory)

            # Load the data from the file and convert it into a dictionary with the following keys
            #   - vel
            #   - rgbd
            #   - pose
            data = dict(np.load(os.path.join(dataset_directory, filename + '.npz')))

            map_id = None
            if("nav01" in filename):
                map_id = 0

                # Set the min and max scale
                scale_min_x = 0
                scale_max_x = 1000
                scale_min_y = 0
                scale_max_y = 500
                self.scale_min_max_x[map_id] = (scale_min_x, scale_max_x)
                self.scale_min_max_y[map_id] = (scale_min_y, scale_max_y)
                
            elif("nav02" in filename):
                map_id = 1

                # Set the min and max scale
                scale_min_x = 0
                scale_max_x = 1500
                scale_min_y = 0
                scale_max_y = 900
                self.scale_min_max_x[map_id] = (scale_min_x, scale_max_x)
                self.scale_min_max_y[map_id] = (scale_min_y, scale_max_y)


            elif("nav03" in filename):
                map_id = 2

                # Set the min and max scale
                scale_min_x = 0
                scale_max_x = 2000
                scale_min_y = 0
                scale_max_y = 1300
                self.scale_min_max_x[map_id] = (scale_min_x, scale_max_x)
                self.scale_min_max_y[map_id] = (scale_min_y, scale_max_y)

            else:
                assert(False)


            # Convert the data into the correct length segments (aka 1000 trajectories with each having 100 steps)            
            for key in data.keys():

                # Old
                s = [1000, 100]

                # new and slightly suspect but this is how the papers are doing it in the code 
                # though this is not what is described in the text....
                # s = [100, 1000]
                s.extend(data[key].shape[1:])
                data[key] = np.reshape(data[key], s)


            # If we have splits then do the split
            if(splits is not None):
                for key in data.keys():
                    data[key] = data[key][splits]

            # Convert all the data to float and into pytorch tensors
            for key in data.keys():
                data[key] = data[key].astype("float32")
                data[key] = torch.from_numpy(data[key])

            for key in ['pose', 'vel']:
                # Convert from degrees to radians:                
                data[key][:,:, 2] *= np.pi / 180

                data[key][:,:, 0] = data[key][:,:, 0] - scale_min_x                    
                data[key][:,:, 0] = data[key][:,:, 0] / (scale_max_x - scale_min_x)

                data[key][:,:, 1] = data[key][:,:, 1] - scale_min_y                    
                data[key][:,:, 1] = data[key][:,:, 1] / (scale_max_y - scale_min_y)


                data[key][:,:, 0] = data[key][:,:, 0] * (self.scale_output_range)
                data[key][:,:, 0] = data[key][:,:, 0] - (self.scale_output_range / 2.0)
                data[key][:,:, 1] = data[key][:,:, 1] * (self.scale_output_range)
                data[key][:,:, 1] = data[key][:,:, 1] - (self.scale_output_range / 2.0)


            # angles should be between -pi and pi
            data['pose'][:,:, 2] = self.wrap_angle(data['pose'][:,:, 2])

            # Crop and add noise to the image
            data["rgbd"] = self.crop_and_add_noise_to_images(data["rgbd"])

            # Compute the empirical velocities
            abs_d_x = (data['pose'][:, 1:, 0:1] - data['pose'][:, :-1, 0:1])
            abs_d_y = (data['pose'][:, 1:, 1:2] - data['pose'][:, :-1, 1:2])
            d_theta = self.wrap_angle(data['pose'][:, 1:, 2:3] - data['pose'][:, :-1, 2:3])
            s = np.sin(data['pose'][:, :-1, 2:3])
            c = np.cos(data['pose'][:, :-1, 2:3])
            rel_d_x = c * abs_d_x + s * abs_d_y
            rel_d_y = s * abs_d_x - c * abs_d_y
            empirical_vel = np.concatenate([rel_d_x, rel_d_y, d_theta], axis=-1)

            # Add noise to actions

            # This is the way the orgional paper did it
            # empirical_vel = empirical_vel * np.random.normal(1.0, 0.1, size=empirical_vel.shape) # paper
            empirical_vel = empirical_vel * np.random.normal(1.0, 0.5, size=empirical_vel.shape) # ali

            # This is my (ali's) way
            # xy_noise_scale = np.mean(np.abs(empirical_vel[:,:, 0])) / 2.33
            # # xy_noise_scale *= 0.5
            # xy_noise_scale *= 0.15
            # # xy_noise_scale = 1.0
            # empirical_vel[:,:, 0:2] += np.random.normal(loc=0, scale=xy_noise_scale, size=empirical_vel[:,:, 0:2].shape)
            # empirical_vel[:, :, 2] += np.random.vonmises(mu=0, kappa=35.0, size=empirical_vel[:, :, 2].shape)
            # empirical_vel[:, :, 2] = self.wrap_angle(empirical_vel[:, :, 2])


            # Cut the sequences so they are the same size as the empirical velocity
            for key in data.keys():
                data[key] = data[key][:,:-1,...]

            # Add the empirical vel to the data
            data["empirical_vel"] = torch.from_numpy(empirical_vel)

            # Make sure we can make sub sequences of the right size
            long_sequence_length = data[list(data.keys())[0]].shape[1]
            if((long_sequence_length % subsequence_length) != 0):

                # compute how much data we need to delete from the sequence to make the long sequence evenly 
                # divisible into subsequences of the specified length
                amount_of_data_to_truncate = long_sequence_length % subsequence_length

                print("WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING")
                print("Dataset file {} has long sequence length of {} which is not evenly divisible by subsequence_length {}:".format(filename, long_sequence_length, subsequence_length))
                print("\t\t Deleting {} samples from the dataset".format(amount_of_data_to_truncate))
                print("WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING")

                # Delete the data
                for key in data.keys():
                    data[key] = data[key][:, :-amount_of_data_to_truncate,...]

            # convert into discrete subsequences
            for key in data.keys():
                data[key] = torch.reshape(data[key], [-1] + list(data[key].shape[2:]))
                data[key] = torch.reshape(data[key], [-1, subsequence_length] + list(data[key].shape[1:]))

            # Save the ID that the map came from
            data["map_id"] = map_id

            # Save the data until we convert it into tensors
            all_data.append(data)

            # compute how many subsequences we have there are
            number_of_subsequences += data[list(data.keys())[0]].shape[0]

        # Create the objects we will be using for loading
        self.states = torch.zeros(size=(number_of_subsequences, subsequence_length, all_data[0]["pose"].shape[-1]))
        self.observations = torch.zeros(size=(number_of_subsequences, subsequence_length, all_data[0]["rgbd"].shape[-3], all_data[0]["rgbd"].shape[-2], all_data[0]["rgbd"].shape[-1]))
        self.actions = torch.zeros(size=(number_of_subsequences, subsequence_length, all_data[0]["empirical_vel"].shape[-1]))
        self.map_id = torch.zeros(size=(number_of_subsequences, ))

        # Fill in the data
        starting_counter = 0
        for data in all_data:

            # compute the starting and stopping indices
            data_length = data[list(data.keys())[0]].shape[0]
            s = starting_counter
            e = starting_counter + data_length
            starting_counter += data_length

            # Fill in the data
            self.states[s:e, ...] = data["pose"] 
            self.observations[s:e, ...] = data["rgbd"] 
            self.actions[s:e, ...] = data["empirical_vel"] 
            self.map_id[s:e] = data["map_id"] 

        # For the observations we have (b, width, height, channels), we want (batch, channels, width, height)
        self.observations = self.observations.permute((0, 1, 4, 2, 3))

    def crop_and_add_noise_to_images(self, image_data, odom_noise_factor=1.0, img_noise_factor=1.0, img_random_shift=True):
        '''
            Some of this code is taken from:
                https://github.com/tu-rbo/differentiable-particle-filters
        '''


        # remove the depth fromt he rgbd so we have rgb
        image_data = image_data[...,:3]

        # Incoming Image size
        incoming_width = image_data.shape[2]
        incoming_height = image_data.shape[3]

        # The output size
        OUTPUT_WIDTH = 24
        OUTPUT_HEIGHT = 24

        # We will crop to 24x24 sized images
        new_observations = torch.zeros(image_data.shape[0], image_data.shape[1], OUTPUT_HEIGHT, OUTPUT_WIDTH, image_data.shape[-1])

        # crop the image to a random 24x24 image from the 32x32 image
        for i in range(image_data.shape[0]):
            for j in range(image_data.shape[1]):
                if img_random_shift:
                        offsets_h = np.random.random_integers(0, incoming_height - OUTPUT_HEIGHT, 1)
                        offsets_w = np.random.random_integers(0, incoming_width - OUTPUT_WIDTH, 1)
                else:
                    offsets = (int((incoming_height-OUTPUT_HEIGHT)/2), int((incoming_width-OUTPUT_WIDTH)/2))

                # Crop
                hs = int(offsets_h)
                he = int(hs + OUTPUT_HEIGHT)
                ws = int(offsets_w)
                we = int(ws + OUTPUT_WIDTH)
                new_observations[i,j] = image_data[i, j, hs:he, ws:we, :]

        # Add random noise to the image
        normal_dist = D.Normal(0.0, 20.0*img_noise_factor)
        new_observations += normal_dist.sample(new_observations.shape)

        # Make the pixels have a range of [0,1] instead of [0,255]
        new_observations /= 255.0

        return new_observations

    def wrap_angle(self, angle):
        return ((angle - np.pi) % (2.0 * np.pi)) - np.pi

    def get_walls(self, map_id):
        if(map_id == 0):
            walls = np.array([
                # horizontal
                [[0, 500], [1000, 500]],
                [[400, 400], [500, 400]],
                [[600, 400], [700, 400]],
                [[800, 400], [1000, 400]],
                [[200, 300], [400, 300]],
                [[100, 200], [200, 200]],
                [[400, 200], [700, 200]],
                [[200, 100], [300, 100]],
                [[600, 100], [900, 100]],
                [[0, 0], [1000, 0]],
                # vertical
                [[0, 0], [0, 500]],
                [[100, 100], [100, 200]],
                [[100, 300], [100, 500]],
                [[200, 200], [200, 400]],
                [[200, 0], [200, 100]],
                [[300, 100], [300, 200]],
                [[300, 400], [300, 500]],
                [[400, 100], [400, 400]],
                [[500, 0], [500, 200]],
                [[600, 100], [600, 200]],
                [[700, 200], [700, 300]],
                [[800, 200], [800, 400]],
                [[900, 100], [900, 300]],
                [[1000, 0], [1000, 500]],
            ])

        elif(map_id == 1):
            walls = np.array([
                # horizontal
                [[0, 900], [1500, 900]],
                [[100, 800], [400, 800]],
                [[500, 800], [600, 800]],
                [[800, 800], [1000, 800]],
                [[1100, 800], [1200, 800]],
                [[1300, 800], [1400, 800]],
                [[100, 700], [600, 700]],
                [[700, 700], [800, 700]],
                [[1000, 700], [1100, 700]],
                [[1200, 700], [1400, 700]],
                [[900, 600], [1200, 600]],
                [[1300, 600], [1500, 600]],
                [[0, 500], [100, 500]],
                [[1300, 500], [1400, 500]],
                [[100, 400], [200, 400]],
                [[1200, 400], [1400, 400]],
                [[300, 300], [800, 300]],
                [[900, 300], [1200, 300]],
                [[400, 200], [600, 200]],
                [[700, 200], [800, 200]],
                [[1200, 200], [1500, 200]],
                [[200, 100], [300, 100]],
                [[500, 100], [700, 100]],
                [[800, 100], [900, 100]],
                [[1100, 100], [1400, 100]],
                [[0, 0], [1500, 0]],
                # vertical
                [[0, 0], [0, 900]],
                [[100, 0], [100, 300]],
                [[100, 500], [100, 600]],
                [[100, 700], [100, 800]],
                [[200, 100], [200, 200]],
                [[200, 300], [200, 400]],
                [[200, 500], [200, 700]],
                [[300, 100], [300, 300]],
                [[400, 0], [400, 200]],
                [[500, 800], [500, 900]],
                [[700, 100], [700, 200]],
                [[700, 700], [700, 800]],
                [[800, 200], [800, 800]],
                [[900, 100], [900, 700]],
                [[1000, 0], [1000, 200]],
                [[1000, 700], [1000, 800]],
                [[1100, 700], [1100, 800]],
                [[1100, 100], [1100, 300]],
                [[1200, 800], [1200, 900]],
                [[1200, 400], [1200, 700]],
                [[1300, 200], [1300, 300]],
                [[1300, 500], [1300, 600]],
                [[1400, 300], [1400, 500]],
                [[1400, 700], [1400, 800]],
                [[1500, 0], [1500, 900]],
            ])
        elif(map_id == 2):
            walls = np.array([
                # horizontal
                [[0, 1300], [2000, 1300]],
                [[100, 1200], [500, 1200]],
                [[600, 1200], [1400, 1200]],
                [[1600, 1200], [1700, 1200]],
                [[0, 1100], [600, 1100]],
                [[1500, 1100], [1600, 1100]],
                [[1600, 1000], [1800, 1000]],
                [[800, 1000], [900, 1000]],
                [[100, 1000], [200, 1000]],
                [[700, 900], [800, 900]],
                [[1600, 900], [1800, 900]],
                [[200, 800], [300, 800]],
                [[800, 800], [1200, 800]],
                [[1300, 800], [1500, 800]],
                [[1600, 800], [1900, 800]],
                [[900, 700], [1400, 700]],
                [[1500, 700], [1600, 700]],
                [[1700, 700], [1900, 700]],
                [[700, 600], [800, 600]],
                [[1400, 600], [1500, 600]],
                [[1600, 600], [1700, 600]],
                [[100, 500], [200, 500]],
                [[300, 500], [500, 500]],
                [[600, 500], [700, 500]],
                [[1400, 500], [1900, 500]],
                [[100, 400], [200, 400]],
                [[400, 400], [600, 400]],
                [[1500, 400], [1600, 400]],
                [[1700, 400], [1800, 400]],
                [[200, 300], [300, 300]],
                [[400, 300], [500, 300]],
                [[600, 300], [800, 300]],
                [[900, 300], [1100, 300]],
                [[1300, 300], [1500, 300]],
                [[1600, 300], [1700, 300]],
                [[100, 200], [200, 200]],
                [[500, 200], [600, 200]],
                [[800, 200], [1100, 200]],
                [[1200, 200], [1400, 200]],
                [[1500, 200], [1600, 200]],
                [[200, 100], [300, 100]],
                [[500, 100], [800, 100]],
                [[1000, 100], [1200, 100]],
                [[1400, 100], [1600, 100]],
                [[1800, 100], [1900, 100]],
                [[0, 0], [2000, 0]],
                # vertical
                [[0, 0], [0, 1300]],
                [[100, 0], [100, 300]],
                [[100, 400], [100, 1000]],
                [[200, 300], [200, 400]],
                [[200, 600], [200, 800]],
                [[200, 900], [200, 1000]],
                [[300, 100], [300, 600]],
                [[300, 800], [300, 1100]],
                [[400, 0], [400, 300]],
                [[400, 1200], [400, 1300]],
                [[500, 100], [500, 200]],
                [[600, 200], [600, 400]],
                [[600, 1100], [600, 1200]],
                [[700, 200], [700, 300]],
                [[700, 400], [700, 1100]],
                [[800, 100], [800, 200]],
                [[800, 300], [800, 500]],
                [[800, 600], [800, 700]],
                [[800, 1000], [800, 1100]],
                [[900, 0], [900, 100]],
                [[900, 300], [900, 600]],
                [[900, 900], [900, 1200]],
                [[1000, 100], [1000, 200]],
                [[1200, 100], [1200, 200]],
                [[1300, 0], [1300, 100]],
                [[1400, 100], [1400, 700]],
                [[1500, 700], [1500, 1000]],
                [[1500, 1100], [1500, 1200]],
                [[1600, 200], [1600, 400]],
                [[1600, 600], [1600, 700]],
                [[1600, 1000], [1600, 1100]],
                [[1600, 1200], [1600, 1300]],
                [[1700, 1100], [1700, 1200]],
                [[1700, 700], [1700, 800]],
                [[1700, 500], [1700, 600]],
                [[1700, 0], [1700, 300]],
                [[1800, 100], [1800, 400]],
                [[1800, 600], [1800, 700]],
                [[1800, 900], [1800, 1200]],
                [[1900, 800], [1900, 1300]],
                [[1900, 100], [1900, 600]],
                [[2000, 0], [2000, 1300]],
            ])

        return walls

    def load_from_save(self, dataset_params, dataset_type):

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
        dataset_params = all_data["dataset_params"]
        dataset_type = all_data["dataset_type"]

        # Not the same so must rebuild the dataset
        if((dataset_params != self.dataset_params) or (dataset_type != self.dataset_type)):
            return False

        # Is the same so unpack the rest
        self.scale_min_max_x = all_data["scale_min_max_x"]
        self.scale_min_max_y = all_data["scale_min_max_y"]
        self.states = all_data["states"]
        self.observations = all_data["observations"]
        self.actions = all_data["actions"]
        self.map_id = all_data["map_id"]
        
        # We have successfully loaded the dataset
        return True


    def save_dataset(self):

        # If we dont have a save location then we cant save
        if(self.save_location is None):
            return

        # Pack everything into a single dict that we can save
        save_dict = dict()
        save_dict["scale_min_max_x"] = self.scale_min_max_x
        save_dict["scale_min_max_y"] = self.scale_min_max_y
        save_dict["states"] = self.states
        save_dict["observations"] = self.observations
        save_dict["actions"] = self.actions
        save_dict["map_id"] = self.map_id
        save_dict["dataset_params"] = self.dataset_params
        save_dict["dataset_type"] = self.dataset_type

        # Save that dict
        torch.save(save_dict, self.save_location)

    def load_splits_file(self, filename, dataset_directory):

        # create the file name
        if("nav01" in filename):
            splits_file_name = "nav01_splits.pt"
        elif("nav02" in filename):
            splits_file_name = "nav02_splits.pt"
        elif("nav03" in filename):
            splits_file_name = "nav03_splits.pt"
        else:
            assert(False)

        # Load the data
        all_splits = torch.load("{}/{}".format(dataset_directory, splits_file_name))

        # Extract only the one we need
        return all_splits[self.dataset_type]
