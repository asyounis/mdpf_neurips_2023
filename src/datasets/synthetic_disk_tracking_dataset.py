# Standard Imports
import os
import numpy as np
import math
import random
import shutil
import PIL
# from PIL import Image
from tqdm import tqdm
import cv2
from matplotlib import colors


# Pytorch Imports
import torch
import torch.distributions as D

# Project imports
from utils import *
from datasets.base_dataset import *

class SyntheticDiskTrackingDataset(BaseDataset):
    def __init__(self, dataset_params, dataset_type):
        '''
            Heavily taken from: https://github.com/akloss/differentiable_filters/blob/main/differentiable_filters/data/create_disc_tracking_dataset.py
        '''

        # All the possible colors for disks
        # all_color_names = ["blue", "green", "red", "cyan", "magenta", "yellow", "tab:blue", "tab:green", "tab:orange", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
        all_color_names = ["blue", "green", "red", "cyan", "magenta", "yellow", "tab:orange", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive"]
        self.all_possible_colors = dict()
        for c_name in all_color_names:

            # dont want the alpha value
            rgb = list(colors.to_rgba(c_name))[:-1]

            # convert from [0, 1] to [0, 255]
            rgb = [int(c*255) for c in rgb]

            self.all_possible_colors[c_name] = rgb

        # Save the inputs in case we need them later
        self.dataset_type = dataset_type
        self.dataset_params = dataset_params

        # Extract the general parameters for this Dataset
        self.subsequence_length = get_parameter_safely("subsequence_length", dataset_params, "dataset_params")
        self.image_size = get_parameter_safely("image_size", dataset_params, "dataset_params")

        # The parameters for the size of the tracked disk
        self.use_static_tracked_disc_size = get_parameter_safely("use_static_tracked_disc_size", dataset_params, "dataset_params")
        if(self.use_static_tracked_disc_size):
            self.tracked_disc_radius = get_parameter_safely("tracked_disc_radius", dataset_params, "dataset_params")
        else:
            self.tracked_disc_min_radius = get_parameter_safely("tracked_disc_min_radius", dataset_params, "dataset_params")
            self.tracked_disc_max_radius = get_parameter_safely("tracked_disc_max_radius", dataset_params, "dataset_params")
            assert(self.tracked_disc_min_radius > 0)
            assert(self.tracked_disc_max_radius > 0)
            assert(self.tracked_disc_min_radius < self.tracked_disc_max_radius)

        # The parameters for the color of the tracked disk
        self.use_static_tracked_disc_color = get_parameter_safely("use_static_tracked_disc_color", dataset_params, "dataset_params")
        if(self.use_static_tracked_disc_size):
            self.tracked_disc_color = get_parameter_safely("tracked_disc_color", dataset_params, "dataset_params")
            assert(self.tracked_disc_color in self.all_possible_colors)

        # Parameters for the distractor disks
        self.number_of_distractor_disks = get_parameter_safely("number_of_distractor_disks", dataset_params, "dataset_params")
        self.distractor_disks_min_radius = get_parameter_safely("distractor_disks_min_radius", dataset_params, "dataset_params")
        self.distractor_disks_max_radius = get_parameter_safely("distractor_disks_max_radius", dataset_params, "dataset_params")

        # Parameters of the system
        self.disk_spring_force = get_parameter_safely("disk_spring_force", dataset_params, "dataset_params")
        self.disk_drag_force = get_parameter_safely("disk_drag_force", dataset_params, "dataset_params")
        self.use_heteroscedastic_process_noise_for_velocity = get_parameter_safely("use_heteroscedastic_process_noise_for_velocity", dataset_params, "dataset_params")
        self.use_correlated_process_noise = get_parameter_safely("use_correlated_process_noise", dataset_params, "dataset_params")
        self.position_noise_scale = get_parameter_safely("position_noise_scale", dataset_params, "dataset_params")
        
        # Cant have both correlated and heteroscedastic noise
        if(self.use_heteroscedastic_process_noise_for_velocity and self.use_correlated_process_noise):
            print("Cant have both correlated and heteroscedastic process noise for disks")
            assert(False)

        # Extract the size of the dataset for this dataset
        dataset_sizes = get_parameter_safely("dataset_sizes", dataset_params, "dataset_params")
        self.dataset_size = get_parameter_safely(dataset_type, dataset_sizes, "dataset_sizes")

        # Get the dataset save location
        dataset_saves = get_parameter_safely("dataset_saves", dataset_params, "dataset_params")
        self.save_location = get_parameter_safely(dataset_type, dataset_saves, "dataset_saves")

        # If the directory does not exist then we want to create it!
        if(not os.path.exists(self.save_location)):
            os.makedirs(self.save_location)

        # Set the directory to save the sequences to and make sure it exists
        self.sequence_directory = "{}/sequences/".format(self.save_location)
        if(not os.path.exists(self.sequence_directory)):
            os.makedirs(self.sequence_directory)

        # self._generate_dataset()
        # exit()

        # try to load the data and if not then generate the data
        if(self._load_dataset() == False):
        
            # Generate the dataset
            self._generate_dataset()

            # Save the dataset parameters if we can
            self._save_dataset_params()


    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):

        # Create the save filename
        current_sequence_dir = "{}/sequence_{:07d}".format(self.sequence_directory, idx)

        # Load the states
        loaded_data = torch.load("{}/states.pt".format(current_sequence_dir))
        loaded_data = torch.from_numpy(loaded_data)

        # Load the images
        images = np.zeros((self.subsequence_length, self.image_size, self.image_size, 3))
        for i in range(self.subsequence_length):
            image_save_filepath = "{}/img_{:04d}.png".format(current_sequence_dir, i)

            # Load the image and convert to RGB not 
            img = cv2.imread(image_save_filepath, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images[i,...] = img

        # Normalize the image to be between 0 and 1
        images = images.astype("float32") / 255.0

        # Convert to pytorch
        images = torch.from_numpy(images)

        # Permute to make sure the channels are how pytorch expects them 
        images = torch.permute(images, (0, 3, 1, 2))        
        
        # Generate the reference patch
        current_disc_color = self.all_disk_colors[idx]
        reference_patch = self._generate_reference_patch(self.all_possible_colors[current_disc_color])

        # Compute the return dictionary
        return_dict = {}
        return_dict["dataset_index"] = idx
        return_dict["states"] = loaded_data.float()
        return_dict["observations"] = images.float()
        return_dict["reference_patch"] = reference_patch.float()

        # # If we dont have actions then we dont have anything to return
        # if((self.actions is not None) and (len(self.actions) > 0)):
        #     return_dict["actions"] = self.actions[idx]

        return return_dict

    def get_subsequence_length(self):
        return self.subsequence_length
        
    def get_x_range(self):
        return (-10, 10)

    def get_y_range(self):
        return (-10, 10)

    def _generate_reference_patch(self, color=[255,0,0]):

        # Radius of the reference circle
        ref_disk_radius = int(15)

        # The size should be just large enough for the red disk
        ref_patch_size = (ref_disk_radius * 2) + 1

        # Create a blank RGB image
        ref_patch = np.zeros((ref_patch_size, ref_patch_size, 3), dtype=np.uint8)

        # Add in the tracked disk. This disk is red
        disk_position = (ref_patch_size//2, ref_patch_size//2)
        cv2.circle(ref_patch, (int(disk_position[0]),int(disk_position[1])), radius=ref_disk_radius, color=color, thickness=-1)

        # Normalize the image to be between 0 and 1
        ref_patch = ref_patch.astype("float32") / 255.0

        # Convert to pytorch
        ref_patch = torch.from_numpy(ref_patch)

        # Permute to make sure the channels are how pytorch expects them 
        ref_patch = torch.permute(ref_patch, (2, 0, 1))        

        return ref_patch



    def _generate_dataset(self):

        self.all_disk_colors = []

        # Generate all the sequences
        print("Generating {} Synthetic Disk sequences".format(self.dataset_size))        
        for seq_num in tqdm(range(self.dataset_size)):  

            # Generate each sequence
            current_tracked_disc_color = self._generate_sequence(seq_num)

            # Save the color for later
            self.all_disk_colors.append(current_tracked_disc_color)

        # Save the disk colors for later
        color_save_filepath = "{}/tracked_disc_colors.pt".format(self.save_location)
        with open(color_save_filepath, 'w') as f:
            for s in self.all_disk_colors:
                f.write(str(s) + '\n')


    def _generate_sequence(self, seq_num):

        # Generate the states for the tracked disk
        tracked_disk_states = self._create_disk_state_sequence()

        # Generate the ones for the distractor disks
        distractor_disks_states = []
        for d_idx in range(self.number_of_distractor_disks):
            distractor_disks_states.append(self._create_disk_state_sequence())
            
        # Pick a size for the tracked disc
        if(self.use_static_tracked_disc_size):
            current_tracked_disk_radius = self.tracked_disc_radius
        else:
            current_tracked_disk_radius = np.random.randint(low=self.tracked_disc_min_radius, high=self.tracked_disc_max_radius)

        # Pick a color for the tracked disk
        if(self.use_static_tracked_disc_color):
            current_tracked_disc_color = self.tracked_disc_color
        else:
            current_tracked_disc_color = random.choice(list(self.all_possible_colors.keys()))

        # Generate the sequence 
        image_sequence = self._generate_image_sequence(tracked_disk_states, distractor_disks_states, current_tracked_disk_radius, current_tracked_disc_color)

        # Convert to an array

        # Scale the states to be in the range -10 and 10 instead of (-self.all_states//2 and self.all_states//2)
        tracked_disk_states = tracked_disk_states / float(self.image_size//2)
        tracked_disk_states *= 10.0

        # Create the directory to save the image to
        current_sequence_dir = "{}/sequence_{:07d}".format(self.sequence_directory, seq_num)
        if(not os.path.exists(current_sequence_dir)):
            os.makedirs(current_sequence_dir)

        # Save each image
        for i in range(len(image_sequence)):
            image_save_filepath = "{}/img_{:04d}.png".format(current_sequence_dir, i)
            cv2.imwrite(image_save_filepath, cv2.cvtColor(image_sequence[i], cv2.COLOR_RGB2BGR))

        # Convert the state from (x, y, vx, vy) to (x, y, width, height) for bounding box problems
        tracked_disk_states[...,2:] = (((current_tracked_disk_radius*2)+1) / float(self.image_size//2)) * 10.0

        # Save the state data
        state_save_filepath = "{}/states.pt".format(current_sequence_dir)
        torch.save(tracked_disk_states, state_save_filepath)


        return current_tracked_disc_color


    def _create_starting_state(self):
        
        # The state consists of the red disc's position and velocity
        
        # Draw a random position in the image
        pos = np.random.uniform(float(-self.image_size//2), float(self.image_size//2), size=(2))
        
        # Draw a random velocity
        VELOCITY_SCALER = 3.0
        vel = np.random.normal(loc=0., scale=1., size=(2)) * VELOCITY_SCALER

        # Put it all together
        return np.array([pos[0], pos[1], vel[0], vel[1]])

    def _create_disk_state_sequence(self):

        disk_states = []
        disk_states.append(self._create_starting_state())

        # Start at 1 here since we already have a starting state
        for i in range(1, self.subsequence_length):

            # Extract the current state
            current_state = disk_states[i-1]

            # Create the new state
            new_state = self._process_model(current_state)
            disk_states.append(new_state)

        # Convert to a numpy array
        disk_states = np.asarray(disk_states)

        return disk_states

    def _process_model(self, state):
        """
        Calculates the next state of the target disc.

        Parameters:
            state: The state (position and velocity) of the target disc

        Returns
            new_state: The next state (position and velocity) of the target disc

        """
        new_state = np.copy(state)
        pull_force = -(self.disk_spring_force * state[:2])
        drag_force = -(self.disk_drag_force * state[2:]**2 * np.sign(state[2:]))

        new_state[0] += state[2]
        new_state[1] += state[3]
        new_state[2] += pull_force[0] + drag_force[0]
        new_state[3] += pull_force[1] + drag_force[1]

        if (self.use_correlated_process_noise == False):
            
            position_noise = np.random.normal(loc=0, scale=self.position_noise_scale, size=(2))

            if(self.use_heteroscedastic_process_noise_for_velocity):
                if(np.abs(state[0]) > (self.im_size//2 - self.im_size//6)) or (np.abs(state[1]) > (self.im_size//2 - self.im_size//6)):
                    velocity_noise = np.random.normal(loc=0, scale=0.1,size=(2))

                elif(np.abs(state[0]) > (self.im_size//2 - self.im_size//3) or np.abs(state[1]) > (self.im_size//2 - self.im_size//3)):
                    velocity_noise = np.random.normal(loc=0, scale=1., size=(2))
                
                else:
                    velocity_noise = np.random.normal(loc=0, scale=3., size=(2))
            else:
                velocity_noise = np.random.normal(loc=0, scale=2., size=(2))

            new_state[:2] += position_noise
            new_state[2:] += velocity_noise

        else:
            pn = self.position_noise_scale
            # pn = 3.0
            cn = 2
            c1 = -0.4
            c2 = 0.2
            c3 = 0.9
            c4 = -0.1
            c5 = 0

            covar = np.array([[pn**2, c1*pn*pn, c2*pn*cn, c3*pn*cn],
                              [c1*pn*pn, pn**2, c4*pn*cn, c5*pn*cn],
                              [c2*pn*cn, c4*pn*cn, cn**2, 0],
                              [c3*pn*cn, c5*pn*cn, 0, cn**2]])

            mean = np.zeros((4))
            noise = np.random.multivariate_normal(mean, covar)
            new_state += noise

        return new_state

    def _generate_image_sequence(self, tracking_disk_states, distractor_disks_states, current_tracked_disk_radius, current_tracked_disk_color):

        # colors for the distractor discs
        # DISTRACTOR_DISK_COLORS = [(0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255),(255, 255, 0), (255, 255, 255)]

        # Create distractor disk parameters
        distractor_params = []
        for i in range(self.number_of_distractor_disks):

            # All the parameters for this disk
            disk_params = dict()

            # Create a random radius for this disk
            disk_params["radius"] = np.random.choice(np.arange(self.distractor_disks_min_radius, self.distractor_disks_max_radius))

            # Select a random color for this disk that is not red!
            while(True):
                distractor_color = random.choice(list(self.all_possible_colors.keys()))
                if(distractor_color != current_tracked_disk_color):
                    break
            disk_params["color_rgb"] = self.all_possible_colors[distractor_color]

            distractor_params.append(disk_params)

        # Create the image sequence
        image_sequence = []
        for i in range(self.subsequence_length):
            image = self._generate_image(i, tracking_disk_states, distractor_disks_states, distractor_params, current_tracked_disk_radius, self.all_possible_colors[current_tracked_disk_color])
            image_sequence.append(image)

        return image_sequence

    def _generate_image(self, state_idx, tracking_disk_states, distractor_disks_states, distractor_params, current_tracked_disk_radius, tracking_disc_color_rgb):
        
        # Create a blank RGB image
        image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        # image[...] = 127.0

        # Add in the tracked disk. This disk is red
        disk_position = np.copy(tracking_disk_states[state_idx][0:2])
        disk_position += (self.image_size//2)
        cv2.circle(image, (int(disk_position[0]),int(disk_position[1])), radius=current_tracked_disk_radius, color=tracking_disc_color_rgb, thickness=-1)

        # draw the distractor disks
        for d_idx in range(self.number_of_distractor_disks):
        
            # Get the disk position
            disk_position = np.copy(distractor_disks_states[d_idx][state_idx][0:2])
            disk_position += (self.image_size//2)

            # Extract the parameters for this disk
            radius = distractor_params[d_idx]["radius"]
            color_rgb = distractor_params[d_idx]["color_rgb"]

            # add the distractor disk
            cv2.circle(image, (int(disk_position[0]),int(disk_position[1])), radius=radius, color=color_rgb, thickness=-1)

        return image

    def _generate_dataset_save_param_dict(self):

        save_param_dict = dict()
        save_param_dict["dataset_size"] = self.dataset_size
        save_param_dict["subsequence_length"] = self.subsequence_length
        save_param_dict["image_size"] = self.image_size
        save_param_dict["number_of_distractor_disks"] = self.number_of_distractor_disks
        save_param_dict["distractor_disks_min_radius"] = self.distractor_disks_min_radius
        save_param_dict["distractor_disks_max_radius"] = self.distractor_disks_max_radius
        save_param_dict["disk_spring_force"] = self.disk_spring_force
        save_param_dict["disk_drag_force"] = self.disk_drag_force
        save_param_dict["use_heteroscedastic_process_noise_for_velocity"] = self.use_heteroscedastic_process_noise_for_velocity
        save_param_dict["use_correlated_process_noise"] = self.use_correlated_process_noise
        save_param_dict["position_noise_scale"] = self.position_noise_scale

        save_param_dict["use_static_tracked_disc_size"] = self.use_static_tracked_disc_size
        if(self.use_static_tracked_disc_size):
            save_param_dict["tracked_disc_radius"] = self.tracked_disc_radius
        else:
            save_param_dict["tracked_disc_min_radius"] = self.tracked_disc_min_radius
            save_param_dict["tracked_disc_max_radius"] = self.tracked_disc_max_radius

        save_param_dict["use_static_tracked_disc_color"] = self.use_static_tracked_disc_color
        if(self.use_static_tracked_disc_color):
            save_param_dict["tracked_disc_color"] = self.tracked_disc_color

        return save_param_dict

    def _save_dataset_params(self):

        # Generate the save dict
        save_dict = self._generate_dataset_save_param_dict()

        # Load the save file parameters
        save_file = "{}/params.pt".format(self.save_location)

        # Save that dict
        torch.save(save_dict, save_file)

    def _load_dataset(self):

        # Load the save file parameters
        save_file = "{}/params.pt".format(self.save_location)

        # Check if the file exists, if not then we cant load 
        if(not os.path.exists(save_file)):
            return False

        # Load and unpack data
        loaded_data = torch.load(save_file)

        # Verify that our dataset matches the parameters
        expected_save_dict = self._generate_dataset_save_param_dict()
        for key in expected_save_dict.keys():

            # If a key is missing then the loaded data is stale and we need to regenerate it
            if(key not in loaded_data):
                return False

            # If we have a mismatch then we need to regenerate the data
            if(expected_save_dict[key] != loaded_data[key]):
                return False

        # All is good so load the colors file
        color_save_filepath = "{}/tracked_disc_colors.pt".format(self.save_location)
        with open(color_save_filepath, 'r') as f:
            self.all_disk_colors = [line.rstrip('\n') for line in f]

        return True

