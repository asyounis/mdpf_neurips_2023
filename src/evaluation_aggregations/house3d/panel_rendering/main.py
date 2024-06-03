# Add the packages we would like to import from to the path
import sys
sys.path.append('../../../../packages/kernel-density-estimator-bandwdidth-prediction/packages/')
sys.path.append('../../../')


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch
import numpy as np
import seaborn as sns
import os
import torch
import pandas as pd
import copy
import yaml
import cv2
import torch.distributions as D
import matplotlib.patheffects as patheffects


# The bandwidth stuff
from kernel_density_estimation.kernel_density_estimator import *


from models.model_creator import *
from models.particle_transformer import *
from datasets.house3d_dataset import *
from problems.house3d.house3d_problem import *


def create_and_load_model(experiment_to_use):

    # Load the config file:
    config_file = "experiments/house3d_experiments/{}/config.yaml".format(experiment_to_use)

    # Read and parse the config file
    with open(config_file) as file:
        
        # Load the whole file into a dictionary
        doc = yaml.load(file, Loader=yaml.FullLoader)

        # The arch params for all the models
        model_architecture_params = dict()

        # Load common model files if present:
        if("model_files" in doc):
            for f in doc["model_files"]:
                with open(f) as m_file:
                    models_doc = yaml.load(m_file, Loader=yaml.FullLoader)
                    model_architecture_params.update(models_doc["models"])               

        # Load the model parameters
        model_architecture_params.update(doc["models"])              


    # Create the model params
    model_params = dict()
    model_params["main_model"] = "dpf_full_training"
    model_params["sub_models"] = dict()
    model_params["sub_models"]["action_encoder_model"] =  "action_encoder"
    model_params["sub_models"]["proposal_model"] =  "proposal_model"
    model_params["sub_models"]["observation_model"] =  "observation_model"
    model_params["sub_models"]["weight_model"] =  "weight_model"
    model_params["sub_models"]["particle_encoder_for_particles_model"] =  "particle_encoder_particles"
    model_params["sub_models"]["particle_encoder_for_weights_model"] =  "particle_encoder_weights"
    model_params["sub_models"]["initializer_model"] =  "initializer_model"
    model_params["sub_models"]["resampling_weight_model"] =  "weight_model"
    model_params["sub_models"]["bandwidth_model"] =  "bandwidth_model_fixed"
    model_params["sub_models"]["resampling_bandwidth_model"] =  "resampling_bandwidth_model_fixed"


    # Create the model that we will be using for this experiment
    model = create_model(model_params, model_architecture_params)

    # Load the model saves
    pre_trained_models = dict()
    pre_trained_models["dpf_model"] = "experiments/house3d_experiments/{}/saves/full_dpf_fixed_bands/models/full_dpf_model_best.pt".format(experiment_to_use)
    model.load_pretrained(pre_trained_models, "cpu")

    return model



def load_dataset():

    # Load the dataset parameters
    with open("./experiments/house3d_experiments/configs/dataset_params_evaluation.yaml") as file:

        # Load the whole file into a dictionary
        doc = yaml.load(file, Loader=yaml.FullLoader)

        # Load the dataset
        dataset = House3DDataset(doc["dataset_params"], "evaluation")
        return dataset



def get_map_render_limits(world_map):
    ''' Taken from:
            https://stackoverflow.com/questions/49907382/how-to-remove-whitespace-from-an-image-in-opencv
    '''

    # Find all non-zero points (text)
    coords = cv2.findNonZero(world_map) 

    # Find minimum spanning bounding box
    x, y, w, h = cv2.boundingRect(coords) 

    # Add some padding
    padding = 0
    x = max(x - padding, 0)
    y = max(y - padding, 0)
    w = w + padding
    h = h + padding

    x_lim = [x, x+w]
    y_lim = [y, y+h]

    return x_lim, y_lim

def get_particle_limits(particles):
    particles_x = particles[..., 0]
    particles_y = particles[..., 0]

    x_min = torch.min(particles_x)
    x_max = torch.max(particles_x)
    y_min = torch.min(particles_y)
    y_max = torch.max(particles_y)

    return [x_min, x_max], [y_min, y_max]


def create_initial_dpf_state_unimodal(model, true_state, observations, number_of_particles,sequence_number):

    # Need the batch size
    batch_size = true_state.shape[0]

    particles =  true_state.unsqueeze(1)
    particles = torch.tile(particles, [1, number_of_particles, 1])
    particles = model.particle_transformer.downscale(particles)

    # Move to the correct device
    particles = particles.to(observations.device)

    # Add noise to the initial particles
    assert(len(model.initial_position_std) == len(model.kde_params["dims"]))
    for d in range(len(model.initial_position_std)):

        # Get the stats and the dist type
        std = model.initial_position_std[d]
        dim_params = model.kde_params["dims"][d]

        # create the correct dist for this dim
        distribution_type = dim_params["distribution_type"]
        if(distribution_type == "Normal"):
            dist = D.Normal(loc=torch.zeros_like(particles[..., d]),  scale=std)
        elif(distribution_type == "Von_Mises"):
            kappa = 1.0 / std
            dist = VonMisesFullDist(loc=torch.zeros_like(particles[..., d]), concentration=kappa)

        # Generate and add some noise
        noise = dist.sample()
        particles[..., d] = particles[..., d] + noise


    # Equally weight all the particles since they are all the same
    particle_weights = torch.ones(size=(batch_size, number_of_particles), device=observations.device) / float(number_of_particles)

    # Tight bands around each particle
    bandwidths = torch.zeros(size=(particles.shape[0], particles.shape[-1])).to(observations.device)
    bandwidths[..., 0] = 1.0
    bandwidths[..., 1] = 1.0
    bandwidths[..., 2] = 0.001

    # Pack everything into an output dict
    output_dict = dict()
    output_dict["particles_downscaled"] = particles
    output_dict["particles"] = model.particle_transformer.upscale(particles)
    output_dict["particle_weights"] = particle_weights
    output_dict["bandwidths_downscaled"] = bandwidths
    output_dict["bandwidths"] = model.particle_transformer.upscale(bandwidths)
    output_dict["importance_gradient_injection"] = torch.ones(size=(batch_size, 1), device=particles.device)

    if(model.decouple_weights_for_resampling):
        output_dict["resampling_particle_weights"] = particle_weights

    if( model.outputs_kde() and model.decouple_bandwidths_for_resampling):
        resampling_bandwidths = model.resampling_weighted_bandwidth_predictor(particles, particle_weights)
        output_dict["resampling_bandwidths_downscaled"] = resampling_bandwidths
        output_dict["resampling_bandwidths"] = model.particle_transformer.upscale(resampling_bandwidths)
    else:
        output_dict["resampling_bandwidths"] = None


    output_dict["do_resample"] = False

    return output_dict




def create_initial_dpf_state_multimodal(model, true_state, observations, number_of_particles, sequence_number, world_map):

    # Need the batch size
    batch_size = true_state.shape[0]

    if(sequence_number == 0):
        mode1 = dict()
        mode1[0] = {"std": 10.0, "pos": true_state[0, 0]}
        mode1[1] = {"std": 10.0, "pos": true_state[0, 1]}
        mode1[2] = {"std": 1.0, "pos": true_state[0, 2]}
        mode2 = dict()
        mode2[0] = {"std": 10.0, "pos": torch.FloatTensor([550])}
        mode2[1] = {"std": 10.0, "pos": torch.FloatTensor([350])}
        mode2[2] = {"std": 1.0, "pos": true_state[0, 2]}

        modes = [mode1, mode2]

    elif(sequence_number == 15):
        mode1 = dict()
        mode1[0] = {"std": 10.0, "pos": true_state[0, 0]}
        mode1[1] = {"std": 10.0, "pos": true_state[0, 1]}
        mode1[2] = {"std": 0.03, "pos": true_state[0, 2]}
        
        mode2 = dict()
        mode2[0] = {"std": 10.0, "pos": torch.FloatTensor([800])}
        mode2[1] = {"std": 10.0, "pos": torch.FloatTensor([800])}
        mode2[2] = {"std": 0.03, "pos": true_state[0, 2]}

        mode3 = dict()
        mode3[0] = {"std": 10.0, "pos": torch.FloatTensor([800])}
        mode3[1] = {"std": 10.0, "pos": torch.FloatTensor([850])}
        mode3[2] = {"std": 0.03, "pos": true_state[0, 2]}

        modes = [mode1, mode2, mode3]

    elif(sequence_number == 505):

        x_lim, y_lim = get_map_render_limits(world_map.squeeze(0).numpy())
        x_range = x_lim[1] - x_lim[0]
        y_range = y_lim[1] - y_lim[0]
        min_range = min(x_range, y_range)
        x_center = (x_lim[1] + x_lim[0]) / 2.0
        y_center = (y_lim[1] + y_lim[0]) / 2.0

        x_std = x_range / 3.0
        y_std = y_range / 3.0

        mode1 = dict()
        mode1[0] = {"std": x_std, "pos": torch.FloatTensor([x_center])}
        mode1[1] = {"std": y_std, "pos": torch.FloatTensor([y_center])}
        mode1[2] = {"std": 0.03, "pos": true_state[0, 2]}
        
        modes = [mode1]


    

    particles_per_mode = [number_of_particles//len(modes) for _ in range(len(modes))]
    particles_per_mode[-1] += number_of_particles - sum(particles_per_mode) 

    particles = []
    for mode_idx, mode in enumerate(modes):

        particle = torch.zeros([particles_per_mode[mode_idx], len(mode.keys())])

        for dim_idx in mode.keys():
            dim_params = model.kde_params["dims"][dim_idx]

            std = mode[dim_idx]["std"]
            pos = mode[dim_idx]["pos"]

            # create the correct dist for this dim
            distribution_type = dim_params["distribution_type"]
            if(distribution_type == "Normal"):
                dist = D.Normal(loc=pos,  scale=std)
            elif(distribution_type == "Von_Mises"):
                kappa = 1.0 / std
                dist = VonMisesFullDist(loc=pos, concentration=kappa)

            # Sample from that mode per dim!
            for i in range(particle.shape[0]):
                sample = dist.sample()
                particle[i, dim_idx] = sample

        particles.append(particle)

    particles = torch.vstack(particles)
    particles = particles.unsqueeze(0)
    particles = particles.to(observations.device)

    # Equally weight all the particles
    particle_weights = torch.ones(size=(batch_size, number_of_particles), device=observations.device) / float(number_of_particles)

    # Tight bands around each particle
    bandwidths = torch.zeros(size=(particles.shape[0], particles.shape[-1])).to(observations.device)
    bandwidths[...] = 2.0

    # Pack everything into an output dict
    output_dict = dict()
    output_dict["particles_downscaled"] = particles
    output_dict["particles"] = model.particle_transformer.upscale(particles)
    output_dict["particle_weights"] = particle_weights
    output_dict["bandwidths_downscaled"] = bandwidths
    output_dict["bandwidths"] = model.particle_transformer.upscale(bandwidths)
    output_dict["importance_gradient_injection"] = torch.ones(size=(batch_size, 1), device=particles.device)

    if(model.decouple_weights_for_resampling):
        output_dict["resampling_particle_weights"] = particle_weights

    if( model.outputs_kde() and model.decouple_bandwidths_for_resampling):
        resampling_bandwidths = model.resampling_weighted_bandwidth_predictor(particles, particle_weights)
        output_dict["resampling_bandwidths_downscaled"] = resampling_bandwidths
        output_dict["resampling_bandwidths"] = model.particle_transformer.upscale(resampling_bandwidths)
    else:
        output_dict["resampling_bandwidths"] = None


    output_dict["do_resample"] = False

    return output_dict


def run_model(model, data, sequence_number ,number_of_particles=50, init_method="unimode"):

    # We dont want the gradient when we run the model
    with torch.no_grad():

        observation_transformer = House3DObservationTransformer()

        # Unpack the data
        observations = data["observations"]
        actions = data["actions"]
        world_map = data["world_map"]
        states = data["states"]

        # Add a batch dim
        observations = observations.unsqueeze(0)
        actions = actions.unsqueeze(0)
        world_map = world_map.unsqueeze(0)
        states = states.unsqueeze(0)

        # transform the observation
        transformed_observation = observation_transformer.forward_tranform(observations)

        # Get some states
        subsequence_length = states.shape[1]

        # Create the initial state for the particle filter
        if(init_method == "unimode"):
            output_dict = create_initial_dpf_state_unimodal(model, states[:,0,:],transformed_observation, number_of_particles, sequence_number)

        elif(init_method == "multimode"):
            output_dict = create_initial_dpf_state_multimodal(model, states[:,0,:],transformed_observation, number_of_particles, sequence_number, world_map)

        else:
            assert(False)


        all_output_dicts = []
        all_output_dicts.append(output_dict)

        # Run through the sub-sequence
        for seq_idx in tqdm(range(subsequence_length-1), leave=False, desc="Running Model"):

            # Get the observation for this step
            observation = transformed_observation[:,seq_idx+1,:]

            # Get the observation for the next step (if there is a next step)
            if((seq_idx+2) >= subsequence_length):
                next_observation = None
            else:
                next_observation = transformed_observation[:,seq_idx+2,:]

            # Create the next input dict from the last output dict
            input_dict = dict()
            for key in output_dict.keys():

                if(isinstance(output_dict[key], torch.Tensor)):
                    input_dict[key] = output_dict[key].clone()
                else:
                    input_dict[key] = output_dict[key]

            # Pack!
            input_dict["observation"] = observation
            input_dict["next_observation"] = next_observation
            input_dict["world_map"] = world_map
            input_dict["actions"] = actions[:,seq_idx, :]
            input_dict["reference_patch"] = None
            input_dict["timestep_number"] = None

            # Run the model on this step
            output_dict = model(input_dict)
            all_output_dicts.append(output_dict)

        return all_output_dicts

def get_limits_and_ranges(data, output_dict, frame_number, do_zoom=True, zoom_factor=1.0):

    # Need a transformer for this
    particle_transformer = House3DTransformer()

    # Get the world map
    world_map = data["world_map"]

    # Convert the world map to numpy so we can fully process it and render it
    world_map = world_map.numpy()

    # Compute the render-able area
    if(do_zoom):

        # Get some data we need
        true_state = data["states"][frame_number]
        
        # Extract some data from the output
        particles = output_dict["particles"].squeeze(0)
        particle_weights = output_dict["particle_weights"].squeeze(0)

        # Get the render limits for the world map
        x_lim, y_lim = get_map_render_limits(world_map)

        # # Update limits with particle limits
        # x_lim_part, y_lim_part = get_particle_limits(particles)
        # # x_lim[0] = min(x_lim[0], x_lim_part[0])
        # # x_lim[1] = max(x_lim[1], x_lim_part[1])
        # # y_lim[0] = min(y_lim[0], y_lim_part[0])
        # # y_lim[1] = max(y_lim[1], y_lim_part[1])


        # Get the range
        x_range = x_lim[1] - x_lim[0]
        y_range = y_lim[1] - y_lim[0]
        
        # How much to zoom
        # ZOOM_FACTOR = 5.0
        # ZOOM_FACTOR = 2.5
        ZOOM_FACTOR = zoom_factor

        # Compute the new widths
        new_x_range = x_range / ZOOM_FACTOR
        new_y_range = y_range / ZOOM_FACTOR

        # Get the current position of the state so we can center it
        current_true_state = true_state
        x_state = current_true_state[0].item()
        y_state = current_true_state[1].item()

        # Compute the mean particle
        particles = particle_transformer.backward_tranform(particles)
        predicted_state = torch.sum(particles * particle_weights.unsqueeze(-1), dim=0)
        # predicted_state = torch.mean(particles, dim=0)
        predicted_state = particle_transformer.forward_tranform(predicted_state).cpu()

        mean_state = (current_true_state + predicted_state) / 2.0
        # mean_state = current_true_state
        x_state = mean_state[0].item()
        y_state = mean_state[1].item()

        # Compute the new mins and maxs
        x_min = x_state - (new_x_range/2)
        x_max = x_state + (new_x_range/2)
        y_min = y_state - (new_y_range/2)
        y_max = y_state + (new_y_range/2)

        x_range = x_max - x_min
        y_range = y_max - y_min 

        if(x_range < y_range):
            diff = y_range - x_range
            x_min -= (diff/2.0)
            x_max += (diff/2.0)

        if(y_range < x_range):
            diff = x_range - y_range
            y_min -= (diff/2.0)
            y_max += (diff/2.0)

        # Make sure they are in range
        if(x_min < x_lim[0]):
            d = x_lim[0] - x_min
            x_max = x_max + d
            x_min = x_min + d

        if(y_min < y_lim[0]):
            d = y_lim[0] - y_min
            y_max = y_max + d
            y_min = y_min + d

        if(x_max >= x_lim[1]):
            diff = x_max - x_lim[1]
            x_max -= diff
            x_min -= diff

        if(y_max >= y_lim[1]):
            diff = y_max - y_lim[1]
            y_max -= diff
            y_min -= diff

        x_lim = [int(x_min), int(x_max)]
        y_lim = [int(y_min), int(y_max)]

    else:

        # Get the render limits for the world map
        x_lim, y_lim = get_map_render_limits(world_map)

        # Get the range
        # x_range = x_lim[1] - x_lim[0]
        # y_range = y_lim[1] - y_lim[0]

    # Get the range
    x_range = x_lim[1] - x_lim[0]
    y_range = y_lim[1] - y_lim[0]

    return x_lim, y_lim, x_range, y_range


def render_sequence_data(ax, data, output_dict, frame_number, zoom_factor, draw_particles):
    
    # Get he limits and ranges
    x_lim, y_lim, x_range, y_range = get_limits_and_ranges(data, output_dict, frame_number, zoom_factor=zoom_factor, do_zoom=True)

    # Set the render base size
    max_range = max(x_range, y_range)
    # rendered_object_base_size = max_range / 20.0
    rendered_object_base_size = max_range / 30.0


    # Extract some data from the output
    particles = output_dict["particles"].squeeze(0)
    particle_weights = output_dict["particle_weights"].squeeze(0)
    bandwidths = output_dict["bandwidths"].squeeze(0)

    # Unpack
    world_map = data["world_map"]
    states = data["states"][frame_number]
    all_states = data["states"]

    # Convert the world map to numpy so we can fully process it and render it
    world_map = world_map.numpy()

    # Convert 
    world_map_rgb = np.zeros((world_map.shape[0], world_map.shape[1], 3))
    world_map_rgb[world_map>0, :] = 255
    # ax.imshow(world_map_rgb.astype("int"), origin="lower")

    # Create the sampling mesh grid
    x = torch.linspace(x_lim[0], x_lim[1], x_range)
    y = torch.linspace(y_lim[0], y_lim[1], y_range)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    points = torch.cat([torch.reshape(grid_x, (-1,)).unsqueeze(-1), torch.reshape(grid_y, (-1,)).unsqueeze(-1)], dim=-1)

    # Create the KDE Params
    kde_params = dict()
    kde_params["dims"] = dict()
    kde_params["dims"][0] = {"distribution_type": "Normal"}
    kde_params["dims"][1] = {"distribution_type": "Normal"}

    # Create the KDE and get the probs over the space
    kde = KernelDensityEstimator(kde_params, particles[:,0:2].unsqueeze(0), particle_weights.unsqueeze(0), bandwidths[0:2].unsqueeze(0))
    log_probs = kde.log_prob(points.unsqueeze(0).to(particles.device)).squeeze(0)
    probs = torch.exp(log_probs).cpu()

    # Scale
    probs_min = torch.min(probs)
    probs_max = torch.max(probs)
    probs -= probs_min
    probs /= (probs_max - probs_min)

    for i in range(points.shape[0]):
        p = points[i].numpy()
        c = probs[i].item()
        # world_map_rgb[int(p[1]), int(p[0]), 0:2] = 255.0 - (c*255.0)
        world_map_rgb[int(p[1]), int(p[0]), 0:2] = int(255.0 - (c*255.0))

    world_map_rgb[world_map==0, :] = 0
    ax.imshow(world_map_rgb.astype("int"), origin="lower")


    if(draw_particles):
        # Draw the particles
        x = particles[:, 0].cpu().numpy()
        y = particles[:, 1].cpu().numpy()
        particles_scatter = ax.scatter(x,y, color="black", s=0.15*rendered_object_base_size)


    # Plot the true location
    true_x = states[0].item()
    true_y = states[1].item()
    true_state_circle = plt.Circle((true_x, true_y), 0.25 * rendered_object_base_size, color=sns.color_palette("bright")[3])
    ax.add_patch(true_state_circle)
    dy = torch.sin(states[2]).item() * 1.0 * rendered_object_base_size
    dx = torch.cos(states[2]).item() * 1.0 * rendered_object_base_size
    true_state_arrow = ax.arrow(true_x, true_y, dx, dy, color=sns.color_palette("bright")[3], width=0.25*rendered_object_base_size)

    # Draw the true path
    x1 = all_states[: ,0].numpy()
    y1 = all_states[: ,1].numpy()
    ax.plot(x1, y1, color="red", label="True State", alpha=1.0, linewidth=0.75)

    # Set the render limits for this plot
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    # Set the aspect ratio to a square 
    # ax.set_aspect("equal")
    ax.set_box_aspect(1)

    # Turn off tick labels
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    # turn off the axis ticks little bars
    ax.tick_params(left=False, bottom=False)

    # Add a box around the outside
    ax.patch.set_edgecolor('black')  
    ax.patch.set_linewidth(1)  

    # Add a text label indicating what frame number we are at
    text_str = "Time-step #{:02d}".format(frame_number+1)
    ax.text(0.01, 0.95, text_str, horizontalalignment='left', verticalalignment='center',transform=ax.transAxes, weight="bold", fontsize=12, path_effects=[patheffects.withStroke(linewidth=4, foreground='white')])


    # Draw the observation
    inset_ax = ax.inset_axes([0.01,0.01,0.3,0.3])
    observation = torch.permute(data["observations"][frame_number].cpu(), (1, 2, 0))
    observation[observation<0] = 0
    observation[observation>1.0] = 1.0
    inset_ax.imshow(observation.numpy())
    inset_ax.set_yticklabels([])
    inset_ax.set_xticklabels([])
    inset_ax.tick_params(left=False, bottom=False)
    inset_ax.patch.set_edgecolor('black')  
    inset_ax.patch.set_linewidth(3)  



def main():

    running_params_1 = dict()
    running_params_1["init_method"] = "multimode"
    running_params_1["sequence_to_render"] = 15
    running_params_1["zoom_factor"] = 2.5
    running_params_1["save_name"] = "multi_init"
    running_params_1["number_of_particles"] = 150
    running_params_1["draw_particles"] = True
    running_params_1["rows"] = 1
    running_params_1["frame_render_mod"] = 3

    running_params_2 = dict()
    running_params_2["init_method"] = "unimode"
    # running_params_2["sequence_to_render"] = 2
    # running_params_2["sequence_to_render"] = 15
    running_params_2["sequence_to_render"] = 502
    running_params_2["zoom_factor"] = 3.0
    running_params_2["save_name"] = "uni_init"
    running_params_2["number_of_particles"] = 50
    running_params_2["draw_particles"] = False
    running_params_2["rows"] = 1
    running_params_2["frame_render_mod"] = 3


    running_params_3 = dict()
    running_params_3["init_method"] = "multimode"
    running_params_3["sequence_to_render"] = 505
    running_params_3["zoom_factor"] = 2.0
    running_params_3["save_name"] = "random_init"
    running_params_3["number_of_particles"] = 1000
    running_params_3["draw_particles"] = True
    running_params_3["rows"] = 1
    running_params_3["frame_render_mod"] = 1


    # all_running_params = [running_params_1]
    # all_running_params = [running_params_2]
    # all_running_params = [running_params_3]
    all_running_params = [running_params_2, running_params_1, running_params_3]

    for running_params in all_running_params:

        # Change directories to make everything very easy 
        absolute_path = os.path.dirname(os.path.realpath(__file__))
        os.chdir("../../..")

        # Constants
        # sequence_to_render = 0
        # sequence_to_render = 15
        sequence_to_render = running_params["sequence_to_render"]
        init_method = running_params["init_method"]
        zoom_factor = running_params["zoom_factor"]
        number_of_particles = running_params["number_of_particles"]
        draw_particles = running_params["draw_particles"]

        # The name of the experiment that we are going to use
        experiment_to_use = "experiment0003_importance_init"

        # Get the model and load it
        model = create_and_load_model(experiment_to_use)

        # Load the dataset
        dataset = load_dataset()

        # Grab a specific sequence
        sequence_data = dataset[sequence_to_render]

        # Plot!!!!!
        rows = running_params["rows"]
        cols = 10
        fig, axes = plt.subplots(rows, cols, sharex=False, figsize=(3*cols, 3*rows), squeeze=False)

        number_of_frames_to_render = rows*cols

        frames_to_render = [i*running_params["frame_render_mod"] for i in range(number_of_frames_to_render)]
        frames_to_render.append(1)
        frames_to_render.append(2)
        frames_to_render.append(3)
        frames_to_render = list(set(frames_to_render))
        frames_to_render = sorted(frames_to_render)


        # run the model
        all_output_dicts = run_model(model, sequence_data, sequence_to_render, number_of_particles=number_of_particles, init_method=init_method)

        for i in range(number_of_frames_to_render):

            # Get the axis to draw on
            r = int(i / cols)
            c = i % cols
            ax = axes[r, c]
            frame_id = frames_to_render[i]

            # Render the sequence data
            render_sequence_data(ax, sequence_data, all_output_dicts[frame_id], frame_id, zoom_factor, draw_particles)


        fig.tight_layout()


        os.chdir(absolute_path)
        fig.savefig("{}.png".format(running_params["save_name"]))
        fig.savefig("{}.pdf".format(running_params["save_name"]))
        print("Saving {}".format(running_params["save_name"]))

    # plt.show()


if __name__ == '__main__':
    main()