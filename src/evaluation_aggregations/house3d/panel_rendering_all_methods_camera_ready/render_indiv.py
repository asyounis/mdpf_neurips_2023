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

def ensure_directory_exists(directory):
    '''
        Makes sure a directory exists.  If it does not exist then the directory is created

        Parameters:
            directory: The directory that needs to exist

        Returns:
            None
    '''
    if(not os.path.exists(directory)):
        os.makedirs(directory)




def render_kde_density(world_map_rgb, particles, particle_weights, bandwidths, x_lim, y_lim, x_range, y_range):

    # Move to the GPU for fast compute
    particles = particles.cuda()
    particle_weights = particle_weights.cuda()
    bandwidths = bandwidths.cuda()


    lim_min = min(x_lim[0], y_lim[0])
    lim_max = max(x_lim[1], y_lim[1])

    # lim_min = 0
    # lim_max = 3000
    lim_range = lim_max - lim_min


    # Create the sampling mesh grid
    x = torch.linspace(lim_min, lim_max, lim_range)
    y = torch.linspace(lim_min, lim_max, lim_range)
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

    points = points.cpu().numpy()
    for i in tqdm(range(points.shape[0]), leave=False):
        p = points[i]
        c = probs[i].item()
        world_map_rgb[int(p[1]), int(p[0]), 0:2] = int(255.0 - (c*255.0))


def forward_tranform(particles):

    assert(particles.shape[-1] == 4)

    new_p = []
    new_p.append(particles[...,0])
    new_p.append(particles[...,1])
    new_p.append(torch.atan2(particles[...,2], particles[...,3]))
    new_p = torch.stack(new_p, dim=-1)

    return new_p

def backward_tranform(particles):
    
    assert(particles.shape[-1] == 3)

    new_p = []
    new_p.append(particles[...,0])
    new_p.append(particles[...,1])
    new_p.append(torch.sin(particles[...,2]))
    new_p.append(torch.cos(particles[...,2]))
    new_p = torch.stack(new_p, dim=-1)

    return new_p



def get_world_map_limits(world_map):

    if(torch.is_tensor(world_map)):
        world_map = world_map.squeeze(0).numpy()
    mask1 = world_map>0.0001
    mask2 = world_map<0.0001
    world_map[mask1] = 1.0
    world_map[mask2] = 0.0

    H, W = world_map.shape


    ####################################################################
    ## Bottom Up
    ####################################################################
    for y in range(H):

        row = world_map[y, :]
        if(np.sum(row) != 0):
            break

    y_min = y

    ####################################################################
    ## Top Down
    ####################################################################
    for y in range(H-1, 1, -1):

        row = world_map[y, :]
        if(np.sum(row) != 0):
            break

    y_max = y

    ####################################################################
    ## Left to Right
    ####################################################################
    for x in range(W):

        col = world_map[:, x]
        if(np.sum(col) != 0):
            break

    x_min = x
    
    ####################################################################
    ## Right to Left
    ####################################################################
    for x in range(W-1, 1, -1):

        col = world_map[:, x]
        if(np.sum(col) != 0):
            break

    x_max = x

    return x_min, x_max, y_min, y_max




def render_sequence_data(ax, data, output_dict, frame_number, render_config):

    # Unpack
    world_map = data["world_map"]
    states = data["states"][frame_number]
    all_states = data["states"]

    # Convert the world map to numpy so we can fully process it and render it
    world_map = world_map.numpy()

    # Unpack the rendering configs
    x_min = render_config["x_min"]
    x_max = render_config["x_max"]
    y_min = render_config["y_min"]
    y_max = render_config["y_max"]    
    draw_particles = render_config["draw_particles"]
    render_kde = render_config["render_kde"]
    tail_render_size = render_config["tail_render_size"]
    obs_inset_location = render_config["obs_inset_location"]

    # x_min, x_max, y_min, y_max = get_world_map_limits(world_map)

    # create the x and y limits
    x_lim = [x_min, x_max]
    y_lim = [y_min, y_max]

    # Compute the ranges
    x_range = x_max - x_min
    y_range = y_max - y_min


    # Set the render base size
    max_range = max(x_range, y_range)
    rendered_object_base_size = max_range / 30.0

    # Extract some data from the output
    particles = output_dict["particles"][0, frame_number]
    particle_weights = output_dict["particle_weights"][0, frame_number]
    bandwidths = output_dict["bandwidths"][0,frame_number]

    all_particles = output_dict["particles"][0]
    all_particle_weights = output_dict["particle_weights"][0]


    # Convert 
    world_map_rgb = np.zeros((world_map.shape[0], world_map.shape[1], 3))
    world_map_rgb[world_map>0, :] = 255

    # Draw the KDE
    if(render_kde):
        render_kde_density(world_map_rgb, particles, particle_weights, bandwidths, x_lim, y_lim, x_range, y_range)

    # Draw the world map
    world_map_rgb[world_map==0, :] = 0
    ax.imshow(world_map_rgb.astype("int"), origin="lower")

    # If we should draw the particles
    if(draw_particles):
        x = particles[:, 0].cpu().numpy()
        y = particles[:, 1].cpu().numpy()
        particles_scatter = ax.scatter(x,y, color="black", s=0.15*rendered_object_base_size)


    # Plot the true location
    true_x = states[0].item()
    true_y = states[1].item()
    true_state_circle = plt.Circle((true_x, true_y), 0.33 * rendered_object_base_size, color=sns.color_palette("bright")[3])
    ax.add_patch(true_state_circle)
    dy = torch.sin(states[2]).item() * 1.5 * rendered_object_base_size
    dx = torch.cos(states[2]).item() * 1.5 * rendered_object_base_size
    true_state_arrow = ax.arrow(true_x, true_y, dx, dy, color=sns.color_palette("bright")[3], width=0.33*rendered_object_base_size)

    # Draw the true path
    x1 = all_states[:frame_number+1 ,0].numpy()
    y1 = all_states[:frame_number+1 ,1].numpy()
    ax.plot(x1, y1, color="red", label="True State", alpha=1.0, linewidth=tail_render_size*rendered_object_base_size, linestyle="dashed")

    # Compute the mean particle
    all_particles = backward_tranform(all_particles)
    mean_state = torch.sum(all_particles * all_particle_weights.unsqueeze(-1), dim=1)
    mean_state = forward_tranform(mean_state)

    # Plot the true location
    x = mean_state[frame_number,0].item()
    y = mean_state[frame_number,1].item()
    mean_particle_circle = plt.Circle((x, y), 0.33 * rendered_object_base_size, color=sns.color_palette("bright")[8])
    ax.add_patch(mean_particle_circle)
    dy = torch.sin(mean_state[frame_number, 2]).item() * 1.5 * rendered_object_base_size
    dx = torch.cos(mean_state[frame_number, 2]).item() * 1.5 * rendered_object_base_size
    mean_particle_arrow = ax.arrow(x, y, dx, dy, color=sns.color_palette("bright")[8], width=0.33*rendered_object_base_size)

    # Draw the true path
    x1 = mean_state[:frame_number+1 ,0].numpy()
    y1 = mean_state[:frame_number+1 ,1].numpy()
    ax.plot(x1, y1, color=sns.color_palette("bright")[8], label="True State", alpha=1.0, linewidth=tail_render_size*rendered_object_base_size, linestyle="dashed")


    # Add a text label indicating what frame number we are at
    text_str = "t={:02d}".format(frame_number)
    ax.text(0.02, 0.93, text_str, horizontalalignment='left', verticalalignment='center',transform=ax.transAxes, weight="bold", fontsize=16, path_effects=[patheffects.withStroke(linewidth=4, foreground='white')])

    # Set the render limits for this plot
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    # Set the aspect ratio to a square 
    ax.set_box_aspect(1)

    # Turn off tick labels
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    # turn off the axis ticks little bars
    ax.tick_params(left=False, bottom=False)

    # Add a box around the outside
    ax.patch.set_edgecolor('black')  
    ax.patch.set_linewidth(1)  


    # Draw the observation
    inset_ax = ax.inset_axes(obs_inset_location)
    observation = torch.permute(data["observations"][frame_number].cpu(), (1, 2, 0))
    observation[observation<0] = 0
    observation[observation>1.0] = 1.0
    inset_ax.imshow(observation.numpy())
    inset_ax.set_yticklabels([])
    inset_ax.set_xticklabels([])
    inset_ax.tick_params(left=False, bottom=False)
    inset_ax.patch.set_edgecolor('black')  
    inset_ax.patch.set_linewidth(3)  

    return true_state_arrow, mean_particle_arrow



def get_name_mapping():

    # The names we want to give to the experiments
    experiment_names = dict()
    experiment_names["lstm_rnn"] = "LSTM"
    # experiment_names["diffy_particle_filter"] = "TG-PF"
    # experiment_names["optimal_transport_pf"] = "OT-PF"
    # experiment_names["soft_resampling_particle_filter"] = "SR-PF"
    # experiment_names["importance_sampling_pf"] = "DIS-PF"

    experiment_names["experiment0001"] = "TG-MDPF"
    experiment_names["experiment0002_importance"] = "MDPF"
    # experiment_names["experiment0003_importance"] = "A-MDPF"
    # experiment_names["experiment0003_importance_init"] = "A-MDPF-Init"
    experiment_names["experiment0003_importance_init"] = "A-MDPF"


    experiment_names["experiment0002_implicit"] = "IRG-MDPF"
    # experiment_names["experiment0003_implicit"] = "A-MDPF-Implicit"

    experiment_names["experiment0002_concrete"] = "MDPF-Concrete"
    experiment_names["experiment0003_concrete"] = "A-MDPF-Concrete"



    experiment_names["diffy_particle_filter_learned_band"] = "TG-PF"
    experiment_names["optimal_transport_pf_learned_band"] = "OT-PF"
    experiment_names["soft_resampling_particle_filter_learned_band"] = "SR-PF"
    experiment_names["importance_sampling_pf_learned_band"] = "DIS-PF"
    experiment_names["discrete_concrete"] = "C-PF"

    return experiment_names





def get_save_mapping():

    # The names we want to give to the experiments
    experiment_names = dict()
    experiment_names["lstm_rnn"] = "lstm"
    experiment_names["experiment0001"] = "tgmdpf"
    experiment_names["experiment0002_importance"] = "mdpf"
    experiment_names["experiment0003_importance_init"] = "amdpf"


    experiment_names["diffy_particle_filter_learned_band"] = "dpf"
    experiment_names["optimal_transport_pf_learned_band"] = "ot"
    experiment_names["soft_resampling_particle_filter_learned_band"] = "sr"
    experiment_names["importance_sampling_pf_learned_band"] = "dis"
    experiment_names["discrete_concrete"] = "C-PF"

    return experiment_names


def get_experiments_to_run():

    experiments = []
    experiments.append("experiment0002_importance")
    experiments.append("experiment0003_importance_init")
    experiments.append("experiment0001")
    experiments.append("diffy_particle_filter_learned_band")
    experiments.append("optimal_transport_pf_learned_band")
    experiments.append("soft_resampling_particle_filter_learned_band")
    experiments.append("importance_sampling_pf_learned_band")
    experiments.append("lstm_rnn")

    return experiments





def pack_and_stack(data):

    keys = []
    keys.append("particles")
    keys.append("particle_weights")
    keys.append("bandwidths_downscaled")
    keys.append("bandwidths")
    # keys.append("resampling_bandwidths")

    new_dict = dict()
    for key in keys:
        if key in data[0]:
            new_dict[key] = []

    for i in range(len(data)):
        for key in new_dict.keys():
            new_dict[key].append(data[i][key])

    for key in new_dict.keys():
        new_dict[key] = torch.stack(new_dict[key], dim=1)

    return new_dict



def ensure_directory_exists(directory):
    '''
        Makes sure a directory exists.  If it does not exist then the directory is created

        Parameters:
            directory: The directory that needs to exist

        Returns:
            None
    '''
    if(not os.path.exists(directory)):
        os.makedirs(directory)


def main():

    # Get the experiments that we should run
    experiments = get_experiments_to_run()

    # Mapping the experiment names to the correct names for the paper
    name_mappings = get_name_mapping()

    # Mapping to save to
    save_mapping = get_save_mapping()

    # Change directories to make everything very easy 
    absolute_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir("../../..")

    # The loading dir
    load_dir = "{}/raw_sequences_data/".format(absolute_path)

    # Which sequence we want to render
    # sequence_to_render = 502
    # sequence_to_render = list(range(820))[20:]
    sequence_to_render = [32]

    # Sequence idxs to render
    sequence_idx_to_render = [i for i in range(1, 41)]
    # sequence_idx_to_render = [i*2 for i in range(1, 21)]

    # For # 32
    render_config = dict()
    render_config["x_min"] = 100
    render_config["x_max"] = 450
    render_config["y_min"] = 2
    render_config["y_max"] = 300
    render_config["draw_particles"] = False
    render_config["render_kde"] = True
    render_config["tail_render_size"] = 0.15
    render_config["obs_inset_location"] = [0.54,0.54,0.45,0.45]

    for seq_idx in tqdm(sequence_to_render):

        # Do each experiments data
        for exp_idx, experiment in enumerate(tqdm(experiments)):

            # Load the data
            saved_dict = torch.load("{}/{}/{:04d}.pt".format(load_dir, experiment, seq_idx))

            # Unpack
            data = saved_dict["data"]

            # The outputs
            output = saved_dict["output"]
            output = pack_and_stack(output)

            # Make the figure
            rows = 4
            cols = 10
            fig, axes = plt.subplots(rows, cols, sharex=False, figsize=(2.8*cols, 3*rows), squeeze=False)

            # Render the frames
            axes_flat = np.reshape(axes, (-1, ))
            for i in tqdm(range(axes_flat.shape[0])):
                frame_number = sequence_idx_to_render[i]
                ax = axes_flat[i]
                true_state_arrow, mean_particle_arrow = render_sequence_data(ax, data, output, frame_number, render_config)


            # Add Legend
            def make_legend_arrow(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
                p = mpatches.FancyArrow(0, 0.5*height, width, 0, length_includes_head=True, head_width=0.9*height, width=height*0.4)
                return p
                
            def make_legend_circle(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
                p = mpatches.Circle((width*0.75,height//2), radius=height*0.5)
                return p

            # Add the Legend
            handles = [true_state_arrow, mean_particle_arrow, plt.Circle((0, 0), 0.25, color="blue")]
            labels = ["True State", "Mean Particle",  "Posterior Density"]
            handler_map={mpatches.FancyArrow : HandlerPatch(patch_func=make_legend_arrow),mpatches.Circle : HandlerPatch(patch_func=make_legend_circle),}
            legend_properties = {'weight':'bold', "size":16}
            fig.legend(handles, labels,handler_map=handler_map, loc='upper center', ncol=5.0, prop=legend_properties)

            # Adjust whitespace
            # fig.subplots_adjust(wspace=0.01, hspace=0.03)
            fig.tight_layout(rect=(0,0,1,0.95))

            # Save
            os.chdir(absolute_path)

            save_dir = "{}/indiv/{}/".format(absolute_path, save_mapping[experiment])
            ensure_directory_exists(save_dir)
            fig.savefig("{}/panel_rendering_{:04d}.png".format(save_dir, seq_idx))
            fig.savefig("{}/panel_rendering_{:04d}.pdf".format(save_dir, seq_idx))

            plt.close("all")


if __name__ == '__main__':
    main()