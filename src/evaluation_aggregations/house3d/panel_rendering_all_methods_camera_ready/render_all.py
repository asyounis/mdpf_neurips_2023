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

    lim_min = 0
    lim_max = 3000
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


def render_sequence_data(ax, data, output_dict, frame_number, render_config):

    # Unpack the rendering configs
    x_min = render_config["x_min"]
    x_max = render_config["x_max"]
    y_min = render_config["y_min"]
    y_max = render_config["y_max"]
    draw_particles = render_config["draw_particles"]
    render_kde = render_config["render_kde"]
    tail_render_size = render_config["tail_render_size"]

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

    # Unpack
    world_map = data["world_map"]
    states = data["states"][frame_number]
    all_states = data["states"]

    # Convert the world map to numpy so we can fully process it and render it
    world_map = world_map.numpy()

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
    # ax.plot(x1, y1, color="red", label="True State", alpha=1.0, linewidth=0.25*rendered_object_base_size, linestyle="solid")
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




def main():

    # Get the experiments that we should run
    experiments = get_experiments_to_run()

    # Mapping the experiment names to the correct names for the paper
    name_mappings = get_name_mapping()

    # Change directories to make everything very easy 
    absolute_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir("../../..")

    # The loading dir
    load_dir = "{}/raw_sequences_data/".format(absolute_path)

    # Which sequence we want to render
    sequence_to_render = 502

    # Sequence idxs to render
    sequence_idx_to_render = [1, 15, 30]


    # render_config = dict()
    # render_config["x_min"] = 750
    # render_config["x_max"] = 1500
    # render_config["y_min"] = 750
    # render_config["y_max"] = 1615
    # render_config["draw_particles"] = False
    # render_config["render_kde"] = False
    # render_config["tail_render_size"] = 0.15




    render_config = dict()
    render_config["x_min"] = 225
    render_config["x_max"] = 350
    render_config["y_min"] = 575
    render_config["y_max"] = 785
    render_config["draw_particles"] = False
    render_config["render_kde"] = True
    render_config["tail_render_size"] = 0.3





    # Make the figure
    rows = len(sequence_idx_to_render)
    cols = len(experiments)+1
    fig, axes = plt.subplots(rows, cols, sharex=False, figsize=(2.8*cols, 3*rows), squeeze=False)

    # Do each experiments data
    for exp_idx, experiment in enumerate(tqdm(experiments)):

        # Load the data
        saved_dict = torch.load("{}/{}/{:04d}.pt".format(load_dir, experiment, sequence_to_render))

        # Unpack
        data = saved_dict["data"]

        # The outputs
        output = saved_dict["output"]
        output = pack_and_stack(output)

        # Render the frames
        for i, frame_number in enumerate(sequence_idx_to_render):
            ax = axes[i, exp_idx+1]
            true_state_arrow, mean_particle_arrow = render_sequence_data(ax, data, output, frame_number, render_config)

        # Say which experiment this is for
        name = name_mappings[experiment]
        axes[0, exp_idx+1].set_title(name, fontsize=16, fontweight="bold")

    # Label the frames with the timestep number
    for i, frame_number in enumerate(sequence_idx_to_render):
            ax = axes[i, 0]
            ax.set_ylabel("Time-step #{}".format(frame_number), fontsize=16, fontweight="bold")


    # Draw the observations
    for i, frame_number in enumerate(sequence_idx_to_render):
        ax = axes[i, 0]
        observation = torch.permute(data["observations"][frame_number].cpu(), (1, 2, 0))
        observation[observation<0] = 0
        observation[observation>1.0] = 1.0
        ax.imshow(observation.numpy())
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.tick_params(left=False, bottom=False)
        ax.patch.set_edgecolor('black')  
        ax.patch.set_linewidth(3)  


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
    fig.savefig("house3d_all_rendered_comparison.png")
    fig.savefig("house3d_all_rendered_comparison.pdf")


if __name__ == '__main__':
    main()