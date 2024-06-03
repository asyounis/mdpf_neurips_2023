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



def load_dataset():

    # Load the dataset parameters
    with open("./experiments/house3d_experiments/configs/dataset_params_evaluation.yaml") as file:

        # Load the whole file into a dictionary
        doc = yaml.load(file, Loader=yaml.FullLoader)

        # Load the dataset
        dataset = House3DDataset(doc["dataset_params"], "evaluation")
        return dataset


def render_kde_density(world_map_rgb, particles, particle_weights, bandwidths, x_lim, y_lim, x_range, y_range):

    # Move to the GPU for fast compute
    particles = particles.cuda()
    particle_weights = particle_weights.cuda()
    bandwidths = bandwidths.cuda()


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
    obs_inset_location = render_config["obs_inset_location"]
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
        particles_scatter = ax.scatter(x,y, color="blue", s=0.15*rendered_object_base_size)


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
    particles = backward_tranform(particles)
    mean_state = torch.sum(particles * particle_weights.unsqueeze(-1), dim=0)
    mean_state = forward_tranform(mean_state)


    # Plot the true location
    true_x = mean_state[0].item()
    true_y = mean_state[1].item()
    mean_particle_circle = plt.Circle((true_x, true_y), 0.33 * rendered_object_base_size, color=sns.color_palette("bright")[8])
    ax.add_patch(mean_particle_circle)
    dy = torch.sin(mean_state[2]).item() * 1.5 * rendered_object_base_size
    dx = torch.cos(mean_state[2]).item() * 1.5 * rendered_object_base_size
    mean_particle_arrow = ax.arrow(true_x, true_y, dx, dy, color=sns.color_palette("bright")[8], width=0.33*rendered_object_base_size)



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

    # Add a text label indicating what frame number we are at
    text_str = "t={:02d}".format(frame_number+1)
    ax.text(0.02, 0.94, text_str, horizontalalignment='left', verticalalignment='center',transform=ax.transAxes, weight="bold", fontsize=16, path_effects=[patheffects.withStroke(linewidth=4, foreground='white')])


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


def get_running_params():

    all_running_params = []

    # #################################################################################
    # ## Running Parameters Unimodal
    # #################################################################################
    running_params_1 = dict()
    running_params_1["sequence_to_render"] = 502
    running_params_1["rows"] = 1
    running_params_1["frame_render_mod"] = 3
    running_params_1["name"] = "Unimodal Init"
    running_params_1["save_name"] = "true_init"

    running_params_1["render_config"] = dict()
    running_params_1["render_config"]["x_min"] = 225
    running_params_1["render_config"]["x_max"] = 350
    running_params_1["render_config"]["y_min"] = 575
    running_params_1["render_config"]["y_max"] = 785

    running_params_1["render_config"]["draw_particles"] = False
    running_params_1["render_config"]["render_kde"] = True

    running_params_1["render_config"]["tail_render_size"] = 0.35

    running_params_1["render_config"]["obs_inset_location"] = [0.54,0.54,0.45,0.45]

    all_running_params.append(running_params_1)


    #################################################################################
    ## Running Parameters Multimodal
    #################################################################################
    running_params_2 = dict()
    running_params_2["sequence_to_render"] = 15
    running_params_2["rows"] = 1
    running_params_2["frame_render_mod"] = 3
    running_params_2["name"] = "Multimodal Init"
    running_params_2["save_name"] = "multi_mode_init"

    running_params_2["render_config"] = dict()
    running_params_2["render_config"]["x_min"] = 400
    running_params_2["render_config"]["x_max"] = 900
    running_params_2["render_config"]["y_min"] = 500
    running_params_2["render_config"]["y_max"] = 900

    running_params_2["render_config"]["draw_particles"] = True
    running_params_2["render_config"]["render_kde"] = False

    running_params_2["render_config"]["tail_render_size"] = 0.15

    running_params_2["render_config"]["obs_inset_location"] = [0.54,0.01,0.45,0.45]

    all_running_params.append(running_params_2)



    #################################################################################
    ## Running Parameters Random Init
    #################################################################################
    running_params_3 = dict()
    running_params_3["sequence_to_render"] = 505
    running_params_3["rows"] = 1
    running_params_3["frame_render_mod"] = 3
    running_params_3["name"] = "Random Init"
    running_params_3["save_name"] = "random_init"

    running_params_3["render_config"] = dict()
    running_params_3["render_config"]["x_min"] = 300
    running_params_3["render_config"]["x_max"] = 1100
    running_params_3["render_config"]["y_min"] = 300
    running_params_3["render_config"]["y_max"] = 1100

    running_params_3["render_config"]["draw_particles"] = True
    running_params_3["render_config"]["render_kde"] = False

    running_params_3["render_config"]["tail_render_size"] = 0.1

    # running_params_3["render_config"]["obs_inset_location"] = [0.6,0.01,0.39,0.39]
    running_params_3["render_config"]["obs_inset_location"] = [0.54,0.01,0.45,0.45]

    all_running_params.append(running_params_3)



    return all_running_params

def main():

    # Get all the running parameters
    all_running_params = get_running_params()

    # Plot!!!!!
    rows = len(all_running_params)
    cols = 10
    # fig, axes = plt.subplots(rows, cols, sharex=False, figsize=(3*cols, 3*rows), squeeze=False)
    fig, axes = plt.subplots(rows, cols, sharex=False, figsize=(2.8*cols, 3*rows), squeeze=False)

    # Change directories to make everything very easy 
    absolute_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir("../../..")

    # Load the dataset
    dataset = load_dataset()

    for row_idx, running_params in enumerate(all_running_params):

        # params
        sequence_to_render = running_params["sequence_to_render"]

        # Grab a specific sequence
        sequence_data = dataset[sequence_to_render]

        # Get the frames we want to render
        number_of_frames_to_render = cols
        frames_to_render = [i*running_params["frame_render_mod"] for i in range(number_of_frames_to_render)]
        frames_to_render.append(1)
        frames_to_render.append(2)
        frames_to_render.append(3)
        frames_to_render = list(set(frames_to_render))
        frames_to_render = sorted(frames_to_render)

        # Load the model outputs
        save_file = "{}/save_{}.pt".format(absolute_path, running_params["save_name"])
        all_output_dicts =  torch.load(save_file)


        for i in tqdm(range(number_of_frames_to_render), desc="Rendering", leave=False):

            # Get the axis to draw on
            ax = axes[row_idx, i]
            frame_id = frames_to_render[i]

            # Render the sequence data
            true_state_arrow, mean_particle_arrow = render_sequence_data(ax, sequence_data, all_output_dicts[frame_id], frame_id, running_params["render_config"])


        axes[row_idx, 0].set_ylabel(running_params["name"], fontsize=16, weight="bold")



    # Add Legend
    def make_legend_arrow(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
        p = mpatches.FancyArrow(0, 0.5*height, width, 0, length_includes_head=True, head_width=0.9*height, width=height*0.4)
        return p
        
    def make_legend_circle(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
        p = mpatches.Circle((width*0.75,height//2), radius=height*0.5)
        return p

    # Add the Legend
    handles = [true_state_arrow, mean_particle_arrow, plt.Circle((0, 0), 0.25, color="blue")]
    labels = ["True State", "Mean Particle",  "Particles / Posterior Density"]
    handler_map={mpatches.FancyArrow : HandlerPatch(patch_func=make_legend_arrow),mpatches.Circle : HandlerPatch(patch_func=make_legend_circle),}
    legend_properties = {'weight':'bold', "size":16}
    fig.legend(handles, labels,handler_map=handler_map, loc='upper center', ncol=5.0, prop=legend_properties)


    # Adjust whitespace
    # fig.subplots_adjust(wspace=0.01, hspace=0.03)
    fig.tight_layout(rect=(0,0,1,0.96))

    os.chdir(absolute_path)
    fig.savefig("combined.png")
    fig.savefig("combined.pdf")


if __name__ == '__main__':
    main()