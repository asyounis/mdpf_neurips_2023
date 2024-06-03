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



# The bandwidth stuff
from kernel_density_estimation.kernel_density_estimator import *
from models.particle_transformer import *
from datasets.deepmind_maze_dataset import *
from problems.deepmind_maze.deepmind_maze_problem import *


def find_median_run_index(experiment, maze_number, number_of_runs=5, metric="nll"):

    # The data for this experiment
    data = []

    for i in range(number_of_runs):

        results_filepath = "experiments/deepmind_maze_experiments_multiple_runs/{}/saves/run_{:03d}/full_dpf_evaluation_maze_{:d}/table_results.pt".format(experiment, i, maze_number)

        # Check if the file exists, if it doesnt then skip this file
        if(os.path.isfile(results_filepath) == False):
            assert(False)

        # Load the data 
        results = torch.load(results_filepath, map_location="cpu")  

        # Extract the NLL
        nll_mean = results[metric][0]

        # Save the data
        data.append(nll_mean)


    # Sort so we can extract the median
    sorted_data = copy.copy(data)
    sorted_data = sorted(sorted_data)

    # Extract the median
    median_value = sorted_data[number_of_runs//2]

    # Get the median index
    for i in range(number_of_runs):
        if(data[i] == median_value):
            return i

    assert(False)


def load_sequence_data(experiment, run_number, maze_number):
    filepath = "experiments/deepmind_maze_experiments_multiple_runs/{}/saves/run_{:03d}/full_dpf_evaluation_maze_{:d}/raw_all_sequence_evaluation_run_data.pt".format(experiment, run_number, maze_number)

    # Check if the file exists, if it doesnt then skip this file
    if(os.path.isfile(filepath) == False):
        return None

    # Load the data 
    data = torch.load(filepath, map_location="cpu")  

    return data


def load_dataset(maze_number):

    # Load the dataset parameters
    with open("./experiments/deepmind_maze_experiments_multiple_runs/config_files/dataset_params_evaluation_maze_{:d}.yaml".format(maze_number)) as file:

        # Load the whole file into a dictionary
        doc = yaml.load(file, Loader=yaml.FullLoader)

        # Load the dataset
        dataset = DeepMindMazeDataset(doc["dataset_params"], "evaluation")
        return dataset


def render_maze(ax, dataset, maze_number):

    # Compute the maze ID
    maze_id = maze_number - 1

    # Set the X and Y limits
    ax.set_xlim(dataset.get_x_range_scaled(maze_id))
    ax.set_ylim(dataset.get_y_range_scaled(maze_id))

    # Draw the walls
    walls = dataset.get_walls(maze_id)
    walls = dataset.scale_data_down(walls, maze_id)
    ax.plot(walls[:, :, 0].T, walls[:, :, 1].T, color="black", linewidth=3)


def render_frame(ax, sequence_data, frame_idx, experiment, true_sequence_data, dataset, maze_number):
            
    # We need the transformers
    particle_transformer = BearingsOnlyTransformer()

    # Constant
    RENDERED_OBJECT_BASE_SIZE = 1.5

    # Extract the relevant data
    particles = sequence_data["particles"][frame_idx]
    particle_weights = sequence_data["particle_weights"][frame_idx]
    bandwidths = sequence_data["bandwidths"][frame_idx]
    states = true_sequence_data["states"][frame_idx]
    observations = true_sequence_data["observations"][frame_idx]

    # Draw the maze
    render_maze(ax, dataset, maze_number)

    # Define the limits of the plot
    # x_lim = [-10, 10]
    x_lim = [-10, 10]
    y_lim = [-10, 10]

    # Create the sampling mesh grid
    density = 400
    x = torch.linspace(x_lim[0], x_lim[1], density)
    y = torch.linspace(y_lim[0], y_lim[1], density)
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
    probs = probs.reshape(grid_x.shape)

    # Plot the probs map
    probs = np.transpose(probs, axes=[1,0])
    ax.imshow(probs, extent=[x_lim[0], x_lim[1], y_lim[0], y_lim[1]], interpolation='bilinear', origin="lower", cmap=sns.cubehelix_palette(start=-0.2, rot=0.0, dark=0.05, light=1.0, reverse=False, as_cmap=True))

    # Draw the particles
    x = particles[:, 0].cpu().numpy()
    y = particles[:, 1].cpu().numpy()
    particles_scatter = ax.scatter(x,y, color="black", s=0.25*RENDERED_OBJECT_BASE_SIZE)

    # Compute the mean particle
    particles = particle_transformer.backward_tranform(particles)
    predicted_state = torch.sum(particles * particle_weights.unsqueeze(-1), dim=0)
    predicted_state = particle_transformer.forward_tranform(predicted_state)

    # Render the mean particle
    x = predicted_state[0].item()
    y = predicted_state[1].item()
    mean_particle_circle = plt.Circle((x, y), 0.25*RENDERED_OBJECT_BASE_SIZE, color=sns.color_palette("bright")[8])
    ax.add_patch(mean_particle_circle)
    dy = torch.sin(predicted_state[2]).item() * 1.0*RENDERED_OBJECT_BASE_SIZE
    dx = torch.cos(predicted_state[2]).item() * 1.0*RENDERED_OBJECT_BASE_SIZE
    mean_particle_arrow = ax.arrow(x, y, dx, dy, color=sns.color_palette("bright")[8], width=0.15*RENDERED_OBJECT_BASE_SIZE)

    # Plot the true location
    true_x = states[0].item()
    true_y = states[1].item()
    true_state_circle = plt.Circle((true_x, true_y), 0.25 * RENDERED_OBJECT_BASE_SIZE, color=sns.color_palette("bright")[3])
    ax.add_patch(true_state_circle)
    dy = torch.sin(states[2]).item() * 1.0 * RENDERED_OBJECT_BASE_SIZE
    dx = torch.cos(states[2]).item() * 1.0 * RENDERED_OBJECT_BASE_SIZE
    true_state_arrow = ax.arrow(true_x, true_y, dx, dy, color=sns.color_palette("bright")[3], width=0.15*RENDERED_OBJECT_BASE_SIZE)


    # # Add a text label indicating what frame number we are at
    # # text_str = "Time-step #{:02d}".format(frame_idx + 1)
    # # ax.text(0.5, 0.93, text_str, horizontalalignment="center", verticalalignment="center",transform=ax.transAxes, weight="bold", fontsize=12)

    # Set the X and Y limits
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    # Turn off the axis labels
    # ax.axis('off')

    # Set the aspect ratio to a square 
    ax.set_aspect("equal")
    
    # Turn off tick labels
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    # turn off the axis ticks little bars
    ax.tick_params(left=False, bottom=False)

    # Add a box around the outside
    ax.patch.set_edgecolor('black')  
    ax.patch.set_linewidth(1)  


    return true_state_arrow, mean_particle_arrow


def render_observation(ax, frame_idx, true_sequence_data):

    observations = true_sequence_data["observations"][frame_idx]

    observation = torch.permute(observations.cpu(), (1, 2, 0))
    observation[observation<0] = 0
    observation[observation>1.0] = 1.0
    ax.imshow(observation.numpy())

    # Set the aspect ratio to a square 
    ax.set_aspect("equal")
    
    # Turn off tick labels
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    # turn off the axis ticks little bars
    ax.tick_params(left=False, bottom=False)

    # Add a box around the outside
    ax.patch.set_edgecolor('black')  
    ax.patch.set_linewidth(1)  


def get_labels_mapping():

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


    experiment_names["experiment0002_implicit"] = "MDPF-Implicit"
    experiment_names["experiment0003_implicit"] = "A-MDPF-Implicit"

    experiment_names["experiment0002_concrete"] = "MDPF-Concrete"
    experiment_names["experiment0003_concrete"] = "A-MDPF-Concrete"



    experiment_names["diffy_particle_filter_learned_band"] = "TG-PF"
    experiment_names["optimal_transport_pf_learned_band"] = "OT-PF"
    experiment_names["soft_resampling_particle_filter_learned_band"] = "SR-PF"
    experiment_names["importance_sampling_pf_learned_band"] = "DIS-PF"
    experiment_names["discrete_concrete"] = "C-PF"

    return experiment_names


def main():


    maze_numbers = [1, 2, 3]
    sequences_to_render = dict()
    # sequences_to_render[1] = 0 # pretty ok
    # sequences_to_render[1] = 1 # pretty ok

    sequences_to_render[1] = 0
    sequences_to_render[2] = 10
    sequences_to_render[3] = 0

    length_of_rendered_path = 25

    # We need the transformers
    particle_transformer = DeepMindTransformer()

    # Generate the labels
    labels_mapping = get_labels_mapping()






    for m_idx, maze_number in enumerate(maze_numbers):

        # Change directories to make everything very easy 
        absolute_path = os.path.dirname(os.path.realpath(__file__))
        os.chdir("../../..")

        # Get which sequence to render
        sequence_to_render = sequences_to_render[maze_number]

        # The Frames to render
        number_of_frames_to_render = 12
        frame_indices_to_render = [i*2 for i in range(number_of_frames_to_render)]

        # The experiments we want to load and the order we want them to be rendered in
        experiments = []
        experiments.append("experiment0003_importance_init")
        experiments.append("experiment0002_importance")
        experiments.append("experiment0001")
        experiments.append("diffy_particle_filter_learned_band")
        experiments.append("importance_sampling_pf_learned_band")
        experiments.append("optimal_transport_pf_learned_band")
        experiments.append("soft_resampling_particle_filter_learned_band")
        experiments.append("discrete_concrete")

        # Load the dataset
        dataset = load_dataset(maze_number)

        # Get the data for this sequence 
        true_sequence_data = dataset[sequence_to_render]

        # Plot!!!!!

        rows = len(experiments) + 1
        cols = len(frame_indices_to_render)
        height_ratios = [1]
        height_ratios.extend([2 for _ in range(len(experiments))])
        fig, axes = plt.subplots(rows, cols, sharex=False, figsize=(3*cols*0.9, 3*(rows-1) + 3*0.25), squeeze=False, gridspec_kw={'height_ratios': height_ratios})


        # Render the Observations
        for j, frame_idx in enumerate(frame_indices_to_render):
            render_observation(axes[0, j], frame_idx, true_sequence_data)

        # Render for each sequence
        for i, experiment in enumerate(tqdm(experiments, desc="Experiment")):

            # Get which run to load
            median_run_idx = find_median_run_index(experiment, maze_number)

            # Load the run data
            run_data = load_sequence_data(experiment, median_run_idx, maze_number)

            # Get this particular sequence
            sequence_data = run_data[sequence_to_render]

            # Render
            for j, frame_idx in enumerate(tqdm(frame_indices_to_render, desc="Frames", leave=False)):
                ax = axes[i+1, j]
                true_state_arrow, mean_particle_arrow = render_frame(ax, sequence_data, frame_idx, experiment, true_sequence_data, dataset, maze_number)

            # Label each row
            axes[i+1, 0].set_ylabel(labels_mapping[experiment], weight="bold", fontsize=16)


        # Set the titles
        for j, frame_idx in enumerate(frame_indices_to_render):
            axes[0, j].set_title("Time-step #{:02d}".format(frame_idx+1), weight="bold", fontsize=14)

        # Add Legend
        def make_legend_arrow(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
            p = mpatches.FancyArrow(0, 0.5*height, width, 0, length_includes_head=True, head_width=0.75*height, width=height*0.25)
            return p
            
        def make_legend_circle(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
            p = mpatches.Circle((width*0.75,height//2), radius=height*0.5)
            return p

        # Add the Legend
        handles = [true_state_arrow, mean_particle_arrow, plt.Circle((0, 0), 0.25, color="black")]
        labels = ["True State", "Mean Particle", "Particles"]
        handler_map={mpatches.FancyArrow : HandlerPatch(patch_func=make_legend_arrow),mpatches.Circle : HandlerPatch(patch_func=make_legend_circle),}
        # lgnd = fig.legend(handles, labels,handler_map=handler_map, loc='upper center', ncol=5.0, fontsize=14, bbox_to_anchor=(0.5, 1.025))
        lgnd = fig.legend(handles, labels,handler_map=handler_map, loc='upper center', ncol=5.0, fontsize=14)

        # Go back to the right spot
        os.chdir(absolute_path)

        # Adjust whitespace
        fig.subplots_adjust(wspace=0.01, hspace=0.03)
        fig.tight_layout(rect=(0,0,1,0.93))

        fig.savefig("maze_{:d}_panel_seq_num_{:05d}.png".format(maze_number, sequence_to_render))
        fig.savefig("maze_{:d}_panel_seq_num_{:05d}.pdf".format(maze_number, sequence_to_render))


if __name__ == '__main__':
    main()


