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

from matplotlib import rc,rcParams
rc('font', weight='bold')




# The bandwidth stuff
from kernel_density_estimation.kernel_density_estimator import *
from models.particle_transformer import *
from datasets.bearings_only_dataset import *
from problems.bearings_only.bearings_only_problem import *


def find_median_run_index(experiment, number_of_runs=11, metric="nll"):

    # The data for this experiment
    data = []

    for i in range(number_of_runs):

        results_filepath = "experiments/bearings_only_experiments_methodical_25_particles_multiple_runs/{}/saves/run_{:03d}/full_dpf_evaluation_fixed_bands/table_results.pt".format(experiment, i)

        # Check if the file exists, if it doesnt then skip this file
        if(os.path.isfile(results_filepath) == False):
            return None

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


def load_sequence_data(experiment, run_number):


    filepath = "experiments/bearings_only_experiments_methodical_25_particles_multiple_runs/{}/saves/run_{:03d}/full_dpf_evaluation_fixed_bands/raw_all_sequence_evaluation_run_data.pt".format(experiment, run_number)

    # Check if the file exists, if it doesnt then skip this file
    if(os.path.isfile(filepath) == False):
        return None

    # Load the data 
    data = torch.load(filepath, map_location="cpu")  

    return data












def render_frame(ax, sequence_data, frame_idx, experiment, true_sequence_data, dataset):
            
    # We need the transformers
    particle_transformer = BearingsOnlyTransformer()
    observation_transformer = BearingsOnlyObservationTransformer()

    # Constant
    RENDERED_OBJECT_BASE_SIZE = 2.5

    # Extract the relevant data
    particles = sequence_data["particles"][frame_idx]
    particle_weights = sequence_data["particle_weights"][frame_idx]
    bandwidths = sequence_data["bandwidths"][frame_idx]
    states = true_sequence_data["states"][frame_idx]
    observations = true_sequence_data["observations"][frame_idx]

    # Transform the observations
    transformed_observation = observation_transformer.forward_tranform(observations)


    # Define the limits of the plot
    # x_lim = [-10, 10]
    x_lim = [-7, 10]
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

    # Draw the sensor information 
    sensors = dataset.get_sensors()
    for i, sensor in enumerate(sensors):

        # The sensor location
        position = sensor.get_position()
        x = position[0].item()
        y = position[1].item()
        ax.add_patch(plt.Circle((x, y), 0.5*RENDERED_OBJECT_BASE_SIZE, color='tab:green'))

        # The sensor observation
        sensor_obs_y = transformed_observation[i].item() * 20
        sensor_obs_x = transformed_observation[int(transformed_observation.shape[-1] / 2) + i].item() * 20
        x_values = [x, sensor_obs_x]
        y_values = [y, sensor_obs_y]
        ax.plot(x_values, y_values, color="tab:green", linewidth=RENDERED_OBJECT_BASE_SIZE*1.0)


    # Add a text label indicating what frame number we are at
    # text_str = "Time-step #{:02d}".format(frame_idx + 1)
    # ax.text(0.5, 0.93, text_str, horizontalalignment="center", verticalalignment="center",transform=ax.transAxes, weight="bold", fontsize=12)

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


def load_dataset():

    # Load the dataset parameters
    with open("./experiments/bearings_only_experiments_methodical_25_particles_multiple_runs/configs/dataset_params_evaluation.yaml") as file:

        # Load the whole file into a dictionary
        doc = yaml.load(file, Loader=yaml.FullLoader)

        # Load the dataset
        dataset = BearingsOnlyDataset(doc["dataset_params"], "evaluation")
        return dataset


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

    # sequences_to_render = [i for i in range(100)]
    # sequences_to_render = [60]
    sequences_to_render = [48]
    # sequences_to_render = [15, 16, 100, 800, 60]
    for sequence_to_render in tqdm(sequences_to_render):

        # Change directories to make everything very easy 
        absolute_path = os.path.dirname(os.path.realpath(__file__))
        os.chdir("../../..")

        # Constants
        # sequence_to_render = 2
        # number_of_frames_to_render = 12
        # frame_indices_to_render = [i*4 for i in range(number_of_frames_to_render)]
    	
        # number_of_frames_to_render = 100
        # frame_indices_to_render = [i for i in range(number_of_frames_to_render)]

        number_of_frames_to_render = 12
        frame_indices_to_render = [i*4+1 for i in range(number_of_frames_to_render)]
        frame_indices_to_render.append(15-1)
        frame_indices_to_render.append(17-1)
        frame_indices_to_render.append(18-1)
        frame_indices_to_render.append(19-1)
        frame_indices_to_render.append(48-1)
        frame_indices_to_render = set(frame_indices_to_render)
        frame_indices_to_render = list(frame_indices_to_render)
        frame_indices_to_render = sorted(frame_indices_to_render)


        # The experiments we want to load and the order we want them to be rendered in
        experiments = []
        experiments.append("experiment0003_importance_init")

        # experiments.append("experiment0002_importance")
        # experiments.append("experiment0001")

        # experiments.append("experiment0001")
        experiments.append("diffy_particle_filter_learned_band")
        experiments.append("importance_sampling_pf_learned_band")
        experiments.append("optimal_transport_pf_learned_band")
        # experiments.append("soft_resampling_particle_filter_learned_band")
        experiments.append("discrete_concrete")
        



        # Load the dataset
        dataset = load_dataset()

        # Generate the labels
        labels_mapping = get_labels_mapping()

        # Plot!!!!!
        rows = len(experiments)
        cols = len(frame_indices_to_render)
        fig, axes = plt.subplots(rows, cols, sharex=True, figsize=(3*cols*0.75, 3*rows*0.9), squeeze=False)


        for i, experiment in enumerate(tqdm(experiments, desc="Experiment")):

            # Get the data for this sequence 
            true_sequence_data = dataset[sequence_to_render]

            # Get which run to load
            median_run_idx = find_median_run_index(experiment)

            # Load the run data
            run_data = load_sequence_data(experiment, median_run_idx)

            # Extract the sequence data for this run
            sequence_data = run_data[sequence_to_render]

            # Render
            for j, frame_idx in enumerate(tqdm(frame_indices_to_render, desc="Frames", leave=False)):
                ax = axes[i, j]
                true_state_arrow, mean_particle_arrow = render_frame(ax, sequence_data, frame_idx, experiment, true_sequence_data, dataset)

            # Label each row
            axes[i, 0].set_ylabel(labels_mapping[experiment], weight="bold", fontsize=20)

        for j, frame_idx in enumerate(frame_indices_to_render):
            axes[0, j].set_title("Time-step: #{:02d}".format(frame_idx+1), weight="bold", fontsize=14)

        # Add Legend
        def make_legend_arrow(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
            p = mpatches.FancyArrow(0, 0.5*height, width, 0, length_includes_head=True, head_width=0.9*height, width=height*0.6)
            return p
            
        def make_legend_circle(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
            p = mpatches.Circle((width*0.75,height//2), radius=height*0.5)
            return p

        # Add the Legend
        handles = [true_state_arrow, mean_particle_arrow, plt.Circle((0, 0), 0.25, color="black"), plt.Circle((0, 0), 0.25, color="green")]
        labels = ["True State", "Mean Particle", "Particles", "Radar Station + Radar Sensor Reading"]
        handler_map={mpatches.FancyArrow : HandlerPatch(patch_func=make_legend_arrow),mpatches.Circle : HandlerPatch(patch_func=make_legend_circle),}
        # lgnd = fig.legend(handles, labels,handler_map=handler_map, loc='upper center', ncol=5.0, fontsize=14, bbox_to_anchor=(0.5, 1.025))
        lgnd = fig.legend(handles, labels,handler_map=handler_map, loc='upper center', ncol=5.0, fontsize=16)

        # Adjust whitespace
        fig.subplots_adjust(wspace=0.01, hspace=0.03)
        fig.tight_layout(rect=(0,0,1,0.96))

        os.chdir(absolute_path)
        fig.savefig("panel_seq_num_{:05d}.png".format(sequence_to_render))
        fig.savefig("panel_seq_num_{:05d}.pdf".format(sequence_to_render))
        print(sequence_to_render)



if __name__ == '__main__':
	main()