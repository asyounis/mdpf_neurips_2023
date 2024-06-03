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

def main():


    running_params_1 = dict()
    running_params_1["sequence_to_render"] = 502
    running_params_1["zoom_factor"] = 3.0
    running_params_1["save_name"] = "uni_init"
    running_params_1["number_of_particles"] = 50
    running_params_1["draw_particles"] = False
    running_params_1["rows"] = 1
    running_params_1["frame_render_mod"] = 3
    running_params_1["name"] = "Unimodal Init"
    running_params_1["save_name"] = "unimode_init"

    # running_params_1["render_config"] = dict()
    # running_params_1["render_config"]["x_min"] = 0
    # running_params_1["render_config"]["x_max"] = 100
    # running_params_1["render_config"]["y_min"] = 0
    # running_params_1["render_config"]["y_max"] = 100

    # running_params_1["render_config"]["draw_particles"] = True

    all_running_params = [running_params_1]


    # Plot!!!!!
    rows = len(all_running_params)
    cols = 10
    fig, axes = plt.subplots(rows, cols, sharex=False, figsize=(3*cols, 3*rows), squeeze=False)

    # Change directories to make everything very easy 
    absolute_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir("../../..")

    # Load the dataset
    dataset = load_dataset()

    for row_idx, running_params in enumerate(all_running_params):

        # params
        sequence_to_render = running_params["sequence_to_render"]
        init_method = running_params["init_method"]
        zoom_factor = running_params["zoom_factor"]
        number_of_particles = running_params["number_of_particles"]
        draw_particles = running_params["draw_particles"]

        # The name of the experiment that we are going to use
        experiment_to_use = "experiment0003_importance_init"

        # Grab a specific sequence
        sequence_data = dataset[sequence_to_render]

        number_of_frames_to_render = cols
        frames_to_render = [i*running_params["frame_render_mod"] for i in range(number_of_frames_to_render)]
        frames_to_render.append(1)
        frames_to_render.append(2)
        frames_to_render.append(3)
        frames_to_render = list(set(frames_to_render))
        frames_to_render = sorted(frames_to_render)


        save_file = "{}/save_{}.pt".format(absolute_path, running_params["save_name"])
        all_output_dicts =  torch.load(save_file)


        # for i in tqdm(range(number_of_frames_to_render), desc="Rendering", leave=False):

        #     # Get the axis to draw on
        #     ax = axes[row_idx, i]
        #     frame_id = frames_to_render[i]

        #     # Render the sequence data
        #     render_sequence_data(ax, sequence_data, all_output_dicts[frame_id], frame_id, zoom_factor, draw_particles)


        # axes[row_idx, 0].set_ylabel(running_params["name"], fontsize=12, weight="bold")



    os.chdir(absolute_path)
    fig.tight_layout()
    fig.savefig("combined.png")
    fig.savefig("combined.pdf")


if __name__ == '__main__':
    main()