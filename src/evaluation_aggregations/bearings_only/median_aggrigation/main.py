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
import shutil


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
        results = torch.load(results_filepath)  

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



def copy_panel(experiment, run_index, absolute_path):

    results_dir = "experiments/bearings_only_experiments_methodical_25_particles_multiple_runs/{}/saves/run_{:03d}/full_dpf_evaluation_fixed_bands/".format(experiment, run_index)

    # Create the directory to save things to
    save_dir = "{}/output/{}/".format(absolute_path, experiment)
    if(not os.path.exists(save_dir)):
        os.makedirs(save_dir)

    files_to_copy = []
    files_to_copy.append("panel_rendering_0.png")
    files_to_copy.append("panel_rendering_0.pdf")
    files_to_copy.append("panel_rendering_1.png")
    files_to_copy.append("panel_rendering_1.pdf")


    for f in files_to_copy:
        src = "{}/{}".format(results_dir, f)
        dst = "{}/{}".format(save_dir, f)
        shutil.copyfile(src, dst)

def main():

    # Change directories to make everything very easy 
    absolute_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir("../../..")

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
    experiments.append("optimal_transport_pf_learned_band")
    
    for i, experiment in enumerate(tqdm(experiments, desc="Experiment")):

        # Get which run to load
        median_run_idx = find_median_run_index(experiment)

        copy_panel(experiment, i, absolute_path)



if __name__ == '__main__':
	main()