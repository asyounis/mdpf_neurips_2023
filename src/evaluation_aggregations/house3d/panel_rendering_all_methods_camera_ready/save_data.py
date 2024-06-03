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
    if ("lstm" not in experiment_to_use):
        pre_trained_models = dict()
        pre_trained_models["dpf_model"] = "experiments/house3d_experiments/{}/saves/full_dpf_fixed_bands/models/full_dpf_model_best.pt".format(experiment_to_use)
        model.load_pretrained(pre_trained_models, "cpu")

    else:
        pre_trained_models = dict()
        pre_trained_models["dpf_model"] = "experiments/house3d_experiments/{}/saves/full_dpf_fixed_bands/models/full_rnn_model_best.pt".format(experiment_to_use)
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



def create_initial_dpf_state_unimodal(model, true_state, observations, number_of_particles):

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


def create_initial_rnn_state(model, true_state, observations):
        # Need the batch size
        batch_size = true_state.shape[0]

        particles =  true_state.unsqueeze(1)
        particles = model.particle_transformer.downscale(particles)

        # Move to the correct device
        particles = particles.to(observations.device)


        # Set the predicted state to be the true state and then create the hidden state with zeros
        predicted_state = true_state.detach().clone().to(observations.device)

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

        # Set the hidden state to be zeros
        hidden_state = model.lstm_internal.create_initial_input_hidden_state(batch_size, predicted_state.device)

        # Pack everything into an output dict
        output_dict = dict()
        output_dict["hidden_state"] = hidden_state
        output_dict["predicted_state"] = predicted_state


        # Create particles
        particles = predicted_state.unsqueeze(1)
        particle_weights = torch.ones_like(particles[:, :, 0])

        bandwidths = torch.zeros(size=(particles.shape[0], particles.shape[-1])).to(observations.device)
        bandwidths[...] = 0.001

        output_dict["particles_downscaled"] = particles
        output_dict["particles"] = model.particle_transformer.upscale(particles)
        output_dict["particle_weights"] = particle_weights
        output_dict["bandwidths_downscaled"] = bandwidths
        output_dict["bandwidths"] = model.particle_transformer.upscale(bandwidths)

        return output_dict


def create_initial_state(model, true_state, observations, number_of_particles):
    if(isinstance(model, LSTMRnn)):
        return create_initial_rnn_state(model, true_state, observations)
    else:
        return create_initial_dpf_state_unimodal(model, true_state, observations, number_of_particles)



def run_model(model, data ,number_of_particles=50):

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

        # Create the initial output
        output_dict = create_initial_state(model, states[:,0,:],transformed_observation, number_of_particles)

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


def get_experiments_to_run():

    experiments = []
    # experiments.append("lstm_rnn")
    # experiments.append("diffy_particle_filter_learned_band")
    experiments.append("optimal_transport_pf_learned_band")
    # experiments.append("soft_resampling_particle_filter_learned_band")
    # experiments.append("importance_sampling_pf_learned_band")
    # experiments.append("experiment0001")
    # experiments.append("experiment0002_importance")
    # experiments.append("experiment0003_importance_init")

    return experiments



def main():

    # Get the experiments that we should run
    experiments = get_experiments_to_run()

    # Change directories to make everything very easy 
    absolute_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir("../../..")

    # Make an overall save directory
    save_dir = "{}/raw_sequences_data/".format(absolute_path)
    ensure_directory_exists(save_dir)

    # Do each experiments data
    for experiment in tqdm(experiments):

        # Make a model specific save file
        specific_save_dir = "{}/{}".format(save_dir, experiment)
        ensure_directory_exists(specific_save_dir)
        
        # Load the model
        model = create_and_load_model(experiment)

        # Load the dataset
        dataset = load_dataset()

        # Process the dataset
        # number_to_process = 250
        number_to_process = len(dataset)
        for i in tqdm(range(0, number_to_process), leave=False, desc="Processing Sequences"):

            # get the data items needed
            data = dataset[i]

            # Get all the outputs
            all_output_dicts = run_model(model, data)

            # Make an overall dict
            save_dict = dict()
            save_dict["data"] = data
            save_dict["output"] = all_output_dicts

            # Save the data
            torch.save(save_dict, "{}/{:04d}.pt".format(specific_save_dir, i))





if __name__ == '__main__':
    main()