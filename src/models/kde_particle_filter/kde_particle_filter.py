# Standard Imports
import numpy as np
import os
import PIL
import math
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from functools import partial

# Pytorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

# For drawing the compute graphs
import torchviz

# Project Imports
from models.particle_transformer import *

# The bandwidth stuff
from bandwidth_selection import bandwidth_selection_models
from bandwidth_selection import blocks
from kernel_density_estimation.kde_computer import *
from kernel_density_estimation.kernel_density_estimator import *
from kernel_density_estimation.epanechnikov import *
from kernel_density_estimation.circular_epanechnikov import *

# Other Models
from models.sequential_models import *
from models.internal_models.proposal_models import *
from models.internal_models.weight_models import *
from models.internal_models.particle_encoder_models import *
from models.internal_models.observation_encoder_models import *
from models.internal_models.initializer_models import *
from models.internal_models.bandwidth_models import *


class KDEParticleFilter(SequentialModels):

    def __init__(self, model_params, model_architecture_params):
        super(KDEParticleFilter, self).__init__(model_params, model_architecture_params)

        # Extract the name of the model so we can get its model architecture params
        main_model_name = model_params["main_model"]
        main_model_arch_params = model_architecture_params[main_model_name]


        # Extract some parameters about this model
        self.weight_divide_by_proposal_probability = main_model_arch_params["weight_divide_by_proposal_probability"]
        self.encode_particles = main_model_arch_params["encode_particles"]
        self.decouple_particle_encoders = main_model_arch_params["decouple_particle_encoders"]
        self.use_differentiable_resampling = main_model_arch_params["use_differentiable_resampling"]
        self.decouple_weights_for_resampling = main_model_arch_params["decouple_weights_for_resampling"]
        self.decouple_bandwidths_for_resampling = main_model_arch_params["decouple_bandwidths_for_resampling"]

        if("do_convert_bandwidth_model_to_bandwidths_for_resampling" in main_model_arch_params):
            self.do_convert_bandwidth_model_to_bandwidths_for_resampling = main_model_arch_params["do_convert_bandwidth_model_to_bandwidths_for_resampling"]
        else:
            self.do_convert_bandwidth_model_to_bandwidths_for_resampling = True

        # If we are initializing with the true state then we may want to add some additional hidden dims to the starting state
        if(self.initilize_with_true_state):
            if("number_of_particle_hidden_dims_to_add_to_init_state" in main_model_arch_params):
                self.number_of_particle_hidden_dims_to_add_to_init_state = main_model_arch_params["number_of_particle_hidden_dims_to_add_to_init_state"]
            else:
                self.number_of_particle_hidden_dims_to_add_to_init_state = 0

            assert("initial_position_std" in main_model_arch_params)
            self.initial_position_std = main_model_arch_params["initial_position_std"]


        # Get the method of differentiable resampling if we are gonna diff resampling
        if(self.use_differentiable_resampling):
            self.differentiable_resampling_method = main_model_arch_params["differentiable_resampling_method"]

            # Get the specific parameters for this type of relaxation
            if(self.differentiable_resampling_method == "ConcreteRelaxation"):
                self.concrete_relaxation_temperature_parameter = main_model_arch_params["concrete_relaxation_temperature_parameter"]


        # Get the parameters for the kde
        assert("kde" in main_model_arch_params)
        self.kde_params = main_model_arch_params["kde"]

        # We can extract the particle dimensions from the KDE parameters
        particle_dims = len(self.kde_params["dims"])

        # Get if we need a particle transformer for this model
        # If not then we will use the identity transformer (aka does not transform)
        if("particle_transformer" in main_model_arch_params):
            particle_transformer_params =  main_model_arch_params["particle_transformer"]
            self.particle_transformer = create_particle_transformer(particle_transformer_params)
        else:
            self.particle_transformer = IdentityTransformer()


        # Create the internal model parameters
        self.create_internal_models(model_params, model_architecture_params, particle_dims)

        self.all_particle_grads = []

    def outputs_kde(self):
        return True

    def outputs_particles_and_weights(self):
        return True

    def outputs_single_solution(self):
        return False

    def create_internal_models(self, model_params, model_architecture_params, particle_dims):

        # All the different networks we can have.  If we dont have a specific network then it is set to None
        # (aka the None value is never overridden)
        self.weight_model = None
        self.weighted_bandwidth_predictor = None
        self.resampling_weighted_bandwidth_predictor = None
        self.proposal_model = None
        self.initializer_model = None
        self.observation_encoder = None
        self.action_encoder_model = None
        self.resampling_weight_model = None

        self.particle_encoder_for_particles_model = None
        self.particle_encoder_for_weights_model = None

        sub_models = model_params["sub_models"]

        # Load the initializer model
        if("initializer_model" in sub_models):
            model_name = sub_models["initializer_model"]
            self.initializer_model = create_initializer_model(model_name, model_architecture_params)
        else:
            print("Need to specify a initializer_model")
            exit()


        # If we are encoding particles then we need a particle encoding model
        if(self.encode_particles):
            if("particle_encoder_for_particles_model" in sub_models):
                model_name = sub_models["particle_encoder_for_particles_model"]
                self.particle_encoder_for_particles_model = create_particle_encoder_model(model_name, model_architecture_params)
            else:
                print("Need to specify a particle encoder model")
                exit()

            if(self.decouple_particle_encoders):
                if("particle_encoder_for_weights_model" in sub_models):
                    model_name = sub_models["particle_encoder_for_weights_model"]
                    self.particle_encoder_for_weights_model = create_particle_encoder_model(model_name, model_architecture_params)
                else:
                    print("Need to specify a particle encoder model")
                    exit()
            else:
                self.particle_encoder_for_weights_model = None

            # If we are encoding then we may need the particle encoder model
            if("action_encoder_model" in sub_models):
                model_name = sub_models["action_encoder_model"]
                self.action_encoder_model = create_particle_encoder_model(model_name, model_architecture_params)
            else:
                self.action_encoder_model = None    
        else:
            self.particle_encoder_for_particles_model = None
            self.action_encoder_model = None

            if("particle_encoder_for_particles_model" in sub_models):
                print("WARNING WARNING WARNING WARNING WARNING WARNING ")
                print("Particle encoder specified BUT flag to encode particles is not set.  Will not be using particle encoder")

            if("action_encoder_model" in sub_models):
                print("WARNING WARNING WARNING WARNING WARNING WARNING ")
                print("Action encoder specified BUT flag to encode particles is not set.  Will not be using action encoder")

        # Load the proposal model
        if("proposal_model" in sub_models):
            model_name = sub_models["proposal_model"]
            self.proposal_model = create_proposal_model(model_name, model_architecture_params)
        else:
            print("Need to specify a proposal_model")
            exit()

        # Load the observation encoder model
        if("observation_model" in sub_models):
            model_name = sub_models["observation_model"]
            self.observation_encoder = create_observation_model(model_name, model_architecture_params)
        else:
            print("Need to specify a observation model")
            exit()


        # Load the weigher model
        if("weight_model" in sub_models):

            # Select which particle encoder to use
            if(self.decouple_particle_encoders):
                encoder_model = self.particle_encoder_for_weights_model
            else:
                encoder_model = self.particle_encoder_for_particles_model

            model_name = sub_models["weight_model"]
            self.weight_model = create_particle_weight_model(model_name, model_architecture_params, self.observation_encoder, encoder_model, self.particle_transformer)
        else:
            print("Need to specify a weight model")
            exit()


        # If we are decoupling the weights model, use the same weight architecture as in the non-decoupled version
        if(self.decouple_weights_for_resampling):
            # Load the weigher model
            if("resampling_weight_model" in sub_models):

                # Select which particle encoder to use
                if(self.decouple_particle_encoders):
                    encoder_model = self.particle_encoder_for_weights_model
                else:
                    encoder_model = self.particle_encoder_for_particles_model


                model_name = sub_models["resampling_weight_model"]
                self.resampling_weight_model = create_particle_weight_model(model_name, model_architecture_params, self.observation_encoder, encoder_model, self.particle_transformer)


                # Check if we have a different weight model input processor
                self.weight_model_input_processors_identical = self.resampling_weight_model.input_processor.is_identical(self.weight_model.input_processor)
            else:
                print("Need to specify a resampling weight model")
                exit()


        # Load the bandwidth predictor model
        if("bandwidth_model" in sub_models):
            model_name = sub_models["bandwidth_model"]
            self.weighted_bandwidth_predictor = create_bandwidth_model(model_name, model_architecture_params)
        else:
            print("Need to specify a bandwidth model")
            exit()

        # If we are decouple the bandwidths for resampling
        if(self.decouple_bandwidths_for_resampling):
            # Load the bandwidth predictor model
            if("resampling_bandwidth_model" in sub_models):
                model_name = sub_models["resampling_bandwidth_model"]
                self.resampling_weighted_bandwidth_predictor = create_bandwidth_model(model_name, model_architecture_params)
            else:
                print("Need to specify a resampling bandwidth model")
                exit()




        # The bandwidth predictor for the KDE
        # self.weighted_bandwidth_predictor = bandwidth_selection_models.BandwidthPredictorNN(particle_dim=particle_dims, output_dim=particle_dims, use_weights=True)

    def freeze_rnn_batchnorm_layers(self):

        # Freeze all models that have batchnorm that are in the rnn path

        if(self.weight_model is not None):
            self.weight_model.freeze_batchnorms()

        if(self.proposal_model is not None):
            self.proposal_model.freeze_batchnorms()

        if(self.initializer_model is not None):
            self.initializer_model.freeze_batchnorms()

        if(self.particle_encoder_for_particles_model is not None):
            self.particle_encoder_for_particles_model.freeze_batchnorms()

        if(self.particle_encoder_for_weights_model is not None):
            self.particle_encoder_for_weights_model.freeze_batchnorms()

        if(self.resampling_weight_model is not None):
            self.resampling_weight_model.freeze_batchnorms()

    def load_pretrained(self, pre_trained_models, device):

        if("dpf_model" in pre_trained_models):
            state_dict = torch.load(pre_trained_models["dpf_model"], map_location=device)

            # If we have a different weight model then we need to convert the weights
            if(self.decouple_weights_for_resampling):
                # Check if we need to convert keys
                convert_keys = True
                for key in state_dict:
                    if("resampling_weight_model" in key):
                        convert_keys = False
                        break

                if(convert_keys):
                    new_state_dict = dict(state_dict)
                    for key in state_dict:
                        if("weight_model" in key):
                            new_state_dict[key.replace("weight_model", "resampling_weight_model")] = state_dict[key].detach().clone()
                    state_dict = new_state_dict


            # If we have a different weight model then we need to convert the weights
            if(self.decouple_bandwidths_for_resampling and self.do_convert_bandwidth_model_to_bandwidths_for_resampling):
                # Check if we need to convert keys
                convert_keys = True
                for key in state_dict:
                    if("resampling_weighted_bandwidth_predictor" in key):
                        convert_keys = False
                        break

                if(convert_keys):
                    new_state_dict = dict(state_dict)
                    for key in state_dict:
                        if("weighted_bandwidth_predictor" in key):
                            new_state_dict[key.replace("weighted_bandwidth_predictor", "resampling_weighted_bandwidth_predictor")] = state_dict[key].detach().clone()
                    state_dict = new_state_dict

            if(self.do_convert_bandwidth_model_to_bandwidths_for_resampling == False):
                self.load_state_dict(state_dict, strict=False)
            else:
                self.load_state_dict(state_dict, strict=True)
            print("loading for \"dpf_model\"")

        if(("particle_encoder_for_particles_model" in pre_trained_models) and (self.particle_encoder_for_particles_model is not None)):
            self.particle_encoder_for_particles_model.load_state_dict(torch.load(pre_trained_models["particle_encoder_for_particles_model"], map_location=device))
        else:
            print("Not loading for \"particle_encoder_for_particles_model\"")

        if(("particle_encoder_for_weights_model" in pre_trained_models) and (self.particle_encoder_for_weights_model is not None)):
            self.particle_encoder_for_weights_model.load_state_dict(torch.load(pre_trained_models["particle_encoder_for_weights_model"], map_location=device))
        else:
            print("Not loading for \"particle_encoder_for_weights_model\"")

        if(("proposal_model" in pre_trained_models) and (self.proposal_model is not None)):
            self.proposal_model.load_state_dict(torch.load(pre_trained_models["proposal_model"], map_location=device))
        else:
            print("Not loading for \"proposal_model\"")

        if(("bandwidth_model" in pre_trained_models) and (self.weighted_bandwidth_predictor is not None)):
            data = torch.load(pre_trained_models["bandwidth_model"], map_location=device)

            if(isinstance(data, dict)):
                self.weighted_bandwidth_predictor.load_state_dict(data)
            else:
                self.weighted_bandwidth_predictor = data
        else:
            print("Not loading for \"bandwidth_model\"")

        if(("initializer_model" in pre_trained_models) and (self.initializer_model is not None)):
            self.initializer_model.load_state_dict(torch.load(pre_trained_models["initializer_model"], map_location=device))
        else:
            print("Not loading for \"initializer_model\"")

        if(("observation_model" in pre_trained_models) and (self.observation_encoder is not None)):
            self.observation_encoder.load_state_dict(torch.load(pre_trained_models["observation_model"], map_location=device))
        else:
            print("Not loading for \"observation_encoder\"")

        if(("weight_model" in pre_trained_models) and (self.weight_model is not None)):                
            self.weight_model.load_state_dict(torch.load(pre_trained_models["weight_model"], map_location=device))
        else:
            print("Not loading for \"weight_model\"")

        if(("action_encoder_model" in pre_trained_models) and (self.action_encoder_model is not None)):                
            self.action_encoder_model.load_state_dict(torch.load(pre_trained_models["action_encoder_model"], map_location=device))
        else:
            print("Not loading for \"action_encoder_model\"")

        # Load the same weights for the resampling to start with
        if(self.decouple_weights_for_resampling and (self.resampling_weight_model is not None)):
            if("resampling_weight_model" in pre_trained_models):                

                # Need to convert the names of the weights first
                state_dict = torch.load(pre_trained_models["resampling_weight_model"], map_location=device)
                
                # Check if we need to convert keys
                convert_keys = True
                for key in state_dict:
                    if("resampling_weight_model" in key):
                        convert_keys = False
                        break

                if(convert_keys):

                    new_state_dict = dict()
                    for key in state_dict:
                        new_state_dict[key.replace("weight_model", "resampling_weight_model")] = state_dict[key]

                    self.resampling_weight_model.load_state_dict(new_state_dict)
                else:
                    self.resampling_weight_model.load_state_dict(state_dict)

            else:
                print("Not loading for \"resampling_weight_model\"")

        if(self.decouple_bandwidths_for_resampling and (self.resampling_weighted_bandwidth_predictor is not None)):
            if("resampling_bandwidth_model" in pre_trained_models):                

                # Need to convert the names of the weights first
                state_dict = torch.load(pre_trained_models["resampling_bandwidth_model"], map_location=device)
                
                # Check if we need to convert keys
                convert_keys = True
                for key in state_dict:
                    if("resampling_bandwidth_model" in key):
                        convert_keys = False
                        break

                if(convert_keys):

                    new_state_dict = dict()
                    for key in state_dict:
                        new_state_dict[key.replace("weighted_bandwidth_predictor", "resampling_weighted_bandwidth_predictor")] = state_dict[key]

                    self.resampling_weighted_bandwidth_predictor.load_state_dict(new_state_dict)
                else:
                    self.resampling_weighted_bandwidth_predictor.load_state_dict(state_dict)

            else:
                print("Not loading for \"resampling_bandwidth_model\"")


    def scale_bandwidths_on_init(self, scale_bandwidths_on_init_params):

        if(self.weighted_bandwidth_predictor is not None):
            self.weighted_bandwidth_predictor.scale_bandwidths_on_init(scale_bandwidths_on_init_params)

        if(self.resampling_weighted_bandwidth_predictor is not None):
            self.resampling_weighted_bandwidth_predictor.scale_bandwidths_on_init(scale_bandwidths_on_init_params)


    def create_and_add_optimizers(self, training_params, trainer, training_type):

        if((self.weight_model is not None) and self.weight_model.is_learned_model()):
            if("weight_model_learning_rate" in training_params):
                lr = training_params["weight_model_learning_rate"]
                trainer.add_optimizer_and_lr_scheduler([self.weight_model], ["weight_model"], lr)

        if((self.weighted_bandwidth_predictor is not None) and self.weighted_bandwidth_predictor.is_learned_model()):
            if("bandwidth_model_learning_rate" in training_params):
                lr = training_params["bandwidth_model_learning_rate"]
                trainer.add_optimizer_and_lr_scheduler([self.weighted_bandwidth_predictor], ["weighted_bandwidth_predictor"], lr)

        if((self.proposal_model is not None) and self.proposal_model.is_learned_model()):
            if("proposal_model_learning_rate" in training_params):
                lr = training_params["proposal_model_learning_rate"]
                trainer.add_optimizer_and_lr_scheduler([self.proposal_model], ["proposal_model"], lr)

        if((self.initializer_model is not None) and self.initializer_model.is_learned_model()):
            if("initializer_model_learning_rate" in training_params):
                lr = training_params["initializer_model_learning_rate"]
                trainer.add_optimizer_and_lr_scheduler([self.initializer_model], ["initializer_model"], lr)

        if((self.observation_encoder is not None) and self.observation_encoder.is_learned_model()):
            if("observation_model_learning_rate" in training_params):
                lr = training_params["observation_model_learning_rate"]
                trainer.add_optimizer_and_lr_scheduler([self.observation_encoder], ["observation_encoder"], lr)

        # Make sure we have the models
        if((self.action_encoder_model is not None) and self.action_encoder_model.is_learned_model()):
            if("action_encoder_model_learning_rate" in training_params):
                lr = training_params["action_encoder_model_learning_rate"]
                trainer.add_optimizer_and_lr_scheduler([self.action_encoder_model], ["action_encoder_model"], lr)

        # Make sure we have the models
        if((self.particle_encoder_for_particles_model is not None) and self.particle_encoder_for_particles_model.is_learned_model()):
            if("particle_encoder_for_particles_learning_rate" in training_params):
                lr = training_params["particle_encoder_for_particles_learning_rate"]
                trainer.add_optimizer_and_lr_scheduler([self.particle_encoder_for_particles_model], ["particle_encoder_for_particles_model"], lr)


        # Make sure we have the models
        if((self.particle_encoder_for_weights_model is not None) and self.particle_encoder_for_weights_model.is_learned_model()):
            if("particle_encoder_for_weights_learning_rate" in training_params):
                lr = training_params["particle_encoder_for_weights_learning_rate"]
                trainer.add_optimizer_and_lr_scheduler([self.particle_encoder_for_weights_model], ["particle_encoder_for_weights_model"], lr)

        # Make sure we have the models
        if((self.resampling_weight_model is not None) and self.resampling_weight_model.is_learned_model()):
            if("resampling_weight_model_learning_rate" in training_params):
                lr = training_params["resampling_weight_model_learning_rate"]
                trainer.add_optimizer_and_lr_scheduler([self.resampling_weight_model], ["resampling_weight_model"], lr)

        # Make sure we have the models
        if((self.resampling_weighted_bandwidth_predictor is not None) and self.resampling_weighted_bandwidth_predictor.is_learned_model()):
            if("resampling_bandwidth_model_learning_rate" in training_params):
                lr = training_params["resampling_bandwidth_model_learning_rate"]
                trainer.add_optimizer_and_lr_scheduler([self.resampling_weighted_bandwidth_predictor], ["resampling_weighted_bandwidth_predictor"], lr)

    def add_models(self, trainer, training_type):

        if(training_type == "full"):
            trainer.add_model(self, "full_dpf_model")
            trainer.add_model_for_training("full_dpf_model")

            if((self.proposal_model is not None) and self.proposal_model.is_learned_model()):
                trainer.add_model(self.proposal_model, "proposal_model")
            
            if((self.weighted_bandwidth_predictor is not None) and self.weighted_bandwidth_predictor.is_learned_model()):
                trainer.add_model(self.weighted_bandwidth_predictor, "weighted_bandwidth_predictor")
            
            if((self.initializer_model is not None) and self.initializer_model.is_learned_model()):
                trainer.add_model(self.initializer_model, "initializer_model")
            
            if((self.observation_encoder is not None) and self.observation_encoder.is_learned_model()):
                trainer.add_model(self.observation_encoder, "observation_encoder")
            
            if((self.weight_model is not None) and self.weight_model.is_learned_model()):
                trainer.add_model(self.weight_model, "weight_model")

            # Make sure we have the models
            if((self.action_encoder_model is not None) and self.action_encoder_model.is_learned_model()):
                trainer.add_model(self.action_encoder_model, "action_encoder_model")

            # Make sure we have the models
            if((self.particle_encoder_for_particles_model is not None) and self.particle_encoder_for_particles_model.is_learned_model()):
                trainer.add_model(self.particle_encoder_for_particles_model, "particle_encoder_for_particles_model")

            # Make sure we have the models
            if((self.particle_encoder_for_weights_model is not None) and self.particle_encoder_for_weights_model.is_learned_model()):
                trainer.add_model(self.particle_encoder_for_weights_model, "particle_encoder_for_weights_model")

            # Make sure we have the models
            if((self.resampling_weight_model is not None) and self.resampling_weight_model.is_learned_model()):
                trainer.add_model(self.resampling_weight_model, "resampling_weight_model")

            # Make sure we have the models
            if((self.resampling_weighted_bandwidth_predictor is not None) and self.resampling_weighted_bandwidth_predictor.is_learned_model()):
                trainer.add_model(self.resampling_weighted_bandwidth_predictor, "resampling_weighted_bandwidth_predictor")


        elif(training_type == "initilizer"):

            if((self.observation_encoder is not None) and self.observation_encoder.is_learned_model()):
                trainer.add_model(self.observation_encoder, "observation_encoder")
            if((self.initializer_model is not None) and self.initializer_model.is_learned_model()):
                trainer.add_model(self.initializer_model, "initializer_model")

        elif(training_type == "proposal"): 

            if((self.proposal_model is not None) and self.proposal_model.is_learned_model()):
                trainer.add_model(self.proposal_model, "proposal_model")

            # Make sure we have the models
            if((self.action_encoder_model is not None) and self.action_encoder_model.is_learned_model()):
                trainer.add_model(self.action_encoder_model, "action_encoder_model")

            # Make sure we have the models
            if((self.particle_encoder_for_particles_model is not None) and self.particle_encoder_for_particles_model.is_learned_model()):
                trainer.add_model(self.particle_encoder_for_particles_model, "particle_encoder_for_particles_model")
        
        elif(training_type == "bandwidth"): 

            if((self.weighted_bandwidth_predictor is not None) and self.weighted_bandwidth_predictor.is_learned_model()):
                trainer.add_model(self.weighted_bandwidth_predictor, "weighted_bandwidth_predictor")

        elif(training_type == "weight"):
            
            if((self.observation_encoder is not None) and self.observation_encoder.is_learned_model()):
                trainer.add_model(self.observation_encoder, "observation_encoder")
            
            if((self.weight_model is not None) and self.weight_model.is_learned_model()):
                trainer.add_model(self.weight_model, "weight_model")

            # Make sure we have the models
            if((self.particle_encoder_for_weights_model is not None) and self.particle_encoder_for_weights_model.is_learned_model()):
                trainer.add_model(self.particle_encoder_for_weights_model, "particle_encoder_for_weights_model")

        else:
            print("Unknown Training Type: {}".format(training_type))
            assert(False)

    def create_initial_dpf_state(self, true_state, observations, number_of_particles):

        # Need the batch size
        batch_size = true_state.shape[0]

        if(self.initilize_with_true_state):
            particles =  true_state.unsqueeze(1)
            particles = torch.tile(particles, [1, number_of_particles, 1])
            particles = self.particle_transformer.downscale(particles)

            # Move to the correct device
            particles = particles.to(observations.device)

            # Add noise to the initial particles
            assert(len(self.initial_position_std) == len(self.kde_params["dims"]))
            for d in range(len(self.initial_position_std)):

                # Get the stats and the dist type
                std = self.initial_position_std[d]
                dim_params = self.kde_params["dims"][d]

                # create the correct dist for this dim
                distribution_type = dim_params["distribution_type"]
                if(distribution_type == "Normal"):
                    dist = D.Normal(loc=torch.zeros_like(particles[..., d]),  scale=std)
                elif(distribution_type == "Von_Mises"):
                    kappa = 1.0 / std
                    dist = VonMisesFullDist(loc=torch.zeros_like(particles[..., d]), concentration=kappa)
                elif(distribution_type == "Epanechnikov"):
                    dist = Epanechnikov(loc=torch.zeros_like(particles[..., d]),  bandwidth=std)
                elif(distribution_type == "CircularEpanechnikov"):
                    dist = CircularEpanechnikov(loc=torch.zeros_like(particles[..., d]),  bandwidth=std)
                else:
                    assert(False)

                # Generate and add some noise
                noise = dist.sample()
                particles[..., d] = particles[..., d] + noise

            # Add the hidden dims
            if(self.number_of_particle_hidden_dims_to_add_to_init_state != 0):
                # create the hidden state
                hidden_state = torch.zeros((particles.shape[0], particles.shape[1], self.number_of_particle_hidden_dims_to_add_to_init_state), device=particles.device)

                # append the hidden state to the particle state
                particles = torch.cat([particles, hidden_state], dim=-1)


        else:

            # Extract the first observation 
            obs = observations[:, 0, :]
            encoded_obs = self.observation_encoder(obs)

            # Create the particles
            particles = self.initializer_model(encoded_obs, number_of_particles)

            # Convert from internal dim to output dim
            particles = self.particle_transformer.forward_tranform(particles)

            # Norm the particles.  Most of the time this wont do anything but sometimes we want to 
            # norm some dims
            particles = self.particle_transformer.apply_norm(particles)

        # Equally weight all the particles since they are all the same
        particle_weights = torch.ones(size=(batch_size, number_of_particles), device=observations.device) / float(number_of_particles)

        if(self.outputs_kde()):
            # Set the bandwidth
            if(self.initilize_with_true_state):
                bandwidths = torch.zeros(size=(particles.shape[0], particles.shape[-1])).to(observations.device)
                bandwidths[...] = 0.001

            else:
                bandwidths = self.weighted_bandwidth_predictor(particles, particle_weights)
        else:
            bandwidths = None

        # Pack everything into an output dict
        output_dict = dict()
        output_dict["particles_downscaled"] = particles
        output_dict["particles"] = self.particle_transformer.upscale(particles)
        output_dict["particle_weights"] = particle_weights
        output_dict["bandwidths_downscaled"] = bandwidths
        output_dict["bandwidths"] = self.particle_transformer.upscale(bandwidths)
        output_dict["importance_gradient_injection"] = torch.ones(size=(batch_size, 1), device=particles.device)

        if(self.decouple_weights_for_resampling):
            output_dict["resampling_particle_weights"] = particle_weights

        if( self.outputs_kde() and self.decouple_bandwidths_for_resampling):
            resampling_bandwidths = self.resampling_weighted_bandwidth_predictor(particles, particle_weights)
            output_dict["resampling_bandwidths_downscaled"] = resampling_bandwidths
            output_dict["resampling_bandwidths"] = self.particle_transformer.upscale(resampling_bandwidths)
        else:
            output_dict["resampling_bandwidths"] = None


        output_dict["do_resample"] = False

        return output_dict

    def resample_particles(self, input_dict):
        ''' Re-sample the particles to generate a new set of particles.
            This can be done in 2 different ways:
                If bandwidths==None:
                    The particles are resampled with replacement using the weight of the particles.
                    This method assumes the particles are not a KDE
                else:
                    The particles are assumed to be a KDE and new particles are generated by sampling the KDE
        '''

        # Unpack the input
        particles = input_dict["particles_downscaled"]
        do_resample = input_dict["do_resample"]

        # Select the weights based on if we are decoupled or not
        if(self.decouple_weights_for_resampling):
            particle_weights = input_dict["resampling_particle_weights"]
        else:
            particle_weights = input_dict["particle_weights"]

        # Select the bandwidths based on if we are decoupled or not
        if(self.decouple_bandwidths_for_resampling):
            bandwidths = input_dict["resampling_bandwidths_downscaled"]
        else:
            bandwidths = input_dict["bandwidths_downscaled"]



        # Extract information about the input
        batch_size = particles.shape[0]
        number_of_particles = particles.shape[1]
        device = particles.device

        if((bandwidths is None) or (not do_resample)):

            # If there are no bandwidths then we just pass the particles through without any changes
            resampled_particles = particles
            resampled_particle_weights = particle_weights

        else:
            

            if(self.use_differentiable_resampling and self.training):

                if(self.differentiable_resampling_method == "ConcreteRelaxation"):
                    
                    # Sample with a differentiable function
                    kde = KernelDensityEstimator(self.kde_params, particles, particle_weights, bandwidths)
                    resampled_particles = kde.sample_concrete_relaxation((particles.shape[1],), self.concrete_relaxation_temperature_parameter)

                    # Generate the new weights which is all ones since we just re-sampled
                    resampled_particle_weights = torch.full(size=(batch_size, number_of_particles), fill_value=(1/float(number_of_particles)), device=device)

                elif(self.differentiable_resampling_method == "ReparameterizationGradients"):

                    # Sample new particles and inject the gradient
                    kde = KernelDensityEstimator(self.kde_params, particles, particle_weights, bandwidths)
                    resampled_particles = kde.sample((particles.shape[1],)).detach()
                    resampled_particles = kde.inject_gradient(resampled_particles)

                    # Generate the new weights 
                    resampled_particle_weights = torch.ones(size=(batch_size, number_of_particles), device=device)

                elif(self.differentiable_resampling_method == "ImportanceSampling"):

                    # Sample new particles and inject the gradient
                    kde = KernelDensityEstimator(self.kde_params, particles, particle_weights, bandwidths)
                    resampled_particles = kde.sample((particles.shape[1],)).detach()
                    resampled_particles = resampled_particles.to(particles.dtype)

                    # Generate the new weights which is all ones since we just re-sampled but inject the weight gradient
                    log_prob = kde.log_prob(resampled_particles)
                    grad_injection = torch.exp(log_prob - log_prob.detach())
                    resampled_particle_weights = grad_injection + 1e-8

                elif(self.differentiable_resampling_method == "ImportanceImplicitHybrid"):

                    # if(particles.requires_grad):
                    #     def print_grad(grad, i):

                    #         # print(torch.norm(grad.detach(), 2.0), i)

                    #         norm = torch.norm(grad.detach(), 2.0)
                    #         self.all_particle_grads.append((norm , i))

                    #         # print(g, i)

                    #     # f = partial(print_grad, 10)

                    #     # particles.register_hook(lambda grad: print("particle", torch.norm(grad.detach(), 2.0)))
                    #     particles.register_hook(partial(print_grad, i=input_dict["timestep_number"]))

                    # Make the different distributions
                    kde_implicit = KernelDensityEstimator(self.kde_params, particles, particle_weights.detach(), bandwidths, validate_args=False)
                    # kde_implicit = KernelDensityEstimator(self.kde_params, particles, particle_weights.detach(), bandwidths.detach(), validate_args=False)
                    kde_importance = KernelDensityEstimator(self.kde_params, particles.detach(), particle_weights, bandwidths.detach(), validate_args=False)

                    # Keep the number of samples constant 
                    number_of_samples = particles.shape[1]

                    # Sample new particles.  Note we can sample from any of the distributions since we detach the samples
                    kde_sampling = kde_implicit
                    resampled_particles = kde_sampling.sample((number_of_samples,)).detach()

                    resampled_particles = resampled_particles.to(particles.dtype)

                    # Inject the implicit reparam gradient
                    resampled_particles = kde_implicit.inject_gradient(resampled_particles)

                    # Generate the new weights which is all ones since we just re-sampled but inject the weight gradient
                    log_prob = kde_importance.log_prob(resampled_particles.detach())
                    grad_injection = torch.exp(log_prob - log_prob.detach())
                    resampled_particle_weights = grad_injection + 1e-8



                elif(self.differentiable_resampling_method == "NoResampling"):
                    resampled_particles = particles
                    resampled_particle_weights = particle_weights

            else:
                kde = KernelDensityEstimator(self.kde_params, particles, particle_weights, bandwidths)
                resampled_particles = kde.sample((particles.shape[1],)).detach()

                # Generate the new weights which is all ones since we just re-sampled
                resampled_particle_weights = torch.full(size=(batch_size, number_of_particles), fill_value=(1/float(number_of_particles)), device=device).detach()

                resampled_particles = resampled_particles.to(particles.dtype)
                resampled_particle_weights = resampled_particle_weights.to(particles.dtype)


        # Norm the particles.  Most of the time this wont do anything but sometimes we want to 
        # norm some dims
        resampled_particles = self.particle_transformer.apply_norm(resampled_particles)

        return resampled_particles, resampled_particle_weights, torch.ones(size=(batch_size, 1), device=particles.device)

    def augment_particles(self, old_particles, actions):

        # Need to transform the particles from output space to internal space
        old_particles = self.particle_transformer.backward_tranform(old_particles)

        if(self.encode_particles):

            # tmp = old_particles.detach()
            tmp = old_particles

            # Encode the old particles
            encoded_old_particles = self.particle_encoder_for_particles_model(tmp)

            # If we have the actions then we will encode those too and add them to the input into the 
            # dynamics model
            if(self.action_encoder_model is not None):
                tiled_actions = torch.tile(actions[:,:].unsqueeze(1),[1,  old_particles.shape[1], 1])
                encoded_actions = self.action_encoder_model(tiled_actions)    
                proposal_model_input = torch.cat([encoded_old_particles, encoded_actions], dim=-1)

            else:
                proposal_model_input = encoded_old_particles            

            # Augment with new particles
            particles, dynamics_weights = self.proposal_model(old_particles, proposal_model_input, None)

        else:

            # Augment with new particles
            particles, dynamics_weights = self.proposal_model(old_particles, old_particles, None)


        # Norm the particles.  Most of the time this wont do anything but sometimes we want to 
        # norm some dims
        particles = self.particle_transformer.apply_norm(particles)

        return particles, dynamics_weights

    def weigh_particles(self, weight_model_input, resampled_particles_weights, dynamics_weights, model, weight_model_processor_input_dict):

        particle_weights = model(weight_model_input)

        # If we need to divide by the weight model then we should do this
        # This will assume that the weight output is in log space 
        if(self.weight_divide_by_proposal_probability):
            assert(False)
            # particle_weights = torch.exp(particle_weights - particles_log_probs)
            # particle_weights = torch.exp(torch.log(particle_weights + 1e-8) - particles_log_probs)        

        if(torch.sum(torch.isnan(particle_weights)) > 0):

            print("")
            print("NAN ERROR 0")
            print("particle_weights", torch.sum(torch.isnan(particle_weights)))
            print("particle_weights", torch.sum(torch.isinf(particle_weights)))
            print("")
            print("")
            print("weight_model_input", torch.sum(torch.isnan(weight_model_input["final_input"])))
            print("weight_model_input", torch.sum(torch.isinf(weight_model_input["final_input"])))

            print("")
            print("")

            internal_particles = weight_model_processor_input_dict["internal_particles"]
            old_particles = weight_model_processor_input_dict["old_particles"]
            print("internal_particles", torch.sum(torch.isnan(internal_particles)))
            print("internal_particles", torch.sum(torch.isinf(internal_particles)))

            print("old_particles", torch.sum(torch.isnan(old_particles)))
            print("old_particles", torch.sum(torch.isinf(old_particles)))

            print("")
            for name, param in model.named_parameters():
                print(name, torch.sum(torch.isnan(param)))
                
            print("")
            for name, param in model.named_parameters():
                print(name, torch.sum(torch.isinf(param)))


            exit()



        if((torch.sum(torch.isnan(particle_weights)) > 0) or (torch.sum(torch.isnan(resampled_particles_weights)) > 0)):
            print("")
            print("NAN ERROR 0.0")
            print(torch.sum(torch.isnan(particle_weights)))
            print(torch.sum(torch.isnan(resampled_particles_weights)))
            exit()


        if((torch.sum(torch.isinf(particle_weights)) > 0) or (torch.sum(torch.isinf(resampled_particles_weights)) > 0)):
            print("")
            print("")
            print("")
            print("")
            print("")
            print("")
            print("INF ERROR 0")
            print(torch.sum(torch.isinf(particle_weights)))
            print(torch.sum(torch.isinf(resampled_particles_weights)))

            internal_particles = weight_model_processor_input_dict["internal_particles"]
            old_particles = weight_model_processor_input_dict["old_particles"]
            print(torch.sum(torch.isinf(internal_particles)))
            print(torch.sum(torch.isinf(old_particles)))

            final_input = weight_model_input["final_input"]
            print(torch.sum(torch.isinf(final_input)))

            exit()



        # Do this so that we can inject importance sampling gradients if we are using them
        particle_weights = particle_weights * resampled_particles_weights * dynamics_weights

        # Normalize the weights
        # norm = torch.sum(particle_weights, dim=1).unsqueeze(1)
        # assert(torch.sum(norm < 1e-8) == 0)
        # particle_weights = particle_weights / (norm + 1e-8)
        particle_weights = torch.nn.functional.normalize(particle_weights, p=1.0, eps=1e-8, dim=1)


        def check_grad(g, name):
            if(torch.sum(torch.isnan(g)) > 0):
                print("{} NOOOOOOOOOO Nan".fomat(name))
                exit()

            if(torch.sum(torch.isinf(g)) > 0):
                print("{} NOOOOOOOOOO inf".fomat(name))
                exit()

        if(particle_weights.requires_grad):
            particle_weights.register_hook(lambda grad: check_grad(grad, "particle_weights"))


        if(torch.sum(torch.isnan(particle_weights)) > 0):
            print("")
            print("NAN ERROR 1")
            print(norm)
            exit()

        if(torch.sum(particle_weights < 0) > 0):
            print("Negative ERROR 1")
            exit()

        if(torch.sum(torch.sum(particle_weights, dim=-1) < 1e-8) > 0):
            print("")
            print("")
            print("")
            print("Zero ERROR")
            print(torch.sum(particle_weights, dim=-1))
            print(norm)
            exit()


        return particle_weights

    def forward(self, input_dict):

        # unpack the inputs
        particles = input_dict["particles_downscaled"]
        particle_weights = input_dict["particle_weights"]
        # bandwidths = input_dict["bandwidths"]
        observation = input_dict["observation"]
        actions = input_dict["actions"]
        reference_patch = input_dict["reference_patch"]
        timestep_number = input_dict["timestep_number"]

        # we might have the next observation, if we do then unpack it
        if("next_observation" in input_dict):
            next_observation = input_dict["next_observation"]
        else:
            next_observation = None

        # we might have a map, if we do then unpack it
        if("world_map" in input_dict):
            world_map = input_dict["world_map"]
        else:
            world_map = None

        # Makes sure that the input is just 1 step in sequence and not a the whole sequence
        assert(len(particles.shape) == 3)

        # Extract information about the input
        batch_size = particles.shape[0]
        
        ############################################################################################# 
        ## Step 1: Re-sample the particles
        ############################################################################################# 
        resampled_particles, resampled_particles_weights, importance_samp_gradients = self.resample_particles(input_dict)

        # Do a copy making sure to detach the computation graph if we are not doing differentiable resampling
        if(self.use_differentiable_resampling):
            particles = resampled_particles
        else:
            particles = resampled_particles.detach()

        ############################################################################################ 
        # Step 2: Augment the particles
        ############################################################################################ 
        
        # Cache the old particles so we can use them in the observation model
        old_particles = particles.clone()

        # Augment the particles
        internal_particles, dynamics_weights = self.augment_particles(old_particles, actions)

        # Convert from internal representation to output representation
        particles = self.particle_transformer.forward_tranform(internal_particles)

        ############################################################################################# 
        ## Step 3: - Weight the particles based on the observation
        ############################################################################################# 

        # Create a dictionary for the inputs
        weight_model_processor_input_dict = dict()
        weight_model_processor_input_dict["observation"] = observation 
        weight_model_processor_input_dict["next_observation"] = next_observation 
        weight_model_processor_input_dict["internal_particles"] = internal_particles 
        weight_model_processor_input_dict["particles"] = particles
        weight_model_processor_input_dict["old_particles"] = old_particles 
        weight_model_processor_input_dict["reference_patch"] = reference_patch
        weight_model_processor_input_dict["world_map"] = world_map




        # weight_model_processor_input_dict["observation"] = observation 
        # weight_model_processor_input_dict["next_observation"] = next_observation 
        # weight_model_processor_input_dict["internal_particles"] = internal_particles 
        # weight_model_processor_input_dict["particles"] = particles.detach()
        # weight_model_processor_input_dict["old_particles"] = old_particles.detach() 
        # weight_model_processor_input_dict["reference_patch"] = reference_patch
        # weight_model_processor_input_dict["world_map"] = world_map


        # Get the pre-processed inputs for the weights model
        weight_model_input = self.weight_model.input_processor.process(weight_model_processor_input_dict)

        # Weigh if we can!
        if(weight_model_input is not None):
            particle_weights = self.weigh_particles(weight_model_input, resampled_particles_weights, dynamics_weights, self.weight_model, weight_model_processor_input_dict)
        else:
            particle_weights = None

        ############################################################################################# 
        ## Step 4: - Compute the decoupled resampling weight if we need to
        ############################################################################################# 

        # If we have enough inputs then do this otherwise we report None for resampling since this is the last step 
        # in the sequence and so we wont be doing any more resampling
        if(self.decouple_weights_for_resampling):

            if(self.weight_model_input_processors_identical == False):
                weight_model_input = self.resampling_weight_model.input_processor.process(weight_model_processor_input_dict)

            if(weight_model_input is not None):
                new_resampling_particle_weights = self.weigh_particles(weight_model_input, resampled_particles_weights, dynamics_weights, self.resampling_weight_model, weight_model_processor_input_dict)
                do_resample = True
            else:
                new_resampling_particle_weights = None    
                do_resample = False
        else:
            new_resampling_particle_weights = None

            # If we are not decoupled and we have weights then resample, otherwise we cant resample
            if(particle_weights is not None):
                do_resample = True
            else:
                do_resample = False

        ############################################################################################# 
        ## Step 5: - Compute the new Bandwidth
        ############################################################################################# 

        # Set the bandwidth
        if(self.outputs_kde()):
            bandwidths = self.weighted_bandwidth_predictor(particles, particle_weights)
        else:
            bandwidths = None

        ############################################################################################# 
        ## Step 6: - Compute the new resampling bandwidth if needed
        ############################################################################################# 

        # Set the resampling bandwidth
        if(self.decouple_bandwidths_for_resampling and do_resample and self.outputs_kde()):

            if(new_resampling_particle_weights is not None):
                resampling_bandwidths = self.resampling_weighted_bandwidth_predictor(particles, new_resampling_particle_weights)    
            else:
                resampling_bandwidths = self.resampling_weighted_bandwidth_predictor(particles, particle_weights)    
        else:
            resampling_bandwidths = None

        # Pack everything in a dict so we can change the outputs as needed
        return_dict = dict()
        return_dict["particles_downscaled"] = particles
        return_dict["particles"] = self.particle_transformer.upscale(particles)
        return_dict["particle_weights"] = particle_weights
        return_dict["bandwidths_downscaled"] = bandwidths
        return_dict["bandwidths"] = self.particle_transformer.upscale(bandwidths)
        return_dict["do_resample"] = do_resample
        return_dict["importance_gradient_injection"] = importance_samp_gradients


        if(self.decouple_weights_for_resampling):
            return_dict["resampling_particle_weights"] = new_resampling_particle_weights
        if(self.decouple_bandwidths_for_resampling):
            return_dict["resampling_bandwidths_downscaled"] = resampling_bandwidths
            return_dict["resampling_bandwidths"] = self.particle_transformer.upscale(resampling_bandwidths)


        return return_dict    




