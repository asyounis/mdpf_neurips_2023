# Standard Imports
import numpy as np
import os
import PIL
import math
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

# Pytorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D



class SequentialModels(nn.Module):
    def __init__(self, model_params, model_architecture_params):
        super(SequentialModels, self).__init__()

        # Extract the name of the model so we can get its model architecture params
        main_model_name = model_params["main_model"]
        main_model_arch_params = model_architecture_params[main_model_name]

        # If we are set to init with the true state at the start
        if("initilize_with_true_state" in main_model_arch_params):
            self.initilize_with_true_state = main_model_arch_params["initilize_with_true_state"]

            assert("initial_position_std" in main_model_arch_params)
            self.initial_position_std = main_model_arch_params["initial_position_std"]

        else:
            self.initilize_with_true_state = False

        


        # Note all models has this so we should set it to false and 
        # The specific models can override this if needed
        self.decouple_weights_for_resampling = False
        self.decouple_bandwidths_for_resampling = False

    def outputs_kde(self):
        raise NotImplemented

    def outputs_particles_and_weights(self):
        raise NotImplemented

    def outputs_single_solution(self):
        raise NotImplemented

    def create_initial_dpf_state(self, true_state, observations, number_of_particles):
        raise NotImplemented

    def forward(self, input_dict):
        raise NotImplemented

    def create_and_add_optimizers(self, training_params, trainer, training_type):
        raise NotImplemented

    def add_models(self, trainer, training_type):
        raise NotImplemented

    def load_pretrained(self, pre_trained_models, device):
        raise NotImplemented
        
    def freeze_rnn_batchnorm_layers(self):
        # Do nothing
        pass
        
    def scale_bandwidths_on_init(self, scale_bandwidths_on_init_params):
        pass 