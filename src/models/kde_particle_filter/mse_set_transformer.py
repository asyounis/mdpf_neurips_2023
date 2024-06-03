# Pytorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

# The bandwidth stuff
from bandwidth_selection.bandwidth_selection_models import * 
from bandwidth_selection.blocks import *

# Project Files
from models.internal_models.internal_model_base import *
from models.kde_particle_filter.kde_particle_filter import *


class SetTransformerVanilla(nn.Module):

    def __init__(self, in_dimension, out_dimension):
        """
        Arguments:
            in_dimension: an integer.
            out_dimension: an integer.
        """
        super().__init__()

        d = 128
        m = 16  # number of inducing points
        h = 4  # number of heads
        k = 4  # number of seed vectors

        self.embed = nn.Sequential(
            nn.Linear(in_dimension, d),
            nn.ReLU(inplace=True)
        )
        self.encoder = nn.Sequential(
            InducedSetAttentionBlock(d, m, h, RFF(d), RFF(d)),
            InducedSetAttentionBlock(d, m, h, RFF(d), RFF(d))
        )
        self.decoder = nn.Sequential(
            PoolingMultiheadAttention(d, k, h, RFF(d)),
            SetAttentionBlock(d, h, RFF(d))
        )
        self.predictor = nn.Linear(k * d, out_dimension)

        def weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        self.apply(weights_init)

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, n, in_dimension].
        Returns:
            a float tensor with shape [b, out_dimension].
        """

        x = self.embed(x)  # shape [b, n, d]
        x = self.encoder(x)  # shape [b, n, d]
        x = self.decoder(x)  # shape [b, k, d]

        b, k, d = x.shape
        x = x.view(b, k * d)
        return self.predictor(x)


class MSESetTransformer(LearnedInternalModelBase):
    def __init__(self, model_parameters):
        super(MSESetTransformer, self).__init__(model_parameters)   

        # Extract the parameters for the model
        assert("particle_dimension" in model_parameters)
        particle_dimension = model_parameters["particle_dimension"]

        # the input dimension is the particles D dim + 1 weight dim + D bandwidth dims
        in_dim = particle_dimension + 1 + particle_dimension

        # create the set transformer.  the output is the same as the particle
        # self.transformer = SetTransformer(in_dim, particle_dimension).float()
        self.transformer = SetTransformerVanilla(in_dim, particle_dimension).float()

    def forward(self, particles, weights, bandwidths):

        # Some stats
        batch_size = particles.shape[0]
        number_of_particles = particles.shape[1]

        # Tile the bandwidths so we have 1 per particle
        tiled_bandwidths = torch.tile(bandwidths.unsqueeze(1), [1, number_of_particles, 1])

        # Make 1 big thing for input into the network
        x = torch.cat([particles, weights.unsqueeze(-1), tiled_bandwidths], dim=-1)

        # Run he set transformer
        out = self.transformer(x)

        return out




class KDEParticleFilterMSESolution(KDEParticleFilter):

    def __init__(self, model_params, model_architecture_params):
        super(KDEParticleFilterMSESolution, self).__init__(model_params, model_architecture_params)

        # Create the set SetTransformer
        sub_models = model_params["sub_models"]
        assert("mse_set_transformer" in sub_models)
        model_name = sub_models["mse_set_transformer"]
        self.mse_set_transformer = MSESetTransformer(model_architecture_params[model_name])


    def outputs_kde(self):
        return True

    def outputs_particles_and_weights(self):
        return True

    def outputs_single_solution(self):
        return True


    def create_and_add_optimizers(self, training_params, trainer, training_type):
        super().create_and_add_optimizers(training_params, trainer, training_type)

        # Add the optimizer if we are gonna train
        if("mse_set_transformer_learning_rate" in training_params):
            lr = training_params["mse_set_transformer_learning_rate"]
            trainer.add_optimizer_and_lr_scheduler([self.mse_set_transformer], ["mse_set_transformer"], lr)

    def add_models(self, trainer, training_type):
        super().add_models(trainer, training_type)

        if(training_type == "full"):
            trainer.add_model(self.mse_set_transformer, "mse_set_transformer")

    def freeze_rnn_batchnorm_layers(self):
        super().freeze_rnn_batchnorm_layers()

        self.mse_set_transformer.freeze_batchnorms()

    def load_pretrained(self, pre_trained_models, device):
        super().load_pretrained(pre_trained_models, device)

        # Load the model!!!
        if("mse_set_transformer" in pre_trained_models):
            self.mse_set_transformer.load_state_dict(torch.load(pre_trained_models["mse_set_transformer"], map_location=device))
        else:
            print("Not loading for \"mse_set_transformer\"")

    def create_initial_dpf_state(self, true_state, observations, number_of_particles):

        output_dict = super().create_initial_dpf_state(true_state, observations, number_of_particles)

        # Create the single solution and add it to the output dict
        output_dict = self.create_single_output_predicted_state(output_dict)

        return output_dict

    def forward(self, input_dict):

        # Run the standard forward for the particle filter
        output_dict = super().forward(input_dict)

        # Create the single solution and add it to the output dict
        output_dict = self.create_single_output_predicted_state(output_dict)

        return output_dict

    def create_single_output_predicted_state(self, output_dict):

        # Unpack from the output dict of the KDE
        particles = output_dict["particles"]
        weights = output_dict["particle_weights"]
        bandwidths = output_dict["bandwidths"]

        # transform 
        particles_transformed = self.particle_transformer.backward_tranform(particles)
        bandwidths_transformed = self.particle_transformer.backward_tranform(bandwidths)

        # Compute the predicted single state
        predicted_state = self.mse_set_transformer(particles_transformed, weights, bandwidths_transformed)

        # transform
        predicted_state = self.particle_transformer.forward_tranform(predicted_state)

        # Append to the output_dict dict
        output_dict["predicted_state"] = predicted_state

        return output_dict
