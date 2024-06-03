# Standard Imports
import numpy as np
from tqdm import tqdm
import math

# Pytorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

# Project Includes
from bandwidth_selection import blocks

# import sys
# sys.path.append('../')
# from kernel_density_estimation import kernel_density_estimation as kde

class BandwidthPredictorNNDeepSets(nn.Module):
    def __init__(self, particle_dim=1, output_dim=1, use_weights=False):
        super(BandwidthPredictorNNDeepSets, self).__init__()


        # Save some stuff we will need later
        self.use_weights = use_weights
        
        if(self.use_weights):
            in_dim = particle_dim + 1
        else:
            in_dim = particle_dim

        LATENT_SPACE = 128

        self.encoder = nn.Sequential(
            nn.Linear(in_features=in_dim,out_features=LATENT_SPACE),
            nn.PReLU(),
            nn.Linear(in_features=LATENT_SPACE, out_features=LATENT_SPACE),
            nn.PReLU(),
            nn.Linear(in_features=LATENT_SPACE, out_features=LATENT_SPACE),
            nn.PReLU(),
            nn.Linear(in_features=LATENT_SPACE, out_features=LATENT_SPACE),
            nn.PReLU(),
            nn.Linear(in_features=LATENT_SPACE, out_features=LATENT_SPACE),
            nn.PReLU(),
        )

        self.regressor = nn.Sequential(
            nn.Linear(in_features=LATENT_SPACE,out_features=LATENT_SPACE),
            nn.PReLU(),
            nn.BatchNorm1d(LATENT_SPACE),
            nn.Linear(in_features=LATENT_SPACE, out_features=LATENT_SPACE),
            nn.PReLU(),
            nn.BatchNorm1d(LATENT_SPACE),
            nn.Linear(in_features=LATENT_SPACE, out_features=LATENT_SPACE),
            nn.PReLU(),
            nn.BatchNorm1d(LATENT_SPACE),
            nn.Linear(in_features=LATENT_SPACE, out_features=LATENT_SPACE),
            nn.PReLU(),
            nn.BatchNorm1d(LATENT_SPACE),
            nn.Linear(in_features=LATENT_SPACE, out_features=output_dim),
            nn.ReLU(),
        )


        # self.n_hidden_sets = LATENT_SPACE
        # self.n_elements = 100
        # self.Wc = Parameter(torch.FloatTensor(LATENT_SPACE, self.n_hidden_sets*self.n_elements))
        # self.Wc.data.uniform_(-1, 1)

    def forward(self, particles, weights):

        if(self.use_weights):
            x = torch.cat([particles, weights.unsqueeze(-1)], dim=-1)
        else:
            x = particles


        # Encode the particles
        encoded_particles = self.encoder(x)

        # encoded_particles = torch.sum(encoded_particles, dim=1)
        encoded_particles = torch.mean(encoded_particles, dim=1)


        # t = F.relu(torch.matmul(encoded_particles, self.Wc))
        # t = t.view(t.size()[0], t.size()[1], self.n_elements, self.n_hidden_sets)
        # t,_ = torch.max(t, dim=2)
        # t = torch.sum(t, dim=1)


        # Regress now!
        regression_output = self.regressor(encoded_particles)
        # regression_output = self.regressor(t)

        # regression_output = torch.abs(regression_output)

        # return regression_output
        return regression_output + 0.00001


class BandwidthPredictorNNHall(nn.Module):
    def __init__(self, particle_dim=1, output_dim=1):
        super(BandwidthPredictorNN, self).__init__()

    def forward(self, particles, weights):

        # Extract some info about the inputs
        batch_size = particles.shape[0]
        number_of_particles = particles.shape[1]
        number_of_dims = particles.shape[2]

        # Compute the standard deviation of the particles
        std = torch.std(particles, dim=1)

        # Compute the pilot distribution bandwidth
        pilot_bandwidth = 1.0592 * std * math.pow(number_of_particles, -1/(4+number_of_dims))

        # compute the delta_x
        x1 = torch.tile(particles.unsqueeze(-2),[1,1, number_of_particles, 1])
        x2 = torch.tile(particles.unsqueeze(1),[1,number_of_particles, 1, 1])
        delta_x = x2 - x1
        delta_x = delta_x / pilot_bandwidth.unsqueeze(1).unsqueeze(1)

        # for i in range(number_of_particles):
            # delta_x[:,i,i,:] = 2e16
 

        # Flatten Delta X
        delta_x = delta_x.view(batch_size, number_of_particles*number_of_particles, number_of_dims)

        # print(delta_x)

        # Find I2 and I3
        I2 = self.find_I2(number_of_particles, delta_x, pilot_bandwidth)
        I3 = self.find_I3(number_of_particles, delta_x, pilot_bandwidth)


        # Some Constants taken from Alex Ihlers code (for Guassians)
        RK = 0.282095
        MU2 = 1.000000
        MU4 = 3.000000; 

        J1 = RK / ((MU2**2) * I2)
        J2 = MU4*I3 / (20 * MU2 * I2)


        # print(torch.sum(torch.isnan(J1)))
        # print(torch.sum(torch.isnan(J2)))


        bandwidth1 = J1 / number_of_particles
        bandwidth1 = torch.sign(bandwidth1) * torch.pow(torch.abs(bandwidth1), 0.2)


        bandwidth2 = J1 / number_of_particles
        bandwidth2 = torch.sign(bandwidth2) * torch.pow(torch.abs(bandwidth2), 0.6)
        bandwidth2 *= J2



        bandwidth = bandwidth1 + bandwidth2

        print(bandwidth)


        # print(torch.sum(torch.isnan(bandwidth)))
        exit()



    def find_I2(self, number_of_particles, delta_x, alpha):

        # Compute the L^(4) sum
        sum_output = torch.sum(-0.5 * (delta_x**2), dim=-1)
        sum_output = torch.exp(sum_output)



        sum_output = torch.tile(sum_output.unsqueeze(-1),[1, 1, delta_x.shape[-1]])
        sum_output *= (delta_x**2 -1)

        print(sum_output[0])


        sum_output *= 1.0 / np.sqrt(2.0 * np.pi)
        sum_output = torch.sum(sum_output, dim=1)

        # Put it all together to get I2
        I2 = sum_output / (alpha**5)
        I2 /=  number_of_particles * (number_of_particles - 1)

        print(I2)
        exit()

        return I2

    def find_I3(self, number_of_particles, delta_x, beta):
        # Compute the L^(4) sum
        sum_output = torch.sum(-0.5 * (delta_x**2), dim=-1)
        sum_output = torch.exp(sum_output)
        sum_output = torch.tile(sum_output.unsqueeze(-1),[1, 1, delta_x.shape[-1]])
        sum_output *= (delta_x**3) - (3 * delta_x)
        sum_output *= 1.0 / np.sqrt(2.0 * np.pi)
        sum_output = torch.sum(sum_output, dim=1)

        # Put it all together to get I2
        I3 = -sum_output / (beta**7)
        I3 /=  number_of_particles * (number_of_particles - 1)

        return I3


# class SetTransformer(nn.Module):

#     def __init__(self, in_dimension, out_dimension):
#         """
#         Arguments:
#             in_dimension: an integer.
#             out_dimension: an integer.
#         """
#         super().__init__()

#         d = 128
#         # m = 16  # number of inducing points
#         h = 1  # number of heads
#         k = 1  # number of seed vectors

#         # m = 100  # number of inducing points
#         # h = 16  # number of heads
#         # k = 32  # number of seed vectors


#         self.embed = nn.Sequential(
#             nn.Linear(in_dimension, d),
#             nn.ReLU(inplace=True),
#             nn.Linear(d, d),
#             nn.ReLU(inplace=True),
#             nn.Linear(d, d),
#             nn.ReLU(inplace=True),
#         )
#         # self.encoder = nn.Sequential(
#         #     blocks.InducedSetAttentionBlock(d, m, h, RFF(d), RFF(d)),
#         #     blocks.InducedSetAttentionBlock(d, m, h, RFF(d), RFF(d)),
#         #     # blocks.InducedSetAttentionBlock(d, m, h, RFF(d), RFF(d)),
#         #     # blocks.InducedSetAttentionBlock(d, m, h, RFF(d), RFF(d)),
#         # )


#         self.encoder = nn.Sequential(
#             blocks.SetAttentionBlock(d, h, RFF(d)),
#             blocks.SetAttentionBlock(d, h, RFF(d)),
#             blocks.SetAttentionBlock(d, h, RFF(d)),
#         )

#         self.decoder = nn.Sequential(
#             blocks.PoolingMultiheadAttention(d, k, h, RFF(d)),
#             blocks.SetAttentionBlock(d, h, RFF(d)),
#             blocks.SetAttentionBlock(d, h, RFF(d)),
#             blocks.SetAttentionBlock(d, h, RFF(d)),
#         )
#         # self.predictor = nn.Linear(k * d, out_dimension)

#         self.predictor = nn.Sequential(
#             # nn.Linear(k * d, out_dimension),
#             nn.Linear(k * d, 128),
#             nn.ReLU(inplace=True),
#             nn.Linear(128, 128),
            

#             # nn.ReLU(inplace=True),
#             # nn.Linear(128, 128),
#             # nn.ReLU(inplace=True),
#             # nn.Linear(128, 128),


#             nn.ReLU(inplace=True),
#             nn.Linear(128, out_dimension),
#             # nn.Sigmoid()
#         )


#         def weights_init(m):
#             if isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal_(m.weight)
#                 nn.init.zeros_(m.bias)
#             elif isinstance(m, nn.LayerNorm):
#                 nn.init.ones_(m.weight)
#                 nn.init.zeros_(m.bias)

#         # self.apply(weights_init)

#     def forward(self, x):
#         """
#         Arguments:
#             x: a float tensor with shape [b, n, in_dimension].
#         Returns:
#             a float tensor with shape [b, out_dimension].
#         """

#         x = self.embed(x)  # shape [b, n, d]
#         x = self.encoder(x)  # shape [b, n, d]
#         x = self.decoder(x)  # shape [b, k, d]

#         b, k, d = x.shape
#         x = x.view(b, k * d)
#         return self.predictor(x)


class SetTransformer(nn.Module):

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
            blocks.InducedSetAttentionBlock(d, m, h, RFF(d), RFF(d)),
            blocks.InducedSetAttentionBlock(d, m, h, RFF(d), RFF(d))
        )
        self.decoder = nn.Sequential(
            blocks.PoolingMultiheadAttention(d, k, h, RFF(d)),
            blocks.SetAttentionBlock(d, h, RFF(d))
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



class RFF(nn.Module):
    """
    Row-wise FeedForward layers.
    """
    def __init__(self, d):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(d, d), 
            nn.ReLU(inplace=True),
            nn.Linear(d, d), 
            nn.ReLU(inplace=True),
            nn.Linear(d, d), 
            nn.ReLU(inplace=True),
            nn.Linear(d, d), 
        )

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, n, d].
        Returns:
            a float tensor with shape [b, n, d].
        """
        return self.layers(x)


class BandwidthPredictorNN(nn.Module):
    def __init__(self, particle_dim=1, output_dim=1, use_weights=False):
        super(BandwidthPredictorNN, self).__init__()

        self.use_weights = use_weights

        if(self.use_weights):
            in_dim = particle_dim + 1
        else:
            in_dim = particle_dim
            
        self.transformer = SetTransformer(in_dim, output_dim).float()
        # self.transformer = SetTransformer(in_dim, output_dim).double()

    def forward(self, particles, weights=None):

        # If we are using weights then make sure that the weights are set
        # otherwise the size of the input will not match the expected size in the first layer
        if(self.use_weights):
            assert(weights is not None)

        # Scale the particles to have 0 mean, unit std
        # particle_mean = torch.mean(particles.view(-1,particles.shape[-1]), dim=0)
        # particle_std = torch.std(particles.view(-1,particles.shape[-1]), dim=0)
        # scaled_particles = (particles - particle_mean) / particle_std

        # Compute the median of each particle set
        median, _ = torch.median(particles, dim=1, keepdim=True)
        median = median.detach()

        # Compute the interquartile range of each particle set
        q1 = torch.quantile(particles.detach(), 0.25, dim=1, keepdim=True).detach()
        q3 = torch.quantile(particles.detach(), 0.75, dim=1, keepdim=True).detach()
        iqr = q3 - q1

        # Scale the particles via a robust scaling and centering method in case we have outliers
        scaled_particles = (particles - median) / iqr

        # # Switch based on if we are using weights or not
        # if(self.use_weights):
        #     x = torch.cat([particles, weights.unsqueeze(-1)], dim=-1)
        # else:
        #     x = particles

        # Switch based on if we are using weights or not
        if(self.use_weights):
            x = torch.cat([scaled_particles, weights.unsqueeze(-1)], dim=-1)
        else:
            x = scaled_particles

        # Pass into the set transformer and compute the log output
        out = self.transformer(x)

        # Since it was the log, exp it to get the actual output
        out = torch.exp(out)

        # re-scale the output to match the expected scale
        iqr = torch.reshape(iqr,(iqr.shape[0], iqr.shape[2]))
        out = out * iqr

        return out + 0.000001



class BandwidthPredictorNNSoftplus(nn.Module):
    def __init__(self, particle_dim=1, output_dim=1, use_weights=False):
        super(BandwidthPredictorNNSoftplus, self).__init__()

        self.use_weights = use_weights

        if(self.use_weights):
            in_dim = particle_dim + 1
        else:
            in_dim = particle_dim
            
        self.transformer = SetTransformer(in_dim, output_dim).float()

        self.softplus = nn.Softplus()

    def forward(self, particles, weights=None):

        # If we are using weights then make sure that the weights are set
        # otherwise the size of the input will not match the expected size in the first layer
        if(self.use_weights):
            assert(weights is not None)

        # Scale the particles to have 0 mean, unit std
        # particle_mean = torch.mean(particles.view(-1,particles.shape[-1]), dim=0)
        # particle_std = torch.std(particles.view(-1,particles.shape[-1]), dim=0)
        # scaled_particles = (particles - particle_mean) / particle_std

        # Compute the median of each particle set
        median, _ = torch.median(particles, dim=1, keepdim=True)
        median = median.detach()

        # Compute the interquartile range of each particle set
        q1 = torch.quantile(particles.detach(), 0.25, dim=1, keepdim=True).detach()
        q3 = torch.quantile(particles.detach(), 0.75, dim=1, keepdim=True).detach()
        iqr = q3 - q1
        
        # Scale the particles via a robust scaling and centering method in case we have outliers
        # scaled_particles = (particles - median) / iqr
        scaled_particles = (particles - median)

        # Switch based on if we are using weights or not
        if(self.use_weights):
            x = torch.cat([scaled_particles, weights.unsqueeze(-1)], dim=-1)
        else:
            x = scaled_particles

        # Pass into the set transformer and compute the log output
        out = self.transformer(x)

        # Since it was the log, exp it to get the actual output
        out = self.softplus(out)
        
        # re-scale the output to match the expected scale
        iqr = torch.reshape(iqr,(iqr.shape[0], iqr.shape[2]))
        # out = out * iqr

        # print(out)

        return out + 0.000001



# class BandwidthPredictorNNPlugIn(nn.Module):
#     def __init__(self, particle_dim=1, output_dim=1, use_weights=False):
#         super(BandwidthPredictorNNPlugIn, self).__init__()

#         self.use_weights = use_weights

#         if(use_weights):
#             self.transformer_1 = SetTransformer(particle_dim+1, output_dim)
#             self.transformer_2 = SetTransformer(particle_dim+2, output_dim)
#         else:
#             self.transformer_1 = SetTransformer(particle_dim, output_dim)
#             self.transformer_2 = SetTransformer(particle_dim+1, output_dim)

#     def forward(self, particles, weights):

#         if(self.use_weights):
#             x = torch.cat([particles, weights.unsqueeze(-1)], dim=-1)
#         else:
#             x = particles

#         # Compute the bandwidths
#         bandwidths = self.transformer_1(x)
#         bandwidths = torch.exp(bandwidths)
#         bandwidths = bandwidths + 0.01

#         # Compute the KDE
#         kde_estimate = kde.compute_weighted_kde(particles, particles, weights, bandwidths)

#         if(self.use_weights):
#             combined = torch.cat([particles,weights.unsqueeze(-1),  kde_estimate.unsqueeze(-1)], dim=-1)
#         else:
#             combined = torch.cat([particles, kde_estimate.unsqueeze(-1)], dim=-1)

#         # Compute the next stage of the bandwidth predictor
#         out = self.transformer_2(combined)
#         out = torch.exp(out) + 0.01

#         return out


class BandwidthPredictorSetBandwidths(nn.Module):
    def __init__(self, particle_dim, output_dim, use_weights=False):
        """
        Arguments:
            particle_dim: an integer.
            output_dim: an integer.
        """
        super().__init__()



        self.use_weights = use_weights

        if(self.use_weights):
            in_dim = particle_dim + 1
        else:
            in_dim = particle_dim

        d = 128
        m = 100  # number of inducing points
        h = 16  # number of heads
        k = 32  # number of seed vectors

        self.embed = nn.Sequential(
            nn.Linear(in_dim, d),
            nn.ReLU(inplace=True),
            nn.Linear(d, d),
            nn.ReLU(inplace=True),
            nn.Linear(d, d),
            nn.ReLU(inplace=True),
        )

        self.predictor = nn.Sequential(
            nn.Linear(d, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, output_dim),
        )

        self.encoder = nn.Sequential(
            blocks.InducedSetAttentionBlock(d, m, h, RFF(d), RFF(d)),
            blocks.InducedSetAttentionBlock(d, m, h, RFF(d), RFF(d)),
            blocks.InducedSetAttentionBlock(d, m, h, RFF(d), RFF(d)),
            blocks.InducedSetAttentionBlock(d, m, h, RFF(d), RFF(d)),
        )

        def weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        self.apply(weights_init)

    def forward(self, particles, weights):
        """
        Arguments:
            x: a float tensor with shape [b, n, particle_dim].
        Returns:
            a float tensor with shape [b, output_dim].
        """

        # print(x.shape)

        if(self.use_weights):
            x = torch.cat([particles, weights.unsqueeze(-1)], dim=-1)
        else:
            x = particles

        x = self.embed(x)  # shape [b, n, d]
        # print(x.shape)

        x = self.encoder(x)  # shape [b, n, d]
        # print(x.shape)

        x = self.predictor(x)
        # print(x.shape)

        out = torch.exp(x) + 0.01

        return out




