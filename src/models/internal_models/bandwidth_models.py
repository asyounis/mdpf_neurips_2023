
# Suppress annoying warnings
import warnings
warnings.filterwarnings('ignore', 'The iteration is not making good progress')


# Imports
from functools import partial
import scipy
import numpy as np


# Pytorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

# The bandwidth stuff
from bandwidth_selection import bandwidth_selection_models
from bandwidth_selection import blocks
from kernel_density_estimation.kde_computer import *
from kernel_density_estimation.kernel_density_estimator import *

# Project Files
from models.internal_models.internal_model_base import *


class FixedBandwith(LearnedInternalModelBase):
    def __init__(self, model_parameters):
        super(FixedBandwith, self).__init__(model_parameters)
        
        # Parse and pytorch-ize the starting bandwidths from the user
        starting_bandwidths = model_parameters["starting_bandwidths"]
        log_bands = torch.FloatTensor(np.log(starting_bandwidths))

        # Parse and pytorch-ize min bandwidths from the user if they are present
        if("min_bandwidths" in model_parameters):
            min_bandwidths = model_parameters["min_bandwidths"]
            self.min_bandwidths = torch.FloatTensor(min_bandwidths)
        else:
            self.min_bandwidths = None


        # Parse and pytorch-ize max bandwidths from the user if they are present
        if("max_bandwidths" in model_parameters):
            max_bandwidths = model_parameters["max_bandwidths"]
            self.max_bandwidths = torch.FloatTensor(max_bandwidths)
        else:
            self.max_bandwidths = None


        # Parse and pytorch-ize max bandwidths from the user if they are present
        if("max_bandwidths_assert" in model_parameters):
            self.max_bandwidths_assert = model_parameters["max_bandwidths_assert"]
        else:
            self.max_bandwidths_assert = False


        # Put into a pytorch item that can "learn"
        self.log_bandwidths = nn.parameter.Parameter(log_bands)


    # def forward(self, particles, weights):

    #     # Make sure the state dimensions match the dimentions of the particles we have
    #     assert(particles.shape[-1] == self.log_bandwidths.shape[-1])

    #     # Tile the log_bandwidths (aka copy the log_bandwidths so we have 1 bandwidth per particle)
    #     return_bands = torch.tile(self.log_bandwidths.unsqueeze(0), (particles.shape[0], 1))

    #     # Return the exponential to make sure the bandwidth
    #     return_bands = torch.exp(return_bands)

    #     # Apply the min bandwidth if needed
    #     if(self.min_bandwidths is not None):

    #         # Move to the correct device
    #         if(self.min_bandwidths.device != return_bands.device):
    #             self.min_bandwidths = self.min_bandwidths.to(return_bands.device)

    #         # Apply it
    #         return_bands = return_bands + self.min_bandwidths


    #     # # Apply the min bandwidth if needed
    #     # if(self.max_bandwidths is not None):

    #     #     # Move to the correct device
    #     #     if(self.max_bandwidths.device != return_bands.device):
    #     #         self.max_bandwidths = self.max_bandwidths.to(return_bands.device)

    #     #     # Apply it
    #     #     return_bands = return_bands + self.max_bandwidths


    #     return return_bands



    def forward(self, particles, weights):

        # Make sure the state dimensions match the dimentions of the particles we have
        assert(particles.shape[-1] == self.log_bandwidths.shape[-1])


        # Return the exponential to make sure the bandwidth
        return_bands = torch.exp(self.log_bandwidths)

        # Apply the min bandwidth if needed
        if(self.min_bandwidths is not None):

            # Move to the correct device
            if(self.min_bandwidths.device != return_bands.device):
                self.min_bandwidths = self.min_bandwidths.to(return_bands.device)

            # Apply it
            return_bands = return_bands + self.min_bandwidths


        # Apply the min bandwidth if needed
        if(self.max_bandwidths is not None):

            # Move to the correct device
            if(self.max_bandwidths.device != return_bands.device):
                self.max_bandwidths = self.max_bandwidths.to(return_bands.device)

            for i in range(self.max_bandwidths.shape[0]):
                if(return_bands[i] > self.max_bandwidths[i]):
                    return_bands[i] = self.max_bandwidths[i]

                    if(self.max_bandwidths_assert):
                        assert(False)



        # Tile the log_bandwidths (aka copy the log_bandwidths so we have 1 bandwidth per particle)
        return_bands = torch.tile(return_bands, (particles.shape[0], 1))

        return return_bands


    def scale_bandwidths_on_init(self, scale_bandwidths_on_init_params):

        print("Scaling Bandwidths...")

        for dim in scale_bandwidths_on_init_params["dims"]:
            params = scale_bandwidths_on_init_params["dims"][dim]

            # Get the scaling type
            scaling_type = params["scaling_type"]

            # Scale!
            if(scaling_type == "linear"):

                # Get the linear scaling parameters
                if("scaling_coefficient" in params):
                    scaling_coefficient = params["scaling_coefficient"]
                else:
                    scaling_coefficient = 1.0

                # Get the linear offset parameters
                if("offset" in params):
                    offset = params["offset"]
                else:
                    offset = 0.0

                with torch.no_grad():
                    current_band = torch.exp(self.log_bandwidths[dim])
                    self.log_bandwidths[dim] = torch.log((scaling_coefficient * current_band) + offset)

            elif(scaling_type == "none"):
                pass

            else:
                print("Unknown Scaling Type")
                assert(False)

class RuleOfThumbBandwidth(NonLearnedInternalModelBase):
    def __init__(self, model_parameters):
        super(NonLearnedInternalModelBase, self).__init__(model_parameters)

        # Load the KDE parameters
        kde_params = model_parameters["kde"]
        self.distribution_types = []
        for d in kde_params["dims"]:
            dim_params = kde_params["dims"][d]
            distribution_type = dim_params["distribution_type"]
            self.distribution_types.append(distribution_type)


    def forward(self, particles, particle_weights):

        # Make sure the dims of the particles is what we expect
        assert(particles.shape[-1] == len(self.distribution_types))

        # Norm the weights in case they are not already normed
        particle_weights = torch.nn.functional.normalize(particle_weights, p=1.0, eps=1e-8, dim=1)

        # Get the bandwidth for each particle set one at a time
        bandwidths = torch.zeros((particles.shape[0], particles.shape[-1]))
        for d, distribution_type in enumerate(self.distribution_types):

            if(distribution_type == "Normal"):
                bandwidths[..., d] = self._compute_bandwidths_normal(particles[..., d], particle_weights)
                
            elif(distribution_type == "Von_Mises"):
                bandwidths[..., d] = self._compute_bandwidths_von_mises(particles[..., d], particle_weights)

            else:
                print("Unknown distribution type \"{}\"").format(distribution_type)
                assert(False)


        # Move the bandwidths to the correct device
        bandwidths = bandwidths.to(particles.device)

        return bandwidths


    def _compute_bandwidths_normal(self, particles, particle_weights):

        # Extract Info
        batch_size = particles.shape[0]
        number_of_particles = particles.shape[1]

        # Estimate the weighted STD in an unbiased way: https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_variance
        weigthed_mean = torch.sum(particles*particle_weights, dim=-1)
        estimated_std = torch.sum(particle_weights * ((particles - weigthed_mean.unsqueeze(-1))**2), dim=-1)
        estimated_std = torch.sqrt(estimated_std)
        
        # Compute the rule of thumb bandwdith
        # Using silvermans method from: https://aakinshin.net/posts/kde-bw/
        iqr = torch.quantile(particles, 0.75, dim=-1) - torch.quantile(particles, 0.25, dim=-1)
        rot_bandwidth = 0.9 * torch.minimum(estimated_std, iqr / 1.35) * (float(number_of_particles) ** (-1.0/5.0)) 

        return rot_bandwidth


    def _compute_bandwidths_von_mises(self, particles, particle_weights):
        
        def _bw_taylor(x, kappa):
            """Taylor's rule for circular bandwidth estimation.
            This function implements a rule-of-thumb for choosing the bandwidth of a von Mises kernel
            density estimator that assumes the underlying distribution is von Mises as introduced in [1]_.
            It is analogous to Scott's rule for the Gaussian KDE.
            Circular bandwidth has a different scale from linear bandwidth. Unlike linear scale, low
            bandwidths are associated with oversmoothing and high values with undersmoothing.
            References
            ----------
            .. [1] C.C Taylor (2008). Automatic bandwidth selection for circular
                   density estimation.
                   Computational Statistics and Data Analysis, 52, 7, 3493–3500.
            """

            # Need to do this in scipy because pytorch doesnt have these values
            kappa_np = kappa.detach().cpu().numpy()
            ive2 = scipy.special.ive(2, 2 * kappa_np)
            ive0 = scipy.special.ive(2, kappa_np)
            
            x_len = x.shape[1]

            num = 3 * x_len * kappa**2 * ive2
            den = 4 * np.pi**0.5 * (ive0** 2)
            return (num / den) ** 0.4

        def func_to_solve(kappa, precomputed_A_K_sum, K):
            
            # Compute I_k(kappa) / I_0(kappa)
            A_K_bessel = scipy.special.ive(K, kappa) / scipy.special.ive(0, kappa)

            # Compute diff
            diff = A_K_bessel - precomputed_A_K_sum

            # print("bessel", A_K_bessel)
            # print("sum", precomputed_A_K_sum)

            return diff

        # Using https://www.sciencedirect.com/science/article/pii/S0167947307004367
        # equation 8
        # Other references:
        #   - "Estimating Overlap of Daily Activity Patterns From Camera Trap Data": https://link.springer.com/content/pdf/10.1198/jabes.2009.08038.pdf?pdf=button
        #   - "topics in circular statistics": https://books.google.com/books?hl=en&lr=&id=6kVqDQAAQBAJ&oi=fnd&pg=PR5&dq=topics+in+circular+statistics&ots=cWS-o0rvUB&sig=Uz3dbxSgC-cY7wgKsgU52YXQ9LM#v=onepage&q=topics%20in%20circular%20statistics&f=false

        # Compute this many kappas
        MAX_K = 10

        # Extract Info
        batch_size = particles.shape[0]
        number_of_particles = particles.shape[1]

        #############################################################
        # Pre_compute some stuff
        #############################################################

        # Create a K matrix
        K_mat = torch.arange(MAX_K) + 1
        K_mat = K_mat.to(particles.device).float()
        K_mat = torch.tile(K_mat.unsqueeze(1).unsqueeze(1),(1, batch_size, number_of_particles))

        # This makes the computation a but easier
        mul = K_mat * particles.unsqueeze(0)

        # Compute the mean
        cos = torch.sum(torch.cos(mul) * particle_weights.unsqueeze(0), dim=-1)
        sin = torch.sum(torch.sin(mul) * particle_weights.unsqueeze(0), dim=-1)
        mean = torch.atan2(sin, cos)

        # Compute the A sum
        A_K_sum = mul - mean.unsqueeze(-1)
        A_K_sum = torch.cos(A_K_sum)
        A_K_sum = torch.sum(A_K_sum * particle_weights.unsqueeze(0), dim=-1) # this keeps with the importance weighted sampling idea
        # A_K_sum = torch.sum(A_K_sum, dim=-1)
        # A_K_sum = A_K_sum / float(number_of_particles)

        # Move to numpy once
        A_K_sum = A_K_sum.detach().cpu().numpy()

        # Compute over batches and kappas
        values_for_kappa = torch.zeros(MAX_K, batch_size)    
        for b in range(batch_size):
            for K in range(1, MAX_K+1):

                # Create a partial function
                f = partial(func_to_solve, precomputed_A_K_sum=A_K_sum[K-1,b], K=K)

                # the bounds on the value of kappa
                low = 0.0001
                # low = 0.0
                high = 10000.0

                # If they have the same sign then there is no zero crossing and so we have no solution for this k, aka use neg inf
                if((f(low)*f(high)) > 0):
                    values_for_kappa[K-1, b] = -np.inf
                    continue

                # Solve using a scalar call
                res = scipy.optimize.root_scalar(f, x0=[100.0], bracket=(low, high), method="brentq")
                values_for_kappa[K-1, b] = res.root

        # torch.set_printoptions(sci_mode=False, linewidth=10000)
        # print(values_for_kappa)
        # # exit()

        # Select the max for each batch
        max_kappas, _ = torch.max(values_for_kappa, dim=0)
        
        # Need to make sure its always a positive value which it should be but this is just in case
        # assert(torch.sum(max_kappas < 0) == 0)
        # Hack....
        max_kappas[max_kappas < 0] = 100.0

        # Using https://github.com/arviz-devs/arviz/blob/main/arviz/stats/density_utils.py#L85
        # also: https://www.sciencedirect.com/science/article/pii/S0167947307004367
        rot_bandwidth = _bw_taylor(particles, kappa=max_kappas)

        return rot_bandwidth




def create_BandwidthPredictorNN(model_name, model_parameters):
    parameters = model_parameters[model_name]

    particle_dims = parameters["particle_dims"]

    return bandwidth_selection_models.BandwidthPredictorNN(particle_dim=particle_dims, output_dim=particle_dims, use_weights=True)

def create_BandwidthPredictorNNSoftPlus(model_name, model_parameters):
    parameters = model_parameters[model_name]

    particle_dims = parameters["particle_dims"]

    return bandwidth_selection_models.BandwidthPredictorNNSoftplus(particle_dim=particle_dims, output_dim=particle_dims, use_weights=True)



def create_bandwidth_model(model_name, model_parameters):

    # Extract the model type
    model_type = model_parameters[model_name]["type"]

    if(model_type == "BandwidthPredictorNN"):
        return  create_BandwidthPredictorNN(model_name, model_parameters)
    elif(model_type == "BandwidthPredictorNNSoftplus"):
        return  create_BandwidthPredictorNNSoftPlus(model_name, model_parameters)
    elif(model_type == "FixedBandwith"):
        parameters = model_parameters[model_name]
        return  FixedBandwith(parameters)

    else:
        print("Unknown initializer_model type \"{}\"".format(model_type))
        exit()



