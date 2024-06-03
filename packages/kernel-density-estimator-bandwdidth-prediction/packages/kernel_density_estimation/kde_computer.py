

# Pytorch Includes
import torch
import torch.distributions as D


from kernel_density_estimation.von_mises_full_dist import *
from kernel_density_estimation.epanechnikov import *
from kernel_density_estimation.circular_epanechnikov import *

class KDEComputer:
    def __init__(self, kde_params):

        # Create the distributions 
        self.distribution_types = []
        for d in kde_params["dims"]:

            # Extract the params for this dim
            dim_params = kde_params["dims"][d]
            distribution_type = dim_params["distribution_type"]

            # Save the dist type for this dimention
            self.distribution_types.append(distribution_type)


    def compute_kde(self, test_points, particles, weights, bandwidths):


        kde = torch.zeros(size=(test_points.shape[1], particles.shape[0], particles.shape[1], particles.shape[-1]), device=particles.device)

        # bandwidths.register_hook(lambda grad: print("bandwidths", grad))


        for d in range(len(self.distribution_types)):



            # create the correct object class for each dist
            distribution_type = self.distribution_types[d]
            if(distribution_type == "Normal"):

                bandwidth = bandwidths[...,d].unsqueeze(-1)
                # bandwidth = 1.0 / bandwidth



                dist = D.Normal(particles[...,d], bandwidth)

            elif(distribution_type == "Von_Mises"):

                # Von Mises uses a concentration parameter which is loosely
                # equal to variance ~= 1/concentration so we should flip the 
                # bandwidth to pass in a concentration

                bandwidth = bandwidths[...,d].unsqueeze(-1)
                bandwidth = 1.0 / bandwidth
                # bandwidth = torch.sqrt(bandwidth)

                # dist = D.VonMises(particles[...,d], bandwidth)
                dist = VonMisesFullDist(particles[...,d], bandwidth)

            elif(distribution_type == "Epanechnikov"):
                bandwidth = bandwidths[...,d].unsqueeze(-1)
                dist = Epanechnikov(particles[...,d], bandwidth)

            elif(distribution_type == "CircularEpanechnikov"):
                bandwidth = bandwidths[...,d].unsqueeze(-1)
                dist = CircularEpanechnikov(particles[...,d], bandwidth)

            else:
                assert(False)


            # Make the test points have only the correct dim and correct shape
            t = test_points[..., d]
            t = torch.transpose(t, 0, 1)
            t = t.unsqueeze(-1)

            # Get the log prob
            kde[...,d] = dist.log_prob(t)

        # Finish off the kde computation

        kde = torch.sum(kde, dim=-1)
        kde = torch.exp(kde)
        kde = kde * weights.unsqueeze(0)
        kde = torch.sum(kde, dim=-1)
        kde = torch.transpose(kde, 0, 1)

        return kde