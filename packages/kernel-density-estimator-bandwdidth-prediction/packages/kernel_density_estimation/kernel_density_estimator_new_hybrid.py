
# Pytorch Includes
import torch
import torch.distributions as D
from functorch import jacrev
import functorch
from tqdm import tqdm

from kernel_density_estimation.von_mises_full_dist import *
from kernel_density_estimation.epanechnikov import *

class KernelDensityEstimatorNewHybrid :
    def __init__(self, kde_params, particles, particle_weights, bandwidths, validate_args=True):
        '''
            particles.shape = [batch, num_particles, dim]
            particle_weights.shape = [batch, num_particles]
            particles.bandwidths = [batch, dim]
        '''

        self.batch_size = particles.shape[0]
        self.number_of_particles = particles.shape[1]

        # The particle weights can be thought as the mixture component weighs
        self.weights = particle_weights
        self.particles = particles
        self.bandwidths = bandwidths

        self.distributions = []
        self.distribution_types = []
        for d in kde_params["dims"]:
            dim_params = kde_params["dims"][d]

            # Get the bandwidth for this dim
            # bandwidth = torch.tile(bandwidths[...,d].unsqueeze(-1), [1, particles.shape[1]])
            bandwidth = self.bandwidths[...,d].unsqueeze(-1)

            # create the correct dist for this dim
            distribution_type = dim_params["distribution_type"]
            self.distribution_types.append(distribution_type)
            if(distribution_type == "Normal"):

                # Create and save the dist
                dist = D.Normal(loc=particles[..., d], scale=bandwidth, validate_args=validate_args)
                self.distributions.append(dist)

            elif(distribution_type == "Von_Mises"):

                bandwidth = 1.0 / (bandwidth + 1e-8)

                # Create and save the dist
                # dist = D.VonMises(loc=particles[..., d], concentration=bandwidth)
                dist = VonMisesFullDist(loc=particles[..., d], concentration=bandwidth, validate_args=validate_args)
                self.distributions.append(dist)


            elif(distribution_type == "Epanechnikov"):

                # Create and save the dist
                dist = Epanechnikov(loc=particles[..., d], bandwidth=bandwidth)
                self.distributions.append(dist)


        self.number_of_dims = len(self.distributions)



    # def sample_new_hybrid(self, shape):
    #     assert(len(shape) == 1)

    #     # Figure out how many samples we need
    #     total_samples = 1
    #     for s in shape:
    #         total_samples *= s

    #     # Create structure that will hold all the samples
    #     all_samples_shape = list()
    #     all_samples_shape.append(total_samples)
    #     all_samples_shape.append(self.batch_size)
    #     all_samples_shape.append(len(self.distributions))
    #     all_samples = torch.zeros(size=all_samples_shape, device=self.weights.device)#.double()

    #     # Select the mixture component to use based on the new C
    #     cat_shape = list()
    #     cat_shape.append(total_samples)
    #     # cat_shape.append(self.batch_size)
    #     cat = D.Categorical(probs=self.weights.detach())
    #     cat_samples = cat.sample(cat_shape).detach()

    #     # Sample from each dim 1 dim at a time
    #     for d, dist in enumerate(self.distributions):

    #         # Sample from all the mixture components and then cut out the
    #         # samples from the components that were selected above
    #         d_samples = dist.rsample(shape)
    #         d_samples = torch.gather(d_samples, -1, cat_samples.unsqueeze(-1))

    #         # Save the sample
    #         all_samples[...,d] = d_samples.squeeze(-1)
    #     all_samples = torch.permute(all_samples,(1, 0, 2))


    #     # Compute the weights using importance sampling

    #     cat_samples = torch.permute(cat_samples,(1, 0))        
    #     w = torch.gather(self.weights, -1, cat_samples)

    #     w = w / (w.detach())
    #     w = w / float(w.shape[1])



    #     return all_samples, w






    def sample_new_hybrid(self, shape):
        assert(len(shape) == 1)

        # Figure out how many samples we need
        total_samples = 1
        for s in shape:
            total_samples *= s

        # Create structure that will hold all the samples
        all_samples_shape = list()
        all_samples_shape.append(total_samples)
        all_samples_shape.append(self.batch_size)
        all_samples_shape.append(len(self.distributions))
        all_samples = torch.zeros(size=all_samples_shape, device=self.weights.device)#.double()

        # Select the mixture component to use based on the new C
        cat_shape = list()
        cat_shape.append(total_samples)
        # cat_shape.append(self.batch_size)
        cat = D.Categorical(probs=self.weights.detach())
        cat_samples = cat.sample(cat_shape).detach()
        cat_samples = torch.permute(cat_samples,(1, 0))        



        new_samples = torch.zeros((cat_samples.shape[0], cat_samples.shape[1], self.particles.shape[-1]))
        new_weights = torch.zeros((cat_samples.shape[0], cat_samples.shape[1]))


        weights_normed = self.weights / torch.sum(self.weights, dim=1)


        for b in range(cat_samples.shape[0]):

            # get the bandwidth to use
            band = self.bandwidths[b].detach()

            for i in range(cat_samples.shape[1]):

                # Get the particle index to use
                p_idx = cat_samples[b, i]

                # Compute the weight
                # w = self.weights[b, p_idx.item()]
                w = weights_normed[b, p_idx.item()]
                new_weights[b, i] = (w / w.detach())


                # Get the particle to sample from
                p = self.particles[b, p_idx.item()]

                # Sample all the dims of the particle
                for d in range(new_samples.shape[-1]):
                    dist = D.Normal(p[d], band[d])
                    new_samples[b, i, d] = dist.rsample()


                # new_weights[b, i] = (w.detach() / w)


        # new_weights = new_weights * (1.0 / float(cat_samples.shape[1]))

        # print(cat_samples.shape)
        # print(new_samples.shape)
        # exit()


        # print(new_samples.shape)
        # print(new_weights.shape)


        return new_samples, new_weights


        # # Sample from each dim 1 dim at a time
        # for d, dist in enumerate(self.distributions):

        #     # Sample from all the mixture components and then cut out the
        #     # samples from the components that were selected above
        #     d_samples = dist.rsample(shape)
        #     d_samples = torch.gather(d_samples, -1, cat_samples.unsqueeze(-1))

        #     # Save the sample
        #     all_samples[...,d] = d_samples.squeeze(-1)
        # all_samples = torch.permute(all_samples,(1, 0, 2))


        # # Compute the weights using importance sampling

        # w = torch.gather(self.weights, -1, cat_samples)

        # w = w / (w.detach())
        # w = w / float(w.shape[1])



        # return all_samples, w


















    # def log_prob(self, x, do_normalize_weights=True):
        
    #     # Right now we only support shapes of size 3 [batch size, samples, dims]
    #     assert(len(x.shape) == 3)

    #     # Need to convert x from [batch, shape , dims] to [shape, batch, dims]
    #     x = torch.permute(x, (1,0,2))

    #     # All the log probabilities
    #     all_log_probs = None 

    #     # Do 1 dim at a time
    #     for d, dist in enumerate(self.distributions):

    #         log_prob = dist.log_prob(x[..., d].unsqueeze(-1))

    #         if(all_log_probs is None):
    #             all_log_probs = log_prob
    #         else:
    #             all_log_probs = all_log_probs + log_prob

    #     # Log the weights
    #     log_weights = torch.log(self.weights.unsqueeze(0) + 1e-8)

    #     # Normalize the weights if we need to
    #     if(do_normalize_weights):
    #         log_weights = log_weights - torch.logsumexp(log_weights, dim=-1, keepdim=True)

    #     # Finish off the computation
    #     all_log_probs = all_log_probs + log_weights
    #     all_log_probs = torch.logsumexp(all_log_probs, dim=-1)

    #     all_log_probs = torch.permute(all_log_probs, (1,0))

    #     return all_log_probs
