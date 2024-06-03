
# Pytorch Includes
import torch
import torch.distributions as D
from functorch import jacrev
import functorch
from tqdm import tqdm

from kernel_density_estimation.von_mises_full_dist import *
from kernel_density_estimation.epanechnikov import *
from kernel_density_estimation.circular_epanechnikov import *

class KernelDensityEstimator:
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

            elif(distribution_type == "CircularEpanechnikov"):

                # Create and save the dist
                dist = CircularEpanechnikov(loc=particles[..., d], bandwidth=bandwidth)
                self.distributions.append(dist)

            else:
                print("Unknown kernel function \"{}\"".format(distribution_type))
                assert("False")


        self.number_of_dims = len(self.distributions)



    def sample_concrete_relaxation(self, sample_shape, temperature_parameter=0.5):


        assert(len(sample_shape) == 1)

        # Figure out how many samples we need
        total_samples = 1
        for s in sample_shape:
            total_samples *= s

        gumbel_samples_shape = list()
        gumbel_samples_shape.append(total_samples)
        gumbel_samples_shape.append(self.batch_size)
        gumbel_samples_shape.append(self.number_of_particles)
        gumbel_samples = torch.rand(gumbel_samples_shape).to(self.weights.device)
        gumbel_samples = -torch.log(-torch.log(gumbel_samples + 1e-10))
        gumbel_samples = gumbel_samples + torch.log(self.weights + 1e-10)

        gumbel_samples = gumbel_samples / temperature_parameter
        gumbel_samples = torch.nn.functional.softmax(gumbel_samples, dim=-1)


        # Create structure that will hold all the samples
        all_samples_shape = list()
        all_samples_shape.append(total_samples)
        all_samples_shape.append(self.batch_size)
        all_samples_shape.append(len(self.distributions))
        all_samples = torch.zeros(size=all_samples_shape, device=self.weights.device)

        all_dim_samples = []
        for d, dist in enumerate(self.distributions):

            # samples = dist.rsample(sample_shape)
            samples = dist.sample(sample_shape)

            if(self.distribution_types[d] == "Normal"):
                samples = samples * gumbel_samples
                samples = torch.sum(samples, dim=-1)

            elif(self.distribution_types[d] == "Von_Mises"):
                # For angles we need to take the average with respect to the angle vectors 
                # So that we  can get the correct averaged angle 
                # See:
                #       - https://stackoverflow.com/questions/491738/how-do-you-calculate-the-average-of-a-set-of-circular-data
                #       - http://catless.ncl.ac.uk/Risks/7.44.html#subj4
                #       - https://stackoverflow.com/questions/1686994/weighted-average-of-angles
                y = torch.sin(samples)
                y = y * gumbel_samples
                y = torch.sum(y, dim=-1)

                x = torch.cos(samples)
                x = x * gumbel_samples
                x = torch.sum(x, dim=-1)

                samples = torch.atan2(y, x)

            else:
                print("Unknown kernel function \"{}\"".format(self.distribution_types[d]))
                assert("False")


            all_samples[...,d] = samples


        all_samples = torch.permute(all_samples,(1, 0, 2))

        return all_samples

    def sample(self, shape):

        # There are no gradients when doing resampling so 
        # we might as well stop the gradients now to save computation time
        with torch.no_grad():
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
            cat = D.Categorical(probs=self.weights)
            cat_samples = cat.sample(cat_shape)

            # Sample from each dim 1 dim at a time
            for d, dist in enumerate(self.distributions):

                # Sample from all the mixture components and then cut out the
                # samples from the components that were selected above
                d_samples = dist.sample(shape)
                d_samples = torch.gather(d_samples, -1, cat_samples.unsqueeze(-1))

                # Save the sample
                all_samples[...,d] = d_samples.squeeze(-1)

            all_samples = torch.permute(all_samples,(1, 0, 2))
            return all_samples.detach()


    def find_peak(self, number_of_iterations=1000, number_of_initial_samples=5000):

        samples = self.sample((number_of_initial_samples,))
        log_probs = self.log_prob(samples)
        highest_prob_particle = torch.argmax(log_probs, dim=-1)

        current_peak = torch.zeros((self.particles.shape[0], self.particles.shape[-1]), device=self.particles.device)
        for i in range(self.particles.shape[0]):
            current_peak[i, :] = samples[i, highest_prob_particle[i], :]

        # There are no gradients when doing resampling so 
        # we might as well stop the gradients now to save computation time
        current_peak_old = current_peak.clone()
        old_log_prob = None
        with torch.enable_grad():

            current_peak.requires_grad = True
            optimizer = torch.optim.Adam([current_peak], 0.01)
            for i in range(number_of_iterations):

                optimizer.zero_grad()

                log_prob = self.log_prob(current_peak.unsqueeze(1))
                loss = torch.sum(-log_prob)
                loss.backward()

                optimizer.step()

                diff = torch.sum(torch.abs(current_peak - current_peak_old))
                current_peak_old = current_peak.clone()
                if(diff < 1e-3):
                    break

                # if(old_log_prob is not None):
                #     log_prob_diff = torch.sum(torch.exp(log_prob) - torch.exp(old_log_prob))
                #     if(log_prob_diff < 1e-5):
                #         print(diff, i)
                #         break
                # old_log_prob = log_prob




            return current_peak

    def log_prob(self, x, do_normalize_weights=True):
        
        # Right now we only support shapes of size 3 [batch size, samples, dims]
        assert(len(x.shape) == 3)

        # Need to convert x from [batch, shape , dims] to [shape, batch, dims]
        x = torch.permute(x, (1,0,2))

        # All the log probabilities
        all_log_probs = None 

        # Do 1 dim at a time
        for d, dist in enumerate(self.distributions):

            log_prob = dist.log_prob(x[..., d].unsqueeze(-1))

            if(all_log_probs is None):
                all_log_probs = log_prob
            else:
                all_log_probs = all_log_probs + log_prob

        # Log the weights
        log_weights = torch.log(self.weights.unsqueeze(0) + 1e-8)

        # Normalize the weights if we need to
        if(do_normalize_weights):
            log_weights = log_weights - torch.logsumexp(log_weights, dim=-1, keepdim=True)

        # Finish off the computation
        all_log_probs = all_log_probs + log_weights
        all_log_probs = torch.logsumexp(all_log_probs, dim=-1)

        all_log_probs = torch.permute(all_log_probs, (1,0))

        return all_log_probs

    def compute_jacobian_manually(self, x):


        # We dont need gradients for this since we will be computing this manually
        with torch.no_grad():

            # The PDF and CDF values for each of the distributions
            shape = list()
            shape.append(x.shape[0])
            shape.append(self.batch_size)
            shape.append(self.number_of_particles)
            shape.append(self.number_of_dims)
            all_probs = torch.zeros(size=shape, device=self.weights.device)
            all_cdfs = torch.zeros(size=shape, device=self.weights.device)
            all_diff_of_probs = torch.zeros(size=shape, device=self.weights.device)

            # Do 1 dim at a time
            for d, dist in enumerate(self.distributions):

                # get the prob
                log_prob = dist.log_prob(x[..., d].unsqueeze(-1))
                all_probs[..., d] = torch.exp(log_prob)

                all_diff_of_probs[..., d] = dist.diff_of_prob(x[..., d].unsqueeze(-1))

                # get the cdf
                all_cdfs[..., d] = dist.cdf(x[..., d].unsqueeze(-1))


            # print(all_probs[:,0,:,0])
            # print(all_cdfs[:,0,:,0])

            # exit()

            # Create the numerator term
            shape = list()
            shape.append(self.number_of_dims)
            shape.append(self.number_of_dims)
            shape.append(x.shape[0])
            shape.append(self.batch_size)
            shape.append(self.number_of_particles)
            c_upper = torch.zeros(size=shape, device=self.weights.device)
            c_lower = torch.zeros(size=shape, device=self.weights.device)
            for c in range(self.number_of_dims):
                for r in range(self.number_of_dims):

                    # print("")
                    # print("--------")
                    # print(r, c)

                    if(c > r):
                        continue

                    # Do the upper coeff
                    c_upper[r,c] = self.weights.unsqueeze(0)
                    for i in range(r+1):
                        if(i==c):
                            continue
                        
                        if(i == r):
                            c_upper[r,c] = c_upper[r,c] * all_cdfs[..., i]
                        else:
                            c_upper[r,c] = c_upper[r,c] * all_probs[..., i]

                        # print(i)


                    # Do the lower coeff
                    if(r != c):

                        c_lower[r,c] = self.weights.unsqueeze(0)

                        for i in range(r):
                            if(i == c):
                                continue
                            c_lower[r,c] = c_lower[r,c] * all_probs[..., i]


            # print(torch.max(all_probs))

            # exit()

            # The jacobian
            shape = list()
            shape.append(x.shape[0])
            shape.append(self.batch_size)
            shape.append(self.number_of_dims)
            shape.append(self.number_of_dims)
            jacobian = torch.zeros(size=shape, device=x.device)


            for c in range(len(self.distributions)):
                for r in range(len(self.distributions)):

                    # We only have the lower triangle so skip the upper triangle of the matrix
                    # Default for the jacobian matrix is zero so not setting just keeps it zero
                    if(c > r):
                        continue

                    if(c == r):
                        # Do the diagonal
                        jacobian[:,:, r, c] = torch.sum(all_probs[..., c] * c_upper[r,c], dim=-1)
                        jacobian[:,:, r, c] /= torch.sum(c_upper[r,c], dim=-1)

                    else:

                        derivA = all_diff_of_probs[..., c] * torch.sum(all_probs[..., c]  * c_lower[r,c], dim=-1,keepdim=True)
                        derivB = all_probs[..., c]  * torch.sum(all_diff_of_probs[..., c] * c_lower[r,c], dim=-1,keepdim=True)
                        derivC = torch.sum(all_probs[..., c] * c_lower[r,c], dim=-1,keepdim=True) ** 2


                        deriv = derivA - derivB
                        deriv /= (derivC + 1e-8)

                        jacobian[:,:, r, c] = torch.sum(c_upper[r,c] * deriv, dim=-1)


            # jacobian += (torch.eye(jacobian.shape[-1]).to(jacobian.device) * 1e-8)
            for c in range(self.number_of_dims):
                for r in range(self.number_of_dims):
                    if(c > r):
                        jacobian[:,:, r, c] = 0  

            return jacobian         

    def compute_gradient_injection(self, x, do_normalize_weights=True):

        # Right now we only support shapes of size 3 [batch size, samples, dims]
        assert(len(x.shape) == 3)


        # # if(len(self.distributions) == 1):

        # #     dist = self.distributions[0]

        # #     ins = x.permute((1,0, 2))

        # #     cdf = dist.cdf(ins.detach())
        # #     cdf = torch.permute(cdf, (1,0,2))
        # #     cdf = torch.sum(self.weights.unsqueeze(1) * cdf, dim=-1)


        # #     pdf = torch.exp(dist.log_prob(ins).detach())
        # #     pdf = torch.permute(pdf, (1,0,2))
        # #     pdf = torch.sum(self.weights.unsqueeze(1) * pdf, dim=-1)

        # #     # print(ins)
        # #     # print(pdf)
        # #     # exit()

        # #     # print(torch.min(pdf).item(), torch.max(pdf).item())

        # #     gradient_injection_value = -cdf / pdf.detach()

        # #     # Make the gradient injection value 0 for this!
        # #     gradient_injection_value = gradient_injection_value - gradient_injection_value.detach()


        # #     return gradient_injection_value.unsqueeze(-1)




        # ###############################################################################################
        # ## Compute ∇zSφ(z))^−1
        # ###############################################################################################

        # def func1(inputs):

        #     # This is the "weight" coeef that will be used in each conditional mixture
        #     shape = list()
        #     shape.append(self.batch_size)
        #     shape.append(inputs.shape[1])
        #     shape.append(self.number_of_particles)
        #     c_numer = torch.zeros(size=shape, device=self.weights.device)
        #     c_numer[:, :, :] = torch.log(self.weights.unsqueeze(1) + 1e-8)


        #     # Sample from each dim 1 dim at a time
        #     all_conditional_cdfs = []
        #     for d, dist in enumerate(self.distributions):

        #         # Compute the normalized C
        #         c_normed = c_numer - torch.logsumexp(c_numer, dim=-1, keepdim=True)

        #         # Create the Input we want to use for this
        #         ins = inputs[..., d].permute((1,0)).unsqueeze(-1)

        #         # Compute the CDF for this dim for all the mixtures
        #         # cdf = dist.cdf(inputs[..., d].permute((1,0)))
        #         cdf = dist.cdf(ins)
        #         cdf = torch.permute(cdf, (1,0,2))

        #         # Compute the conditional CDF
        #         conditional_cdf = torch.sum(cdf * torch.exp(c_normed), dim=-1)
        #         all_conditional_cdfs.append(conditional_cdf)

        #         # If we have all the conditional CDFs then we are done
        #         if(len(all_conditional_cdfs) == len(self.distributions)):
        #             break

        #         # Update the numerator for the next dim
        #         log_prob = dist.log_prob(ins)
        #         log_prob = torch.permute(log_prob, (1,0,2))
        #         c_numer = c_numer + log_prob

        #     all_conditional_cdfs = torch.stack(all_conditional_cdfs, -1)

        #     return all_conditional_cdfs

   

        # ###############################################################################################
        # ## Compute ∇φSφ(z)
        # ###############################################################################################
        # # all_conditional_cdfs1 = func1(x)


        # def batch_jacobian(func, inputs, create_graph=False):
        #     # x in shape (particles, Batch, dims)
        #     def _func_sum(inputs2):
        #         return func(inputs2).sum(dim=0).sum(dim=0)

        #     with torch.autograd.set_detect_anomaly(False):
        #         # jac = torch.autograd.functional.jacobian(_func_sum, inputs, create_graph=create_graph, vectorize=True)
        #         jac = jacrev(_func_sum)(inputs)

        #     # which one????
        #     # jac = jac.permute(1, 2, 3 , 0) # Gives Upper triangular jacobian
        #     jac = jac.permute(1, 2, 0 , 3) # Gives lower triangular jacobian, USE THIS ONE! 
        #     return jac


        ################################################################################
        # Functorch way.  This was very easy to compute!!!
        ################################################################################
        def func_for_vmap(inputs_func_for_vmap, p, w, b):

            c_numer = torch.log(w + 1e-12)


            all_conditional_cdfs = []
            for d, dist in enumerate(self.distributions):

                # Get the bandwidth for this dim
                bandwidth = b[...,d].unsqueeze(-1)

                # Create the distributions
                if(self.distribution_types[d] == "Normal"):
                    dist = D.Normal(loc=p[..., d], scale=bandwidth, validate_args=False)

                elif(self.distribution_types[d] == "Von_Mises"):
                    bandwidth = 1.0 / bandwidth
                    dist = VonMisesFullDist(loc=p[..., d], concentration=bandwidth, validate_args=False)

                elif(self.distribution_types[d] == "Epanechnikov"):
                    dist = Epanechnikov(loc=p[..., d], bandwidth=bandwidth)

                elif(self.distribution_types[d] == "CircularEpanechnikov"):
                    dist = CircularEpanechnikov(loc=p[..., d], bandwidth=bandwidth)
                else:
                    assert(False)


                # Compute the normalized C
                if((d == 0) and (do_normalize_weights==False)):
                    c_normed = c_numer
                else:
                    c_normed = c_numer - torch.logsumexp(c_numer, dim=0, keepdim=True)

                # Compute the CDF for this dim for all the mixtures
                cdf = dist.cdf(inputs_func_for_vmap[d])


                # Compute the conditional CDF
                conditional_cdf = torch.sum(cdf * torch.exp(c_normed), dim=-1)
                all_conditional_cdfs.append(conditional_cdf)

                # Update the numerator for the next dim
                log_prob = dist.log_prob(inputs_func_for_vmap[d])
                c_numer = c_numer + log_prob



            all_conditional_cdfs = torch.stack(all_conditional_cdfs)

            return all_conditional_cdfs




        # Create the vmap and compute the jacs!!!!
        vmap1 = functorch.vmap(func_for_vmap, in_dims=(0, None, None, None)) # particles dim
        vmap2 = functorch.vmap(vmap1) # batch dim
        all_conditional_cdfs1 = vmap2(x, self.particles, self.weights, self.bandwidths)


        with torch.no_grad():

            # Old way.  This should work and Ive been using it for a while
            # jacobian1 = batch_jacobian(func1, x).detach()



            # Create the vmap and compute the jacs!!!!
            vmap1 = functorch.vmap(functorch.jacrev(func_for_vmap), in_dims=(0, None, None, None)) # particles dim
            vmap2 = functorch.vmap(vmap1) # batch dim
            jacobian1 = vmap2(x, self.particles, self.weights, self.bandwidths)


            # check the ranks of the matrix.  Sometimes we generate a matrix that is not full rank.  Thus causes the solver to fail
            # So if the rank is not correct then we should fix it. BUT only if there are a few entries.  Too many entries and we should fail
            ranks = torch.linalg.matrix_rank(jacobian1)
            not_large_enough = torch.sum(ranks!=jacobian1.shape[-1])
            if((not_large_enough > 0) and (not_large_enough < 5)):

                # Not too many errors so fix the ones we have
                invalid_idxs = (ranks!=jacobian1.shape[-1])
                jacobian1[invalid_idxs] += torch.tril(torch.rand(jacobian1[invalid_idxs].shape) * 1e-4).to(jacobian1.device)



        # Working
        gradient_injection_value = -(torch.linalg.solve_triangular(jacobian1.detach(), all_conditional_cdfs1.unsqueeze(-1), upper=False)).squeeze(-1)

        if(torch.sum(torch.isnan(gradient_injection_value)) > 0):

            torch.set_printoptions(threshold=100000)
            print("")
            print("NAN for Gradient")


            # Compute rank of matrix to see if we are getting Nans from not having full ranks
            ranks = torch.linalg.matrix_rank(jacobian1)
            print(ranks)

            exit()

        # Do some clipping for stability
        # This is a hack.  It doesnt happen all that often so it shouldnt have much effect on things
        inf_replacement_value = 1e8
        gradient_injection_value = torch.nan_to_num(gradient_injection_value, nan=float("nan"), posinf=float(inf_replacement_value), neginf=-float(inf_replacement_value))





        # Make the gradient injection value 0 for this!
        gradient_injection_value = gradient_injection_value - gradient_injection_value.detach()


        if(torch.sum(torch.isnan(gradient_injection_value)) > 0):

            torch.set_printoptions(threshold=100000)
            print("")
            print("NAN for Gradient 2")


        return gradient_injection_value


    def inject_gradient(self, x, do_normalize_weights=True):

        # We need to detach x
        x = x.detach()

        grad_inject = self.compute_gradient_injection(x, do_normalize_weights=do_normalize_weights)

        # if(torch.sum(torch.isnan(grad_inject)) > 0):

        #     torch.set_printoptions(threshold=100000)
        #     print("")
        #     print("NAN for Gradient 3")


        # Inject the gradient
        x = x + grad_inject

        return x

    def marginal_log_prob(self, x, variables_to_marginilize_out):
        
        # Right now we only support shapes of size 3 [batch size, samples, dims]
        assert(len(x.shape) == 3)

        # Make sure the number of dims supplied is correct
        assert(x.shape[-1] == (len(self.distributions) - len(variables_to_marginilize_out)))

        # Need to convert x from [batch, samples , dims] to [samples, batch, dims]
        new_shape = list()
        new_shape.extend(list(x.shape[1:-1]))
        new_shape.append(self.batch_size)
        new_shape.append(x.shape[-1])
        x = torch.reshape(x, new_shape)


        kde = torch.zeros(size=(x.shape[0], x.shape[1], self.number_of_particles, x.shape[2]), device=x.device)
        curr_x_dim = 0
        for d, dist in enumerate(self.distributions):

            # Skip dims we are marginalizing out
            if(d in variables_to_marginilize_out):
                continue

            # Get the log prob
            kde[...,curr_x_dim] = dist.log_prob(x[..., curr_x_dim].unsqueeze(-1))
            curr_x_dim += 1


        # Finish off the kde computation
        kde = torch.sum(kde, dim=-1)
        kde += torch.log(self.weights.unsqueeze(0) + 1e-10)
        kde = torch.logsumexp(kde, dim=-1)
        kde = torch.transpose(kde, 0, 1)

        return kde



    def mdt_cdf(self, x, do_normalize_weights=True):

        ################################################################################
        # Functorch way.  This was very easy to compute!!!
        ################################################################################
        def func_for_vmap(inputs_func_for_vmap, p, w, b):

            c_numer = torch.log(w + 1e-12)


            all_conditional_cdfs = []
            for d, dist in enumerate(self.distributions):

                # Get the bandwidth for this dim
                bandwidth = b[...,d].unsqueeze(-1)

                # Create the distributions
                if(self.distribution_types[d] == "Normal"):
                    dist = D.Normal(loc=p[..., d], scale=bandwidth, validate_args=False)

                elif(self.distribution_types[d] == "Von_Mises"):
                    bandwidth = 1.0 / bandwidth
                    dist = VonMisesFullDist(loc=p[..., d], concentration=bandwidth, validate_args=False)

                elif(self.distribution_types[d] == "Epanechnikov"):
                    dist = Epanechnikov(loc=p[..., d], bandwidth=bandwidth)

                elif(self.distribution_types[d] == "CircularEpanechnikov"):
                    dist = CircularEpanechnikov(loc=p[..., d], bandwidth=bandwidth)
                else:
                    assert(False)


                # Compute the normalized C
                if((d == 0) and (do_normalize_weights==False)):
                    c_normed = c_numer
                else:
                    c_normed = c_numer - torch.logsumexp(c_numer, dim=0, keepdim=True)

                # Compute the CDF for this dim for all the mixtures
                cdf = dist.cdf(inputs_func_for_vmap[d])


                # Compute the conditional CDF
                conditional_cdf = torch.sum(cdf * torch.exp(c_normed), dim=-1)
                all_conditional_cdfs.append(conditional_cdf)

                # Update the numerator for the next dim
                log_prob = dist.log_prob(inputs_func_for_vmap[d])
                c_numer = c_numer + log_prob



            all_conditional_cdfs = torch.stack(all_conditional_cdfs)

            return all_conditional_cdfs




        # Create the vmap and compute the jacs!!!!
        vmap1 = functorch.vmap(func_for_vmap, in_dims=(0, None, None, None)) # particles dim
        vmap2 = functorch.vmap(vmap1) # batch dim
        all_conditional_cdfs1 = vmap2(x, self.particles, self.weights, self.bandwidths)

        return all_conditional_cdfs1