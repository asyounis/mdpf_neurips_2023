# Imports
import torch
import torch.distributions as D
import numpy as np
import matplotlib.pyplot as plt


class CircularEpanechnikov():

    def __init__(self, loc, bandwidth, validate_args=None):
        super().__init__()
        self.loc = loc
        self.bandwidth = bandwidth

        if(torch.is_tensor(self.bandwidth) ==  False):
            self.bandwidth = torch.FloatTensor([self.bandwidth]).to(self.loc.device)

        # Need to make sure that the bandwidth is always constrained to be below pi so that 
        # this implementation can actually work (aka dont need to worry about wrapping)
        assert(torch.all(self.bandwidth < np.pi))

    def log_prob(self, x):

        z = x - self.loc

        # Constrain to -pi and pi
        
        num_periods = torch.round(z / (2. * np.pi)).detach()
        z = z - ((2. * np.pi) * num_periods).detach()

        z = z / self.bandwidth

        pdf = 1.0 - (z**2)
        pdf *= (3.0 / 4.0)
        pdf /= self.bandwidth

        eps = torch.FloatTensor([1e-8]).to(pdf.device)
        pdf = torch.where(z < -1, eps, pdf)
        pdf = torch.where(z > 1, eps, pdf)
        pdf = torch.where(pdf < 1e-8, eps, pdf)

        # print(pdf)

        output = torch.log(pdf)

        # if(torch.sum(torch.isnan(output)) > 0):
        #     print(output)
        #     print(torch.sum(pdf < 1e-8))
        #     assert(False)

        return output

    def sample(self, s=(1,)):

        s = list(s)
        s.extend(self.loc.shape)
        # s.extend(self.bandwidth.shape)

        # print(s)
        # exit()

        # Sample a uniform distribution
        dist = D.Uniform(torch.tensor([0.0]), torch.tensor([1.0]))
        uniform_samples = dist.sample(s).to(self.loc.device).squeeze(-1)

        # Convert to samples from the Epanechnikov
        samples = (2.0 * uniform_samples) - 1
        samples = torch.asin(samples) / 3.0
        samples = torch.sin(samples) * 2 * self.bandwidth

        samples += self.loc

        # Constrain to -pi and pi
        num_periods = torch.round(samples / (2. * np.pi)).detach()
        samples = samples - ((2. * np.pi) * num_periods).detach()


        # if(torch.sum(torch.isnan(samples)) > 0):
        #     print(samples)
        #     assert(False)

        return samples
