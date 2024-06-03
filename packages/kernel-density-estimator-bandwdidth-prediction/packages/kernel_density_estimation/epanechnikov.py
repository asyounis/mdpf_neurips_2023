# Imports
import torch
import torch.distributions as D
import numpy as np
import matplotlib.pyplot as plt


class Epanechnikov():

    def __init__(self, loc, bandwidth, validate_args=None):
        super().__init__()
        self.loc = loc
        self.bandwidth = bandwidth

        if(torch.is_tensor(self.bandwidth) ==  False):
            self.bandwidth = torch.FloatTensor([self.bandwidth]).to(self.loc.device)


    def cdf(self, x):
        diff = x - self.loc
        diff2 = diff / self.bandwidth

        output = 3.0 * (self.bandwidth**2)*diff
        output -= diff**3
        output /= (4*self.bandwidth**3)
        output += 0.5

        zero = torch.FloatTensor([0.0]).to(output.device)
        one = torch.FloatTensor([1.0]).to(output.device)
        output = torch.where(diff2 < -1, zero, output)
        output = torch.where(diff2 > 1, one, output)


        # if(torch.sum(torch.isnan(samples)) > 0):
        #     print(samples)
        #     assert(False)

        return output




    def log_prob(self, x):


        z = x - self.loc
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

        # if(torch.sum(torch.isnan(samples)) > 0):
        #     print(samples)
        #     assert(False)

        return samples


# x_min = -5
# x_max = 5
# x = torch.linspace(x_min, x_max, steps=1000)

# dist = Epanechnikov(torch.ones((1,)), 2)

# samples = dist.sample((1000000,)).numpy()
# plt.hist(samples, density=True, bins=1000, range=[x_min, x_max])


# pdf = torch.exp(dist.log_prob(x)).squeeze(0)
# plt.plot(x, pdf.numpy())


# cdf = dist.cdf(x).squeeze(0).numpy()
# plt.plot(x, cdf)


# print(torch.trapezoid(pdf, x))



# # plt.grid(axis='x', color='0')
# # plt.grid(axis='y', color='0')


# plt.show()