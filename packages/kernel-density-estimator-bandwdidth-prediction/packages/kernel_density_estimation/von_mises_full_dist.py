# Imports
import torch
import torch.distributions as D
import numpy as np



class VonMisesFullDist(D.VonMises):

    def __init__(self, loc, concentration, validate_args=None):
        super().__init__(loc, concentration, validate_args)






    def cdf(self, x):
        # Hacking around
        diff = x-self.loc

        num_periods = torch.round(diff / (2. * np.pi)).detach()
        diff = diff - ((2. * np.pi) * num_periods).detach()

        sigma_squared = 1 / self.concentration
        dist = D.Normal(0, torch.sqrt(sigma_squared), validate_args=False)


        cdf = dist.cdf(diff)
        cdf = cdf + num_periods.detach()

        return cdf




    def rsample(self, sample_shape):

        samples = self.sample(sample_shape).detach()

        # Compute the CDF and PDF so we can do Implicit re parameterized gradients
        cdf_values = self.cdf(samples)
        log_prob_values = self.log_prob(samples)

        # Compute the gradient injection
        gradient_injection_value = -cdf_values / (torch.exp(log_prob_values).detach() + 1e-10)
        gradient_injection_value = gradient_injection_value - gradient_injection_value.detach()

        samples = samples + gradient_injection_value

        return samples






    # def cdf(self, x):
    #     diff = x-self.loc

    #     num_periods = torch.round(diff / (2. * np.pi))
    #     diff = diff - ((2. * np.pi) * num_periods)

    #     sigma_squared = 1 / self.concentration
    #     dist = D.Normal(0, torch.sqrt(sigma_squared), validate_args=False)


    #     cdf = dist.cdf(diff)

    #     return cdf






    # def cdf(self, x):
    #     """
    #     Returns the cumulative density/mass function evaluated at
    #     `x`.

    #     Note: This function is based off of the "von_mises_cdf" function from tensorflow
    #           and parts are copy pasted from that function.


    #     Denote the density of vonMises(loc=0, concentration=concentration) by p(t).
    #     Note that p(t) is periodic, p(t) = p(t + 2 pi).
    #     The CDF at the point x is defined as int_{-pi}^x p(t) dt.
    #     Thus, when x in [-pi, pi], the CDF is in [0, 1]; when x is in [pi, 3pi], the
    #     CDF is in [1, 2], etc.
        
    #     The CDF is not available in closed form. Instead, we use the method [1]
    #     which uses either a series expansion or a Normal approximation, depending on
    #     the x of concentration.


    #     References:
    #         [1] G. Hill "Algorithm 518: Incomplete Bessel Function I_0. The Von Mises
    #         Distribution." ACM Transactions on Mathematical Software, 1977


    #     Args:
    #         x (Tensor):
    #     """

    #     # print("")
    #     # print("")
    #     # print("")
    #     # print("")
    #     # print("")
    #     # print("")
    #     # print("")
    #     # print("")
    #     # print(x.shape)
    #     # print(self.loc.shape)



    #     # Make sure its a tensor
    #     x = torch.as_tensor(x)
    #     concentration = torch.as_tensor(self.concentration)


    #     # We need to standardize x first
    #     x = x - self.loc

    #     # print(x.shape)

    #     # Map x to [-pi, pi].
    #     num_periods = torch.round(x / (2. * np.pi))
    #     x = x - ((2. * np.pi) * num_periods)


    #     # We take the hyperparameters from Table I of [1], the row for D=8
    #     # decimal digits of accuracy. ck is the cut-off for concentration:
    #     # if concentration < ck,  the series expansion is used;
    #     # otherwise, the Normal approximation is used.
    #     ck = 10.5

    #     ck = 0.0

    #     # If we should use the series approx of the normal dist approx
    #     use_series = concentration < ck

    #     # The number of terms in the series expansion. [1] chooses it as a function
    #     # of concentration, n(concentration). This is hard to implement in TF.
    #     # Instead, we upper bound it over concentrations:
    #     #   num_terms = ceil ( max_{concentration <= ck} n(concentration) ).
    #     # The maximum is achieved for concentration = ck.
    #     # Note this shouldnt be hard in pytorch but this is an optimization 
    #     # we can add later
    #     num_terms = 20

    #     # Compute the CDF with the series implementation
    #     cdf_series = self._von_mises_cdf_series(x, concentration, num_terms)


    #     # # Now compute the normal distribution one for ONLY the samples that need to use this approximation
    #     # # This is to prevent NANS
    #     # false_mask = torch.zeros_like(x, dtype=torch.bool)
    #     # true_mask = ~false_mask
    #     # mask = torch.where(use_series, false_mask, true_mask)

    #     # if(torch.sum(mask).item() != 0):


    #     #     x_normal = x[mask]
    #     #     # concentration_normal = torch.tile(concentration.unsqueeze(0), [x.shape[0], 1])
    #     #     # concentration_normal = concentration_normal[mask]

    #     #     concentration_normal = concentration[~use_series]

    #     #     cdf_normal = self._von_mises_cdf_normal(x_normal, concentration_normal)

    #     #     # print(x_normal.shape)
    #     #     # print(concentration.shape)
    #     #     # print(concentration_normal.shape)
    #     #     # print(cdf_normal.shape)

    #     #     # exit()


    #     #     # Set the normal distribution ones
    #     #     false_mask = torch.zeros_like(cdf_series, dtype=torch.bool)
    #     #     true_mask = ~false_mask
    #     #     mask = torch.where(use_series, false_mask, true_mask)
    #     #     cdf_series[mask] = cdf_normal

    #     # cdf = cdf_series



    #     cdf_normal = self._von_mises_cdf_normal(x, concentration)
    #     cdf = torch.where(use_series, cdf_series, cdf_normal)
        
    #     # print(torch.sum(use_series).item(), torch.sum(~use_series).item())

    #     # x.register_hook(lambda x: print("x", torch.mean(torch.abs(x))))
    #     # concentration.register_hook(lambda x: print("cons", torch.mean(torch.abs(x))))



    #     # Tensor flow implementation does the following line BUT algorithm 518 does not
    #     # This also messes everything up (causes values to turn to NAN) so we should not do this???
    #     # cdf = cdf + num_periods

    #     return cdf

    # def _von_mises_cdf_series(self, x, concentration, num_terms):
    #     """Computes the von Mises CDF and its derivative via series expansion."""
    #     # Keep the number of terms as a float. It should be a small integer, so
    #     # exactly representable as a float.

    #     def loop_body(n, rn, vn):
    #         """One iteration of the series loop."""

    #         denominator = 2.0 * n / concentration + rn
    #         rn = 1.0 / denominator

    #         multiplier = torch.sin(n * x) / n + vn
    #         vn = rn * multiplier

    #         return rn, vn


    #     rn = torch.zeros_like(x, device=x.device)
    #     vn = torch.zeros_like(x, device=x.device)
    #     for n in range(num_terms, 0, -1):
    #         rn, vn = loop_body(n , rn , vn)

    #     cdf = 0.5 + x / (2.0 * np.pi) + vn / np.pi

    #     # print("GT", torch.sum(cdf>1.0))
    #     # print("LT", torch.sum(cdf<0.0))
    #     # Clip the result to [0, 1].
    #     cdf_clipped = torch.clamp(cdf, 0.0, 1.0)

    #     return cdf_clipped




    # def _von_mises_cdf_normal(self, x, concentration):
    #     """Computes the von Mises CDF and its derivative via Normal approximation."""

    #     # a1 = 12.0
    #     # a2 = 0.8
    #     # a3 = 8.0
    #     # a4 = 1.0
    #     # c1 = 56.0


    #     # c = 24.0 * concentration
    #     # v = c - c1
    #     # r = torch.sqrt((54.0 / (347.0/v+26.0-c)-6.0+c)/6.0)

    #     # z = torch.sin(0.5*x)*r

    #     # s = z * z
    #     # v = v - s + 3.0
    #     # y = (c-s-s-16.0)/3.0
    #     # y = ((s+1.75)*s+83.5)/v-y

    #     # tmp = z-s/(y*y)*z
    #     # distrib = D.Normal(tmp, 1.0)
    #     # return distrib.cdf(xi)


    #     # print(concentration.shape)
    #     # print(x.shape)
    #     # exit()

    #     # print("Is Nan x",torch.sum(torch.isnan(x)).item())
    #     z = (np.sqrt(2. / np.pi) / torch.special.i0e(concentration)) * torch.sin(0.5 * x)


    #     # This is the correction described in [1] which reduces the error
    #     # of the Normal approximation.
    #     z2 = z ** 2.0
    #     z3 = z2 * z
    #     z4 = z2 ** 2.0
    #     c = 24.0 * concentration

    #     c1 = 56.0




    #     a = (c - (2.0 * z2) - 16.0) / 3.0
    #     b = (z4 + (1.75 * z2) + 83.5)
    #     c = (c - c1 - z2 + 3.0)

    #     d = a - (b / (c+1e-8))
    #     # print("Is Nan d1",torch.sum(torch.isnan(d)).item())
    #     # print("Is Negative d",torch.sum(d < 0).item())


    #     # WARNING WARNING WARNING WARNING WARNING
    #     # We do this to prevent pytorch from crashing:
    #     # If d is negative then we have a problem where pytorch could crash because of a NAN gradient BUT we will 
    #     # this only occurs for concentration values below 10.5 and we will never use this method for those values
    #     # So if d is negative then we should set it to not be negative as to prevent NANs
    #     # d = torch.abs(d)
    #     # d[d<1] = 10.0
    #     # d_lt_1 = d < 1
    #     # d[d_lt_1] = 1.0
    #     # d[torch.isnan(d)] = 1.0

    #     # print(torch.sum(torch.isnan(d)))


    #     # print("d_lt_1", torch.sum(d_lt_1))



    #     d = d ** 2.0
    #     # print("Is Nan d2",torch.sum(torch.isnan(d)).item())
    #     xi = z - (z3 / (d+1e-8))

    #     # See WARNING above
    #     # xi[d_lt_1] = 0
    #     # xi[d_lt_1] = xi[d_lt_1].detach()


    #     # print("Is Nan xi",torch.sum(torch.isnan(xi)).item())




    #     # xi = z - z3 / ((c - 2.0 * z2 - 16.0) / 3.0 - (z4 + (7.0 / 4.0) * z2 + 167.0 / 2.0) / (c - c1 - z2 + 3.0)) ** 2.0

    #     distrib = D.Normal(0.0, 1.0)

    #     return distrib.cdf(xi)



    # def diff_of_prob(self, value):
    #     prob = torch.exp(self.log_prob(value))



    #     diff_of_prob = -self.concentration * torch.sin(value - self.loc)
    #     diff_of_prob = diff_of_prob * prob

    #     return diff_of_prob



