import torch
import torch.distributions as D
import matplotlib.pyplot as plt
import numpy as np


from kernel_density_estimator import *





# Generate a distribution for testing


mixture_weights = torch.tensor([0.1, 0.3, 0.6])
means = torch.tensor([[-10, 0, 10], [0.0, 1.0, 1], [10, -10, 0]])
# stds = torch.tensor([[1, 1], [1, 1], [1, 1]])
stds = 1.0
# stds = torch.tensor([1, 5, 1])



mix = D.Categorical(mixture_weights)
comp = D.Independent(D.Normal(means, stds), 1)
gmm = D.MixtureSameFamily(mix, comp)

samples = gmm.sample((100000, ))
x = samples[:, 0].numpy()
y = samples[:, 1].numpy()
plot1 = plt.figure(1)
plt.hist2d(x, y, bins=(100, 100), cmap=plt.cm.jet)
# plt.show()


kde_params = dict()
dim0 = dict()
dim0["distribution_type"] = "Normal"
dim1 = dict()
dim1["distribution_type"] = "Normal"

dim2 = dict()
dim2["distribution_type"] = "Normal"

all_dims = dict()
all_dims[0] = dim0
all_dims[1] = dim1
all_dims[2] = dim2
kde_params["dims"] = all_dims

# print(kde_params)

# print(mixture_weights.shape)
# print(means.shape)
# print(stds.shape)

b = 1
batched_means = torch.zeros((b, 3, 3))
batched_stds = torch.zeros((b, 3))
batched_mixture_weights = torch.zeros((b, 3))

for i in range(b):
	batched_means[i,...] = means	
	batched_stds[i,...] = stds
	batched_mixture_weights[i,...] = mixture_weights


# kde = KernelDensityEstimator(kde_params, means.unsqueeze(0), mixture_weights.unsqueeze(0), stds.unsqueeze(0))
kde = KernelDensityEstimator(kde_params, batched_means, batched_mixture_weights, batched_stds)
kde_samples = kde.sample((100000, ))






log_probs = kde.log_prob(kde_samples)
truth_probs = gmm.log_prob(kde_samples)





print("")
print("")
print("")
print("")
print("")
print("")
print(kde_samples.shape)
print(log_probs.shape)
print(truth_probs.shape)

diff = log_probs - truth_probs
diff = torch.abs(diff)
diff_mean = torch.mean(diff)
diff = torch.sum(diff)
print(diff)
print(diff_mean)


# # kde.compute_gradient_injection(kde_samples)
# kde_samples = kde.inject_gradient(kde_samples)

# # kde_samples = kde_samples[,,:]


# exit()

x = kde_samples[0, :, 0].numpy()
y = kde_samples[0, :, 1].numpy()
plot1 = plt.figure(2)
plt.hist2d(x, y, bins=(100, 100), cmap=plt.cm.jet)
plt.show()
