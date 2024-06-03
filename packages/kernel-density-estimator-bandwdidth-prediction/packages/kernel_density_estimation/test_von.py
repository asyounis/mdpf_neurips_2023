import torch
import torch.distributions as D
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt

from tqdm import tqdm

# from kernel_density_estimator import *
from von_mises_full_dist import *



x = torch.linspace(-np.pi, np.pi, 10000)


concs = [0.1, 1, 10, 20]


for c in concs:

	dist = VonMisesFullDist(loc=0, concentration=c)
	probs = torch.exp(dist.log_prob(x))
	plt.plot(x, probs, label="{}".format(c))

plt.legend()
plt.show()







# # ends = np.linspace(-np.pi, np.pi, 1000)
# ends = np.linspace(-np.pi + 1.0, np.pi, 1000)

# for end in tqdm(ends):

# 	x = torch.linspace(-np.pi, end, 10000)
# 	probs = torch.exp(dist.log_prob(x))

# 	# print(probs)

# 	# plt.plot(x, probs)
# 	# plt.show()


# 	integral = torch.trapezoid(probs, x)

# 	cdf = dist.cdf(end)


	
# 	diff = torch.abs(integral-cdf).item()

# 	if(diff > 1e-6):
# 		print("end", end)
# 		print("integral", integral)
# 		print("cdf", cdf)
# 		print("diff", diff)

# 		assert(False)
# 	# assert(diff < 1e-6)

