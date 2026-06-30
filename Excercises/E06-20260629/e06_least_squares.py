# written by Stefan Leutenegger, TU Munich, November 2021

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib import lines

# here's the ground truth model that generates the data:
sigma = 0.1 # the standard deviation of the normally distributed noise
P_outlier = 0.0 # outlier probability
order = 5 # order of the polynomial
x_gt = np.random.normal(0.0, 1.0, size = order + 1)

# let's generate the (noisy!) data samples
numSamples = 100
p_tilde = np.random.uniform(-1, 1, size = numSamples)
z_tilde = np.zeros(len(p_tilde))
for j in range(0,len(x_gt)):
    z_tilde += x_gt[j]*np.power(p_tilde,j)
z_tilde += np.random.normal(0.0,sigma, size = len(z_tilde))

# add outliers
m = min(z_tilde)-sigma
p = max(z_tilde)+sigma
for i in range(0,len(z_tilde)):
    rnd = np.random.uniform(0,1)
    if rnd < P_outlier:
        z_tilde[i] = np.random.uniform(m, p)
        
# TODO: implement your estimator(s) for the parameters x!
    

# now do the plotting
p_gt = np.linspace(-1,1)
z_gt = np.zeros(len(p_gt))
for i in range(0,len(x_gt)):
    z_gt += x_gt[i]*np.power(p_gt,i)
fig, ax = plt.subplots()
ax.plot(p_gt, z_gt, 'gray', label='Ground Truth')
ax.scatter(p_tilde, z_tilde, marker=(5, 2), label='Measurements')
# TODO: plot your estimated model, too!
plt.show()
