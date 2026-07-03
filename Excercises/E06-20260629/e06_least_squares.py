# written by Stefan Leutenegger, TU Munich, November 2021

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib import lines

# here's the ground truth model that generates the data:
sigma = 0.1 # the standard deviation of the normally distributed noise
P_outlier = 0.5 # outlier probability
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
        
H = np.empty((len(z_tilde), len(x_gt)))
for i in range(0,len(z_tilde)):
    for j in range(0,len(x_gt)):
        H[i][j] = p_tilde[i]**j

x_min = np.linalg.inv(np.transpose(H)@H)@np.transpose(H)@z_tilde

# Gauss Newton implementation:
x_k = x_min.copy().reshape((order+1, 1)) # Take the direct maybe biased estimate as starting point.
E = -H
iter_count = 0
while True:
    iter_count+=1
    # 1. compute current model prediction and residuals
    z_k = H@x_k
    err = z_tilde.reshape(-1,1) - z_k
    # 2. build A, b for this iteration
    A = np.zeros((order+1, order+1))
    b = np.zeros((order+1, 1))

    for i in range(0,numSamples):
        ei = err[i, 0]
        ri = np.abs(ei)
        Ei = E[i:i+1, :]
        
    

    # 3. solve for dx, update xk
    # 4. check convergence
    break


# now do the plotting
p_gt = np.linspace(-1,1)
z_gt = np.zeros(len(p_gt))
for i in range(0,len(x_gt)):
    z_gt += x_gt[i]*np.power(p_gt,i)
fig, ax = plt.subplots()
ax.plot(p_gt, z_gt, 'gray', label='Ground Truth')
ax.scatter(p_tilde, z_tilde, marker=(5, 2), label='Measurements')
# TODO: plot your estimated model, too!
# Direct Solution
z_est = np.zeros(len(p_gt))
for i in range(0, len(x_min)):
    z_est += x_min[i] * np.power(p_gt, i)
ax.plot(p_gt, z_est, 'red', label='Solution of normal equation')
# Gauss newton method:



ax.legend()
plt.show()
