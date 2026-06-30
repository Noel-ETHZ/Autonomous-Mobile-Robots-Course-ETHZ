# written by Stefan Leutenegger, TU Munich, November 2021

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib import lines

# Define parameters for GN Optimization here:
k = .1 # huber threshold
c = .1 # cauchy parameter
conv_crit_dx = 1e-05 # criterion for convergence based on norm of update step
loss_type = 'cauchy' # either 'cauchy' or 'huber'

# here's the ground truth model that generates the data:
sigma = 0.1 # the standard deviation of the normally distributed noise
P_outlier = 0.20 # outlier probability
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
        
# implementing estimator(s) for the parameters x

# ----- Least squares estimator, closed-form solution (normal equations) ----
N = len(z_tilde)
H = p_tilde.reshape((-1,1)) ** np.array(range(order+1)).reshape(-1,order+1)
x_min = np.linalg.inv(np.matmul(np.transpose(H),H))
x_min = np.matmul(x_min, np.transpose(H))
x_min = np.matmul(x_min, z_tilde)

# ----- GN Solver and Robustification -----

xk = x_min.copy() # initialisation point for GN
xk = xk.reshape((order+1,1))
iter_count = 0
while(1):
    iter_count += 1
    
    zk = np.zeros(len(p_tilde))
    for j in range(0,len(xk)):
        zk += xk[j] * np.power(p_tilde,j)
    
    err = z_tilde - zk
    A = np.zeros((order+1,order+1))
    b = np.zeros((order+1,1))
    
    for i in range(0,N):
        
        ei = err[i]
        Ei = - p_tilde[i] ** np.array(range(order+1)).reshape(-1,order+1)
        ri = np.abs(ei)
        
        if loss_type == 'huber':
            if ri <= k:
                w = 1
                w_dash = 0
            else:
                w = np.sqrt(2*k*ri - k**2)/ri
                w_dash = k/(ri*np.sqrt(2.0*k*ri - k**2)) - np.sqrt(- k**2 + 2.0*ri*k)/ri**2
 
        elif loss_type == 'cauchy':
            if np.abs(ri) < 1.0e-5:
                w = 1.0
                w_dash = 0.0
            else:
                log_arg = 1.0 + (ri/c)**2
                log = np.log(log_arg)
                w = np.sqrt(c**2*log)/ri
                w_dash = 1.0/(log_arg*np.sqrt(c**2*log)) - np.sqrt(c**2*log)/ri**2
        
        ei_dash = w*ei
        Ei_dash = w_dash*ei/ri*Ei*ei + w*Ei
        A += np.matmul(np.transpose(Ei_dash), Ei_dash)
        b -= np.transpose(Ei_dash)*ei_dash

    # obtain GN step
    dx = np.matmul(np.linalg.inv(A), b)
    xk += dx

    if np.linalg.norm(dx) < conv_crit_dx:
        print("Converged in ", iter_count, " steps.")
        break

# ---------------------------------

# now do the plotting
p_gt = np.linspace(-1,1)
z_gt = np.zeros(len(p_gt))
for i in range(0,len(x_gt)):
    z_gt += x_gt[i]*np.power(p_gt,i)
fig, ax = plt.subplots()
ax.plot(p_gt, z_gt, 'gray', label='Ground Truth')
ax.scatter(p_tilde, z_tilde, marker=(5, 2), label='Measurements')

# plot estimated model, too:
z_normal = np.zeros(len(p_gt))
for i in range(0,len(x_min)):
    z_normal += x_min[i]*np.power(p_gt,i)
z_robustGN = np.zeros(len(p_gt))
for i in range(0,len(xk)):
    z_robustGN += xk[i]*np.power(p_gt,i)
ax.plot(p_gt, z_normal, 'red', label='Solution of normal equation')
ax.plot(p_gt, z_robustGN, 'green', label='Solution using '+loss_type+' loss')
ax.legend()
plt.show()
