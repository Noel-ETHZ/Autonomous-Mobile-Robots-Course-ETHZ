# written by Stefan Leutenegger, MRL (ETH Zürch), November 2021

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib import lines
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

# draw the BA problem
class BaVisualiser:
    def __init__(self, W_t_C, rot_WC, W_l):
        self.W_t_C = W_t_C
        self.rot_WC = rot_WC
        self.W_l = W_l
        
        # start plotting
        self.fig = plt.figure
        self.ax = plt.axes(projection='3d')
        self.ex = np.array([[1.0],[0.0],[0.0]])*0.2
        self.ey = np.array([[0.0],[1.0],[0.0]])*0.2
        self.ez = np.array([[0.0],[0.0],[1.0]])*0.2
        
    def drawGroundTruth(self):
        
        # plot ground truth poses
        for i in range(0,len(self.W_t_C)):
            label = ["Camera pose x (ground truth)",
                     "Camera pose y (ground truth)",
                     "Camera pose z (ground truth)"] if i == 0 else 3*[None]
            W_t_C_i = self.W_t_C[i]
            rot_WC_i = self.rot_WC[i]
            endx = W_t_C_i + rot_WC_i.apply(self.ex.reshape(3)).reshape(3,1)
            endy = W_t_C_i + rot_WC_i.apply(self.ey.reshape(3)).reshape(3,1)
            endz = W_t_C_i + rot_WC_i.apply(self.ez.reshape(3)).reshape(3,1)
            self.ax.plot([W_t_C_i[0,0],endx[0,0]],
                           [W_t_C_i[1,0],endx[1,0]],
                           [W_t_C_i[2,0],endx[2,0]],
                           color='red', alpha=0.2, label=label[0])
            self.ax.plot([W_t_C_i[0,0],endy[0,0]],
                           [W_t_C_i[1,0],endy[1,0]],
                           [W_t_C_i[2,0],endy[2,0]],
                           color='green', alpha=0.2, label=label[1])
            self.ax.plot([W_t_C_i[0,0],endz[0,0]],
                           [W_t_C_i[1,0],endz[1,0]],
                           [W_t_C_i[2,0],endz[2,0]],
                           color='blue', alpha=0.2, label=label[2])
        
        # plot ground truth landmarks
        for i, W_l_j in enumerate(self.W_l):
            label = "Landmarks (ground truth)" if i == 0 else None
            self.ax.scatter(W_l_j[0,0], W_l_j[1,0], W_l_j[2,0],  marker='o', color='black', alpha=0.1, label=label)
        
    def draw(self, W_t_C_bar, rot_WC_bar, W_l_bar):
        self.drawGroundTruth()
        for i in range(0,len(W_t_C_bar)):
            label = ["Camera pose x (estimated)",
                     "Camera pose y (estimated)",
                     "Camera pose z (estimated)"] if i == 0 else 3*[None]
            W_t_C_i = W_t_C_bar[i]
            rot_WC_i = rot_WC_bar[i]
            endx = W_t_C_i + rot_WC_i.apply(self.ex.reshape(3)).reshape(3,1)
            endy = W_t_C_i + rot_WC_i.apply(self.ey.reshape(3)).reshape(3,1)
            endz = W_t_C_i + rot_WC_i.apply(self.ez.reshape(3)).reshape(3,1)
            self.ax.plot([W_t_C_i[0,0],endx[0,0]],
                           [W_t_C_i[1,0],endx[1,0]],
                           [W_t_C_i[2,0],endx[2,0]],
                           color='red', label=label[0])
            self.ax.plot([W_t_C_i[0,0],endy[0,0]],
                           [W_t_C_i[1,0],endy[1,0]],
                           [W_t_C_i[2,0],endy[2,0]],
                           color='green', label=label[1])
            self.ax.plot([W_t_C_i[0,0],endz[0,0]],
                           [W_t_C_i[1,0],endz[1,0]],
                           [W_t_C_i[2,0],endz[2,0]],
                           color='blue', label=label[2])
        for i, W_l_j in enumerate(W_l_bar):
            label = "Landmarks (estimated)" if i == 0 else None
            self.ax.scatter(W_l_j[0,0], W_l_j[1,0], W_l_j[2,0],  marker='o', color='black', alpha=0.3, label=label)
        self.ax.legend()
        plt.show()
        
# configurable parameters:
b = 8.0 # room side length
h = 3.5 # room height
hc = 1.2 # height of camera
r = 2.0 # circle radius
image_width = 640.0 # image width in pixels
image_height = 480.0 # image height in pixels
f = 500.0 # focal length in pixels
P_outlier = 0.0 # outlier probability

# camera projection helper class:
class Camera:
    def __init__(self, image_width, image_height, f):
        self.image_width = image_width # image width in pixels
        self.image_height = image_height # image height in pixels
        self.f = f # focal length in pixels
        self.c_1 = image_width/2.0 - 0.5
        self.c_2 = image_height/2.0 - 0.5
    def project(self, C_l):
        C_l = C_l.reshape(3)
        # TODO: you might want to implement the Jacobian computation in here.
        if C_l[2]<1.0e-10:
            return []
        x_dash = np.array([[C_l[0]/C_l[2]],[C_l[1]/C_l[2]]])
        u = self.f*x_dash + np.array([[self.c_1],[self.c_2]])
        if u[0,0]<-0.5 or u[0,0]>self.image_width-0.5:
            return []
        if u[1,0]<-0.5 or u[1,0]>self.image_height-0.5:
            return []
        return u
    
# make up some random landmarks
W_l = []
for j in range(0,20):
    x = np.random.uniform(-b/2.0, b/2.0)
    z = np.random.uniform(0, h)
    W_l.append(np.array([[x],[-b/2.0],[z]]))
for j in range(0,20):
    x = np.random.uniform(-b/2.0, b/2.0)
    z = np.random.uniform(0, h)
    W_l.append(np.array([[x],[b/2.0],[z]]))
for j in range(0,20):
    y = np.random.uniform(-b/2.0, b/2.0)
    z = np.random.uniform(0, h)
    W_l.append(np.array([[-b/2.0],[y],[z]]))
for j in range(0,20):
    y = np.random.uniform(-b/2.0, b/2.0)
    z = np.random.uniform(0, h)
    W_l.append(np.array([[b/2.0],[y],[z]]))
for j in range(0,20):
    x = np.random.uniform(-b/2.0, b/2.0)
    y = np.random.uniform(-b/2.0, b/2.0)
    W_l.append(np.array([[x],[y],[0]]))
for j in range(0,20):
    x = np.random.uniform(-b/2.0, b/2.0)
    y = np.random.uniform(-b/2.0, b/2.0)
    W_l.append(np.array([[x],[y],[h]]))

# simulate poses and keypoint measurments
W_t_C = []
rot_WC = []
rot_WC0 = R.from_quat([0, np.sin(np.pi/2.0/2.0), 0,  np.cos(np.pi/2.0/2.0)])
u_tilde = []
camera = Camera(image_width, image_height, f)
for i in range(0,50):
    theta = i * 2.0 * np.pi / 50
    W_t_C_i = np.array([[r*np.cos(theta)],[r*np.sin(theta)],[hc]])
    W_t_C.append(W_t_C_i)
    rot_WC_i = rot_WC0*R.from_quat([np.sin(-theta/2.0), 0, 0,  np.cos(-theta/2.0)])
    rot_WC.append(rot_WC_i)
    for j in range(0,len(W_l)):
        W_l_j = W_l[j]
        C_l_j = rot_WC_i.inv().apply(W_l_j.reshape(1,3)) - rot_WC_i.inv().apply(W_t_C_i.reshape(1,3))
        u = camera.project(C_l_j)
        if len(u)>0:
            u = u + np.random.normal(0,0.3,size=(2,1)) # add some realistic noise
            X = np.random.uniform(0,1)
            if X<P_outlier:
                u_tilde.append((u, i, np.random.randint(len(W_l))))
            else:
                u_tilde.append((u, i, j))

# the estimator:
max_iter = 100

# since this is a simulation, we can use as a simple starting point a perturbation of the ground truth:
W_t_C_bar = []
rot_WC_bar = []
W_l_bar = []
for W_t_C_i in W_t_C:
    W_t_C_bar.append(W_t_C_i + np.random.normal(0,0.1,size=(3,1))) # perturb camera positions with 10 cm stdev.
for rot_WC_i in rot_WC:
    dq = R.from_rotvec(np.random.normal(0,0.001,size=(3))) # rotation vector perturbation with 0.001 rad stdev.
    rot_WC_bar.append(dq*rot_WC_i)
for W_l_j in W_l:
    W_l_bar.append(W_l_j + np.random.normal(0,0.2,size=(3,1))) # perturb landmark positions with 20 cm stdev.
    
# let's have a visualiser
baVisualiser = BaVisualiser(W_t_C, rot_WC, W_l)

# now for the Levenberg-Marquardt minimisation:
#for i in range(0,max_iter):
    # TODO: compute A*dx = b as well as cost
    # TODO: augment the system according to the Levenberg-Marquardt scheme
    # TODO: solve for dx and apply it to x_bar
    # TODO: check for progress
    # TODO: convergence
    
baVisualiser.draw(W_t_C_bar, rot_WC_bar, W_l_bar)
