# written by Stefan Leutenegger, MRL (ETH Zürch), December 2021

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
r = 1.0 # circle radius
image_width = 640.0 # image width in pixels
image_height = 480.0 # image height in pixels
f = 500.0 # focal length in pixels
P_outlier = 0.01 # outlier probability

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
        if C_l[2]<1.0e-10:
            return [], []
        x_dash = np.array([[C_l[0]/C_l[2]],[C_l[1]/C_l[2]]])
        u = self.f*x_dash + np.array([[self.c_1],[self.c_2]])
        if u[0,0]<-0.5 or u[0,0]>self.image_width-0.5:
            return [], []
        if u[1,0]<-0.5 or u[1,0]>self.image_height-0.5:
            return [], []
        # also do the Jacobian
        J = self.f * np.array([[1.0/C_l[2], 0.0, -C_l[0]/C_l[2]**2],[0.0, 1.0/C_l[2], -C_l[1]/C_l[2]**2]])
        return u, J
    
# make up some random landmarks
W_l = []
for j in range(0,40):
    x = np.random.uniform(-b/2.0, b/2.0)
    z = np.random.uniform(0, h)
    W_l.append(np.array([[x],[-b/2.0],[z]]))
for j in range(0,40):
    x = np.random.uniform(-b/2.0, b/2.0)
    z = np.random.uniform(0, h)
    W_l.append(np.array([[x],[b/2.0],[z]]))
for j in range(0,40):
    y = np.random.uniform(-b/2.0, b/2.0)
    z = np.random.uniform(0, h)
    W_l.append(np.array([[-b/2.0],[y],[z]]))
for j in range(0,40):
    y = np.random.uniform(-b/2.0, b/2.0)
    z = np.random.uniform(0, h)
    W_l.append(np.array([[b/2.0],[y],[z]]))
for j in range(0,40):
    x = np.random.uniform(-b/2.0, b/2.0)
    y = np.random.uniform(-b/2.0, b/2.0)
    W_l.append(np.array([[x],[y],[0]]))
for j in range(0,40):
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
        u, J = camera.project(C_l_j)
        if len(u)>0:
            u = u + np.random.normal(0,0.3,size=(2,1)) # add some realistic noise
            X = np.random.uniform(0,1)
            #print(C_l_j, C_l_j_dash)
            if X<P_outlier:
                u_tilde.append((u, i, np.random.randint(len(W_l))))
            else:
                u_tilde.append((u, i, j))

def crossmx(a):
    a = a.reshape(3)
    c = np.array([[0.0, -a[2], a[1]],[a[2], 0.0, -a[0]],[-a[1], a[0], 0.0]])
    return c

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
max_iter = 100
lambd = 0.0
for n in range(0,max_iter):
    # prepare GN system and ordering bookkeeping
    dim_i = len(W_t_C_bar)*6
    dim_j = len(W_l_bar)*3
    dim = dim_i + dim_j
    A = np.zeros((dim,dim))
    b = np.zeros((dim,1))
    c = 0.0;
        
    # compute A*dx = b as well as cost
    for (u_tilde_ij, i, j) in u_tilde:
        R_WCi_bar = R.as_matrix(rot_WC_bar[i])
        R_CiW_bar = np.transpose(R_WCi_bar)
        W_t_Ci_bar = W_t_C_bar[i]
        # evaluate error
        Ci_lj_bar = np.matmul(R_CiW_bar,(W_l_bar[j]-W_t_Ci_bar))
        u, J = camera.project(Ci_lj_bar)
        if(len(u) == 0):
            continue # invalid projection
        e_ij = u_tilde_ij - u
        # update cost
        if np.matmul(np.transpose(e_ij), e_ij) > 10000.0:
            c += 0.5 * 10000.0
            continue # outlier threshold 100 pixels
        c += 0.5 * np.matmul(np.transpose(e_ij), e_ij)

        # compute Jacobians
        E_i = -np.matmul(J,np.concatenate((-R_CiW_bar, np.matmul(R_CiW_bar,crossmx(W_l_bar[j]-W_t_Ci_bar))),axis=1))
        E_j = -np.matmul(J,R_CiW_bar)
       
        # update A*dx = b
        A[i*6:i*6+6, i*6:i*6+6] += np.matmul(np.transpose(E_i),E_i)
        b[i*6:i*6+6] += -np.matmul(np.transpose(E_i),e_ij)
        A[dim_i+j*3:dim_i+j*3+3, i*6:i*6+6] += np.matmul(np.transpose(E_j),E_i)
        A[i*6:i*6+6, dim_i+j*3:dim_i+j*3+3] += np.matmul(np.transpose(E_i),E_j)
        A[dim_i+j*3:dim_i+j*3+3, dim_i+j*3:dim_i+j*3+3] += np.matmul(np.transpose(E_j),E_j)
        b[dim_i+j*3:dim_i+j*3+3] += -np.matmul(np.transpose(E_j),e_ij)
    
    # augment the system according to the Levenberg-Marquardt scheme
    if lambd == 0.0:
        lambd = 1000 # slight hack to find lambda0 in the beginning
    A += lambd*np.eye(dim)
        
    # solve for dx
    dx = np.linalg.solve(A,b)

    # apply it to x_bar preliminarily
    for i in range(0, len(W_t_C_bar)):
        W_t_C_bar[i] += dx[i*6:i*6+3] # apply by addition
    for i in range(0, len(rot_WC_bar)):
        dq = R.from_rotvec(dx[i*6+3:i*6+6].reshape(3)) # rotation vector to quaternion
        rot_WC_bar[i] = dq*rot_WC_bar[i] # apply delta quaternion
    for j in range(0, len(W_l_bar)):
        W_l_bar[j] += dx[dim_i + j*3:dim_i + j*3+3] # apply by addition
        
    # recompute cost
    c1 = 0.0
    for (u_tilde_ij, i, j) in u_tilde:
        R_WCi_bar = R.as_matrix(rot_WC_bar[i])
        R_CiW_bar = np.transpose(R_WCi_bar)
        W_t_Ci_bar = W_t_C_bar[i]
        # evaluate error
        Ci_lj_bar = np.matmul(R_CiW_bar,(W_l_bar[j]-W_t_Ci_bar))
        u, J = camera.project(Ci_lj_bar)
        if(len(u) == 0):
            continue # invalid projection
        e_ij = u_tilde_ij - u
        # update cost
        if np.matmul(np.transpose(e_ij), e_ij) > 10000.0:
            c1 += 0.5 * 10000.0
            continue # outlier threshold 100 pixels
        c1 += 0.5 * np.matmul(np.transpose(e_ij), e_ij)
        
    # check for progress
    if(c1>c):
        # undo update to x_bar
        for i in range(0, len(W_t_C_bar)):
            W_t_C_bar[i] -= dx[i*6:i*6+3] # apply by addition
        for i in range(0, len(rot_WC_bar)):
            dq = R.from_rotvec(dx[i*6+3:i*6+6].reshape(3)) # rotation vector to quaternion
            rot_WC_bar[i] = dq.inv()*rot_WC_bar[i] # apply delta quaternion
        for j in range(0, len(W_l_bar)):
            W_l_bar[j] -= dx[dim_i + j*3:dim_i + j*3+3] # apply by addition
        lambd *= 2
    else:
        lambd /= 2
        
    print("iteration ", n, ", cost = ", c, "success = ", c1<c, "lambda = ", lambd)
    
    # check convergence
    if abs(c1-c)/c1 < 1.0e-7:
        break
    
baVisualiser.draw(W_t_C_bar, rot_WC_bar, W_l_bar)
