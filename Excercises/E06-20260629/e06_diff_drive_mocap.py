# written by Stefan Leutenegger, TU Munich, November 2021

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib import lines
import matplotlib.patches as mpatches

# definition of camera poses (in world frame)
T_WC1 = np.array([[0.0, 0.0, -1.0, 6.0],
                [1.0, 0.0, 0.0, 1.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]])
T_WC2 = np.array([[-1.0, 0.0, 0.0, 2.0],
                [0.0, 0.0, -1.0, 7.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]])
T_WC3 = np.array([[0.0, 0.0, 1.0, -1.0],
                [-1.0, 0.0, 0.0, 3.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]])
T_WC = [T_WC1, T_WC2, T_WC3]

# now unfortunately the ground truth will be a bit perturbed
T_WC_gt = []
for i in range(0,len(T_WC)):
    T_WCi = T_WC[i]
    alpha = np.random.normal(0.0,np.pi/180.0)
    t = np.random.normal(0.0, 0.01, size = 2)
    T_WgtW = np.array([[np.cos(alpha), -np.sin(alpha), 0.0, t[0]],
                         [np.sin(alpha), np.cos(alpha), 0.0, t[1]],
                         [0.0, 0.0, 1.0, 0.0],
                         [0.0, 0.0, 0.0, 1.0]])
    if i==len(T_WC)-1:
        T_WgtW = np.eye(4) # not altering the last one, we keep this to "reference" the world frame
    T_WC_gt.append(np.matmul(T_WgtW, T_WCi))

# cameras are assumed with limit in field of view:
fov = np.pi/2.0 # pi/2 = 90 deg

# definition of the marker positions (in robot body frame)
B_t_1 = np.array([[0.1],[0.1],[0.0]])
B_t_2 = np.array([[-0.1],[0.1],[0.0]])
B_t_3 = np.array([[-0.02],[-0.05],[0.0]])
B_t_4 = np.array([[-0.10],[0.05],[0.0]])
B_t = [B_t_1, B_t_2, B_t_3, B_t_4]

# here's the estimator class
class RobotPoseEstimator:
    def __init__(self, T_WC, B_t):
        self.T_WC = T_WC
        self.B_t = B_t
    
    def estimateRobotPose(self, z, x0):
        # TODO: implement!
        return x0

# draw the robot at a specific state
class RobotSimulator:
    def __init__(self, T_WC, fov, B_t, estimator):
        # external parameters: camera poses, field of view, marker positions, and estimator
        self.T_WC = T_WC
        self.fov = fov
        self.B_t = B_t
        self.estimator = estimator
        
        # some constants -- change if you need to
        self.fps = 20.0 # frames per second for animation [1/sec]
        self.duration = 30.0 # animation duration [sec]
        self.R = 1.2 # circle radius the robot will be on [m]
        self.centre = np.array([[2],[2]])
        self.Omega = 2.0*np.pi/self.duration*2 # do 2 circles
        self.b = 0.4 # robot base [m]
        self.l = 0.3 # robot length [m]
        
        # initialise plotting
        self.fig = plt.figure()
        self.ax = plt.axes(xlim=(-1.1, 6.1), ylim=(-1.1, 7.1))
        plt.gca().set_aspect('equal', adjustable='box')
        self.box, = self.ax.plot([], [], 'gray')
        self.xaxis, = self.ax.plot([], [], color='gray')
        self.box_estimated, = self.ax.plot([], [], color='C0')
        self.xaxis_estimated, = self.ax.plot([], [], color='red')
        self.cameras_x = []
        self.cameras_z = []
        est_patch = mpatches.Patch(edgecolor='C0', fill=False, label='Estimated Robot Pose', linewidth=1.5)
        gt_patch = mpatches.Patch(edgecolor='gray', fill=False, label='GT Robot Pose', linewidth=1.5)
        self.ax.legend(handles=[est_patch, gt_patch], handlelength=3*self.b, handleheight=2*self.b)
        
    # initialisation function: plot the background
    def init(self):
        # draw camera coordinate frames
        self.cameras_x = []
        self.cameras_z = []
        for T_WCi in self.T_WC:
            start = np.array([T_WCi[0,3],T_WCi[1,3]])
            endz = T_WCi[0:2,2] + start
            endx = T_WCi[0:2,0] + start
            camera_z, = self.ax.plot([start[0],endz[0]], [start[1],endz[1]], color='blue')
            camera_x, = self.ax.plot([start[0],endx[0]], [start[1],endx[1]], color='red')
            self.cameras_x.append(camera_x)
            self.cameras_z.append(camera_z)
        return self.box, self.xaxis, self.box_estimated, self.xaxis_estimated, *self.cameras_x, *self.cameras_z
        
    def drawRobot(self, x, x_estimated):
        l=self.l
        b=self.b
        B_points = np.array([[-l/2,-b/2],[l/2,-b/2], [l/2,b/2], [-l/2,b/2], [-l/2,-b/2]]).transpose()
        # draw the ground truth robot:
        R_WB = np.array([[np.cos(x[2]), -np.sin(x[2])],
                         [np.sin(x[2]), np.cos(x[2])]])
        W_points = np.dot(R_WB, B_points) + np.array(
            [[x[0], x[0], x[0], x[0], x[0]],[x[1], x[1], x[1], x[1], x[1]]])
        self.box.set_data(W_points[0,:], W_points[1,:])
        startx = np.array([x[0],x[1]])
        endx = R_WB[:,0] + startx
        self.xaxis.set_data([startx[0],endx[0]], [startx[1],endx[1]])
        # draw the estimated robot:
        R_WB_estimated = np.array([[np.cos(x_estimated[2]), -np.sin(x_estimated[2])],
                                   [np.sin(x_estimated[2]), np.cos(x_estimated[2])]])
        W_points_estimated = np.dot(R_WB_estimated, B_points) + np.array(
            [[x_estimated[0],x_estimated[0], x_estimated[0], x_estimated[0], x_estimated[0]],
             [x_estimated[1],x_estimated[1], x_estimated[1], x_estimated[1], x_estimated[1]]])
        self.box_estimated.set_data(W_points_estimated[0,:], W_points_estimated[1,:])
        startx_estimated = np.array([x_estimated[0],x_estimated[1]])
        endx_estimated = R_WB_estimated[:,0] + startx_estimated
        self.xaxis_estimated.set_data(
            [startx_estimated[0],endx_estimated[0]], [startx_estimated[1],endx_estimated[1]])
            
    # update the robot
    def step(self, k):
        # generate ground truth trajectory
        t = k/self.fps
        x = np.array([self.R*np.cos(self.Omega*t)+self.centre[0,0],
             self.R*np.sin(self.Omega*t)+self.centre[1,0],
             np.pi/2 + self.Omega*t]);
        
        # generate measurements
        T_WB = np.array([[np.cos(x[2]), -np.sin(x[2]), 0, x[0]],
                         [np.sin(x[2]), np.cos(x[2]), 0, x[1]],
                         [0.0, 0.0, 1.0, 0.0],
                         [0.0, 0.0, 0.0, 1.0]])
        z=[]
        for i in range(0,len(self.T_WC)):
            T_WCi = self.T_WC[i]
            T_CiW = np.linalg.inv(T_WCi) # TODO: use the formula for homeogeneous transformations instead...
            z_i=[]
            for j in range(0,len(self.B_t)):
                B_tj = self.B_t[j]
                B_tj_h = np.array([[B_tj[0,0]], [B_tj[1,0]], [B_tj[2,0]], [1.0]]) # to homogeneous, 4x1
                Ci_tj = np.dot(T_CiW, np.dot(T_WB, B_tj_h))
               
                Ci_tj_xz = Ci_tj[[0, 2]]
                Ci_tj_xz = np.squeeze(Ci_tj_xz)
                z_ij = Ci_tj_xz/np.linalg.norm(Ci_tj_xz) + np.random.normal(0.0, 0.01, size=2)
                z_ij = z_ij/np.linalg.norm(z_ij)
                
                cosfov = Ci_tj[2,0]/ np.linalg.norm(Ci_tj[:3,0])
                if np.arccos(cosfov)<self.fov:
                    measurement = (i, j, z_ij)
                    z.append(measurement)
        
        # call the estimatorLh
        x0 = np.array([self.R+self.centre[0,0], self.centre[1,0], np.pi/2])
        x_estimated = self.estimator.estimateRobotPose(z, x0)
        self.drawRobot(x, x_estimated)
        
        return self.box, self.xaxis, self.box_estimated, self.xaxis_estimated, *self.cameras_x, *self.cameras_z
    
    # start animation
    def run(self):
        self.anim = animation.FuncAnimation(
            self.fig, self.step, init_func=self.init,
            frames=int(self.duration*self.fps), interval=int(1000.0/self.fps), blit=True)
        plt.show()

# simulate the robot
robotPoseEstimator = RobotPoseEstimator(T_WC, B_t) # estimator does not know ground truth camera poses
robotSimulator = RobotSimulator(T_WC_gt, fov, B_t, robotPoseEstimator) # simulator needs ground truth camera poses
robotSimulator.run()
