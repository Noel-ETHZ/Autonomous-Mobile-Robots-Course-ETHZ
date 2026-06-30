import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import PyQt5.Qt as Qt
import PyQt5.QtCore as QtCore
import sys

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.patches import Ellipse, Rectangle
from PyQt5.QtWidgets import QWidget, QApplication, QHBoxLayout


FPS = 10                   # simulation frames per second
X_MIN, X_MAX = -10, 10     # simulation environment size along x
Y_MIN, Y_MAX = -10, 10     # simulation environment size along y
CAR_WIDTH = 1.1            # car width
CAR_WHEEL_BASE = 1.2       # car wheel base
CAR_WHEEL_RADIUS = 0.1    # car wheel radius
CAR_LENGTH = 1.4           # car length
NUM_LANDMARKS = 15         # number of landmarks to be used
CAM_F = 100                # camera focal length
CAM_W = 200                # image size (pixels)
CAM_C = 99.5               # image centre (pixels)
MATCH_OUTLIER_PROB = 0.05  # outlier probability of a measurement
ANGULAR_NOISE = 0.01       # noise on angle measurement (radians)
PIXEL_NOISE = 6.5          # noise on keypoint measurement (pixels)
RESAMPLING_NOISE_XY = 0.1 # noise added to particles during resampling (meters)
RESAMPLING_NOISE_THETA = 0.05 # noise added to particles during resampling (radians)
NUM_PARTICLES = 200          # number of particles to use in the particle filter
RESAMPLE_EVERY_N_UPDATES = 20       # how often to re-sample in the particle filter (in number of updates)
GLOBAL_INIT = False # whether to initialise the filter with the true pose (cheating) or not (global localisation problem)

def normpdf(x, mean, sd):
    var = float(sd)**2
    denom = (2*np.pi*var)**.5
    num = np.exp(-(float(x)-float(mean))**2/(2*var))
    return num/denom

class ParticleFilter:
    def __init__(self, landmarks):
        self.landmarks = landmarks
        self.need_initialisation = True
        self.N_SAMPLES = NUM_PARTICLES
        self.update_counter = 0
        self.consecutive_failed_updates = 0
        self.particles = np.zeros((self.N_SAMPLES,3)) # each row is a particle, with [x, y, theta]
        self.weights = np.zeros((self.N_SAMPLES,1))
        return
    
    def reinitialise(self, x):
        if GLOBAL_INIT:
            self.particles[:,0] = np.random.uniform(X_MIN, X_MAX, self.N_SAMPLES)
            self.particles[:,1] = np.random.uniform(Y_MIN, Y_MAX, self.N_SAMPLES)
            self.particles[:,2] = np.random.uniform(0, 2*np.pi, self.N_SAMPLES)
        else:
            self.particles[:,0] = x[0] + np.random.normal(0, RESAMPLING_NOISE_XY, self.N_SAMPLES)
            self.particles[:,1] = x[1] + np.random.normal(0, RESAMPLING_NOISE_XY, self.N_SAMPLES)
            self.particles[:,2] = x[2] + np.random.normal(0, RESAMPLING_NOISE_THETA, self.N_SAMPLES)
        self.weights = np.ones((self.N_SAMPLES,1))*1.0/self.N_SAMPLES
        self.need_initialisation = False # since we are cheating a bit, this will always be successful...
        return
    
    def predict(self, u):
        # TODO: implement me...
        return

    def update(self, z):
        # TODO: implement me...
        return
    
    
    def getParticles(self):
        return self.particles

    def getStateAndCov(self):
        x = 0.0
        y = 0.0
        theta = 0.0
        P = None # we don't have a covariance for the particle filter, this function need to return something -> None

        # TODO: implement me...

        return np.array([x, y, theta]), P

        
class Ekf:
    def __init__(self, landmarks):
        self.need_initialisation = True
        self.landmarks = landmarks
        self.x = np.zeros(3) # [x, y, theta]
        self.P = np.zeros((3,3)) # covariance matrix
        return
    
    def reinitialise(self, x):
        if GLOBAL_INIT:
            self.x = np.array([np.random.uniform(X_MIN, X_MAX), np.random.uniform(Y_MIN, Y_MAX), np.random.uniform(0, 2*np.pi)])
            self.P = (np.diag([X_MAX-X_MIN, Y_MAX-Y_MIN, 2*np.pi])/10)**2
        else:
            self.x = x.astype(float)
            self.P = np.array([[0.01, 0.0, 0.0],[0.0, 0.01, 0.0],[0.0, 0.0, 0.0004]])
        self.need_initialisation = False
        return
    
    def predict(self, u):
        # TODO: implement me...
        return
    
    def update(self, z):
        # TODO: implement me...
        return

    def getStateAndCov(self):
        x = self.x
        P = self.P
        return x, P

# helper function to plot a 2D uncertainty ellipse (n-sigma)
def nSigmaEllipse(mean, cov, color, n=3):
    lambda_, v = np.linalg.eig(cov)
    lambda_ = np.sqrt(lambda_)
    ell = Ellipse(xy=(mean[0], mean[1]),
                  width=lambda_[0]*n*2, height=lambda_[1]*n*2,
                  angle=np.rad2deg(np.arccos(v[0, 0])),
                  color=color, fill=color, alpha=0.5, label=f'{n}-sigma pos. ellipse')
    return ell

class DiffDriveSimulator:
    def __init__(self, x: float, y: float, theta: float, estimator_type='ekf'):
        self.b = CAR_WHEEL_BASE # wheel base
        self.r = CAR_WHEEL_RADIUS # wheel radius
        self.num_landmarks = NUM_LANDMARKS # number of landmarks to generate
        self.pose = np.array([x, y, theta])
        self.controls = np.zeros(2)  # speed, rotation rate
        self.previous_controls = np.zeros(2)  # speed, rotation rate
        self.key_state = np.zeros(2) # fwd/bwd, rot-left/rot-right
        self.landmarks = np.random.uniform(low=[X_MIN, Y_MIN], high=[X_MAX, Y_MAX], size=(self.num_landmarks,2))
        if estimator_type == 'pf':
            self.estimator = ParticleFilter(self.landmarks) # select/initialise
        else:
            self.estimator = Ekf(self.landmarks) # select/initialise

        if self.estimator.need_initialisation:
            self.estimator.reinitialise(self.pose)

    def update(self, dt: float):
        # some higher-order dynamics from the key state
        self.controls = self.previous_controls* 0.8 + 0.2*self.key_state*np.array([2.0, np.pi / 4.0])
        self.previous_controls = self.controls
        
        v, omega = self.controls
        _, _, theta = self.pose
        
        # kinematics using Euler-forward discretisation
        x_dot = np.cos(theta) * v
        y_dot = np.sin(theta) * v
        theta_dot = omega
        self.pose = self.pose + dt * np.array([x_dot, y_dot, theta_dot])
        self.pose[2] = self.pose[2] % (2 * np.pi)  # theta -> [0, 2pi)
        
        # simulate the wheel speed measurements (with some noise, if turning)
        wheel_rotvel_diff = omega*self.b/self.r
        wheel_rotvel_mean = v/self.r
        u = np.zeros(2)
        u[0] = wheel_rotvel_mean - 0.5*wheel_rotvel_diff
        u[1] = wheel_rotvel_mean + 0.5*wheel_rotvel_diff
        if abs(u[0]) > 0.0001:
            u[0] += np.random.normal(0,ANGULAR_NOISE/dt)
        if abs(u[1]) > 0.0001:
            u[1] += np.random.normal(0,ANGULAR_NOISE/dt)
        
        # generate measurements
        z = []
        for i in range(0, len(self.landmarks)):
            lm = self.landmarks[i,:]
            R_WB = np.array([[np.cos(self.pose[2]), -np.sin(self.pose[2])],
                             [np.sin(self.pose[2]), np.cos(self.pose[2])]])
            lm_B = np.matmul(np.transpose(R_WB), (lm-self.pose[0:2]).reshape(2,1))
            if lm_B[0]>1.0e-10 :
                cam_u = -CAM_F*lm_B[1]/lm_B[0]+CAM_C
                if cam_u > 0.5 and cam_u < CAM_W-0.5:
                    zi = cam_u + np.random.normal(0,PIXEL_NOISE)
                    o = np.random.uniform(0,1)
                    if o<MATCH_OUTLIER_PROB:
                        zi = [np.random.uniform(-0.5,CAM_W-0.5)]
                    z.append((i, zi[0]))
        
        # call prediction and update functions on your estimator
        if self.estimator.need_initialisation:
            self.estimator.reinitialise(self.pose)
        else:
            self.estimator.predict(u)
            self.estimator.update(z)
                
    @property
    def speed(self) -> float:
        return self.controls[0]

    @property
    def rotation_rate(self) -> float:
        return self.controls[1]

    def __str__(self) -> str:
        return f"DiffDriveSimulator(pose={self.pose})"


class Window(QWidget):

    def __init__(self, parent=None, estimator_type='ekf'):
        super(Window, self).__init__(parent)

        self.diffDriveSimulator = DiffDriveSimulator(x=0, y=0, theta=0, estimator_type=estimator_type)
        self.k = 0
        self.estimator_type = estimator_type

        # set the layout
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        
        layout = QHBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        screen_size = self.screen().geometry().size()
        min_size = min(screen_size.width(), screen_size.height())
        self.resize(QtCore.QSize(int(min_size / 2), int(min_size / 2)))

        self.to_update = False

        # Timer for updating the view, with a delta t of 1s / fps between frames.
        self.timer = Qt.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(int(1000.0 / FPS))  # in milliseconds

    def update(self):
        if self.to_update:
            self.diffDriveSimulator.update(dt=1 / FPS)
            self.k += 1
        self.plot()

    ####################################################################################################################
    # Interactions #####################################################################################################
    ####################################################################################################################
    def keyPressEvent(self, event):
        key_pressed = event.key()
        
        self.to_update = False
        
        if key_pressed == QtCore.Qt.Key.Key_Right:
            self.diffDriveSimulator.key_state[1] = -1
            self.to_update = True
        if key_pressed == QtCore.Qt.Key.Key_Left:
            self.diffDriveSimulator.key_state[1] = 1
            self.to_update = True

        if key_pressed == QtCore.Qt.Key.Key_Up:
            self.diffDriveSimulator.key_state[0] = 1
            self.to_update = True
        if key_pressed == QtCore.Qt.Key.Key_Down:
            self.diffDriveSimulator.key_state[0] = -1
            self.to_update = True

        if key_pressed == QtCore.Qt.Key.Key_Escape:
            self.close()
            
    def keyReleaseEvent(self, event):
        key_pressed = event.key()

        if key_pressed == QtCore.Qt.Key.Key_Right:
            self.diffDriveSimulator.key_state[1] = 0
        if key_pressed == QtCore.Qt.Key.Key_Left:
            self.diffDriveSimulator.key_state[1] = 0

        if key_pressed == QtCore.Qt.Key.Key_Up:
            self.diffDriveSimulator.key_state[0] = 0
        if key_pressed == QtCore.Qt.Key.Key_Down:
            self.diffDriveSimulator.key_state[0] = 0

        if self.diffDriveSimulator.key_state[0] == 0 and self.diffDriveSimulator.key_state[1] == 0:
            self.to_update = False
        else:
            self.to_update = True

    ####################################################################################################################
    # Rendering ########################################################################################################
    ####################################################################################################################
    def plot(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # plot the particles (if using particle filter):
        if self.estimator_type == 'pf':
            particles = self.diffDriveSimulator.estimator.getParticles()
            for i, particle in enumerate(particles):
                [x, y, theta] = particle
                R = np.array([[np.cos(theta), -np.sin(theta)], 
                              [np.sin(theta), np.cos(theta)]])
                dxdy = np.matmul(R, np.array([-CAR_LENGTH / 2, - CAR_WIDTH / 2]))
                dx = np.matmul(R, np.array([[CAR_LENGTH], [0]]))
                car_rect = Rectangle((x + dxdy[0], y + dxdy[1]), angle=theta / np.pi * 180, width=CAR_LENGTH, height=CAR_WIDTH, color='blue', alpha=0.1)
                if i==0:
                    car_x = plt.plot([x,x+dx[0,0]],[y,y+dx[1,0]], color='blue', alpha=0.1, label='particles')
                else:
                    car_x = plt.plot([x,x+dx[0,0]],[y,y+dx[1,0]], color='blue', alpha=0.1)
                ax.add_patch(car_rect)


        x, y, theta = self.diffDriveSimulator.pose
        R = np.array([[np.cos(theta), -np.sin(theta)], 
                      [np.sin(theta), np.cos(theta)]])
        dxdy = np.matmul(R, np.array([-CAR_LENGTH / 2, - CAR_WIDTH / 2]))
        dx = np.matmul(R, np.array([[CAR_LENGTH], [0]]))

        car_rect = Rectangle((x + dxdy[0], y + dxdy[1]), angle=theta / np.pi * 180, width=CAR_LENGTH, height=CAR_WIDTH, color='gray', alpha=0.5)
        car_x = plt.plot([x,x+dx[0,0]],[y,y+dx[1,0]], color='gray', alpha=0.5, label='true pose')
        ax.add_patch(car_rect)

        # Add the camera's field of view
        fov_angle = np.arctan((CAM_W/2)/CAM_F)  # field of view angle in radians
        fov_length = 3.0  # length of the field of view lines
        left_fov_x = x + fov_length * np.cos(theta + fov_angle)
        left_fov_y = y + fov_length * np.sin(theta + fov_angle)
        right_fov_x = x + fov_length * np.cos(theta - fov_angle)
        right_fov_y = y + fov_length * np.sin(theta - fov_angle)
        plt.plot([x, left_fov_x], [y, left_fov_y], color='red', alpha=0.5, label='camera FOV')
        plt.plot([x, right_fov_x], [y, right_fov_y], color='red', alpha=0.5)
        plt.plot([left_fov_x, right_fov_x], [left_fov_y, right_fov_y], color='red', alpha=0.5)

        
        plt.scatter(self.diffDriveSimulator.landmarks[:,0], self.diffDriveSimulator.landmarks[:,1], label='landmarks', color='red')
        # Display the landmark ids
        for i, lm in enumerate(self.diffDriveSimulator.landmarks):
            plt.text(lm[0], lm[1], str(i), fontsize=12, ha='right', va='bottom')

        # plot the estimate of the state (either particle distribution or Gaussian):
        x_est, P_est = self.diffDriveSimulator.estimator.getStateAndCov()
        [x, y, theta] = x_est
        R = np.array([[np.cos(theta), -np.sin(theta)], 
                      [np.sin(theta), np.cos(theta)]])
        dxdy = np.matmul(R, np.array([-CAR_LENGTH / 2, - CAR_WIDTH / 2]))
        dx = np.matmul(R, np.array([[CAR_LENGTH], [0]]))
        car_rect = Rectangle((x + dxdy[0], y + dxdy[1]), angle=theta / np.pi * 180, width=CAR_LENGTH, height=CAR_WIDTH, color='green', alpha=0.5)
        car_x = plt.plot([x,x+dx[0,0]],[y,y+dx[1,0]], color='green', alpha=1.0, label='estimate mean')
        ax.add_patch(car_rect)

        if P_est is not None:        
            ell = nSigmaEllipse(x_est[0:2], P_est[0:2,0:2], color='blue', n=5)
            ax.add_patch(ell)
            # Display the orientation uncertainty as a shaded sector
            theta_std = 5*np.sqrt(P_est[2,2])
            theta_range = np.linspace(theta - theta_std, theta + theta_std, 100)
            sector_x = np.concatenate(([x], x + fov_length/1.5 * np.cos(theta_range)))
            sector_y = np.concatenate(([y], y + fov_length/1.5 * np.sin(theta_range)))
            plt.fill(sector_x, sector_y, color='blue', alpha=0.2, label='5-sigma rot. uncertainty')


        plt.legend(loc = 'upper left')
        plt.axis('equal')
        ax.set_xlim(X_MIN - 0.25*(X_MAX-X_MIN), X_MAX + 0.25*(X_MAX-X_MIN))
        ax.set_ylim(Y_MIN - 0.25*(Y_MAX-Y_MIN), Y_MAX + 0.25*(Y_MAX-Y_MIN))
        self.canvas.draw()


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'pf':
        estimator_type = 'pf'
    else:
        estimator_type = 'ekf'

    if estimator_type != 'pf' and estimator_type != 'ekf':
        print("Usage: python E7_diff_drive_sim_sol.py [pf|ekf]")
        sys.exit(1)
    app = QApplication(sys.argv)
    main = Window(estimator_type=estimator_type)
    main.show()
    sys.exit(app.exec_())
