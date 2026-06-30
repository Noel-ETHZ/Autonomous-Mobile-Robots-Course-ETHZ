import matplotlib.pyplot as plt
import numpy as np
import PyQt5.Qt as Qt
import PyQt5.QtCore as QtCore
import sys

from collections import namedtuple
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.patches import Ellipse, Rectangle
from matplotlib import lines
from PyQt5.QtWidgets import QWidget, QApplication, QHBoxLayout


FPS = 20                   # simulation frames per second
X_MIN, X_MAX = -10, 10    # simulation environment size along x
Y_MIN, Y_MAX = -10, 10    # simulation environment size along y
CAR_WIDTH = 1.1            # car width
CAR_WHEEL_BASE = 1.2       # car wheel base
CAR_WHEEL_RADIUS = 0.1     # car wheel radius
CAR_LENGTH = 1.4           # car length
NUM_BEAMS = 9              # number of laser range sensors
BEAM_ANGLE = 0.1           # angle between laser beams [rad]
MAX_RANGE = 20.0           # range sensor maximum distance
P_RANGE_OUTLIER = 0.01     # range sensor measurement outlier probability
RANGE_STD = 0.1            # range sensor standard deviation
MAP_RESOLUTION = 0.5       # resolution of the occupancy map

RANGE_SIGMA = 3.0*RANGE_STD # wall measurement uncertainty for inverse sensor model
WALL_THICKNESS = np.max([MAP_RESOLUTION+RANGE_SIGMA, 2*RANGE_SIGMA]) # wall thickness for inverse sensor model
L_MAX = 10.0  # roughly P=1-1e-6
L_OCC = 1.0   # roughly P=75%
L_MIN = -10.0 # roughly P=1e-6
L_FREE = -1.0 # roughly P=25%

class OccupancyMap:
    def __init__(self):
        x_axis = np.arange(X_MIN, X_MAX+MAP_RESOLUTION, MAP_RESOLUTION)
        y_axis = np.arange(Y_MIN, Y_MAX+MAP_RESOLUTION, MAP_RESOLUTION)
        self.x_axis, self.y_axis = np.meshgrid(x_axis, y_axis, sparse=False, indexing='xy')
        self.logOdds = np.zeros((len(x_axis),len(y_axis)))
    def update(self, x, z):
        R_WB = np.array([[np.cos(x[2]), -np.sin(x[2])],
                         [np.sin(x[2]), np.cos(x[2])]])
        for u in range(0,np.size(self.logOdds,1)):
            for v in range(0,np.size(self.logOdds,0)):
                cx = self.x_axis[u,v]
                cy = self.y_axis[u,v]
                corners = np.array([[cx-x[0],cy-x[1]],
                                    [cx-x[0],cy+MAP_RESOLUTION-x[1]],
                                    [cx+MAP_RESOLUTION-x[0],cy-x[1]],
                                    [cx+MAP_RESOLUTION-x[0],cy+MAP_RESOLUTION-x[1]]])
                centre = np.array([cx+0.5*MAP_RESOLUTION-x[0],cy+0.5*MAP_RESOLUTION-x[1]])

                # first check if beyond range
                if np.linalg.norm(centre)>MAX_RANGE+WALL_THICKNESS:
                    continue
                for i in range(0,NUM_BEAMS):
                    zi = z[i]
                    if zi == MAX_RANGE:
                        continue # invalid measurement, ignore.
                    if np.linalg.norm(centre)>zi+WALL_THICKNESS:
                        continue # beyond the wall
                    a = BEAM_ANGLE*(NUM_BEAMS-1)/2.0 - BEAM_ANGLE*i
                    Ri = np.array([[np.cos(a), -np.sin(a)],
                              [np.sin(a),  np.cos(a)]])
                    B_e = np.matmul(Ri,np.array([[1.0],[0.0]]))
                    W_e = np.dot(R_WB,B_e)
                    # check if center of the cell is forward of the ray
                    if np.dot(centre,W_e) < 0.0:
                        continue
                    # next see if corners are mixed left/right of the beam
                    signs = np.zeros(4)
                    for c in range(0,4):
                        if (corners[c,0]*W_e[1]-corners[c,1]*W_e[0]) < 0.0: # 2D cross product
                            signs[c] = -1
                        else:
                            signs[c] = 1
                    if np.max(signs)-np.min(signs) >0:
                        d = np.linalg.norm(centre)
                        if d<zi-3.0*RANGE_STD:
                            dl = L_FREE
                        elif d>=zi-RANGE_SIGMA and d<=zi+RANGE_SIGMA:
                            dl = (d-(zi-RANGE_SIGMA))/(2.0*RANGE_SIGMA)*(L_OCC-L_FREE)+L_FREE
                        elif d>zi+RANGE_SIGMA and d<=zi+WALL_THICKNESS-RANGE_SIGMA:
                            dl = L_OCC
                        elif d>zi+WALL_THICKNESS-RANGE_SIGMA and d<zi+WALL_THICKNESS:
                            dl = (d-(zi+WALL_THICKNESS-RANGE_SIGMA))/(RANGE_SIGMA)*(L_OCC)
                        else:
                            dl = 0.0
                        # now update
                        self.logOdds[u,v] += dl
                        # ... and saturate:
                        if self.logOdds[u,v] < L_MIN:
                            self.logOdds[u,v] = L_MIN
                        if self.logOdds[u,v] > L_MAX:
                            self.logOdds[u,v] = L_MAX

# helper class to store wall segments and compute intersections
class Wall:
    def __init__(self, start, end):
        self.start = np.array([[start[0]],[start[1]]]);
        self.end = np.array([[end[0]],[end[1]]]);

        # now compute line equation
        # (end[0]-start[0])*start[1]- start[0]*(end[1]-start[1]) - (end[0]-start[0])*y + x*(end[1]-start[1]) = 0
        self.a_1 = (end[1]-start[1])
        self.a_2 = -(end[0]-start[0])
        self.a_3 = (end[0]-start[0])*start[1] - start[0]*(end[1]-start[1])
    def computeIntersection(self, startPoint, direction):
        denom = (self.a_1*direction[0] + self.a_2*direction[1])
        if(abs(denom) < 1.0e-8):
            return []
        lambd = (- self.a_3 - self.a_1*startPoint[0] - self.a_2*startPoint[1])/denom
        if lambd < 0:
            return []
        x_star = startPoint + lambd*direction
        if np.dot(np.transpose(x_star-self.end), self.end-self.start)>0:
            return [] # intersection past end
        if np.dot(np.transpose(x_star-self.start), self.end-self.start)<0:
            return [] # intersection before start
        return x_star

class DiffDriveSimulator:

    def __init__(self, x: float, y: float, theta: float):
        self.w = CAR_WHEEL_BASE # wheel base
        self.r = CAR_WHEEL_RADIUS # wheel radius
        self.pose = np.array([x, y, theta])
        self.controls = np.zeros(2)  # speed, rotation rate
        self.previous_controls = np.zeros(2)  # speed, rotation rate
        self.key_state = np.zeros(2) # fwd/bwd, rot-left/rot-right
        self.occupancyMap = OccupancyMap()
        self.walls = [Wall([-9,-9],[9,-9]), Wall([9,-9],[9,3]), Wall([9,9],[9,9]), Wall([9,9],[-9,9]), Wall([-9,9],[-9,-9])]
        self.z = np.zeros(NUM_BEAMS)

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
        wheel_rotvel_diff = omega*self.w/self.r
        wheel_rotvel_mean = v/self.r
        u = np.zeros(2)
        u[0] = wheel_rotvel_mean - wheel_rotvel_diff
        u[1] = wheel_rotvel_mean + wheel_rotvel_diff
        if abs(u[0]) > 0.0001:
            u[0] += np.random.normal(0,0.001)
        if abs(u[1]) > 0.0001:
            u[1] += np.random.normal(0,0.001)

        # generate measurements
        for i in range(0,NUM_BEAMS):
            zi = MAX_RANGE
            for wall in self.walls:
                a = BEAM_ANGLE*(NUM_BEAMS-1)/2.0 - BEAM_ANGLE*i
                Ri = np.array([[np.cos(a), -np.sin(a)],
                              [np.sin(a),  np.cos(a)]])
                R_WB = np.array([[np.cos(self.pose[2]), -np.sin(self.pose[2])],
                                 [np.sin(self.pose[2]),  np.cos(self.pose[2])]])
                B_e = np.matmul(Ri,np.array([[1.0],[0.0]]))
                W_e = np.dot(R_WB,B_e)
                W_r = np.array([[self.pose[0]],[self.pose[1]]])
                W_intersection = wall.computeIntersection(W_r, W_e)
                if len(W_intersection)>0:
                    d=np.linalg.norm(W_intersection - W_r)
                    zi = min(d,zi)
            # add Gaussian noise
            if zi!=MAX_RANGE:
                zi += np.random.normal(0,RANGE_STD)
            if zi>MAX_RANGE:
                zi=MAX_RANGE
            # add outliers
            u = np.random.uniform(0,1)
            if u<P_RANGE_OUTLIER:
                zi = np.random.uniform(0,MAX_RANGE)
            self.z[i] = zi

        # update map
        self.occupancyMap.update(self.pose, self.z)

    @property
    def speed(self) -> float:
        return self.controls[0]

    @property
    def rotation_rate(self) -> float:
        return self.controls[1]

    def __str__(self) -> str:
        return f"DiffDriveSimulator(pose={self.pose})"


class Window(QWidget):

    def __init__(self, parent=None):
        super(Window, self).__init__(parent)

        self.diffDriveSimulator = DiffDriveSimulator(x=0, y=0, theta=0)
        self.k = 0

        # set the layout
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)

        layout = QHBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        screen_size = self.screen().geometry().size()
        min_size = min(screen_size.width(), screen_size.height())
        self.resize(QtCore.QSize(int(min_size / 2), int(min_size / 2)))

        # Timer for updating the view, with a delta t of 1s / fps between frames.
        self.timer = Qt.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(int(1000.0 / FPS))  # in milliseconds

    def update(self):
        self.diffDriveSimulator.update(dt=1 / FPS)
        self.plot()
        self.k += 1

    ####################################################################################################################
    # Interactions #####################################################################################################
    ####################################################################################################################
    def keyPressEvent(self, event):
        key_pressed = event.key()

        if key_pressed == QtCore.Qt.Key.Key_Right:
            self.diffDriveSimulator.key_state[1] = -1
        if key_pressed == QtCore.Qt.Key.Key_Left:
            self.diffDriveSimulator.key_state[1] = 1

        if key_pressed == QtCore.Qt.Key.Key_Up:
            self.diffDriveSimulator.key_state[0] = 1
        if key_pressed == QtCore.Qt.Key.Key_Down:
            self.diffDriveSimulator.key_state[0] = -1

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

    ####################################################################################################################
    # Rendering ########################################################################################################
    ####################################################################################################################
    def plot(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_xlim(X_MIN, X_MAX)
        ax.set_ylim(Y_MIN, Y_MAX)

        # plot the occupancy map
        plt.pcolor(self.diffDriveSimulator.occupancyMap.x_axis,
                   self.diffDriveSimulator.occupancyMap.y_axis,
                   self.diffDriveSimulator.occupancyMap.logOdds,
                   cmap=plt.get_cmap('Greys'), vmin=-10, vmax=10, edgecolors='b', linewidths=0.1)

        x, y, theta = self.diffDriveSimulator.pose
        R_WB = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        dxdy = np.matmul(R_WB, np.array([-CAR_LENGTH / 2, - CAR_WIDTH / 2]))
        dx = np.matmul(R_WB, np.array([[CAR_LENGTH], [0]]))

        car_rect = Rectangle((x + dxdy[0], y + dxdy[1]), angle=theta / np.pi * 180, width=CAR_LENGTH, height=CAR_WIDTH)
        car_x = plt.plot([x,x+dx[0,0]],[y,y+dx[1,0]], color='red')
        ax.add_patch(car_rect)

        # plot walls
        for wall in self.diffDriveSimulator.walls:
            wallLine = lines.Line2D([wall.start[0],wall.end[0]], [wall.start[1],wall.end[1]])
            ax.add_line(wallLine)

        # also plot the measurements
        for i in range(0,len(self.diffDriveSimulator.z)):
            zi = self.diffDriveSimulator.z[i]
            if zi<MAX_RANGE:
                a = BEAM_ANGLE*(NUM_BEAMS-1)/2.0 - BEAM_ANGLE*i
                Ri = np.array([[np.cos(a), -np.sin(a)],
                              [np.sin(a),  np.cos(a)]])
                B_e = np.matmul(Ri,np.array([[1.0],[0.0]]))
                W_e = np.matmul(R_WB,B_e)
                W_measurement = np.array([[x],[y]]) + zi*W_e
                ax.plot(W_measurement[0,0], W_measurement[1,0], '*y')

        self.canvas.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = Window()
    main.show()
    sys.exit(app.exec_())
