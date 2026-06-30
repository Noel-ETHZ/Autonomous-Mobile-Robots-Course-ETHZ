# written by Stefan Leutenegger, ETH Zurich, November 2021

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib import lines


# helper class generate a noisy measurement from the ground truth distance d
class UltrasonicSensor:
    def __init__(self, d_min, d_max, sigma, P_outlier):
        self.d_min = d_min
        self.d_max = d_max
        self.sigma = sigma
        self.P_outlier = P_outlier

    def generateMeasurement(d):
        # TODO: implement
        return []


# helper class to store wall segments and compute intersections
class Wall:
    def __init__(self, start, end):
        self.start = start
        self.end = end

        # now compute line equation
        # self.a_1 =
        # self.a_2 =

    def computeIntersection(startPoint, direction):
        # TODO: implement. Return [] if not intersecting.
        return []


# define walls:
walls = [
    Wall([0, 0], [5, 0]),
    Wall([5, 0], [5, 3]),
    Wall([5, 4], [5, 6]),
    Wall([5, 6], [0, 6]),
    Wall([0, 6], [0, 0]),
]

# define sensors:
d_min = 0.1  # minimum distance [m]
d_max = 2.5  # maximum distance [m]
sigma = 0.02  # measurement Gaussian noise stdev [m],
P_outlier = 0.05  #
sensor_1 = UltrasonicSensor(d_min, d_max, sigma, P_outlier)
sensor_2 = UltrasonicSensor(d_min, d_max, sigma, P_outlier)
sensor_3 = UltrasonicSensor(d_min, d_max, sigma, P_outlier)


# draw the robot at a specific state
class RobotVisualiser:
    def __init__(self):
        # some constants -- change if you need to
        self.fps = 20.0  # frames per second for animation [1/sec]
        self.duration = 30.0  # animation duration [sec]
        self.R = 1.2  # circle radius the robot will be on [m]
        self.centre = np.array([[2], [2]])
        self.Omega = 2.0 * np.pi / self.duration * 2  # do 2 circles
        self.b = 0.4  # robot base [m]
        self.l = 0.3  # robot length [m]

        # initialise plotting
        self.fig = plt.figure()
        self.ax = plt.axes(xlim=(-1, 6), ylim=(-1, 7))
        plt.gca().set_aspect("equal", adjustable="box")
        (self.box,) = self.ax.plot([], [])
        (self.xaxis,) = self.ax.plot([], [], color="red")
        self.measurement_data_x = []
        self.measurement_data_y = []
        (self.measurements,) = self.ax.plot(
            self.measurement_data_x, self.measurement_data_y, "*y"
        )

    # initialisation function: plot the background
    def init(self):
        self.wallLines = []
        for wall in walls:
            wallLine = lines.Line2D(
                [wall.start[0], wall.end[0]], [wall.start[1], wall.end[1]]
            )
            self.ax.add_line(wallLine)
            self.wallLines.append(wallLine)
        return self.box, self.xaxis, *self.wallLines, self.measurements

    def drawRobot(self, x):
        l = self.l
        b = self.b
        B_points = np.array(
            [
                [-l / 2, -b / 2],
                [l / 2, -b / 2],
                [l / 2, b / 2],
                [-l / 2, b / 2],
                [-l / 2, -b / 2],
            ]
        ).transpose()
        R_WB = np.array([[np.cos(x[2]), -np.sin(x[2])], [np.sin(x[2]), np.cos(x[2])]])
        W_points = np.dot(R_WB, B_points) + np.array(
            [[x[0], x[0], x[0], x[0], x[0]], [x[1], x[1], x[1], x[1], x[1]]]
        )
        self.box.set_data(W_points[0, :], W_points[1, :])
        startx = np.array([x[0], x[1]])
        endx = R_WB[:, 0] + startx
        self.xaxis.set_data([startx[0], endx[0]], [startx[1], endx[1]])
        self.measurement_data_x = []
        self.measurement_data_y = []

        # TODO: also overlay the distance measurements drawn into world frame...
        # (use fill self.measurement_data_x / self.measurement_data_y and .set_data on self.measurements)

    # update the robot
    def update(self, k):
        t = k / self.fps
        x = [
            self.R * np.cos(self.Omega * t) + self.centre[0, 0],
            self.R * np.sin(self.Omega * t) + self.centre[1, 0],
            np.pi / 2 + self.Omega * t,
        ]
        self.drawRobot(x)
        return self.box, self.xaxis, *self.wallLines, self.measurements

    # start animation
    def animate(self):
        self.anim = animation.FuncAnimation(
            self.fig,
            self.update,
            init_func=self.init,
            frames=int(self.duration * self.fps),
            interval=int(1000.0 / self.fps),
            blit=True,
        )
        plt.show()


# simulate the robot
robotVisualiser = RobotVisualiser()
robotVisualiser.animate()
