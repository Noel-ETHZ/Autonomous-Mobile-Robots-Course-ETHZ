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

    def generateMeasurement(self, d):
        if d < self.d_min or d > self.d_max:
            return []
        X = np.random.uniform(0, 1)
        if X < self.P_outlier:
            return np.random.uniform(self.d_min, self.d_max)
        return np.random.normal(d, self.sigma)


# helper class to store wall segments and compute intersections
class Wall:
    def __init__(self, start, end):
        self.start = np.array([[start[0]], [start[1]]])
        self.end = np.array([[end[0]], [end[1]]])

        # now compute line equation
        # (end[0]-start[0])*start[1]- start[0]*(end[1]-start[1]) - (end[0]-start[0])*y + x*(end[1]-start[1]) = 0
        self.a_1 = end[1] - start[1]
        self.a_2 = -(end[0] - start[0])
        self.a_3 = (end[0] - start[0]) * start[1] - start[0] * (end[1] - start[1])

    def computeIntersection(self, startPoint, direction):
        denom = self.a_1 * direction[0] + self.a_2 * direction[1]
        if abs(denom) < 1.0e-8:
            return []
        lambd = (
            -self.a_3 - self.a_1 * startPoint[0] - self.a_2 * startPoint[1]
        ) / denom
        if lambd < 0:
            return []
        x_star = startPoint + lambd * direction
        if np.dot(np.transpose(x_star - self.end), self.end - self.start) > 0:
            return []  # intersection past end
        if np.dot(np.transpose(x_star - self.start), self.end - self.start) < 0:
            return []  # intersection before start
        return x_star


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
sensor = UltrasonicSensor(d_min, d_max, sigma, P_outlier)


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
        # also overlaying the distance measurements drawn into world frame...
        es = []
        es.append(np.array([[0], [1]]))
        es.append(np.array([[1], [0]]))
        es.append(np.array([[0], [-1]]))
        self.measurement_data_x = []
        self.measurement_data_y = []
        for wall in walls:
            for e in es:
                W_e = np.dot(R_WB, e)
                W_r = np.array([[x[0]], [x[1]]])
                W_intersection = wall.computeIntersection(W_r, W_e)
                if len(W_intersection) > 0:
                    d = np.linalg.norm(W_intersection - W_r)
                    d_measured = sensor.generateMeasurement(d)
                    if d_measured:
                        W_measurement = W_r + d_measured * W_e
                        self.measurement_data_x.append(W_measurement[0, 0])
                        self.measurement_data_y.append(W_measurement[1, 0])
        self.measurements.set_data(self.measurement_data_x, self.measurement_data_y)

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
