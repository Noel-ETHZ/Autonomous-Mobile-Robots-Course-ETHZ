import matplotlib.pyplot as plt
import numpy as np
import PyQt5.Qt as Qt
import PyQt5.QtCore as QtCore
import sys

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QWidget, QApplication, QHBoxLayout, QRadioButton, QVBoxLayout, QPushButton

from qpsolvers import solve_qp

# todo: add no-controller option
#       graphs of control signal and state

FPS = 20                # simulation frames per second
X_HILL_S = -3*np.pi/2   # x at start of hilly section [m]
X_HILL_E = 3*np.pi/2    # x at end of hilly section [m]
X_WALL = 10             # x at cliff/end of environment [m]

m = 1.0                 # car mass [kg]
car_length = 0.5        # car length
a = 0.5                 # hill scaling [m]
u_min = -5.0            # min force [N]
u_max = 5.0             # max force [N]

k1 = 0.1  #0.6          # dynamic friction [1/s] (only for simulator)
k2 = 0.03 #0.3          # drag [1/m] (only for simulator)
k_drag = 0.9            # fraction of drag considered by controller
g = 9.81                # acceleration due to gravity (DO NOT CHANGE -- unless you change planet...)
sigma_x = 0.00          # simulated noise magnitude on position [m]
sigma_v = 0.0 # 0.1     # simulated noise magnitude on velocity [m/s]

class PidController:

    def __init__(self, kp=10.0, ki=0.1, kd=5.0, enable_feedforward=False):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.enable_feedforward = enable_feedforward

        self.integral = 0.0
        self.previous_error = 0.0
        self.dt = 1.0 / FPS

    def control(self, x: float, v: float, r_x: float):
        
        # TODO 1. PID feedback control
        # mind integrator windup
        error = r_x - x
        self.integral = min(max(self.integral + error * self.dt, -2.0), 2.0)
        derivative = (error - self.previous_error) / self.dt
        
        pid_u = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        self.previous_error = error

        # TODO 2. Feedforward control
        x_bar = min(x, X_HILL_E) # linearization point to account for flat section
        slope_factor = np.sqrt((a * np.cos(x_bar))**2 + 1.0)
        u_ff = (m * a * g * np.cos(x_bar)) / slope_factor + m*(k1*v + k2*v*np.abs(v))*k_drag
        
        #u_pid = 0.0
        #u_ff = 0.0

        # total control input
        if self.enable_feedforward:
            return pid_u + u_ff
        else:
            return pid_u

class MpcController:

    def __init__(self):
        self.P_1 = 100.0
        self.P_2 = 0.1
        self.Q_1 = 100.0
        self.Q_2 = 0.1
        self.R = 0.001
        self.N = 50 # horizon
        self.predicted_trajectory = None  # Store predicted trajectory for plotting

    def control(self, x: float, v: float, r_x: float):
        N = self.N
        dt = 1.0/FPS

        #######################################################################################
        # Note for indexing and problem setup. The employed qp solver requires the mpc problem 
        # to be transcribed into the following form
        #
        #       J_opt = min 0.5*x_q'*P_q*x_q + q_q'*x_q 
        #       
        #           s.t.  G_q*x_q <= h_q        # inequality constraints (state and input bounds)
        #                 A_q*x_q = b_q         # equality constraints (dynamics)
        # 
        # where (2*(N+1) + N) variables of the optimization problem (augmented states + controls 
        # vector) are:
        #
        #       x_q = [dx_0, dv_0,..., dx_N, dv_N, du_0, ..., du_N] 
        #
        #       where   dx_i = x_i - x_bar 
        #               dv_i = v_i - v_bar
        #               du_i = u_i - u_bar (s.t. d(v_0)/dt=0)
        #               i = [0, ..., N]
        #
        # Solve with: x_q_opt = solve_qp(P_q, q_q, G_q, h_q, A_q, b_q, solver='daqp')
        #######################################################################################

        #######################################################################################
        # TODO 1:   For both the flat and hill section, compute the nominal input u_bar so that 
        #           dv_0/dt = 0 and linearize the system around (x_bar, v_bar, u_bar), i.e.
        #           compute both continuous (_c) and discrete time system matrices
        #
        #               d/dt[dx_i, dv_i] = F_c*[dx_i, dv_i] + G_c*du_i (+ k)
        #               [dx_i+1, dv_i+1] = F_c*[dx_i, dv_i] + G_c*du_i (+ k*i)
        #
        #######################################################################################

        # first, we compute the linearised, discrete-time system:
        x_bar = x 
        v_bar = v # equilibrium at current pos (if > v -> drag overcompensation -> crash risk)

        if (x >= X_HILL_E): 
            # flat section
            u_bar = m*(k1*v_bar + k2*v_bar*np.abs(v_bar))*k_drag
            F_c = np.array([[0.0, 1.0],[0.0, -(k1 + 2.0*np.abs(v_bar)*k2)/m*k_drag]])

        else: 
            # hill
            u_bar = a*g*m*np.cos(x_bar)/np.sqrt((a*np.cos(x_bar))**2 + 1) + m*(k1*v_bar + k2*v_bar*np.abs(v_bar))*k_drag
            F_c = np.array([[ (a**2*v_bar*np.cos(x_bar)*np.sin(x_bar))/(a**2*np.cos(x_bar)**2 + 1)**1.5, 1/np.sqrt(a**2*np.cos(x_bar)**2 + 1)],
                            [ (a*g*np.sin(x_bar))/(a**2*np.cos(x_bar)**2 + 1)**1.5, -(k1 + 2.0*np.abs(v_bar)*k2)/m*k_drag ]]) # note the first entry is 0.0 with v_bar=0.0...

        G_c = np.array([[0],[1.0/m]])

        F = np.identity(2) + dt * F_c
        G = dt * G_c

        #######################################################################################
        # TODO 1: end
        #######################################################################################

        # next, we assemble the QP ...

        # since we control to a steady position, the velocity reference is set to zero
        r_v = 0.0

        # quadratic cost matrix P_q:
        P_q = np.zeros((2*(N+1)+N,2*(N+1)+N))
        for k in range(0,N):
            P_q[2*k:2*k+2, 2*k:2*k+2] = 2.0*np.array([[self.Q_1, 0],[0, self.Q_2]])
        P_q[2*N:2*N+2, 2*N:2*N+2] = 2.0*np.array([[self.P_1, 0],[0, self.P_2]])
        for k in range(0,N):
            P_q[2*(N+1)+k,2*(N+1)+k] = 2.0*self.R
        
        # linear cost vector q_q:
        q_q = np.zeros((2*(N+1)+N))
        for k in range(0,N):
            q_q[2*k:2*k+2] = 2.0*np.matmul(np.array([[self.Q_1, 0],[0, self.Q_2]]),np.array([[x_bar-r_x],[v_bar-r_v]])).reshape((2))
        
        q_q[2*N:2*N+2] = 2.0*np.matmul(np.array([[self.P_1, 0],[0, self.P_2]]),np.array([[x_bar-r_x],[v_bar-r_v]])).reshape((2))
        
        for k in range(0,N):
            q_q[2*(N+1)+k] = 2.0*self.R*u_bar
        
        #######################################################################################
        # TODO 2:   inequality constraints:
        #           G_q*x_q <= h_q
        #######################################################################################

        # control constraints:
        G_ulim = np.zeros((2*N,2*(N+1)+N))
        for k in range(0,N):
            G_ulim[2*k,2*(N+1)+k] = 1.0
            G_ulim[2*k+1,2*(N+1)+k] = -1.0
        
        h_ulim = np.zeros((2*N))
        for k in range(0,N):
            h_ulim[2*k] = u_max - u_bar
            h_ulim[2*k+1] = -u_min + u_bar

        # position constraint (don't hit wall)
        G_pos = np.zeros((N, 2*(N+1)+N))
        for k in range(0, N):
            G_pos[k, 2*k] = 1.0 
            
        h_pos = np.zeros((N))
        for k in range(0, N):
            h_pos[k] = X_WALL - car_length*0.75 - x_bar  # car cannot go beyond this position
        
        # combine control and position constraints
        G_q = np.vstack([G_ulim, G_pos])
        h_q = np.concatenate([h_ulim, h_pos])

        #######################################################################################
        # TODO 2: end
        #######################################################################################

        #######################################################################################
        # TODO 3:   (equality) constraints imposed by dynamics
        #           A_q*x_q = b_q
        #######################################################################################

        A_q = np.zeros((2+2*N,2*(N+1)+N))
        A_q[0:2,0:2] = np.eye(2)
        for k in range(0,N):
            A_q[2+2*k:2+2*k+2,2*k:2*k+2] = F
            A_q[2+2*k:2+2*k+2,2*k+2:2*k+4] = -np.eye(2)
            A_q[2+2*k:2+2*k+2,2*(N+1)+k:2*(N+1)+k+1] = G

        #######################################################################################
        # TODO 3: end
        #######################################################################################
        
        b_q = np.zeros((2+2*N))
        
        # initial state constraints:
        b_q[0] = x-x_bar # = dx_0
        b_q[1] = v-v_bar # = dv_0

        # lin around non-zero vel.
        for k in range(0,N):
            b_q[2+k*2] = -v_bar * dt
        
        # solve QP
        x_q = solve_qp(P_q, q_q, G_q, h_q, A_q, b_q, solver='daqp')

        if x_q is not None:
            u = u_bar + x_q[2*(N+1)] # don't forget to apply delta to linearisation point...
            
            # Extract predicted trajectory for plotting
            self.predicted_trajectory = []
            for k in range(N+1):
                x_pred = x_q[2*k] + x_bar
                v_pred = x_q[2*k+1] + v_bar
                self.predicted_trajectory.append((x_pred, v_pred))
        else: 
            print("QP solver failed, applying feedforward control only") 
            u = u_bar
            self.predicted_trajectory = None

        return u

class CarSimulator:

    def __init__(self, x: float, v: float):
        self.x = x
        self.v = v
        self.r_x = np.pi/2.0

    def update(self, dt: float, u_cmd: float):

        x_bar = min(self.x, X_HILL_E) # account for flat section

        # saturate control commands (e.g. for PID which is unconstrained)
        u = min(max(u_cmd, u_min), u_max)

        # kinematics using Euler-forward discretisation
        x_dot = self.v/np.sqrt((a*np.cos(x_bar))**2 + 1.0)
        v_dot = -a*g*np.cos(x_bar)/np.sqrt((a*np.cos(x_bar))**2 + 1.0) - (k1*self.v + k2*self.v*np.abs(self.v))/m + u/m
        
        self.x = self.x + dt * x_dot + np.random.normal(0, sigma_x)
        self.v = self.v + dt * v_dot + np.random.normal(0, sigma_v)

    def x2h(self, X):

        if not isinstance(X, np.ndarray):
            X = np.array([X])

        h = []
        for x in X:
            if (x<X_HILL_E):
                h.append(a + a*np.sin(x)) 
            else:
                h.append(a + a*np.sin(X_HILL_E) + (x-X_HILL_E)*a*np.cos(X_HILL_E))

        return np.array(h)

class Window(QWidget):

    def __init__(self, parent=None):
        super(Window, self).__init__(parent)
        self.setMouseTracking(True)

        self.sim = CarSimulator(x=0.5*np.pi, v=0)

        # Initialize all controllers
        self.mpc_controller = MpcController()
        self.pid_controller = PidController()
        self.pid_ff_controller = PidController(enable_feedforward=True)

        self.k = 0 # counter
        
        # Time-series data for plotting (circular buffer)
        self.time_history = []
        self.position_history = []
        self.velocity_history = []
        self.input_history = []
        self.reference_history = []
        self.time_window = 10.0  # Moving window size in seconds

        # Create the main layout (Vertical)
        self.main_layout = QVBoxLayout()
        
        # Create control buttons layout
        self.button_layout = QHBoxLayout()
        self.reset_button = QPushButton("Reset Timeseries")
        self.reset_button.clicked.connect(self.reset_timeseries)
        self.button_layout.addWidget(self.reset_button)
        
        # Create the plot canvas
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.cid = self.canvas.mpl_connect('button_press_event', self)
        
        # Controller Selection Layout
        self.control_layout = QHBoxLayout()
        self.no_control_radio = QRadioButton("No Ctrl")
        self.pid_radio = QRadioButton("PID")
        self.pid_ff_radio = QRadioButton("PID+FF")
        self.mpc_radio = QRadioButton("MPC")
        
        self.no_control_radio.setChecked(True) # Default
        
        self.control_layout.addWidget(self.no_control_radio)
        self.control_layout.addWidget(self.pid_radio)
        self.control_layout.addWidget(self.pid_ff_radio)
        self.control_layout.addWidget(self.mpc_radio)

        # Add widgets to main layout
        self.main_layout.addLayout(self.button_layout)
        self.main_layout.addLayout(self.control_layout)
        self.main_layout.addWidget(self.canvas)
        
        self.setLayout(self.main_layout)
        
        screen_size = self.screen().geometry().size()
        min_size = min(screen_size.width(), screen_size.height())
        self.resize(QtCore.QSize(int(min_size / 1.5), int(min_size / 1.5)))

        # Timer for updating the view, with a delta t of 1s / fps between frames.
        self.timer = Qt.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(int(1000.0 / FPS))  # in milliseconds
        
    def __call__(self, event):
        modifiers = QApplication.keyboardModifiers()
        Mmodo = QApplication.mouseButtons()
        if Mmodo == QtCore.Qt.LeftButton:
            print('reset car x =',event.xdata)
            self.sim.x = event.xdata
            self.sim.v = 0.0
        else:
            print('reset reference position x =',event.xdata)
            self.sim.r_x = event.xdata

        if  self.sim.x is None:
             self.sim.x = 0.0

        if  self.sim.r_x is None:
             self.sim.r_x = 0.0

    def reset_timeseries(self):
        """Reset (clear) all time-series data"""
        self.time_history = []
        self.position_history = []
        self.velocity_history = []
        self.input_history = []
        self.reference_history = []
        self.plot()

    def update(self):

        # call selected controller
        self.mpc_controller.predicted_trajectory = None # reset predicted trajectory (only for plotting) at each step, will be updated if MPC is selected and QP solver succeeds
        if self.mpc_radio.isChecked():
            u = self.mpc_controller.control(self.sim.x, self.sim.v, self.sim.r_x)
        elif self.pid_ff_radio.isChecked():
            u = self.pid_ff_controller.control(self.sim.x, self.sim.v, self.sim.r_x)
        elif self.pid_radio.isChecked():
            u = self.pid_controller.control(self.sim.x, self.sim.v, self.sim.r_x)
        else:
            u = 0.0

        # Store history data before simulation update
        current_time = self.k / FPS
        self.time_history.append(current_time)
        self.position_history.append(self.sim.x)
        self.velocity_history.append(self.sim.v)
        self.input_history.append(u)
        self.reference_history.append(self.sim.r_x)
        
        # Implement circular buffer (moving window) - keep only last 10 seconds
        if len(self.time_history) > 0 and current_time - self.time_history[0] > self.time_window:
            self.time_history.pop(0)
            self.position_history.pop(0)
            self.velocity_history.pop(0)
            self.input_history.pop(0)
            self.reference_history.pop(0)

        # simulate all together
        self.sim.update(dt=1 / FPS, u_cmd=u)
        self.plot()
        self.k += 1

    def plot(self):

        self.figure.clear()
        
        # Create grid layout: terrain on top, time-series plots below
        # hspace controls vertical spacing, height_ratios controls relative heights
        gs = self.figure.add_gridspec(2, 3, hspace=1.0, wspace=0.3, height_ratios=[1.2, 1])
        ax_terrain = self.figure.add_subplot(gs[0, :])  # Top row, all columns
        ax_pos = self.figure.add_subplot(gs[1, 0])      # Bottom left - position
        ax_vel = self.figure.add_subplot(gs[1, 1])      # Bottom center - velocity
        ax_input = self.figure.add_subplot(gs[1, 2])    # Bottom right - input
        
        # ===== TERRAIN VIEW =====
        ax_terrain.set_xlim(X_HILL_S, 11)
        ax_terrain.set_ylim(-0.2*a, a*2.5)
        ax_terrain.set_title('Terrain with Car')
        ax_terrain.set_xlabel('Position [m]')
        ax_terrain.set_ylabel('Height [m]')

        # current position of the car
        x = self.sim.x
        
        # draw terrain
        x_line = list(np.arange(X_HILL_S, X_WALL, 0.01))   # start, stop, step
        h_line = list(self.sim.x2h(np.array(x_line)))
        
        x_line += [X_WALL, X_WALL+1]
        h_line += [h_line[-1]+0.2, h_line[-1]+0.2]
        ax_terrain.plot(x_line, h_line, 'black', linewidth=2)
        
        # Fill area below terrain
        ax_terrain.fill_between(x_line, h_line, -0.2*a, alpha=0.3, color='grey')
        
        # reference
        ax_terrain.plot(self.sim.r_x, self.sim.x2h(self.sim.r_x), '|g', markersize=12)
        
        crashed = (x>X_WALL - car_length/2)
        if (crashed):
            x = X_WALL-car_length/2
            self.sim.x = x+1e-3
            self.sim.v = 0.0

        # car as a side view (simple shape)
        h = float(self.sim.x2h(x))
        
        # Calculate slope to determine car orientation
        dx = 0.01
        if x < X_WALL:
            h_forward = float(self.sim.x2h(min(x + dx, X_WALL)))
            slope = (h_forward - h) / dx
            angle = np.arctan(slope)
        else:
            angle = 0
            
        # Create car polygon (side view - trapezoid/rectangle with wheels)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        # Rotation matrix
        rot_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ])
        
        # Car body in local frame (before rotation)
        # Wider at bottom, narrower at top for side view
        corners_local = np.array([
            [-2,0.3],
            [2,0.3],
            [2,0.8],
            [1,1],
            [0,1.5],
            [-1,1.5],
            [-2,1]
        ])/4.0*car_length
        
        # Rotate and translate using matrix multiplication
        corners_rotated = corners_local @ rot_matrix.T
        corners_global = corners_rotated + np.array([x, h])
        
        # Draw car body
        car_polygon = plt.Polygon(corners_global, fill=True, edgecolor='darkred', 
                                  facecolor='red', linewidth=2)
        ax_terrain.add_patch(car_polygon)
        
        # Draw wheels
        wheel_radius = 0.08*car_length
        
        # Front wheel
        front_wheel_x = x + 0.3*car_length * cos_a + wheel_radius * sin_a
        front_wheel_y = self.sim.x2h(front_wheel_x) + wheel_radius * cos_a
        front_wheel = plt.Circle((front_wheel_x, front_wheel_y), wheel_radius, 
                                fill=True, edgecolor='black', facecolor='black')

        # Back wheel
        back_wheel_x = x - 0.3*car_length * cos_a + wheel_radius * sin_a
        back_wheel_y = self.sim.x2h(back_wheel_x) + wheel_radius * cos_a
        back_wheel = plt.Circle((back_wheel_x, back_wheel_y), wheel_radius, 
                               fill=True, edgecolor='black', facecolor='black')
        
        ax_terrain.add_patch(front_wheel)
        ax_terrain.add_patch(back_wheel)

        # Plot MPC horizon if available
        if self.mpc_controller.predicted_trajectory is not None:
            traj_x = []
            traj_h = []
            for x_pred, v_pred in self.mpc_controller.predicted_trajectory:
                traj_x.append(x_pred)
                traj_h.append(float(self.sim.x2h(x_pred)))
            
            # Plot predicted trajectory as dashed line with circles
            ax_terrain.plot(traj_x, traj_h, 'b--', linewidth=1.5, alpha=0.7, label='MPC Horizon')
            ax_terrain.plot(traj_x, traj_h, 'bo', markersize=4, alpha=0.7)
            
            # Add legend
            ax_terrain.legend(loc='upper left', fontsize=8)

        # kaboom
        if crashed:
            ax_terrain.plot(X_WALL, self.sim.x2h(X_WALL)+0.05, marker='*', markerfacecolor='yellow', markeredgecolor='red', markeredgewidth=4, markersize=30)
        
        # explanation
        ax_terrain.text(X_HILL_E+0.1, a*1.3, 'left click: reset car position x')
        ax_terrain.text(X_HILL_E+0.1, a*1.0, 'right click: reset car reference r_x')

        # ===== TIME-SERIES PLOTS =====
        # Position plot
        ax_pos.set_title('Position vs. Time')
        ax_pos.set_xlabel('Time [s]')
        ax_pos.set_ylabel('Position [m]')
        ax_pos.grid(True, alpha=0.3)
        if len(self.time_history) > 0:
            ax_pos.plot(self.time_history, self.position_history, 'r-', linewidth=2, label='Position')
            ax_pos.plot(self.time_history, self.reference_history, 'g--', linewidth=1.5, label='Reference')
            ax_pos.legend(fontsize=8)
        
        # Velocity plot
        ax_vel.set_title('Velocity vs. Time')
        ax_vel.set_xlabel('Time [s]')
        ax_vel.set_ylabel('Velocity [m/s]')
        ax_vel.grid(True, alpha=0.3)
        if len(self.time_history) > 0:
            ax_vel.plot(self.time_history, self.velocity_history, 'b-', linewidth=2)
        
        # Input/Control plot
        ax_input.set_title('Control Input vs. Time')
        ax_input.set_xlabel('Time [s]')
        ax_input.set_ylabel('Force [N]')
        ax_input.grid(True, alpha=0.3)
        if len(self.time_history) > 0:
            ax_input.plot(self.time_history, self.input_history, 'm-', linewidth=2)
            ax_input.axhline(y=u_max, color='r', linestyle='--', alpha=0.5, linewidth=1)
            ax_input.axhline(y=u_min, color='r', linestyle='--', alpha=0.5, linewidth=1)
        
        self.canvas.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = Window()
    main.show()
    sys.exit(app.exec_())
