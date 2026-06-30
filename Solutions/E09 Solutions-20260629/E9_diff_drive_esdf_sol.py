import sys

import matplotlib.pyplot as plt
import numpy as np
import PyQt5.Qt as Qt
import PyQt5.QtCore as QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.patches import Circle
from PyQt5.QtWidgets import QWidget, QApplication, QHBoxLayout

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CELL_SIZE        = 2.0           # metres per ETH-mask cell
X_MIN, X_MAX     = 0.0, 18.0    # world x range
Y_MIN, Y_MAX     = 0.0, 10.0    # world y range (y points UP)
MAP_RESOLUTION   = 0.1          # ESDF grid spacing [m]
TRUNC_DIST       = 0.6          # ESDF truncation distance [m]

NUM_BEAMS        = 50
MAX_RANGE        = 20.0
# MAX_RANGE        = 2.0 # TODO: Uncomment this line for vi)
RANGE_STD        = 0.1
P_RANGE_OUTLIER  = 0.01

SPEED_NOISE_STD = 0.80   # m/s   additive speed noise
ANGLE_NOISE_STD = 0.30   # rad/s additive angular noise

MAX_GN_ITER      = 5
GN_CONV_THR      = 1e-4
REFINE_EVERY     = 10            # frames between GN LiDAR corrections
LAMBDA_LM_DIAG   = 0             # LM: weight each dimension by its curvature; suppressed in degenerate dirs
LAMBDA_LM_I      = 1e-4          # LM: uniform shift; guarantees full rank even when entire diag(A) row is zero

CAR_RADIUS   = 0.5   # robot body radius [m]; crash when ESDF(centre) < CAR_RADIUS

# ETH-logo maze: True = navigable corridor, False = solid wall
# Columns 0-2 = E,  3-5 = T,  6-8 = H;  row 0 = top of world
ETH_MASK = np.array([
    [1, 1, 1, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 1, 0, 1, 0, 1],
    [1, 1, 1, 0, 1, 0, 1, 1, 1],
    [1, 0, 0, 0, 1, 0, 1, 0, 1],
    [1, 1, 1, 0, 1, 0, 1, 0, 1],
], dtype=bool)

FPS          = 20
GOAL_RADIUS  = 0.5
DEBUG        = False

# Mask indexing: row r, col c  →  world centre ((c+0.5)·2, (4−r+0.5)·2)
# Start: mask(4,8) = bottom-right of H's right upright → world (17, 1), heading up
# Goal:  mask(4,2) = bottom-right of E's bottom bar   → world  (5, 1)
START_POSE = np.array([17.0, 1.0, np.pi / 2])   # (x, y, θ)  heading = π/2 = up
GOAL_POS   = np.array([5.0,  1.0])

# ---------------------------------------------------------------------------
# PoseEstimator  (Gauss-Newton over ESDF residuals)
# ---------------------------------------------------------------------------
class PoseEstimator:
    def __init__(self, esdf_map):
        self.esdf_map = esdf_map

    def estimate(self, z, x0):
        """
        Refine pose x = [x_R, y_R, θ] from LiDAR measurement z.
        For each beam i with valid range z[i]:
          p_i = [x_R + z_i cos(θ+αi),  y_R + z_i sin(θ+αi)]
          e_i = ESDF(p_i)          (error — should be 0 on a wall)
          E_i = ∇ESDF(p_i) @ dp_i/dx
        Gauss-Newton update:  (∑ EᵢᵀEᵢ) dx = −∑ Eᵢ eᵢ
        Levenberg-Marquardt when A is rank-deficient:
          (A + λ_diag·diag(A) + λ_I·I) dx = b
        """
        x = x0.copy().astype(float)
        for _ in range(MAX_GN_ITER):
            A = np.zeros((3, 3))
            b = np.zeros(3)
            for i in range(NUM_BEAMS):
                zi = z[i]
                if zi >= MAX_RANGE:
                    continue
                alpha_i = 2 * np.pi * i / NUM_BEAMS
                p_i = np.array([x[0] + zi * np.cos(x[2] + alpha_i),
                                x[1] + zi * np.sin(x[2] + alpha_i)])
                esdf_val, grad = self.esdf_map.query(p_i)
                if abs(esdf_val) >= TRUNC_DIST:
                    continue
                dp_dx = np.array([
                    [1.0, 0.0, -zi * np.sin(x[2] + alpha_i)],
                    [0.0, 1.0,  zi * np.cos(x[2] + alpha_i)],
                ])
                E_i = grad @ dp_dx          # (3,)
                A  += np.outer(E_i, E_i)
                b  -= E_i * esdf_val
            if np.linalg.matrix_rank(A) < 3:
                # Levenberg-Marquardt
                A_lm = A + LAMBDA_LM_DIAG * np.diag(np.diag(A)) + LAMBDA_LM_I * np.eye(3)
                dx = np.linalg.solve(A_lm, b)
            else:
                dx = np.linalg.solve(A, b)
            x  = x + dx
            if np.linalg.norm(dx) < GN_CONV_THR:
                break
        return x

# ---------------------------------------------------------------------------
# Wall
# ---------------------------------------------------------------------------
class Wall:
    def __init__(self, start, end):
        self.start = np.array([[start[0]], [start[1]]])
        self.end   = np.array([[end[0]],   [end[1]]])
        self.a_1 = end[1] - start[1]
        self.a_2 = -(end[0] - start[0])
        self.a_3 = (end[0] - start[0]) * start[1] - start[0] * (end[1] - start[1])

    def computeIntersection(self, startPoint, direction):
        denom = self.a_1 * direction[0] + self.a_2 * direction[1]
        if abs(denom) < 1e-8:
            return []
        lambd = (-self.a_3 - self.a_1 * startPoint[0] - self.a_2 * startPoint[1]) / denom
        if lambd < 0:
            return []
        x_star = startPoint + lambd * direction
        if np.dot(np.transpose(x_star - self.end),   self.end - self.start) > 0:
            return []
        if np.dot(np.transpose(x_star - self.start), self.end - self.start) < 0:
            return []
        return x_star

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def generate_lidar(pose, walls):
    x, y, theta = pose
    R_WB = np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta),  np.cos(theta)]])
    W_r  = np.array([[x], [y]])
    z    = np.full(NUM_BEAMS, MAX_RANGE)
    for i in range(NUM_BEAMS):
        a   = 2 * np.pi * i / NUM_BEAMS
        Ri  = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        W_e = R_WB @ Ri @ np.array([[1.0], [0.0]])
        zi  = MAX_RANGE
        for wall in walls:
            hit = wall.computeIntersection(W_r, W_e)
            if len(hit) > 0:
                zi = min(zi, float(np.linalg.norm(hit - W_r)))
        if zi < MAX_RANGE:
            zi += np.random.normal(0, RANGE_STD)
        zi = float(np.clip(zi, 0.0, MAX_RANGE))
        if np.random.uniform() < P_RANGE_OUTLIER:
            zi = np.random.uniform(0, MAX_RANGE)
        z[i] = zi
    return z


def draw_robot(ax, pose, face_color, label, edgecolor='black', linestyle='-', alpha=0.85):
    """Draw the robot as a filled circle with a red heading line."""
    x, y, theta = pose
    circle = Circle((x, y), radius=CAR_RADIUS,
                    facecolor=face_color, edgecolor=edgecolor,
                    linestyle=linestyle, linewidth=1.5,
                    alpha=alpha, zorder=5)
    ax.add_patch(circle)
    ax.plot([x, x + (CAR_RADIUS*1.5) * np.cos(theta)],
            [y, y + (CAR_RADIUS*1.5) * np.sin(theta)],
            color='red', lw=2, zorder=6)
    ax.plot([], [], color=face_color, label=label)

# ---------------------------------------------------------------------------
# Wall extraction from the ETH mask
# ---------------------------------------------------------------------------
def extract_walls(mask, cell_size=CELL_SIZE):
    """
    For every navigable cell, emit a Wall segment on each side that borders
    a non-navigable cell or the grid boundary.  This produces the full set of
    wall segments that enclose the ETH-logo maze.
    """
    rows, cols = mask.shape
    walls = []
    for r in range(rows):
        for c in range(cols):
            if not mask[r, c]:
                continue
            x0, x1 = c * cell_size, (c + 1) * cell_size
            # y increases upward; mask row 0 is the top of the world
            y0 = (rows - 1 - r) * cell_size   # bottom of this cell in world
            y1 = (rows - r) * cell_size        # top of this cell in world

            # left boundary (world −x)
            if c == 0 or not mask[r, c - 1]:
                walls.append(Wall([x0, y0], [x0, y1]))
            # right boundary (world +x)
            if c == cols - 1 or not mask[r, c + 1]:
                walls.append(Wall([x1, y1], [x1, y0]))
            # top boundary in world (mask row r−1)
            if r == 0 or not mask[r - 1, c]:
                walls.append(Wall([x0, y1], [x1, y1]))
            # bottom boundary in world (mask row r+1)
            if r == rows - 1 or not mask[r + 1, c]:
                walls.append(Wall([x1, y0], [x0, y0]))
    return walls

# ---------------------------------------------------------------------------
# ESDFMap
# ---------------------------------------------------------------------------
class ESDFMap:
    def __init__(self, mask, cell_size=CELL_SIZE):
        self.mask      = mask
        self.cell_size = cell_size
        self.rows, self.cols = mask.shape
        self.walls = extract_walls(mask, cell_size)
        self.x1d   = np.arange(X_MIN - TRUNC_DIST, X_MAX + TRUNC_DIST + MAP_RESOLUTION, MAP_RESOLUTION)
        self.y1d   = np.arange(Y_MIN - TRUNC_DIST, Y_MAX + TRUNC_DIST + MAP_RESOLUTION, MAP_RESOLUTION)
        self.esdf  = self._compute_esdf()

    def _compute_esdf(self):
        """Vectorised truncated ESDF over the full grid."""
        rows, cols, cs = self.rows, self.cols, self.cell_size

        # All grid points as (N, 2) array, indexing='ij' → shape (Nx, Ny)
        P_xx, P_yy = np.meshgrid(self.x1d, self.y1d, indexing='ij')
        P = np.column_stack([P_xx.ravel(), P_yy.ravel()])   # (Nx*Ny, 2)

        # Minimum distance to any wall segment
        D = np.full(len(P), np.inf)
        for w in self.walls:
            a  = w.start.flatten()
            b  = w.end.flatten()
            ab = b - a
            t  = np.dot(P - a, ab) / np.dot(ab, ab)
            proj = a + np.clip(t, 0.0, 1.0)[:, np.newaxis] * ab
            d = np.linalg.norm(P - proj, axis=1)
            D = np.minimum(D, d)

        # Sign: +1 inside a navigable cell, −1 elsewhere
        mc = np.floor(P[:, 0] / cs).astype(int)
        mr = rows - 1 - np.floor(P[:, 1] / cs).astype(int)
        in_bounds  = (mc >= 0) & (mc < cols) & (mr >= 0) & (mr < rows)
        mc_c = np.clip(mc, 0, cols - 1)
        mr_c = np.clip(mr, 0, rows - 1)
        navigable  = in_bounds & self.mask[mr_c, mc_c]
        sign       = np.where(navigable, 1.0, -1.0)

        esdf_flat  = sign * np.minimum(D, TRUNC_DIST)
        return esdf_flat.reshape(len(self.x1d), len(self.y1d))   # (Nx, Ny)

    def query(self, p):
        """
        Return (esdf_value, gradient) at world point p via bilinear interpolation.
        Out-of-bounds points return (−TRUNC_DIST, [0, 0]) (treated as wall).
        """
        px, py = float(p[0]), float(p[1])
        ix = (px - self.x1d[0]) / MAP_RESOLUTION
        iy = (py - self.y1d[0]) / MAP_RESOLUTION
        i0, j0 = int(np.floor(ix)), int(np.floor(iy))
        if i0 < 0 or i0 >= len(self.x1d) - 1 or j0 < 0 or j0 >= len(self.y1d) - 1:
            return -TRUNC_DIST, np.zeros(2)
        fx, fy = ix - i0, iy - j0
        v00 = self.esdf[i0,     j0    ]
        v10 = self.esdf[i0 + 1, j0    ]
        v01 = self.esdf[i0,     j0 + 1]
        v11 = self.esdf[i0 + 1, j0 + 1]
        val = (1-fx)*(1-fy)*v00 + fx*(1-fy)*v10 + (1-fx)*fy*v01 + fx*fy*v11
        h   = MAP_RESOLUTION
        gx  = (1/h) * ((1-fy)*(v10 - v00) + fy*(v11 - v01))
        gy  = (1/h) * ((1-fx)*(v01 - v00) + fx*(v11 - v10))
        return val, np.array([gx, gy])

# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------
class ETHDriveSimulator:
    """
    Tracks both the true (noisy) pose and the estimated pose simultaneously.

    real_pose  — integrates user input with additive odometry noise.
    est_pose   — integrates clean user input; corrected by GN every REFINE_EVERY frames.

    The player sees only est_pose. A crash is detected when real_pose enters a wall.
    """
    def __init__(self, mask, debug=False):
        self.esdf_map  = ESDFMap(mask)
        self.walls     = self.esdf_map.walls
        self.estimator = PoseEstimator(self.esdf_map)
        self.debug     = debug

        self.real_pose     = START_POSE.copy().astype(float)
        self.est_pose      = START_POSE.copy().astype(float)
        self.crashed       = False
        self.won           = False
        self.frame         = 0
        self.controls      = np.zeros(2)
        self.prev_controls = np.zeros(2)
        self.key_state     = np.zeros(2)   # [fwd/bwd, rot-left/rot-right]
        self.lidar_hits    = None          # world-frame hit positions, frozen until next scan

    def update(self, dt):
        if self.crashed or self.won:
            return

        # 1. Smooth controls from key state
        self.controls = (0.8 * self.prev_controls
                         + 0.2 * self.key_state * np.array([2.0, np.pi / 4]))
        self.prev_controls = self.controls.copy()
        v, omega = self.controls

        # 2. Integrate REAL pose with noise
        if abs(v) > 0.01 or abs(omega) > 0.01:
            v_noisy     = v     + np.random.normal(0.0, SPEED_NOISE_STD)
            omega_noisy = omega + np.random.normal(0.0, ANGLE_NOISE_STD)
        else:
            v_noisy, omega_noisy = v, omega
        tr = self.real_pose[2]
        self.real_pose += dt * np.array([np.cos(tr) * v_noisy,
                                         np.sin(tr) * v_noisy,
                                         omega_noisy])

        # 3. Integrate ESTIMATED pose with clean controls
        te = self.est_pose[2]
        self.est_pose += dt * np.array([np.cos(te) * v,
                                        np.sin(te) * v,
                                        omega])

        # 4. Periodic GN correction using sensor from REAL pose
        self.frame += 1
        if self.frame % REFINE_EVERY == 0:
            z = generate_lidar(self.real_pose, self.walls)
            self.est_pose = self.estimator.estimate(z, self.est_pose)
            # Project hit positions into world frame at the moment of measurement
            th = self.est_pose[2]
            R  = np.array([[np.cos(th), -np.sin(th)],
                           [np.sin(th),  np.cos(th)]])
            hits = []
            for i in range(NUM_BEAMS):
                if z[i] < MAX_RANGE:
                    a   = 2 * np.pi * i / NUM_BEAMS
                    Ri  = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
                    hits.append(self.est_pose[:2] + z[i] * (R @ Ri @ np.array([1.0, 0.0])))
            self.lidar_hits = np.array(hits) if hits else None

        # 5. Crash: centre is within CAR_RADIUS of a wall
        if self.esdf_map.query(self.real_pose[:2])[0] < CAR_RADIUS:
            self.crashed = True

        # 6. Win: real pose within goal radius
        if np.linalg.norm(self.real_pose[:2] - GOAL_POS) < GOAL_RADIUS:
            self.won = True

# ---------------------------------------------------------------------------
# Window
# ---------------------------------------------------------------------------
class Window(QWidget):
    def __init__(self, mask, debug=DEBUG, parent=None):
        super().__init__(parent)
        self.sim = ETHDriveSimulator(mask, debug=debug)

        self.figure = plt.figure(figsize=(11, 7))
        self.canvas = FigureCanvas(self.figure)
        layout = QHBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        screen = self.screen().geometry().size()
        self.resize(int(screen.width() * 0.75), int(screen.height() * 0.65))
        self.setWindowTitle("E9 Task 2 – ESDF Localization Game")

        # Pre-build meshgrid for ESDF pcolormesh (constant across frames)
        esdf = self.sim.esdf_map
        self._MX, self._MY = np.meshgrid(esdf.x1d, esdf.y1d)   # (Ny, Nx) each

        self.timer = Qt.QTimer()
        self.timer.timeout.connect(self._step)
        self.timer.start(int(1000.0 / FPS))

    def _step(self):
        self.sim.update(dt=1.0 / FPS)
        self._plot()
        if self.sim.crashed or self.sim.won:
            self.timer.stop()

    def keyPressEvent(self, event):
        k = event.key()
        if k == QtCore.Qt.Key.Key_Up:    self.sim.key_state[0] =  1
        if k == QtCore.Qt.Key.Key_Down:  self.sim.key_state[0] = -1
        if k == QtCore.Qt.Key.Key_Left:  self.sim.key_state[1] =  1
        if k == QtCore.Qt.Key.Key_Right: self.sim.key_state[1] = -1
        if k == QtCore.Qt.Key.Key_D:
            self.sim.debug = not self.sim.debug
        if k == QtCore.Qt.Key.Key_Escape:
            self.close()

    def keyReleaseEvent(self, event):
        k = event.key()
        if k in (QtCore.Qt.Key.Key_Up, QtCore.Qt.Key.Key_Down):
            self.sim.key_state[0] = 0
        if k in (QtCore.Qt.Key.Key_Left, QtCore.Qt.Key.Key_Right):
            self.sim.key_state[1] = 0

    def _plot(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_xlim(X_MIN - TRUNC_DIST, X_MAX + TRUNC_DIST)
        ax.set_ylim(Y_MIN - TRUNC_DIST, Y_MAX + TRUNC_DIST)
        ax.set_aspect('equal')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_title(
            'Lost in ETH\n'
            '↑↓ move  ←→ steer  D = toggle debug',
            fontsize=11)

        # ESDF background (blue = navigable interior, red = wall)
        c = ax.pcolormesh(self._MX, self._MY,
                          self.sim.esdf_map.esdf.T,          # .T → (Ny, Nx)
                          cmap='RdBu',
                          vmin=-TRUNC_DIST, vmax=TRUNC_DIST,
                          shading='auto')
        self.figure.colorbar(c, ax=ax, label='ESDF (m)', fraction=0.025, pad=0.02)

        # Wall segments
        for w in self.sim.walls:
            ax.plot([w.start[0, 0], w.end[0, 0]],
                    [w.start[1, 0], w.end[1, 0]], 'k-', lw=1.2)

        # Goal marker
        ax.plot(*GOAL_POS, '*', color='gold', ms=20, zorder=7, label='Goal')
        circle = plt.Circle(GOAL_POS, GOAL_RADIUS,
                            color='gold', fill=False, linestyle='--', lw=1.5, zorder=6)
        ax.add_patch(circle)

        # LiDAR hits — only in debug mode; frozen in world frame until next scan
        if self.sim.debug and self.sim.lidar_hits is not None:
            ax.plot(self.sim.lidar_hits[:, 0], self.sim.lidar_hits[:, 1],
                    '*y', ms=5, zorder=6)

        # Robot poses
        draw_robot(ax, self.sim.est_pose, 'tab:blue', 'Estimated pose')
        if self.sim.debug:
            draw_robot(ax, self.sim.real_pose, 'lightgray', 'Real pose (debug)',
                       edgecolor='gray', linestyle='--', alpha=0.6)
        elif self.sim.crashed:
            draw_robot(ax, self.sim.real_pose, 'red', 'Real pose (revealed)')

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10),
                  ncol=4, fontsize=9, framealpha=0.8)
        self.figure.tight_layout()

        # Game-over overlays
        if self.sim.won:
            ax.text(0.5, 0.5, 'You reached the goal!',
                    transform=ax.transAxes, fontsize=22, color='lime',
                    ha='center', va='center', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.65))
        elif self.sim.crashed:
            ax.text(0.5, 0.5, 'CRASHED!',
                    transform=ax.transAxes, fontsize=30, color='red',
                    ha='center', va='center', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.65))

        self.canvas.draw()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = Window(ETH_MASK, debug=DEBUG)
    win.show()
    sys.exit(app.exec_())
