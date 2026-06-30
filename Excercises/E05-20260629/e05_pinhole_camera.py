# Autonomous Mobile Robots - Exercise 5 - Pinhole Camera Model

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize
import plotly.graph_objects as go

# helper class that just does the radial-tangentialDistortion
class RadialTangentialDistortion:
    def __init__(self, k1, k2, p1, p2):
        self.k1 = k1
        self.k2 = k2
        self.p1 = p1
        self.p2 = p2
        
    def distort(self, uUndistorted):
        uDistorted = uUndistorted # TODO: implement distortion model
        return uDistorted
    
    def undistort(self, uDistorted):
        num_points = uDistorted.shape[0]
        uUnDistorted = np.zeros_like(uDistorted)
        # make it work for many points:
        for k in range(num_points):
            uDistortedk = uDistorted[k, :]
            diff = lambda uUndistortedk, uDistorted_k : np.linalg.norm(self.distort(uUndistortedk)-uDistortedk)
            res = minimize(diff, uDistortedk, args=(uDistortedk), method='Nelder-Mead', tol=1e-6)
            uUnDistorted[k, :] = res.x
        return uUnDistorted
        
# now the pinhole camera class
class PinholeCamera:
    def __init__(self, width, height, f1, f2, c1, c2, distortion=None):
        self.width = width
        self.height = height
        self.f1 = f1
        self.f2 = f2
        self.c1 = c1
        self.c2 = c2
        self.distortion = distortion

    def project(self, x):
        # TODO
        return []
        
    def p(self, x):
        # TODO
        return []
    
    def p_inverse(self, x_dash):
        # TODO
        return []
            
    def d(self, x_dash):
        if self.distortion is not None:
            return self.distortion.distort(x_dash)
        else:
            return x_dash
    
    def d_inverse(self, x_ddash):
        if self.distortion is not None:
            return self.distortion.undistort(x_ddash)
        else:
            return x_ddash

    def k(self, x_ddash):
        # TODO
        return []
    
    def k_inverse(self, u):
        # TODO
        return []
    
    def backproject(self, u):
        x_ddash = self.k_inverse(u)
        x_dash = self.d_inverse(x_ddash)
        return self.p_inverse(x_dash)

# helper function to plot point clouds
def plot_pointcloud(points_3d, colors = None, title = "Point Cloud", sample_size = 1e5):
    """
    Plots a 3D point cloud using Plotly.
    
    Args:
        points_3d (np.ndarray): Array of 3D points (N, 3).
        colors (np.ndarray or None): Optional RGB colors for each point (N, 3) or scalar values.
        title (str): Title of the plot.
        sample_size (int): Maximum number of points to plot (randomly sampled if more).
    """
    points_3d = np.asarray(points_3d)
    if points_3d.shape[0] > sample_size:
        indices = np.random.choice(points_3d.shape[0], int(sample_size), replace = False)
        points_3d = points_3d[indices]
        if colors is not None:
            colors = np.asarray(colors)[indices]
    x = points_3d[:, 0]
    y = points_3d[:, 1]
    z = points_3d[:, 2]
    marker_dict = dict(size = 2)
    if colors is not None:
        colors = np.asarray(colors)
        if colors.ndim == 2 and colors.shape[1] == 3:
            if np.issubdtype(colors.dtype, np.floating):
                colors_vis = np.clip(colors * 255, 0, 255).astype(np.uint8)
            else:
                colors_vis = colors.astype(np.uint8)
            marker_colors = [f'rgb({r},{g},{b})' for r, g, b in colors_vis]
            marker_dict["color"] = marker_colors
        else:
            marker_dict["color"] = colors
    fig = go.Figure(data = [go.Scatter3d(x = x, y = y, z = z, mode = 'markers', marker = marker_dict)])
    xyz_min = points_3d.min(axis = 0)
    xyz_max = points_3d.max(axis = 0)
    center = (xyz_min + xyz_max) / 2.0
    scale = (xyz_max - xyz_min).max() / 2.0
    fig.update_layout(title = title, 
                      scene = dict(xaxis = dict(title = "X", range = [center[0] - scale, center[0] + scale]), 
                                   yaxis = dict(title = "Y", range = [center[1] - scale, center[1] + scale]), 
                                   zaxis = dict(title = "Z", range = [center[2] - scale, center[2] + scale]), 
                                   aspectmode = 'cube',
                                   camera = dict(eye=dict(x=0, y=0, z=-1), up=dict(x=0, y=-1, z=0)),
                                   ), 
                      margin = dict(l = 0, r = 0, b = 0, t = 40))
    fig.show()

# helper function to plot the overlap of two point clouds
def plot_pointcloud_overlap(points_gt, points_est, colors=None, title="Point Cloud Overlap", sample_size=1e5):
    """
    Plots the overlap of two 3D point clouds using Plotly.

    Args:
        points_gt (np.ndarray): Ground truth 3D points (N, 3).
        points_est (np.ndarray): Estimated 3D points (N, 3).
        colors (np.ndarray or None): Optional RGB colors for each point (N, 3) or scalar values.
        title (str): Title of the plot.
        sample_size (int): Maximum number of points to plot (randomly sampled if more).
    """
    points_gt = np.asarray(points_gt)
    points_est = np.asarray(points_est)

    if points_gt.shape[0] > sample_size:
        indices = np.random.choice(points_gt.shape[0], int(sample_size), replace=False)
        points_gt = points_gt[indices]
        points_est = points_est[indices]
        if colors is not None:
            colors = np.asarray(colors)[indices]

    marker_dict = dict(size=2)
    if colors is not None:
        colors = np.asarray(colors)
        if colors.ndim == 2 and colors.shape[1] == 3:
            if np.issubdtype(colors.dtype, np.floating):
                colors_vis = np.clip(colors * 255, 0, 255).astype(np.uint8)
            else:
                colors_vis = colors.astype(np.uint8)
            marker_dict["color"] = [f'rgb({r},{g},{b})' for r, g, b in colors_vis]
        else:
            marker_dict["color"] = colors
    else:
        marker_dict["color"] = points_est[:, 2]

    fig = go.Figure(data=[
        go.Scatter3d(
            x=points_gt[:, 0], y=points_gt[:, 1], z=points_gt[:, 2],
            mode='markers',
            marker=dict(size=2, color='rgb(0,144,189)'), # blue
            # marker=marker_dict,
            name='GT'
        ),
        go.Scatter3d(
            x=points_est[:, 0], y=points_est[:, 1], z=points_est[:, 2],
            mode='markers',
            # marker=marker_dict,
            marker=dict(size=2, color='rgb(217,83,25)'), # orange
            name='Est'
        )
    ])

    points_all = np.concatenate([points_gt, points_est], axis=0)
    xyz_min = points_all.min(axis=0)
    xyz_max = points_all.max(axis=0)
    center = (xyz_min + xyz_max) / 2.0
    scale = (xyz_max - xyz_min).max() / 2.0

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title="X", range=[center[0] - scale, center[0] + scale]),
            yaxis=dict(title="Y", range=[center[1] - scale, center[1] + scale]),
            zaxis=dict(title="Z", range=[center[2] - scale, center[2] + scale]),
            aspectmode='cube',
            camera=dict(eye=dict(x=0, y=0, z=-1), up=dict(x=0, y=-1, z=0)),
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    fig.show()

if __name__ == "__main__":
    # ============================================================
    # Test the full projection pipeline on a synthetic 3D cube
    # ============================================================
    b = 1.0 # sidelength
    z_distance = 2.0 # distance from the camera along the z-axis
    N = 20 # number of points on edge
    
    spacing = np.linspace(0, b, N).reshape(N, 1)
    e1 = np.array([-b/2, -b/2.0, -b/2.0 + z_distance]).reshape(1, 3) + spacing.dot(np.array([1, 0, 0]).reshape(1, 3))
    edges = [e1]
    # TODO: generate the other 11 edges equivalently and connect them into a single array of shape (12*N, 3)
    #edges = [e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12]
    
    # plot the cube edges in 3D
    ax = plt.figure().add_subplot(projection='3d')
    ax.scatter(edges[:, 0], edges[:, 1], edges[:, 2])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show(block=True)

    # create a plausible pinhole camera model, VGA resolution
    pinholeCamera = PinholeCamera(640, 480, 450, 450, 319.5, 239.5,
                                RadialTangentialDistortion(-0.3, 0.1, -0.0001, -0.00005))

    u,is_valid = pinholeCamera.project(edges)
    
    # plot the projected points in the image plane
    fig, ax = plt.subplots()
    ax.scatter(u[:, 0], u[:, 1])
    ax.set_xlim(0, 640)
    ax.set_ylim(0, 480)
    top_left_origin = True
    if top_left_origin:
        ax.set_ylim(480, 0)
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
    plt.show(block=True)

    # ============================================================
    # Unit test for projection and back-projection consistency
    # ============================================================
    distortion = RadialTangentialDistortion(-0.3, 0.1, -0.0001, -0.00005)
    pinholeCamera = PinholeCamera(640, 480, 450, 450, 319.5, 239.5, distortion)

    success = True
    print('Running unit test...')
    # TODO: generate random visible point in image
    # TODO: back-project and assign random distance
    # TODO: project again
    # TODO: check the projection is the same as the generated initial image point

    # ============================================================
    # Test with real data
    # ============================================================
    # load image and get the dimensions
    img = plt.imread('./data/pinhole/rgb.jpg')
    H, W = img.shape[:2]

    # use the ./data/pinhole/intrinsic.txt parameters
    intrinsic = np.loadtxt('./data/pinhole/intrinsic.txt')
    # TODO: create the pinhole camera model with the loaded parameters

    # load and visualize depth map
    depth_gt = np.load('./data/pinhole/depth_gt.npy')
    depth_est = np.load('./data/pinhole/depth_est.npy')
    # plot the depth maps
    plt.figure(figsize=(6,6))
    plt.subplot(2,1,1)
    plt.imshow(depth_gt, cmap='plasma')
    plt.title('Ground Truth Depth')
    plt.subplot(2,1,2)
    plt.imshow(depth_est, cmap='plasma')
    plt.title('Monocular Depth Estimation (Metric)')
    # remove the border and ticks
    for ax in plt.gcf().axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    plt.show()

    # back-project the depth maps into 3D using different depths
    # TODO: use np.meshgrid to create the pixel coordinates
    # TODO: use the back-projected rays and the depth maps to get the 3D point clouds
    # TODO: point_cloud_gt = ...
    # TODO: point_cloud_est = ..

    # visualize the 3D point clouds
    # TODO: plot the back-projected ground truth point clouds uisng the plot_pointcloud function defined above
    # set threshold to remove the far points, only consider points within 200 meters
    # TODO: compare the back-projected estimated point cloud uisng the plot_pointcloud_overlap function defined above
    # compute absolute relative error for the estimated point cloud