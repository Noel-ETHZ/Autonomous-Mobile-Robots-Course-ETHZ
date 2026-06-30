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
        
    def distort(self, u_undistorted):
        u_undistorted = u_undistorted.reshape(-1, 2)
        u, v = u_undistorted.T
        r2 = u**2 + v**2
        
        u_residual = 2 * self.p1 * u * v + self.p2 * (r2 + 2 * u**2)
        v_residual = self.p1 * (r2 + 2 * v**2) + 2 * self.p2 * u * v
        residual = np.stack([u_residual, v_residual], axis=1)

        radial_coeff = (1 + self.k1 * r2 + self.k2 * r2**2).reshape(-1, 1)
        return radial_coeff * u_undistorted + residual
    
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
        x_dash, is_valid = self.p(x)
        x_ddash = self.d(x_dash)
        u = self.k(x_ddash)
        
        is_valid_index = np.argwhere(is_valid)[:, 0]
        is_in_image_u = np.logical_and(u[:, 0] >= -0.5, u[:, 0] <= self.width - 0.5)
        is_in_image_v = np.logical_and(u[:, 1] >= -0.5, u[:, 1] <= self.height - 0.5)
        is_in_image = np.logical_and(is_in_image_u, is_in_image_v)
        is_valid[is_valid_index] = is_in_image
        
        if not np.any(is_valid):
            return [], is_valid
        return u[is_in_image, :], is_valid

    def p(self, x):
        is_z_positive = x[:, 2] > 1e-10
        u = x[is_z_positive, 0] / x[is_z_positive, 2]
        v = x[is_z_positive, 1] / x[is_z_positive, 2]
        return np.stack([u, v], axis=1), is_z_positive
    
    def p_inverse(self, x_dash):
        return np.stack([x_dash[:,0], x_dash[:,1], np.ones(np.size(x_dash[:,0]))], axis=1)

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
        u = self.f1 * x_ddash[:, 0] + self.c1
        v = self.f2 * x_ddash[:, 1] + self.c2
        return np.stack([u, v], axis=1)
    
    def k_inverse(self, u):
        x_ddash1 = 1.0/self.f1 * (u[:, 0] - self.c1)
        x_ddash2 = 1.0/self.f2 * (u[:, 1] - self.c2)
        return np.stack([x_ddash1, x_ddash2], axis=1)
    
    def backproject(self, u):
        x_ddash = self.k_inverse(u)
        x_dash = self.d_inverse(x_ddash)
        return self.p_inverse(x_dash)

# helper function to plot point clouds
def plot_pointcloud(points_3d, colors = None, title = "Point Cloud", sample_size = 1e5):
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
    e2 = np.array([-b/2, -b/2.0, b/2.0 + z_distance]).reshape(1, 3) + spacing.dot(np.array([1, 0, 0]).reshape(1, 3))
    e3 = np.array([-b/2, b/2.0, -b/2.0 + z_distance]).reshape(1, 3) + spacing.dot(np.array([1, 0, 0]).reshape(1, 3))
    e4 = np.array([-b/2, b/2.0, b/2.0 + z_distance]).reshape(1, 3) + spacing.dot(np.array([1, 0, 0]).reshape(1, 3))
    e5 = np.array([b/2, -b/2.0, -b/2.0 + z_distance]).reshape(1, 3) + spacing.dot(np.array([0, 1, 0]).reshape(1, 3))
    e6 = np.array([b/2, -b/2.0, b/2.0 + z_distance]).reshape(1, 3) + spacing.dot(np.array([0, 1, 0]).reshape(1, 3))
    e7 = np.array([-b/2, -b/2.0, -b/2.0 + z_distance]).reshape(1, 3) + spacing.dot(np.array([0, 1, 0]).reshape(1, 3))
    e8 = np.array([-b/2, -b/2.0, b/2.0 + z_distance]).reshape(1, 3) + spacing.dot(np.array([0, 1, 0]).reshape(1, 3))
    e9 = np.array([b/2, -b/2.0, -b/2.0 + z_distance]).reshape(1, 3) + spacing.dot(np.array([0, 0, 1]).reshape(1, 3))
    e10 = np.array([b/2, b/2.0, -b/2.0 + z_distance]).reshape(1, 3) + spacing.dot(np.array([0, 0, 1]).reshape(1, 3))
    e11 = np.array([-b/2, -b/2.0, -b/2.0 + z_distance]).reshape(1, 3) + spacing.dot(np.array([0, 0, 1]).reshape(1, 3))
    e12 = np.array([-b/2, b/2.0, -b/2.0 + z_distance]).reshape(1, 3) + spacing.dot(np.array([0, 0, 1]).reshape(1, 3))
    edges = np.concatenate([e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12], axis=0)

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
    # method 1: loop over points and test one by one
    for i in range(0,1000):
        # generate random visible point in image
        u_1 = np.random.uniform(-0.49,639.49)
        u_2 = np.random.uniform(-0.49,479.49)
        # back-project and assign random distance
        randomPoint = np.array([[u_1,u_2]])
        ray = pinholeCamera.backproject(np.array([[u_1,u_2]]))
        # project again
        point, is_valid = pinholeCamera.project(ray)
        # check the projection is the same as the generated initial image point
        if np.linalg.norm(point - randomPoint)>0.001:
            success = False
            break

    if success:
        print('[PASSED]')
    else:
        print('[FAILED]')

    # method 2: test all points at once
    # generate random visible point in image
    u_1 = np.random.uniform(-0.49,639.49,size=1000)
    u_2 = np.random.uniform(-0.49,479.49,size=1000)
    randomPoint = np.stack([u_1, u_2], axis=1)

    # back-project and assign random distance
    ray = pinholeCamera.backproject(randomPoint)

    # project again
    point, is_valid = pinholeCamera.project(ray)

    # check the projection is the same as the generated initial image point
    error = np.linalg.norm(point - randomPoint[is_valid], axis=1)
    assert np.all(error < 0.001)
    
    # ============================================================
    # Test with real data
    # ============================================================
    # load image and get the dimensions
    img = plt.imread('./data/pinhole/rgb.jpg')
    H, W = img.shape[:2]

    # use the ./data/pinhole/intrinsic.txt parameters
    intrinsic = np.loadtxt('./data/pinhole/intrinsic.txt')
    pinholeCamera = PinholeCamera(W, H, intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3], distortion=None)

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

    # back-project the depth maps into 3D
    # use np.meshgrid to create the pixel coordinates
    u_1 = np.arange(W)
    u_2 = np.arange(H)
    u_1, u_2 = np.meshgrid(u_1, u_2)
    u = np.stack([u_1.flatten(), u_2.flatten()], axis=1)
    ray = pinholeCamera.backproject(u)
    # use the back-projected rays and the depth maps to get the 3D point clouds
    point_cloud_gt = ray * depth_gt.flatten().reshape(-1, 1)
    point_cloud_est = ray * depth_est.flatten().reshape(-1, 1)

    # visualize the 3D point clouds
    colors = img[u_2.flatten(), u_1.flatten(), :]/255.0
    # set threshold to remove the far points, only consider points within 200 meters
    thres = 200.0
    valid = (depth_gt.flatten() > 0) & (depth_gt.flatten() < 200.0)
    # plot the back-projected ground truth point clouds
    plot_pointcloud(point_cloud_gt[valid], colors=colors[valid], title='Ground Truth Point Cloud')
    # compare the back-projected estimated point clouds
    plot_pointcloud_overlap(point_cloud_gt[valid], point_cloud_est[valid], colors=colors[valid], title='Point Cloud Overlap (Metric Depth)')
    absrel_err = np.mean(np.abs(depth_gt.flatten()[valid] - depth_est.flatten()[valid]) / (depth_gt.flatten()[valid] + 1e-10))
    print(f'Absolute Relative Error (Metric Depth): {absrel_err:.4f}')
    
    
    