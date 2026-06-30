# Autonomous Mobile Robots - Exercise 5 - Stereo Vision

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# helper function to plot 3D point clouds
def plot_pointcloud_mpl(xyz, name):
    """
    Plots a 3D point cloud using Matplotlib.

    Args:
        xyz (np.ndarray): Array of 3D points (N, 3).
        name (str): Name or title for the plot window.
    """
    ax = plt.figure(num=name).add_subplot(projection='3d')
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=xyz[:, 2])
    ax.set_box_aspect((np.ptp(xyz[:, 0]),
                        np.ptp(xyz[:, 1]),
                        np.ptp(xyz[:, 2])))

# helper function to plot the projected points on the left and right images
def plotProjection(u_left, u_right, H, W, top_left_origin=True):
    """
    Plots the projections of 3D points onto left and right image planes, and their overlay.

    Args:
        u_left (np.ndarray): Projected points in the left image (N, 2).
        u_right (np.ndarray): Projected points in the right image (N, 2).
        H (int): Image height.
        W (int): Image width.
        top_left_origin (bool): If True, origin is at the top-left of the image.
    """
    plt.figure(num='Projection', figsize=(3 * W / 10, H / 10))

    plt.subplot(1, 3, 1)
    plt.scatter(u_left[:, 0], u_left[:, 1], c='r', label='Left Image')
    plt.xlim(0, W)
    plt.ylim(0, H)
    ax = plt.gca()
    ax.set_aspect('equal')
    if top_left_origin:
        plt.ylim(H, 0)   # top-left origin
        ax.xaxis.tick_top()                  # move x ticks to top
        ax.xaxis.set_label_position('top')   # move x label to top
    plt.xlabel('u [px]')
    plt.ylabel('v [px]')
    plt.title('Projection of Teapot Point Cloud (Left Image)')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.scatter(u_right[:, 0], u_right[:, 1], c='b', label='Right Image')
    plt.xlim(0, W)
    plt.ylim(0, H)
    ax = plt.gca()
    ax.set_aspect('equal')
    if top_left_origin:
        plt.ylim(H, 0)   # top-left origin
        ax.xaxis.tick_top()                  # move x ticks to top
        ax.xaxis.set_label_position('top')   # move x label to top
    plt.xlabel('u [px]')
    plt.ylabel('v [px]')
    plt.title('Projection of Teapot Point Cloud (Right Image)')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.scatter(u_left[:, 0], u_left[:, 1], c='r', label='Left Image')
    plt.scatter(u_right[:, 0], u_right[:, 1], c='b', label='Right Image')
    plt.xlim(0, W)
    plt.ylim(0, H)
    ax = plt.gca()
    ax.set_aspect('equal')
    if top_left_origin:
        plt.ylim(H, 0)   # top-left origin
        ax.xaxis.tick_top()                  # move x ticks to top
        ax.xaxis.set_label_position('top')   # move x label to top
    plt.xlabel('u [px]')
    plt.ylabel('v [px]')
    plt.title('Overlay Projection')
    plt.legend()

    plt.tight_layout()


# helper function to plot disparity for projected points
def plotDisparity(u, disparity, H, W, top_left_origin=True):
    """
    Plots a sparse disparity map for projected points.

    Args:
        u (np.ndarray): Projected points (N, 2).
        disparity (np.ndarray): Disparity values for each point (N,).
        H (int): Image height.
        W (int): Image width.
        top_left_origin (bool): If True, origin is at the top-left of the image.
    """
    x = u[:, 0]
    y = u[:, 1]

    fig, ax = plt.subplots(num='Disparity Map', figsize=(W / 10, H / 10))

    sc = ax.scatter(x, y, c=disparity, cmap='magma')
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect('equal')

    if top_left_origin:
        ax.set_ylim(H, 0)
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')

    ax.set_title('Sparse Disparity Map')
    ax.set_xlabel('u [px]')
    ax.set_ylabel('v [px]')
    fig.colorbar(sc, ax=ax, label='Disparity')
    plt.tight_layout()

# helper function to add noise to disparity
def addNoise(disparity, noise_std=0.05):
    noise = noise_std * np.random.randn(*disparity.shape)
    return disparity + noise

if __name__ == "__main__":
    # ============================================================
    # Stereo triangulation of synthetic teapot point cloud
    # ============================================================
    # visualize the 3D point cloud of the teapot
    xyzPoints = np.genfromtxt('./data/stereo/teapot.xyz', delimiter=',')
    plot_pointcloud_mpl(xyzPoints, 'Utah Teapot (3D)')
    # Only consider front half.
    xyzPoints = xyzPoints[xyzPoints[:, 1] < 0, :]

    # Intrinsic
    H, W, f = 40, 80, 70.0
    K = np.array([[f, 0, W/2-0.5], [0, f, H/2-0.5], [0, 0, 1]])
    # Extrinsic
    baseline = 1.0
    offset_y = 1.5
    offset_z = 10.0
    R_left = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    t_left = np.array([[baseline/2.0], [offset_y], [offset_z]])
    R_right = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    t_right = np.array([[-baseline/2.0], [offset_y], [offset_z]])

    from e05_pinhole_camera import PinholeCamera
    # TODO: create pinholeCamera object using the intrinsic parameters defined above

    # project the teapot point cloud to the left and right images
    # TODO: transform the point cloud to the left and right camera frames
    # TODO: project to the image planes
    # TODO: plot the projected points on the left and right images using the helper function plot_projection

    # TODO: compute the disparity for each point
    # TODO: plot disparity map using the helper function plot_sparse_disparity_map

    # stereo triangulation
    def stereoTriangulation(u, disparity, camera, baseline):
        # TODO
        return []
    
    # transform points in the camera frame to world frame
    def cam2world(points_cam, R, t):
        # TODO

        return []
    # TODO: perform stereo triangulation to get 3D points in the camera frame using stereo_tringulation function
    # TODO: then transform the points to world frame using cam2world function
    # TODO: plot the triangulated point cloud using the helper function point_cloud_plot
    
    # add noise to the disparity and see how it affects the triangulation
    # TODO: add noise to the disparity using the helper function add_noise
    # TODO: perform stereo triangulation with the noisy disparity and plot the triangulated point cloud to see how noise affects the result
    # TODO: change this baseline and see how it affects the triangulation result    

    plt.show()
    # NOTE: you need to close the pop-up windows to proceed to the next part of the code. You can also choose to move plt.show() only once at the end of the script to show all the plots together.
    
    # ============================================================
    # Stereo triangulation of real-world scene
    # ============================================================
    # TODO: load camera parameters. Note the baseline is in milimeters, so we need to convert it to meters.
    # load image and mask
    image_left = plt.imread('./data/stereo/im0.png')
    mask_left = plt.imread('./data/stereo/mask.png')
    valid = mask_left.astype(bool)
    # TODO: create coordinates and get color for all valid pixels
    
    # TODO: load ground truth and predicted disparities
    # TODO: disparity_gt = ...
    # TODO: disparity_pred = ...
    
    # TODO: stereo triangulation for all valid pixels using disparities using stereo_triangulation function
    
    from e05_pinhole_camera import plot_pointcloud, plot_pointcloud_overlap
    # TODO: plot the triangulated point clouds
    # TODO: compare the point clouds from predicted disparity, compute the absolute relative error for depth
    
    
    