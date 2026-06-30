# Autonomous Mobile Robots - Exercise 5 - Stereo Vision

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# helper function to plot 3D point clouds
def plot_pointcloud_mpl(xyz, name):
    ax = plt.figure(num=name).add_subplot(projection='3d')
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=xyz[:, 2])
    ax.set_box_aspect((np.ptp(xyz[:, 0]),
                        np.ptp(xyz[:, 1]),
                        np.ptp(xyz[:, 2])))

# helper function to plot the projected points on the left and right images
def plotProjection(u_left, u_right, H, W, top_left_origin=True):
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

    from e05_pinhole_camera_sol import PinholeCamera
    pinholeCamera = PinholeCamera(W, H, K[0,0], K[1,1], K[0,2], K[1,2], None)

    # project the teapot point cloud to the left and right images
    # transform the point cloud to the left and right camera frames
    xyzPoints_left = (R_left @ xyzPoints.T + t_left).T
    xyzPoints_right = (R_right @ xyzPoints.T + t_right).T
    # project to the image planes
    u_left, is_valid_left = pinholeCamera.project(xyzPoints_left)
    u_right, is_valid_right = pinholeCamera.project(xyzPoints_right)
    # plot the projected points on the left and right images
    plotProjection(u_left, u_right, H, W)

    # compute the disparity for each point
    disparity = u_left[:, 0] - u_right[:, 0]
    # plot disparity map
    plotDisparity(u_left, disparity, H, W)

    # stereo triangulation
    def stereoTriangulation(u, disparity, camera, baseline):
        f = camera.f1
        z = f * baseline / disparity
        # back-project the points to 3D
        ray = pinholeCamera.backproject(u)
        points = ray * z.reshape(-1, 1)
        return points
    
    # transform points in the camera frame to world frame
    def cam2world(points_cam, R, t):
        R_inv = np.linalg.inv(R)
        t_inv = -R_inv @ t
        points_world = (R_inv @ points_cam.T + t_inv).T
        return points_world
    
    points_cam = stereoTriangulation(u_left, disparity, pinholeCamera, baseline)
    points_world = cam2world(points_cam, R_left, t_left)
    # plot the triangulated point cloud
    plot_pointcloud_mpl(points_world, 'Triangulated Point Cloud')
    
    # add noise to the disparity and see how it affects the triangulation
    disparity_noisy = addNoise(disparity)
    points_cam_noisy = stereoTriangulation(u_left, disparity_noisy, pinholeCamera, baseline)
    points_world_noisy = cam2world(points_cam_noisy, R_left, t_left)
    plot_pointcloud_mpl(points_world_noisy, 'Triangulated Point Cloud with Noisy Disparity')
    # change the baseline to 0.1 and run the triangulation again, see how it affects the triangulation result
    plt.show()
    
    # ============================================================
    # Stereo triangulation of real-world scene
    # ============================================================a
    # load camera parameters. Note the baseline is in milimeters, so we need to convert it to meters.
    baseline = 536.62 / 1000.0
    pinholeCamera = PinholeCamera(1920, 1080, 1733.74, 1733.74, 792.2, 541.89, distortion=None)
    # load image and mask
    image_left = plt.imread('./data/stereo/im0.png')
    mask_left = plt.imread('./data/stereo/mask.png')
    valid = mask_left.astype(bool)
    # create coordinates for all valid pixels
    H, W = image_left.shape[:2]
    u_1, u_2 = np.meshgrid(np.arange(W), np.arange(H))
    u = np.stack([u_1[valid], u_2[valid]], axis=-1)
    # get color for each valid pixel
    color = image_left[u[:, 1], u[:, 0], :]
    
    # load disparity maps
    disparity_map_gt = np.load('./data/stereo/disp_gt.npy')
    disparity_gt = disparity_map_gt[u[:, 1], u[:, 0]]
    disparity_map_pred = np.load('./data/stereo/disp_est.npy')
    disparity_pred = disparity_map_pred[u[:, 1], u[:, 0]]
    
    # stereo triangulation using disparity maps
    points_cam_gt = stereoTriangulation(u, disparity_gt, pinholeCamera, baseline)
    points_cam_pred = stereoTriangulation(u, disparity_pred, pinholeCamera, baseline)
    
    # plot the triangulated point clouds
    from e05_pinhole_camera_sol import plot_pointcloud, plot_pointcloud_overlap
    plot_pointcloud(points_cam_gt, color, 'Triangulated Point Cloud (GT Disparity)')
    plot_pointcloud_overlap(points_cam_gt, points_cam_pred, colors=color, title='Triangulated Point Cloud Overlap')
    absrel_err_rel = np.mean(np.abs(points_cam_gt[:, 2] - points_cam_pred[:, 2]) / (points_cam_gt[:, 2] + 1e-10))
    print(f'AbsRel error: {absrel_err_rel:.4f}')
    
    
    