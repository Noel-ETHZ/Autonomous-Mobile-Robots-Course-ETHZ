# written by Yannick Burkhardt, MRL (ETH Zürch), April 2026

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

### Ground truth poses (only given to assess matches) and camera calibration parameters ###
q_RB1 = np.array([-0.662169,-0.428829,-0.434934,0.434148])
t_RB1 = np.array([-0.923925,7.741909,-0.415333])
q_RB2 = np.array([-0.625698,-0.594441,-0.385229,0.326719])
t_RB2 = np.array([0.018943,6.456890,-0.112635])

T_BC = np.array([0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975,
        0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768,
        -0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949,
        0.0, 0.0, 0.0, 1.0]).reshape(4,4)

T_RB1 = np.eye(4)
T_RB1[:3,:3] = R.from_quat(q_RB1).as_matrix()
T_RB1[:3, 3] = t_RB1

T_RB2 = np.eye(4)
T_RB2[:3,:3] = R.from_quat(q_RB2).as_matrix()
T_RB2[:3, 3] = t_RB2

T_RC1 = T_RB1 @ T_BC
T_RC2 = T_RB2 @ T_BC

intrinsics=[458.654, 457.296, 367.215, 248.375]
K = np.array([
        [intrinsics[0], 0, intrinsics[2]],
        [0, intrinsics[1], intrinsics[3]],
        [0, 0, 1],
    ])
distortion_coeffs=np.array([-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05])
###########################################################################################

def assessMatchesReprojection(
    pts1, pts2, threshold=10.0):
    """Checks matches by triangulating 3D points and computing reprojection error in pixels.

    Assumes input R and t are Camera Poses in world coordinates.
    """

    # Preprocessing of points and transformations
    q1_w2c = R.from_matrix(T_RC1[:3, :3]).as_quat()
    t1_w2c = T_RC1[:3, 3]
    q2_w2c = R.from_matrix(T_RC2[:3, :3]).as_quat()
    t2_w2c = T_RC2[:3, 3]

    R1_w2c = R.from_quat(q1_w2c).as_matrix()
    R2_w2c = R.from_quat(q2_w2c).as_matrix()

    pts1 = np.array([p.pt for p in pts1])
    pts2 = np.array([p.pt for p in pts2])

    # Undistort and normalize points
    pts1_reshaped = pts1.reshape(-1, 1, 2).astype(np.float32)
    pts2_reshaped = pts2.reshape(-1, 1, 2).astype(np.float32)

    pts1_norm = cv2.undistortPoints(pts1_reshaped, K, distortion_coeffs).reshape(-1, 2)
    pts2_norm = cv2.undistortPoints(pts2_reshaped, K, distortion_coeffs).reshape(-1, 2)

    # Define projection matrices for triangulation
    t1_w2c = - R1_w2c.T @ t1_w2c
    R1_w2c = R1_w2c.T   
    t2_w2c = - R2_w2c.T @ t2_w2c
    R2_w2c = R2_w2c.T   
    P1 = np.hstack((R1_w2c, t1_w2c.reshape(3, 1)))
    P2 = np.hstack((R2_w2c, t2_w2c.reshape(3, 1)))

    # Triangulate points in 3D (yields 4D homogeneous coordinates)
    pts4D_hom = cv2.triangulatePoints(
        P1, P2, pts1_norm.T, pts2_norm.T
    ).T  # Nx4
    pts3D_world = pts4D_hom[:, :3] / pts4D_hom[:, [3]]  # Normalize to Nx3

    # Project 3D points back to Image 1 and Image 2 (in pixels)
    # Camera 1 projection
    rvec1_w2c, _ = cv2.Rodrigues(R1_w2c)
    proj_pts1, _ = cv2.projectPoints(
        pts3D_world, rvec1_w2c, t1_w2c, K, distortion_coeffs
    )
    proj_pts1 = proj_pts1.reshape(-1, 2)

    # Camera 2 projection
    rvec2_w2c, _ = cv2.Rodrigues(R2_w2c)
    proj_pts2, _ = cv2.projectPoints(
        pts3D_world, rvec2_w2c, t2_w2c, K, distortion_coeffs
    )
    proj_pts2 = proj_pts2.reshape(-1, 2)

    # Compute Euclidean distance (reprojection error) in pixels
    err1 = np.linalg.norm(pts1 - proj_pts1, axis=1)
    err2 = np.linalg.norm(pts2 - proj_pts2, axis=1)

    # Average error across both images
    total_error = (err1 + err2) / 2.0

    # Mask valid points
    correct_matches_mask = total_error < threshold

    # Depth check (points must be in front of both cameras)
    # Convert world points to camera coordinate systems to check Z
    pts3D_cam1 = (R1_w2c @ pts3D_world.T + t1_w2c.reshape(3, 1)).T
    pts3D_cam2 = (R2_w2c @ pts3D_world.T + t2_w2c.reshape(3, 1)).T

    positive_depth = (pts3D_cam1[:, 2] > 0) & (pts3D_cam2[:, 2] > 0)

    return correct_matches_mask & positive_depth

def drawMatches(kpts_arr1, kpts_arr2):
    """
    Plots matches in two images in two colors.
    Mtaches with small reprojection error are green (inliers), outliers are red.
    """

    matches_img = np.hstack([img1, img2])
    matches_img = cv2.cvtColor(matches_img, cv2.COLOR_GRAY2BGR)
    correct_matches_mask = assessMatchesReprojection(kpts_arr1, kpts_arr2)

    for i, k in enumerate(kpts_arr1):
        if correct_matches_mask[i]:
            color = (0, 255, 0)
        else:
            color = (255, 0, 0)
        pt0 = k.pt
        pt1 = kpts_arr2[i].pt
        pt0 = (round(pt0[0]), round(pt0[1]))
        pt1 = (round(pt1[0] + img1.shape[1])), round(pt1[1])
        cv2.line(matches_img, pt0, pt1,
                color=color, thickness=1, lineType=cv2.LINE_AA)
        # display line end-points as circles
        cv2.circle(matches_img, pt0, radius=3, color=color, thickness=1)
        cv2.circle(matches_img, pt1, radius=3, color=color, thickness=1)
    
    print("---------------------------------------------")
    print(f"Number of inliers: {np.sum(correct_matches_mask)}")
    print(f"Number of outliers: {np.sum(~correct_matches_mask)}")
    print(f"Ratio of correct matches: {100. * np.sum(correct_matches_mask) / len(correct_matches_mask)}%")
    print("---------------------------------------------")
    plt.imshow(matches_img,),plt.show()

### These functions should be implemented ###
def calculateDescriptorDistances(des1, des2):
    # Compute L2 distance squared using matrix multiplication
    # ||des1 - des2||^2 = ||des1||^2 + ||des2||^2 - 2 * des1 @ des2.T
    norm1 = np.sum(des1**2, axis=1, keepdims=True)
    norm2 = np.sum(des2**2, axis=1)
    # Cip to ensure no negative values due to floating point errors
    dist_sq = np.maximum(norm1 + norm2 - 2 * np.dot(des1, des2.T), 0)
    dist = np.sqrt(dist_sq)
    return dist

def getMutualMatchingMask(dist, best_index):
    # Find the best match for des2 relative to des1
    best_index_back = np.argmin(dist, axis=0)
    mutual_mask = np.arange(len(des1)) == best_index_back[best_index]
    return mutual_mask

def getRatioMask(dist, idx_sorted, threshold=0.7):
    d1 = dist[np.arange(len(dist)), idx_sorted[:, 0]]
    d2 = dist[np.arange(len(dist)), idx_sorted[:, 1]]

    ratio_mask = d1 < threshold * d2
    return ratio_mask
#############################################

# We want to find landmarks in these two images
img1 = cv2.imread('1403636671213555456.png')
img2 = cv2.imread('1403636672113555456.png')
plt.imshow(np.hstack([img1,img2])),plt.show()
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Initiate SIFT detector
sift = cv2.SIFT_create()
 
# find the keypoints and descriptors with SIFT
# if you are interested on how SIFT works, check this summary: https://docs.opencv2.org/4.x/da/df5/tutorial_py_sift_intro.html
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# Draw detections
kp_img1=cv2.drawKeypoints(img1,kp1,img1,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
kp_img2=cv2.drawKeypoints(img2,kp2,img2,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.imshow(np.hstack([kp_img1,kp_img2])),plt.show()

# Match the keypoints by finding the corresponding, most similar descriptors
# for each descriptor in the first image, find its nearest neighbor in the second image by the L2 norm
dist = calculateDescriptorDistances(des1, des2)

# sorted_indices gives us the indices of the closest descriptors
idx_sorted = np.argsort(dist, axis=1)
best_index = idx_sorted[:, 0]

# Let's look at the matching results
drawMatches(np.array(kp1), np.array(kp2)[best_index])
# There are lots of outliers...

# Filter by cross-checking that we match the nearest neighbor in both directions
mutual_mask = getMutualMatchingMask(dist, best_index)

# Let's look at the matches again
drawMatches(np.array(kp1)[mutual_mask], np.array(kp2)[best_index][mutual_mask])
# That's better, but still not great

# Let's filter out the matches that are close to other neighbors
ratio_mask = getRatioMask(dist, idx_sorted)

# Combine both constraints
match_mask = ratio_mask & mutual_mask

# How do the matches look now?
drawMatches(np.array(kp1)[match_mask], np.array(kp2)[best_index][match_mask])