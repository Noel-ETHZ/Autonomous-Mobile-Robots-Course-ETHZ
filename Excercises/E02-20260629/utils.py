import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import meshcat.geometry as g
import meshcat.transformations as tf

def draw_eight_3D_trajectory(size=0.15, center=[0.5, 0.0, 0.5]):
    "draw a 3D trajectory in the shape of an eight"
    n = 5
    t = np.linspace(0, n*2*np.pi, n*60)
    x = np.full_like(t, center[0])
    y = center[1] + size * np.sin(t)
    z = center[2] + size * np.sin(2*t) / 2
    return np.column_stack((x, y, z))


def get_ee_frame_id(model: pin.Model) -> int:
    for name in ("wrist_ft_tool_link", "arm_7_link"):
        if model.existFrame(name):
            return model.getFrameId(name)
    return model.nframes - 1

def fk_position(model: pin.Model, data: pin.Data, q: np.ndarray, ee_frame_id: int) -> np.ndarray:
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    return data.oMf[ee_frame_id].translation.copy()


def target_pose_from_position(position: np.ndarray, orientation_quat: np.ndarray) -> np.ndarray:
    return np.concatenate([position, orientation_quat])


def parse_ik_solution(sol: object) -> np.ndarray:
    if sol is None:
        return np.array([])
    try:
        arr = np.asarray(sol, dtype=float).reshape(-1)
    except Exception:
        return np.array([])
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return np.array([])
    return arr

def create_sphere(viz: MeshcatVisualizer, node_name: str, radius: float, color_hex: int):
    material = g.MeshPhongMaterial(color=color_hex, opacity=1.0)
    viz.viewer[node_name].set_object(g.Sphere(radius), material)


def set_sphere_xyz(viz: MeshcatVisualizer, node_name: str, xyz: np.ndarray):
    viz.viewer[node_name].set_transform(tf.translation_matrix([float(xyz[0]), float(xyz[1]), float(xyz[2])]))

def remove_node(viz: MeshcatVisualizer, node_name: str):
    """
    Removes a node and its geometry from the Meshcat scene.
    """
    viz.viewer[node_name].delete()


