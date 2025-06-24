import numpy as np
import torch
import ti.math.vec3 as Vec3
import genesis as gs
import torch
from typing import List


def wrap_to_pi(angles):
    angles %= 2 * np.pi
    angles -= 2 * np.pi * (angles > np.pi)
    return angles

@torch.jit.script
def torch_wrap_to_pi_minuspi(theta):
    # type: (Tensor) -> Tensor
    return (theta + np.pi) % (2 * np.pi) - np.pi


def gs_get_dof_state(entity: gs.HybridEntity, num_envs: int, num_dofs: int, num_gripper_joints: int):
    """
    Replaces Gym's DOF state capture.
    
    Returns:
        dof_pos: (num_envs, num_dofs)
        dof_vel: (num_envs, num_dofs)
        dof_pos_wo_gripper: (num_envs, num_dofs - num_gripper_joints)
        dof_vel_wo_gripper: (num_envs, num_dofs - num_gripper_joints)
    """
    pos = entity.get_dofs_position()  # shape: num_envs × num_dofs
    vel = entity.get_dofs_velocity()  # shape: num_envs × num_dofs

    # Optionally convert to PyTorch
    dof_pos = torch.tensor(pos, dtype=torch.float32)
    dof_vel = torch.tensor(vel, dtype=torch.float32)

    dof_pos_wo_gripper = dof_pos[:, :-num_gripper_joints]
    dof_vel_wo_gripper = dof_vel[:, :-num_gripper_joints]

    return dof_pos, dof_vel, dof_pos_wo_gripper, dof_vel_wo_gripper


def gs_get_root_states(sim: gs.Simulator, num_envs: int):
    """
    Retrieves actor states from a Genesis Simulator and returns:
      - root_states: (num_envs, 13) for actor 0
      - box_root_state: (num_envs, 13) for actor 1

    Assumes sim.get_state().root returns a tensor of shape (num_envs, 2, 13)
    """
    state = sim.get_state()  # snapshot of all actors
    # Expecting shape [num_envs, 2, 13] for two actors per env
    root_states_all = state.root  # or state.pos concatenated with vel/quaternion
    assert root_states_all.shape == (num_envs, 2, 13), \
        f"Unexpected root shape {root_states_all.shape}, expected ({num_envs},2,13)"

    root_states = root_states_all[:, 0, :]     # first actor per env
    box_root_state = root_states_all[:, 1, :]  # second actor per env
    return root_states, box_root_state


def gs_get_contact_forces(entity: gs.RigidEntity, num_envs: int, num_bodies: int):
    """
    Returns (contact_forces, box_contact_force): torch.FloatTensor
      - contact_forces: (num_envs, num_bodies, 3)
      - box_contact_force: (num_envs, 3)
    """
    # get_links_net_contact_force returns list of tensors per link
    all_forces = [link.get_net_contact_force() for link in entity.links]
    # Stack into (n_links, num_envs, 3)
    stacked = torch.stack(all_forces, dim=1)
    # split—assuming the last link is your box
    contact_forces = stacked[:, :num_bodies, :]
    box_contact_force = stacked[:, num_bodies, :]
    return contact_forces, box_contact_force

def gs_get_rigid_body_states(entity: gs.RigidEntity, num_envs: int, num_bodies: int, feet_indices, gripper_idx):
    """
    Returns:
      - rigid_body_state: (num_envs, num_bodies + 1, 13)
      - foot_velocities: (num_envs, len(feet_indices), 3)
      - foot_positions: (num_envs, len(feet_indices), 3)
      - ee_pos, ee_orn, ee_vel: each (num_envs, dim)
    """
    qs = entity.get_dofs_position()
    vel = entity.get_dofs_velocity()
    orn = entity.get_links_quat()
    pos = entity.get_links_pos()
    # stack for all links (base + links)
    body_state = torch.cat([pos, orn, vel], dim=-1)  # shape (num_envs, n_links, 13)

    rigid_body_state = body_state
    foot_velocities = body_state[:, feet_indices, 7:10]
    foot_positions = body_state[:, feet_indices, 0:3]
    ee_pos = body_state[:, gripper_idx, 0:3]
    ee_orn = body_state[:, gripper_idx, 3:7]
    ee_vel = body_state[:, gripper_idx, 7:]
    return rigid_body_state, foot_velocities, foot_positions, ee_pos, ee_orn, ee_vel

def gs_get_jacobian(entity: gs.RigidEntity, gripper_idx: int, num_gripper_joints: int):
    """
    Returns: ee_j_eef (jacobian at end-effector)
      shape: (num_envs, end_dim, num_active_dofs)
    """
    # full jacobian at link-level
    J_full = entity.get_jacobian(link=entity.links[gripper_idx])
    # slice to skip gripper joints
    ee_j_eef = J_full[:, :6, -(6 + num_gripper_joints):-num_gripper_joints]
    return ee_j_eef

def gs_get_force_sensors(sensor_entities: List[gs.RigidEntity], num_envs: int):
    """
    Returns a stacked tensor of size (num_envs, n_sensors, 6)
    """
    sensor_values = [sensor.get_dofs_force() for sensor in sensor_entities]
    return torch.stack(sensor_values, dim=1)

def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


def gs_inv_quat(quat):
    qw, qx, qy, qz = quat.unbind(-1)
    inv_quat = torch.stack([1.0 * qw, -qx, -qy, -qz], dim=-1)
    return inv_quat


def gs_transform_by_quat(pos, quat):
    qw, qx, qy, qz = quat.unbind(-1)

    rot_matrix = torch.stack(
        [
            1.0 - 2 * qy**2 - 2 * qz**2,
            2 * qx * qy - 2 * qz * qw,
            2 * qx * qz + 2 * qy * qw,
            2 * qx * qy + 2 * qz * qw,
            1 - 2 * qx**2 - 2 * qz**2,
            2 * qy * qz - 2 * qx * qw,
            2 * qx * qz - 2 * qy * qw,
            2 * qy * qz + 2 * qx * qw,
            1 - 2 * qx**2 - 2 * qy**2,
        ],
        dim=-1,
    ).reshape(*quat.shape[:-1], 3, 3)
    rotated_pos = torch.matmul(rot_matrix, pos.unsqueeze(-1)).squeeze(-1)

    return rotated_pos


def gs_quat2euler(quat):  # xyz
    # Extract quaternion components
    qw, qx, qy, qz = quat.unbind(-1)

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    pitch = torch.where(
        torch.abs(sinp) >= 1,
        torch.sign(sinp) * torch.tensor(torch.pi / 2),
        torch.asin(sinp),
    )

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack([roll, pitch, yaw], dim=-1)


def gs_euler2quat(xyz):  # xyz

    roll, pitch, yaw = xyz.unbind(-1)

    cosr = (roll * 0.5).cos()
    sinr = (roll * 0.5).sin()
    cosp = (pitch * 0.5).cos()
    sinp = (pitch * 0.5).sin()
    cosy = (yaw * 0.5).cos()
    siny = (yaw * 0.5).sin()

    qw = cosr * cosp * cosy + sinr * sinp * siny
    qx = sinr * cosp * cosy - cosr * sinp * siny
    qy = cosr * sinp * cosy + sinr * cosp * siny
    qz = cosr * cosp * siny - sinr * sinp * cosy

    return torch.stack([qw, qx, qy, qz], dim=-1)


def gs_quat_from_angle_axis(angle, axis):
    theta = (angle / 2).unsqueeze(-1)
    xyz = normalize(axis) * theta.sin()
    w = theta.cos()
    return normalize(torch.cat([w, xyz], dim=-1))


def normalize(x, eps: float = 1e-9):
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)


def gs_quat_mul(a, b):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    w1, x1, y1, z1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    w2, x2, y2, z2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = torch.stack([w, x, y, z], dim=-1).view(shape)

    return quat


def gs_quat_apply(a, b):
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    xyz = a[:, 1:]
    t = xyz.cross(b, dim=-1) * 2
    return (b + a[:, :1] * t + xyz.cross(t, dim=-1)).view(shape)


def gs_quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, 1:3] = 0.
    quat_yaw = normalize(quat_yaw)
    return gs_quat_apply(quat_yaw, vec)


def gs_quat_conjugate(a):
    shape = a.shape
    a = a.reshape(-1, 4)
    return torch.cat((a[:, :1], -a[:, 1:], ), dim=-1).view(shape)


## added by pghezzi

"""
Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""


def gs_orientation_error(desired, current):
    cc = gs_quat_conjugate(current)
    q_r = gs_quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)


@torch.jit.script
def gs_quat_rotate_inverse(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c


@torch.jit.script
def gs_quat_from_euler_xyz(roll, pitch, yaw):
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)

    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp

    return torch.stack([qx, qy, qz, qw], dim=-1)

@torch.jit.script
def gs_euler_from_quat(quat_angle):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    x = quat_angle[:,0]
    y = quat_angle[:,1]
    z = quat_angle[:,2]
    w = quat_angle[:,3]
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll_x = torch.atan2(t0, t1)
    
    t2 = 2.0 * (w * y - z * x)
    t2 = torch.clip(t2, -1, 1)
    pitch_y = torch.asin(t2)
    
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw_z = torch.atan2(t3, t4)
    
    return roll_x, pitch_y, yaw_z # in radians

@torch.jit.script
def gs_sphere2cart(sphere_coords):
    # type: (Tensor) -> Tensor
    """ Convert spherical coordinates to cartesian coordinates
    Args:
        sphere_coords (torch.Tensor): Spherical coordinates (l, pitch, yaw)
    Returns:
        cart_coords (torch.Tensor): Cartesian coordinates (x, y, z)
    """
    l = sphere_coords[:, 0]
    pitch = sphere_coords[:, 1]
    yaw = sphere_coords[:, 2]
    cart_coords = torch.zeros_like(sphere_coords)
    cart_coords[:, 0] = l * torch.cos(pitch) * torch.cos(yaw)
    cart_coords[:, 1] = l * torch.cos(pitch) * torch.sin(yaw)
    cart_coords[:, 2] = l * torch.sin(pitch)
    return cart_coords

@torch.jit.script
def gs_cart2sphere(cart_coords):
    # type: (Tensor) -> Tensor
    """ Convert cartesian coordinates to spherical coordinates
    Args:
        cart_coords (torch.Tensor): Cartesian coordinates (x, y, z)
    Returns:
        sphere_coords (torch.Tensor): Spherical coordinates (l, pitch, yaw)
    """
    sphere_coords = torch.zeros_like(cart_coords)
    xy_len = torch.norm(cart_coords[:, :2], dim=1)
    sphere_coords[:, 0] = torch.norm(cart_coords, dim=1)
    sphere_coords[:, 1] = torch.atan2(cart_coords[:, 2], xy_len)
    sphere_coords[:, 2] = torch.atan2(cart_coords[:, 1], cart_coords[:, 0])
    return sphere_coords

def to_torch(x, dtype=torch.float, device='cuda:0', requires_grad=False):
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)

def get_axis_params(value, axis_idx, x_value=0., dtype=float, n_dims=3):
    """construct arguments to `Vec` according to axis index.
    """
    zs = np.zeros((n_dims,))
    assert axis_idx < n_dims, "the axis dim should be within the vector dimensions"
    zs[axis_idx] = 1.
    params = np.where(zs == 1., value, zs)
    params[0] = x_value
    return list(params.astype(dtype))