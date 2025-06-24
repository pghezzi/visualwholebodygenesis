import torch
from torch import Tensor
import numpy as np
from typing import Tuple

# @torch.jit.script
def normalize(x, eps: float = 1e-9):
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)

# @torch.jit.script
def quat_apply(a, b):
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    xyz = a[:, :3]
    t = xyz.cross(b, dim=-1) * 2
    return (b + a[:, 3:] * t + xyz.cross(t, dim=-1)).view(shape)

# @ torch.jit.script
def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)

# @ torch.jit.script
def wrap_to_pi(angles):
    angles %= 2*np.pi
    angles -= 2*np.pi * (angles > np.pi)
    return angles

# @ torch.jit.script
def torch_rand_sqrt_float(lower, upper, shape, device):
    # type: (float, float, Tuple[int, int], str) -> Tensor
    r = 2*torch.rand(*shape, device=device) - 1
    r = torch.where(r<0., -torch.sqrt(-r), torch.sqrt(r))
    r =  (r + 1.) / 2.
    return (upper - lower) * r + lower


def quat_from_euler_xyz(roll, pitch, yaw):
    """
    Convert Euler angles to quaternion
    """
    # Handle both scalar and tensor inputs
    if isinstance(roll, (int, float)):
        roll = torch.tensor(roll)
    if isinstance(pitch, (int, float)):
        pitch = torch.tensor(pitch)
    if isinstance(yaw, (int, float)):
        yaw = torch.tensor(yaw)
    
    # Ensure all inputs are on the same device
    device = yaw.device if hasattr(yaw, 'device') else roll.device if hasattr(roll, 'device') else pitch.device if hasattr(pitch, 'device') else 'cpu'
    roll = roll.to(device)
    pitch = pitch.to(device)
    yaw = yaw.to(device)
    
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    
    # Handle broadcasting for different input shapes
    if roll.dim() == 0:
        # Scalar inputs
        quat = torch.zeros(4, device=device)
        quat[0] = cy * cp * cr + sy * sp * sr
        quat[1] = cy * cp * sr - sy * sp * cr
        quat[2] = cy * sp * cr + sy * cp * sr
        quat[3] = sy * cp * cr - cy * sp * sr
    else:
        # Tensor inputs
        quat = torch.zeros((len(roll), 4), device=device)
        quat[:, 0] = cy * cp * cr + sy * sp * sr
        quat[:, 1] = cy * cp * sr - sy * sp * cr
        quat[:, 2] = cy * sp * cr + sy * cp * sr
        quat[:, 3] = sy * cp * cr - cy * sp * sr
    
    return quat

def euler_from_quat(quat):
    """
    Convert quaternion to Euler angles
    """
    x = quat[:, 0]
    y = quat[:, 1]
    z = quat[:, 2]
    w = quat[:, 3]
    roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = torch.asin(2 * (w * y - z * x))
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return torch.stack((roll, pitch, yaw), dim=-1)

def quat_rotate_inverse(quat, vec):
    """
    Rotate vector by inverse of quaternion
    """
    qw = quat[:, 3]
    qx = quat[:, 0] 
    qy = quat[:, 1]
    qz = quat[:, 2]

    # Compute rotation matrix
    R00 = 1 - 2 * (qy * qy + qz * qz)
    R01 = 2 * (qx * qy - qw * qz)
    R02 = 2 * (qx * qz + qw * qy)
    R10 = 2 * (qx * qy + qw * qz)
    R11 = 1 - 2 * (qx * qx + qz * qz) 
    R12 = 2 * (qy * qz - qw * qx)
    R20 = 2 * (qx * qz - qw * qy)
    R21 = 2 * (qy * qz + qw * qx)
    R22 = 1 - 2 * (qx * qx + qy * qy)

    # Stack into rotation matrix
    rot = torch.stack([R00, R01, R02, R10, R11, R12, R20, R21, R22], dim=-1)
    rot = rot.reshape(quat.shape[0], 3, 3)

    # Rotate vector
    rotated = torch.matmul(rot.transpose(-2,-1), vec.unsqueeze(-1)).squeeze(-1)
    return rotated



