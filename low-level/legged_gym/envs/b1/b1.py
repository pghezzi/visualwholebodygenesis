import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
from genesis.engine.solvers.rigid.rigid_solver_decomp import RigidSolver
from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
import numpy as np
import os

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.math import wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.gs_utils import *
from .b1_config import B1Cfg

class B1(LeggedRobot):
    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos * gs_rand_float(0.8, 1.2, (len(env_ids), self.num_dofs), device=self.device)
        self.dof_vel[env_ids] = 0.

        self.robot.set_dofs_position(
            position=self.dof_pos[env_ids],
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
            envs_idx=env_ids,
        )
        self.robot.zero_all_dofs_velocity(env_ids)
    

    #def _reward_feet_air_time(self):
    #    # Reward long steps
    #    # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
    #    first_contact = (self.env.feet_air_time > 0.) * self.env.foot_contacts_from_sensor  #self.env.contact_filt
    #    self.env.feet_air_time += self.env.dt
    #
    #    if self.env.cfg.rewards.feet_aritime_allfeet:
    #        rew_airTime = torch.sum((self.env.feet_air_time - 0.5) * first_contact, dim=1)
    #    else:
    #        rew_airTime = torch.sum((self.env.feet_air_time[:, :2] - 0.5) * first_contact[:, :2], dim=1)
    #    
    #    rew_airTime *= self.env._get_walking_cmd_mask()  # reward for stepping for any of the 3 motions
    #    self.env.feet_air_time *= ~ self.env.foot_contacts_from_sensor  #self.env.contact_filt
    #    return rew_airTime, rew_airTime
    
    def _get_walking_cmd_mask(self, env_ids=None, return_all=False):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        walking_mask0 = torch.abs(self.commands[env_ids, 0]) > self.cfg.commands.lin_vel_x_clip
        walking_mask1 = torch.abs(self.commands[env_ids, 1]) > self.cfg.commands.lin_vel_x_clip
        walking_mask2 = torch.abs(self.commands[env_ids, 2]) > self.cfg.commands.ang_vel_yaw_clip
        walking_mask = walking_mask0 | walking_mask1 | walking_mask2
        if return_all:
            return walking_mask0, walking_mask1, walking_mask2, walking_mask
        return walking_mask

    def _reward_feet_air_time(self):
        # Reward long steps
        contact = self.link_contact_forces[:, self.feet_indices, 2] > 2.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        
        if self.cfg.rewards.feet_aritime_allfeet:
           rew_airTime = torch.sum((self.env.feet_air_time - 0.5) * first_contact, dim=1)
        else:
            rew_airTime = torch.sum((self.env.feet_air_time[:, :2] - 0.5) * first_contact[:, :2], dim=1)
        rew_airTime = torch.sum((self.feet_air_time - 0.2) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime