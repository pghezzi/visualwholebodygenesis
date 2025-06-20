import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
from genesis.engine.solvers.rigid.rigid_solver_decomp import RigidSolver
from legged_gym import LEGGED_GYM_ROOT_DIR, envs
import time
from warnings import WarningMessage
import numpy as np
import os

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.math import wrap_to_pi
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.gs_utils import *
from collections import deque


class GO2Deploy(LeggedRobot):

    def compute_observations(self):
        """ Computes observations
        """
        obs_buf = torch.cat((
            self.commands[:, :3] * self.commands_scale,    # cmd     3
            self.projected_gravity,                        # g       3
            self.base_ang_vel * self.obs_scales.ang_vel,   # omega   3
            (self.dof_pos - self.default_dof_pos) *
            self.obs_scales.dof_pos,                       # p_t     12
            self.dof_vel * self.obs_scales.dof_vel,        # dp_t    12
            self.actions,                                  # a_{t-1} 12
        ), dim=-1)
        
        if self.num_privileged_obs is not None:  # critic_obs, no noise
            self.privileged_obs_buf = torch.cat((
                self.base_lin_vel * self.obs_scales.lin_vel,   # v_t     3
                self.base_ang_vel * self.obs_scales.ang_vel,   # omega_t 3
                self.projected_gravity,                        # g_t     3
                self.commands[:, :3] * self.commands_scale,    # cmd_t   3
                (self.dof_pos - self.default_dof_pos) * 
                self.obs_scales.dof_pos,                       # p_t     12
                self.dof_vel * self.obs_scales.dof_vel,        # dp_t    12
                self.actions,                                  # a_{t-1} 12
                # add privileged inputs, to foster learning under domain randomization
                self._rand_push_vel[:, :2],                    # 2
                self._base_mass,                               # 1
                self._env_frictions,                           # 1
                self._com_displacements,                       # 3
            ), dim=-1)
            self.critic_history.append(self.privileged_obs_buf)
            self.privileged_obs_buf = torch.cat(
            [self.critic_history[i] for i in range(self.critic_history.maxlen)], dim=-1
            )

        # add perceptive inputs if not blind
        # if self.cfg.terrain.measure_heights:
        #     heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
        #     self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        
        # add noise if needed
        if self.add_noise:
            obs_now = obs_buf.clone()
            obs_now += (2 * torch.rand_like(obs_now) - 1) * self.noise_scale_vec
        else:
            obs_now = obs_buf.clone()
        
        self.obs_history.append(obs_now)
        self.obs_buf = torch.cat(
            [self.obs_history[i] for i in range(self.obs_history.maxlen)], dim=-1
        )

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        # clear obs and critic history for the envs that are reset
        for i in range(self.obs_history.maxlen):
            self.obs_history[i][env_ids] *= 0
        for i in range(self.critic_history.maxlen):
            self.critic_history[i][env_ids] *= 0

    # ------------- Callbacks --------------

    def _get_noise_scale_vec(self):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros(self.cfg.env.num_single_obs, dtype=gs.tc_float, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = 0.  # commands
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = noise_scales.ang_vel * \
            noise_level * self.obs_scales.ang_vel
        noise_vec[9:9+1*self.num_actions] = noise_scales.dof_pos * \
            noise_level * self.obs_scales.dof_pos                    # p_t
        noise_vec[9+1*self.num_actions:9+2*self.num_actions] = noise_scales.dof_vel * \
            noise_level * self.obs_scales.dof_vel  # dp_t
        noise_vec[9+2*self.num_actions:9+3*self.num_actions] = 0.  # a_{t-dt}

        # if self.cfg.terrain.measure_heights:
        #     noise_vec[48:235] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements
        
        return noise_vec

    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        super()._init_buffers()

        # obs_history
        self.obs_history = deque(maxlen=self.cfg.env.frame_stack)
        self.critic_history = deque(maxlen=self.cfg.env.c_frame_stack)
        for _ in range(self.cfg.env.frame_stack):
            self.obs_history.append(
                torch.zeros(
                    self.num_envs,
                    self.cfg.env.num_single_obs,
                    dtype=gs.tc_float,
                    device=self.device,
                )
            )
        for _ in range(self.cfg.env.c_frame_stack):
            self.critic_history.append(
                torch.zeros(
                    self.num_envs,
                    self.cfg.env.single_num_privileged_obs,
                    dtype=gs.tc_float,
                    device=self.device,
                )
            )

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset, create entity
             2. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)
        
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=os.path.join(asset_root, asset_file),
                merge_fixed_links = True,  # if merge_fixed_links is True, then one link may have multiple geometries, which will cause error in set_friction_ratio
                links_to_keep = self.cfg.asset.links_to_keep,
                pos= np.array(self.cfg.init_state.pos),
                quat=np.array(self.cfg.init_state.rot),
                fixed = self.cfg.asset.fix_base_link,
            ),
            visualize_contact=self.debug,
        )
        
        # build
        self.scene.build(n_envs=self.num_envs)
        
        self._get_env_origins()
        
        # name to indices
        self.motor_dofs = [self.robot.get_joint(name).dof_idx_local for name in self.dof_names]
        
        # find link indices, termination links, penalized links, and feet
        def find_link_indices(names):
            link_indices = list()
            for link in self.robot.links:
                flag = False
                for name in names:
                    if name in link.name:
                        flag = True
                if flag:
                    link_indices.append(link.idx - self.robot.link_start)
            return link_indices
        self.termination_indices = find_link_indices(self.cfg.asset.terminate_after_contacts_on)
        all_link_names = [link.name for link in self.robot.links]
        print(f"all link names: {all_link_names}")
        print("termination link indices:", self.termination_indices)
        self.penalized_indices = find_link_indices(self.cfg.asset.penalize_contacts_on)
        print(f"penalized link indices: {self.penalized_indices}")
        self.feet_indices = find_link_indices(self.cfg.asset.foot_name)
        print(f"feet link indices: {self.feet_indices}")
        assert len(self.termination_indices) > 0
        assert len(self.feet_indices) > 0
        self.feet_link_indices_world_frame = [i+1 for i in self.feet_indices]
        
        # dof position limits
        self.dof_pos_limits = torch.stack(self.robot.get_dofs_limit(self.motor_dofs), dim=1)
        self.torque_limits = self.robot.get_dofs_force_range(self.motor_dofs)[1]
        for i in range(self.dof_pos_limits.shape[0]):
            # soft limits
            m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
            r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
            self.dof_pos_limits[i, 0] = (
                m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
            )
            self.dof_pos_limits[i, 1] = (
                m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        )
            
        self._init_domain_rand_params()
            
        # randomize friction
        if self.cfg.domain_rand.randomize_friction:
            self._randomize_friction(np.arange(self.num_envs))
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            self._randomize_base_mass(np.arange(self.num_envs))
        # randomize COM displacement
        if self.cfg.domain_rand.randomize_com_displacement:
            self._randomize_com_displacement(np.arange(self.num_envs))

    def _init_domain_rand_params(self):
        """ Initialize domain randomization parameters
        """
        self._rand_push_vel = torch.zeros(
            self.num_envs, 3, dtype=gs.tc_float, device=self.device
        )
        self._base_mass = torch.zeros(
            self.num_envs, 1, dtype=gs.tc_float, device=self.device
        )
        self._env_frictions = torch.zeros(
            self.num_envs, 1, dtype=gs.tc_float, device=self.device
        )
        self._com_displacements = torch.zeros(
            self.num_envs, 3, dtype=gs.tc_float, device=self.device
        )
    
    def _randomize_friction(self, env_ids=None):
        ''' Randomize friction of all links'''
        min_friction, max_friction = self.cfg.domain_rand.friction_range

        solver = self.rigid_solver
        
        # different geoms of same env has the same friction ratio
        ratios = gs.rand((len(env_ids), 1), dtype=float).repeat(1, solver.n_geoms) \
                 * (max_friction - min_friction) + min_friction
        self._env_frictions[env_ids, :] = ratios[:, 0].unsqueeze(1)  # (N, 1)
        
        solver.set_geoms_friction_ratio(ratios, torch.arange(0, solver.n_geoms), env_ids)
    
    def _randomize_base_mass(self, env_ids=None):
        ''' Randomize base mass'''
        min_mass, max_mass = self.cfg.domain_rand.added_mass_range
        base_link_id = 1
        added_mass = gs.rand((len(env_ids), 1), dtype=float) * (max_mass - min_mass) + min_mass
        self._base_mass[env_ids, :] = added_mass[:]
        self.rigid_solver.set_links_mass_shift(added_mass, [base_link_id, ], env_ids)
    
    def _randomize_com_displacement(self, env_ids):

        min_displacement, max_displacement = self.cfg.domain_rand.com_displacement_range
        base_link_id = 1

        self._com_displacements[env_ids] = gs.rand((len(env_ids), 3), dtype=float) \
                            * (max_displacement - min_displacement) + min_displacement
        com_displacement = self._com_displacements[env_ids].unsqueeze(1)  # (N, 1, 3)
        
        self.rigid_solver.set_links_COM_shift(com_displacement, [base_link_id,], env_ids)
        
    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        if self.push_interval_s > 0 and not self.debug:
            max_push_vel_xy = self.cfg.domain_rand.max_push_vel_xy
            # in Genesis, base link also has DOF, it's 6DOF if not fixed.
            dofs_vel = self.robot.get_dofs_velocity() # (num_envs, num_dof) [0:3] ~ base_link_vel
            self._rand_push_vel[:, :2] = gs_rand_float(
                -max_push_vel_xy, max_push_vel_xy, (self.num_envs, 2), self.device)
            self._rand_push_vel[((self.common_step_counter + self.env_identities) % int(self.push_interval_s / self.dt) != 0)] = 0
            dofs_vel[:, :2] += self._rand_push_vel[:, :2]
            self.robot.set_dofs_velocity(dofs_vel)
    
    def _reward_action_smoothness(self):
        '''Penalize action smoothness'''
        action_smoothness_cost = torch.sum(torch.square(self.actions - 2*self.last_actions + self.llast_actions), dim=-1)
        return action_smoothness_cost