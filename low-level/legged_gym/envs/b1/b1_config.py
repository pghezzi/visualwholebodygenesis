from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from legged_gym.envs.go2.go2_config import GO2Cfg, GO2CfgPPO
import numpy as np

class B1Cfg( GO2Cfg ):
    
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.5] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.2,   # [rad]
            'FL_thigh_joint': 0.8,     # [rad]
            'FL_calf_joint': -1.5,   # [rad]

            'RL_hip_joint': 0.2,   # [rad]
            'RL_thigh_joint': 0.8,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]

            'FR_hip_joint': -0.2 ,  # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'FR_calf_joint': -1.5,  # [rad]

            'RR_hip_joint': -0.2,   # [rad]
            'RR_thigh_joint': 0.8,   # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }
        rand_yaw_range = np.pi/2
        origin_perturb_range = 0.5
        init_vel_perturb_range = 0.1

    class control ( LeggedRobotCfg.control ):
        stiffness = {'joint': 80}  # [N*m/rad] # Kp: 80, 150, 200
        damping = {'joint': 2.0}     # [N*m*s/rad]

        adaptive_arm_gains = False
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = [0.4, 0.45, 0.45] * 2 + [0.4, 0.45, 0.45] * 2
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        torque_supervision = False

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/b1/urdf/b1.urdf'
        dof_names = [
            'FL_hip_joint',
            'FL_thigh_joint',
            'FL_calf_joint',
            'RL_hip_joint',
            'RL_thigh_joint',
            'RL_calf_joint',
            'FR_hip_joint',
            'FR_thigh_joint',
            'FR_calf_joint',
            'RR_hip_joint',
            'RR_thigh_joint',
            'RR_calf_joint',
        ]
        foot_name = ["foot"]
        penalize_contacts_on = ["thigh", "base", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        collapse_fixed_joints = True # Specific fixed joints can be kept by adding " <... dont_collapse="true">
        links_to_keep = ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']
        fix_base_link = False

    class rewards (LeggedRobotCfg.rewards):
        feet_aritime_allfeet = False

    class commands ( LeggedRobotCfg.commands ) :
        lin_vel_x_schedule = [0, 0.5]
        ang_vel_yaw_schedule = [0, 1]
        tracking_ang_vel_yaw_schedule = [0, 1]

        ang_vel_yaw_clip = 0.5
        lin_vel_x_clip = 0.2

        class ranges ( LeggedRobotCfg.commands.ranges ):
            lin_vel_x = [-0.8, 0.8] # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0]    # min max [rad/s]

class B1CfgPPO( GO2CfgPPO ):
    experiment_name = 'b1'
    class runner( GO2CfgPPO.runner ):
        run_name = ''
        experiment_name = 'b1'
        save_interval = 100
        load_run = "Jul23_17-09-36_"
        checkpoint = -1
        max_iterations = 6400