from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class GO2DeployCfg( LeggedRobotCfg ):
    
    class env( LeggedRobotCfg.env ):
        num_envs = 4096
        env_spacing = 3.  # not used with heightfields/trimeshes
        num_actions = 12
        # observation history
        frame_stack = 5   # policy frame stack
        c_frame_stack = 3 # critic frame stack
        num_single_obs = 45
        num_observations = int( num_single_obs * frame_stack )
        single_num_privileged_obs = 55
        num_privileged_obs = int( c_frame_stack * single_num_privileged_obs )
    
    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'plane' # "heightfield" # none, plane, heightfield or trimesh
        friction = 1.0
        restitution = 0.
        
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }
        # initial state randomization
        yaw_angle_range = [0., 3.14] # min max [rad]

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        # control_type = 'P'
        stiffness = {'joint': 20.}   # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad]
        action_scale = 0.25 # action scale: target angle = actionScale * action + defaultAngle
        dt =  0.02  # control frequency 50Hz
        decimation = 4 # decimation: Number of control action updates @ sim DT per policy DT

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        dof_names = [        # specify yhe sequence of actions
            'FR_hip_joint',
            'FR_thigh_joint',
            'FR_calf_joint',
            'FL_hip_joint',
            'FL_thigh_joint',
            'FL_calf_joint',
            'RR_hip_joint',
            'RR_thigh_joint',
            'RR_calf_joint',
            'RL_hip_joint',
            'RL_thigh_joint',
            'RL_calf_joint',]
        foot_name = ["foot"]
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        links_to_keep = ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']
        self_collisions = True
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.36
        class scales( LeggedRobotCfg.rewards.scales ):
            # limitation
            dof_pos_limits = -2.0
            collision = -1.0
            # command tracking
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            # smooth
            lin_vel_z = -2.0
            base_height = -1.0
            ang_vel_xy = -0.05
            orientation = -1.0
            dof_vel = -5.e-4
            dof_acc = -2.e-7
            action_rate = -0.01
            torques = -2.e-4
            # gait
            feet_air_time = 1.0
            dof_close_to_default = -0.1
    
    class commands( LeggedRobotCfg.commands ):
        curriculum = True
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10.  # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges( LeggedRobotCfg.commands.ranges ):
            lin_vel_x = [-0.5, 0.5] # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]
    
    class domain_rand:
        randomize_friction = True
        friction_range = [0.2, 1.7]
        randomize_base_mass = True
        added_mass_range = [-1., 1.]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 0.5
        simulate_action_latency = False # 1 step delay
        randomize_com_displacement = True
        com_displacement_range = [-0.01, 0.01]
    
    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [10, 0, 6]       # [m]
        lookat = [11., 5, 3.]  # [m]
        num_rendered_envs = 10  # number of environments to be rendered
        add_camera = False

class GO2DeployCfgPPO( LeggedRobotCfgPPO ):
    seed = 0
    runner_class_name = "OnPolicyRunner"
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'go2_deploy'
        save_interval = 100
        load_run = "Dec22_21-05-25_"
        checkpoint = 1000
        max_iterations = 3000