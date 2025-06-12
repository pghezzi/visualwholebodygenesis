import genesis as gs
import numpy as np

########################## init ##########################
gs.init()

########################## create a scene ##########################

scene = gs.Scene(
    viewer_options = gs.options.ViewerOptions(
        camera_pos    = (0, -3.5, 2.5),
        camera_lookat = (0.0, 0.0, 0.5),
        camera_fov    = 30,
        max_FPS       = 60,
    ),
    sim_options = gs.options.SimOptions(
        dt = 0.01,
    ),
    show_viewer = True,
)

########################## entities ##########################
plane = scene.add_entity(gs.morphs.Plane())

b1z1_joints = [
        'joint_base', 
        'FR_hip_joint', 
        'FL_hip_joint', 
        'RR_hip_joint', 
        'RL_hip_joint', 
        'z1_waist', 
        'FR_thigh_joint', 
        'FL_thigh_joint', 
        'RR_thigh_joint', 
        'RL_thigh_joint', 
        'z1_shoulder', 
        'FR_calf_joint', 
        'FL_calf_joint', 
        'RR_calf_joint', 
        'RL_calf_joint', 
        'z1_elbow', 
        'z1_wrist_angle', 
        'z1_forearm_roll', 
        'z1_wrist_rotate', 
        'z1_jointGripper'
]

b1z1 = scene.add_entity(gs.morphs.URDF(file="/home/lily-hcrlab/genesis_b1z1/robots/b1z1/urdf/b1z1.urdf"))

dofs_idx = [b1z1.get_joint(name).dof_idx_local for name in b1z1_joints]


print(dofs_idx)

########################## build ##########################

# create 20 parallel environments
B = 20
scene.build(n_envs=20, env_spacing=(10.0, 10.0))

for i in range(1000):
    scene.step()


########################## run   ##########################
for i in range(1000):
   scene.step()
