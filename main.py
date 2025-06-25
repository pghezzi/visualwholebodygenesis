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

b1z1 = scene.add_entity(gs.morphs.URDF(file="/home/lily-hcrlab/visualwholebodygenesis/low-level/resources/robots/b1/urdf/b1.urdf"))

########################## build ##########################

# create 20 parallel environments
B = 20
scene.build(n_envs=4, env_spacing=(1.0, 1.0))

for i in range(1000):
    scene.step()


########################## run   ##########################
for i in range(1000):
   scene.step()


