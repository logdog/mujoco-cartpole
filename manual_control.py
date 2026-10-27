import platform
import distutils.util
import os
import subprocess
import mujoco
import time
import itertools
import numpy as np

# Graphics and plotting.
# import mediapy as media
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from collections import deque

# Use GPU rendering
import os
os.environ["MUJOCO_GL"] = "egl"

# More legible printing from numpy.
np.set_printoptions(precision=3, suppress=True, linewidth=100)

model = mujoco.MjModel.from_xml_path('cart-pole.xml')
data = mujoco.MjData(model)

cam = mujoco.MjvCamera()
mujoco.mjv_defaultCamera(cam)
cam.distance = 0.5

scene_option = mujoco.MjvOption()
# scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True


# initial conditions
print(data.qpos, data.qvel)
print(f'mass = {sum(model.body_mass)}')

data.qpos[:] = [0.0, np.pi]
data.qvel[:] = [0, 0.0]

# # store the sensor data
times = []
sensordata = []
statedata = []
trajectory = []
inputdata = []

# controller = BicopterController(model, data)
    

import mujoco.viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.opt.flags[:] = scene_option.flags
    idx = 0
    while viewer.is_running():
        step_start = time.time()
        
        # Compute forward kinematics to update visual geometries (body positions, geoms)
        mujoco.mj_step(model, data)
        statedata.append([*data.qpos.copy(), *data.qvel.copy()])
        sensordata.append(data.sensordata.copy())
        times.append(data.time)
        inputdata.append(data.ctrl.copy())
        
        # Push the updated data to the GUI
        mujoco.mj_forward(model, data)
        
        # Maintain real-time playback pacing
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
        viewer.sync()
    
times = np.asarray(times)
sensordata = np.asarray(sensordata)
statedata = np.asarray(statedata)
# inputdata = np.asarray(inputdata)

fig,axs = plt.subplots(2,3,figsize=(12,8),sharex=True)
axs = axs.flatten()

axs[0].plot(times, statedata[:,0], label='state')
axs[0].plot(times, sensordata[:,0], label='sensor')
axs[0].set_ylabel('m')
axs[0].set_title('position')

axs[1].plot(times, np.rad2deg(statedata[:,1]), label='state')
axs[1].plot(times, np.rad2deg(sensordata[:,6]), label='sensor')
axs[1].set_title('theta (deg)')

axs[2].plot(times, statedata[:,2], label='state')
axs[2].plot(times, sensordata[:,3], label='sensor')
axs[2].set_ylabel('m/s')
axs[2].set_title('lin velocity')

axs[3].plot(times, np.rad2deg(statedata[:,3]), label='state')
axs[3].set_ylabel('deg/s')
axs[3].set_title('ang velocity')

axs[4].plot(times, inputdata, label="u")
axs[4].set_ylabel('Force (N)')
axs[4].set_title('Input Force')


for ax in axs:
    ax.legend()
    ax.set_xlabel('Time (s)')
    
plt.tight_layout()
plt.show()