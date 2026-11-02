from pathlib import Path

import mujoco
import numpy as np

# Load the assignment scene.
scene = (
    Path(__file__).resolve().parent
    / "models/robotis_tb3/scene_assignment.xml"
)
model = mujoco.MjModel.from_xml_path(str(scene))
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

site_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_SITE, "lidar_site"
)
if site_id == -1:
    raise RuntimeError("Add lidar_site to the robot XML and save it first.")

base_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_BODY, "base"
)

# Sensor position and orientation, used only to simulate measurements.
origin = data.site_xpos[site_id].copy()
rotation = data.site_xmat[site_id].reshape(3, 3)

MAX_RANGE = 12.0  # metres; chosen for initial test
ranges = np.full(360, np.inf)
hit_geom = np.empty(1, dtype=np.int32)

# One measurement per degree around the robot.
for angle_deg in range(360):
    angle = np.deg2rad(angle_deg)
    local_direction = np.array([np.cos(angle), np.sin(angle), 0.0])
    world_direction = rotation @ local_direction

    distance = mujoco.mj_ray(
        model,
        data,
        origin,
        world_direction,
        None,       # Include all geometry groups.
        True,       # Include static walls and shelves.
        base_id,    # Ignore geometry attached to the robot base.
        hit_geom,
    )

    if 0 <= distance <= MAX_RANGE:
        ranges[angle_deg] = distance

print("360-degree LiDAR scan complete.")
for label, index in [("Front", 0), ("Left", 90),
                     ("Back", 180), ("Right", 270)]:
    distance = ranges[index]
    reading = f"{distance:.2f} m" if np.isfinite(distance) else "No return"
    print(f"{label}: {reading}")