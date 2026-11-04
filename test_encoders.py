from pathlib import Path
import time

import mujoco
import mujoco.viewer

ROOT = Path(__file__).resolve().parent
scene = ROOT / "models/robotis_tb3/scene_assignment.xml"

model = mujoco.MjModel.from_xml_path(str(scene))
data = mujoco.MjData(model)


def required_id(kind, name):
    result = mujoco.mj_name2id(model, kind, name)
    if result == -1:
        raise RuntimeError(f"Could not find: {name}")
    return result


# Find the wheel joints and their angle locations.
left_joint = required_id(mujoco.mjtObj.mjOBJ_JOINT, "wheel_left")
right_joint = required_id(mujoco.mjtObj.mjOBJ_JOINT, "wheel_right")

left_address = model.jnt_qposadr[left_joint]
right_address = model.jnt_qposadr[right_joint]

# Find the motors independently of the joint indices.
left_motor = required_id(mujoco.mjtObj.mjOBJ_ACTUATOR, "wheel_left")
right_motor = required_id(mujoco.mjtObj.mjOBJ_ACTUATOR, "wheel_right")


def read_encoders():
    """Ideal wheel-angle measurements in radians."""
    return (
        float(data.qpos[left_address]),
        float(data.qpos[right_address]),
    )


# Allow the robot to settle with zero motor targets.
for _ in range(round(1.0 / model.opt.timestep)):
    mujoco.mj_step(model, data)

initial_left, initial_right = read_encoders()
start_time = data.time
next_print = 0.0

print("Angles are measured relative to the start of this test.")
print("Time(s)   Target(rad/s)   Left(rad)   Right(rad)")

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        wall_start = time.perf_counter()
        elapsed = data.time - start_time

        if elapsed >= 7.0:
            break

        # Drive for five seconds, then command a stop for two seconds.
        target = 4.0 if elapsed < 5.0 else 0.0
        data.ctrl[left_motor] = target
        data.ctrl[right_motor] = target

        mujoco.mj_step(model, data)

        elapsed = data.time - start_time
        left, right = read_encoders()

        if elapsed >= next_print:
            print(
                f"{elapsed:7.2f}   {target:13.2f}   "
                f"{left - initial_left:9.4f}   "
                f"{right - initial_right:10.4f}"
            )
            next_print += 0.5

        viewer.sync()

        remaining = model.opt.timestep - (
            time.perf_counter() - wall_start
        )
        if remaining > 0:
            time.sleep(remaining)

data.ctrl[left_motor] = 0.0
data.ctrl[right_motor] = 0.0

print("Encoder test finished.")