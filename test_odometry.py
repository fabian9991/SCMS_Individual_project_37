from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parent
scene = ROOT / "models/robotis_tb3/scene_assignment.xml"

model = mujoco.MjModel.from_xml_path(str(scene))
data = mujoco.MjData(model)

WHEEL_RADIUS = 0.033  # Provisional: must verify against tyre mesh.
WHEEL_SEPARATION = 0.288


def required_id(kind, name):
    result = mujoco.mj_name2id(model, kind, name)
    if result == -1:
        raise RuntimeError(f"Could not find: {name}")
    return result


left_joint = required_id(mujoco.mjtObj.mjOBJ_JOINT, "wheel_left")
right_joint = required_id(mujoco.mjtObj.mjOBJ_JOINT, "wheel_right")

left_address = model.jnt_qposadr[left_joint]
right_address = model.jnt_qposadr[right_joint]

left_motor = required_id(mujoco.mjtObj.mjOBJ_ACTUATOR, "wheel_left")
right_motor = required_id(mujoco.mjtObj.mjOBJ_ACTUATOR, "wheel_right")

base_id = required_id(mujoco.mjtObj.mjOBJ_BODY, "base")


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def read_encoders():
    """Ideal wheel measurements; no robot position is read here."""
    return np.array([
        data.qpos[left_address],
        data.qpos[right_address],
    ])


def update_odometry(pose, wheel_changes):
    """Update local x, y and heading using only wheel rotation."""
    left_distance, right_distance = WHEEL_RADIUS * wheel_changes

    distance = (left_distance + right_distance) / 2
    heading_change = (
        right_distance - left_distance
    ) / WHEEL_SEPARATION

    # Exact circular-arc integration for this wheel increment.
    half_turn = heading_change / 2
    arc_factor = np.sinc(half_turn / np.pi)
    direction = pose[2] + half_turn

    pose[0] += distance * arc_factor * np.cos(direction)
    pose[1] += distance * arc_factor * np.sin(direction)
    pose[2] = wrap_angle(pose[2] + heading_change)

def predict_ekf(pose, P, wheel_changes):
    """Predict pose and covariance using measured wheel increments."""
    left_distance, right_distance = WHEEL_RADIUS * wheel_changes

    distance = (left_distance + right_distance) / 2
    heading_change = (
        right_distance - left_distance
    ) / WHEEL_SEPARATION

    half_turn = heading_change / 2
    arc_factor = np.sinc(half_turn / np.pi)
    direction = pose[2] + half_turn

    dx = distance * arc_factor * np.cos(direction)
    dy = distance * arc_factor * np.sin(direction)

    # Motion Jacobian with respect to [x, y, heading].
    F = np.array([
        [1.0, 0.0, -dy],
        [0.0, 1.0,  dx],
        [0.0, 0.0, 1.0],
    ])

    predicted_pose = pose.copy()
    predicted_pose[0] += dx
    predicted_pose[1] += dy
    predicted_pose[2] = wrap_angle(pose[2] + heading_change)

    # For this prediction-equivalence check only.
    # Add a justified process covariance before camera fusion.
    Q = np.zeros((3, 3))

    predicted_P = F @ P @ F.T + Q
    predicted_P = 0.5 * (predicted_P + predicted_P.T)

    return predicted_pose, predicted_P

def evaluation_pose():
    """Simulator truth for scoring only."""
    rotation = data.xmat[base_id].reshape(3, 3)
    heading = np.arctan2(rotation[1, 0], rotation[0, 0])

    return np.array([
        data.xpos[base_id, 0],
        data.xpos[base_id, 1],
        heading,
    ])


def relative_truth(start, end):
    """Express actual movement in the robot's starting frame."""
    dx, dy = end[:2] - start[:2]
    cosine = np.cos(start[2])
    sine = np.sin(start[2])

    return np.array([
        cosine * dx + sine * dy,
        -sine * dx + cosine * dy,
        wrap_angle(end[2] - start[2]),
    ])


# Each test resets the robot to its original starting pose.
# Format: name, left target (rad/s), right target (rad/s), duration (s).
tests = [
    ("Straight drive", 4.0, 4.0, 5.0),
    ("Turn left", -4.0, 4.0, 3.0),
    ("Reverse drive", -4.0, -4.0, 1.5),
    ("Turn right", 4.0, -4.0, 3.0),
    ("Curve left", 3.0, 5.0, 2.0),
    ("Curve right", 5.0, 3.0, 2.0),
]

print("Local coordinates: x forward, y left, heading positive left.")
print("Wheel radius is provisionally 0.033 m.")

with mujoco.viewer.launch_passive(model, data) as viewer:
    for name, left_target, right_target, drive_duration in tests:
        if not viewer.is_running():
            break

        mujoco.mj_resetData(model, data)

        # Settle the robot before starting measurements.
        for _ in range(round(1.0 / model.opt.timestep)):
            mujoco.mj_step(model, data)

        mujoco.mj_forward(model, data)

        # Odometry starts at its own local origin.
        # Ground truth is stored separately for evaluation.
        odometry = np.zeros(3)

        ekf_pose = np.zeros(3)
        # Initial uncertainty assumptions for this local-frame test:
        # 1 cm in x/y and 1 degree in heading.
        P = np.diag([
            0.01**2,
            0.01**2,
            np.deg2rad(1.0)**2,
        ])
        truth_start = evaluation_pose()
        print("\nCONTACTS AFTER SETTLING")

        for i in range(data.ncon):
            contact = data.contact[i]
            labels = []

            for geom_id in (contact.geom1, contact.geom2):
                body_name = mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_BODY,
                    int(model.geom_bodyid[geom_id])
                )
                mesh_id = int(model.geom_dataid[geom_id])

                if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH:
                    shape = mujoco.mj_id2name(
                        model, mujoco.mjtObj.mjOBJ_MESH, mesh_id
                    )
                else:
                    shape = mujoco.mj_id2name(
                        model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
                    ) or f"geom_{geom_id}"

                labels.append(f"{body_name}/{shape}")

            print(
                f"{labels[0]} <-> {labels[1]} | "
                f"distance={contact.dist * 1000:.3f} mm | "
                f"contact dimension={contact.dim} | "
                f"sliding friction={contact.friction[0]:.4f}"
            )
        previous_encoders = read_encoders()

        start_time = data.time
        next_print = 1.0

        print(f"\n{name}")
        print("Time(s)   Odom x(m)   Odom y(m)   Odom heading(deg)")

        # Include two seconds of stopping motion.
        while (
            viewer.is_running()
            and data.time - start_time < drive_duration + 2.0
        ):
            wall_start = time.perf_counter()
            elapsed = data.time - start_time

            driving = elapsed < drive_duration
            data.ctrl[left_motor] = left_target if driving else 0.0
            data.ctrl[right_motor] = right_target if driving else 0.0

            mujoco.mj_step(model, data)

            encoders = read_encoders()
            wheel_changes = encoders - previous_encoders
            previous_encoders = encoders

            update_odometry(odometry, wheel_changes)
            ekf_pose, P = predict_ekf(
                ekf_pose, P, wheel_changes
            )

            elapsed = data.time - start_time

            if elapsed >= next_print:
                print(
                    f"{elapsed:7.2f}   "
                    f"{odometry[0]:9.4f}   "
                    f"{odometry[1]:9.4f}   "
                    f"{np.degrees(odometry[2]):17.3f}"
                )
                next_print += 1.0

            viewer.sync()

            remaining = model.opt.timestep - (
                time.perf_counter() - wall_start
            )
            if remaining > 0:
                time.sleep(remaining)

        # Refresh body transforms for the final simulation state.
        mujoco.mj_forward(model, data)

        actual = relative_truth(truth_start, evaluation_pose())
        position_error = np.linalg.norm(odometry[:2] - actual[:2])
        heading_error = wrap_angle(odometry[2] - actual[2])

        print("\nFINAL COMPARISON")
        print("                x(m)       y(m)    Heading(deg)")
        print(
            f"Odometry:   {odometry[0]:9.4f}  {odometry[1]:9.4f}"
            f"  {np.degrees(odometry[2]):12.3f}"
        )
        print(
            f"Actual:     {actual[0]:9.4f}  {actual[1]:9.4f}"
            f"  {np.degrees(actual[2]):12.3f}"
        )
        print(f"Position error: {100 * position_error:.2f} cm")
        print(f"Heading error: {np.degrees(heading_error):+.3f} degrees")

        prediction_difference = ekf_pose - odometry
        prediction_difference[2] = wrap_angle(
            prediction_difference[2]
        )

        print(
            "EKF prediction vs odometry: "
            f"{np.linalg.norm(prediction_difference[:2]):.3e} m, "
            f"{np.degrees(prediction_difference[2]):.3e} degrees"
        )
        data.ctrl[left_motor] = 0.0
        data.ctrl[right_motor] = 0.0

print("\nTests finished.")