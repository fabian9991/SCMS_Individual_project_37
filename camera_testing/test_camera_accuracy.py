from pathlib import Path
from itertools import product
from datetime import datetime
import csv

import cv2
import mujoco
import numpy as np
import sys

ROOT = Path(__file__).resolve().parent
if not (ROOT / "models").is_dir():
    ROOT = ROOT.parent

sys.path.insert(0, str(ROOT))
from test_ekf import heading_from_normal, wrap_angle
scene = ROOT / "models/robotis_tb3/scene_assignment.xml"

model = mujoco.MjModel.from_xml_path(str(scene))
data = mujoco.MjData(model)


def required_id(kind, name):
    result = mujoco.mj_name2id(model, kind, name)
    if result == -1:
        raise RuntimeError(f"Could not find: {name}")
    return result


joint_id = required_id(mujoco.mjtObj.mjOBJ_JOINT, "base_joint")
camera_id = required_id(mujoco.mjtObj.mjOBJ_CAMERA, "front_camera")
marker_geom_id = required_id(mujoco.mjtObj.mjOBJ_GEOM, "charger_marker")
base_id = required_id(mujoco.mjtObj.mjOBJ_BODY, "base")
address = model.jnt_qposadr[joint_id]

# Load the existing calibration.
with np.load(ROOT / "calibration/camera_calibration.npz") as calibration:
    camera_matrix = calibration["camera_matrix"].astype(float).copy()
    distortion = calibration["distortion"].copy()
    calibration_width, calibration_height = map(
        int, calibration["image_size"]
    )

# New rendering resolution.
width = 2560
height = 1920

# Adjust the calibration to the new resolution.
scale_x = width / calibration_width
scale_y = height / calibration_height

camera_matrix[0, 0] *= scale_x
camera_matrix[1, 1] *= scale_y
camera_matrix[0, 2] = (
    camera_matrix[0, 2] + 0.5
) * scale_x - 0.5
camera_matrix[1, 2] = (
    camera_matrix[1, 2] + 0.5
) * scale_y - 0.5

model.vis.global_.offwidth = width
model.vis.global_.offheight = height

# Keep the selected detector settings unchanged.
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
parameters.cornerRefinementWinSize = 8
parameters.relativeCornerRefinmentWinSize = 0.5
parameters.cornerRefinementMaxIterations = 30
parameters.cornerRefinementMinAccuracy = 0.001
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

marker_size = 0.36
half = marker_size / 2
object_points = np.array([
    [-half,  half, 0],
    [ half,  half, 0],
    [ half, -half, 0],
    [-half, -half, 0],
], dtype=np.float64)


def estimate_from_image(rgb):
    """Estimator receives only image pixels, not simulator pose."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None:
        return {"status": "not_detected"}

    matches = np.flatnonzero(ids.flatten() == 0)
    if len(matches) == 0:
        return {"status": "not_detected"}
    if len(matches) > 1:
        return {"status": "multiple_id_0"}

    image_points = corners[matches[0]].reshape(4, 2).astype(np.float64)
    marker_width = np.linalg.norm(image_points[1] - image_points[0])

    try:
        success, rvec, tvec = cv2.solvePnP(
            object_points, image_points, camera_matrix, distortion,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    except cv2.error:
        return {"status": "solver_error"}

    if (
        not success
        or not np.all(np.isfinite(rvec))
        or not np.all(np.isfinite(tvec))
    ):
        return {"status": "invalid_pose"}

    x_right, y_down, z_forward = tvec.flatten()
    if z_forward <= 0:
        return {"status": "behind_camera"}

    horizontal_range = np.hypot(x_right, z_forward)
    bearing = np.degrees(np.arctan2(-x_right, z_forward))

    rotation_matrix, _ = cv2.Rodrigues(rvec)
    toward_camera = -tvec.flatten() / np.linalg.norm(tvec)
    facing = np.degrees(np.arccos(np.clip(
        np.dot(rotation_matrix[:, 2], toward_camera), -1, 1
    )))

    projected, _ = cv2.projectPoints(
        object_points, rvec, tvec, camera_matrix, distortion
    )
    residual = image_points - projected.reshape(4, 2)
    reprojection_rms = np.sqrt(np.mean(np.sum(residual**2, axis=1)))

    normal = rotation_matrix[:, 2]

    if np.dot(normal, -tvec.flatten()) <= 0:
        return {"status": "back_facing"}

    if not np.isfinite(reprojection_rms) or reprojection_rms > 2.0:
        return {"status": "high_reprojection_error"}

    estimated_heading = heading_from_normal(normal)

    return {
        "status": "ok",
        "estimated_range_m": horizontal_range,
        "estimated_bearing_deg": bearing,
        "estimated_facing_deg": facing,
        "marker_width_px": marker_width,
        "reprojection_rms_px": reprojection_rms,
        "estimated_heading_rad": float(estimated_heading),
    }


def evaluation_truth():
    """Ground truth for scoring only; never passed to the estimator."""
    marker_rotation = data.geom_xmat[marker_geom_id].reshape(3, 3)
    marker_normal = marker_rotation[:, 2]

    # Centre of the visible panel surface, not the thin box's centre.
    marker_position = (
        data.geom_xpos[marker_geom_id]
        + marker_normal * model.geom_size[marker_geom_id, 2]
    )
    camera_position = data.cam_xpos[camera_id]
    camera_rotation = data.cam_xmat[camera_id].reshape(3, 3)
    delta = marker_position - camera_position

    # MuJoCo camera axes: right, up, backward.
    # OpenCV camera axes: right, down, forward.
    local = camera_rotation.T @ delta
    x_right, y_down, z_forward = local * np.array([1, -1, -1])

    toward_camera = -delta / np.linalg.norm(delta)
    facing = np.degrees(np.arccos(np.clip(
        np.dot(marker_normal, toward_camera), -1, 1
    )))

    base_rotation = data.xmat[base_id].reshape(3, 3)
    true_heading = np.arctan2(base_rotation[1, 0], base_rotation[0, 0])

    return {
        "true_range_m": np.hypot(x_right, z_forward),
        "true_bearing_deg": np.degrees(np.arctan2(-x_right, z_forward)),
        "true_facing_deg": facing,
        "true_heading_rad": float(true_heading),
    }


# Corridor positions. No physics stepping: each case is a static image.
tests = list(product(
    [0.0, -1.5, -3.0, -4.5, -6.0, -7.5, -8.0],
    [-0.15, 0.0, 0.25],
    [160.0, 170.0, 180.0, 190.0, 200.0],
))

# A separate results folder for each run.
output = ROOT / "camera_accuracy_results" / datetime.now().strftime(
    "%Y%m%d_%H%M%S_%f"
)
output.mkdir(parents=True)
rows = []

with mujoco.Renderer(model, height=height, width=width) as renderer:
    for number, (x, y, heading_deg) in enumerate(tests, start=1):
        mujoco.mj_resetData(model, data)
        data.qpos[address] = x
        data.qpos[address + 1] = y

        heading = np.deg2rad(heading_deg)
        data.qpos[address + 3:address + 7] = [
            np.cos(heading / 2), 0, 0, np.sin(heading / 2)
        ]
        mujoco.mj_forward(model, data)

        renderer.update_scene(data, camera="front_camera")
        rgb = renderer.render().copy()

        measurement = estimate_from_image(rgb)
        truth = evaluation_truth()

        row = {
            "test": number,
            "robot_x_m": x,
            "robot_y_m": y,
            "heading_deg": heading_deg,
            **truth,
            **measurement,
        }

        if measurement["status"] == "ok":
            row["range_error_m"] = (
                measurement["estimated_range_m"] - truth["true_range_m"]
            )
            difference = (
                measurement["estimated_bearing_deg"] - truth["true_bearing_deg"]
            )
            row["bearing_error_deg"] = (difference + 180) % 360 - 180
            row["facing_error_deg"] = (
                measurement["estimated_facing_deg"] - truth["true_facing_deg"]
            )
            row["heading_error_rad"] = float(wrap_angle(
                measurement["estimated_heading_rad"] - truth["true_heading_rad"]
            ))

            print(
                f"{number:03d}/{len(tests)} "
                f"range error={row['range_error_m']:+.3f} m, "
                f"bearing error={row['bearing_error_deg']:+.2f} deg"
            )
        else:
            print(f"{number:03d}/{len(tests)} {measurement['status']}")

        # Save failed cases and substantial range errors for inspection.
        if (
            measurement["status"] != "ok"
            or abs(row["range_error_m"]) > 0.10
        ):
            cv2.imwrite(
                str(output / f"inspect_{number:03d}.png"),
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            )

        rows.append(row)

fields = list(dict.fromkeys(key for row in rows for key in row))
csv_path = output / "results.csv"
with csv_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

print("\nSUMMARY — errors calculated only for valid pose estimates")
good = [row for row in rows if row["status"] == "ok"]
print(f"Valid estimates: {len(good)}/{len(rows)}")

for x in sorted(set(row["robot_x_m"] for row in rows), reverse=True):
    cases = [row for row in rows if row["robot_x_m"] == x]
    valid = [row for row in cases if row["status"] == "ok"]

    if not valid:
        print(f"X={x:5.1f}: 0/{len(cases)} valid")
        continue

    range_errors = np.array([row["range_error_m"] for row in valid])
    bearing_errors = np.array([row["bearing_error_deg"] for row in valid])

    print(
        f"X={x:5.1f}: {len(valid):2d}/{len(cases)} valid | "
        f"mean range error={100 * range_errors.mean():+.2f} cm | "
        f"range RMSE={100 * np.sqrt(np.mean(range_errors**2)):.2f} cm | "
        f"max range error={100 * np.max(np.abs(range_errors)):.2f} cm | "
        f"bearing RMSE={np.sqrt(np.mean(bearing_errors**2)):.3f} deg"
    )

print(f"\nResults saved to: {csv_path}")

# Use the same measured-range limit as the mission.
usable = [
    row for row in rows
    if row["status"] == "ok"
    and row["estimated_range_m"] <= 3.0
]

print(f"\nSamples within the mission range gate: {len(usable)}")

if len(usable) >= 4:
    errors = np.array([
        [
            row["range_error_m"],
            np.deg2rad(row["bearing_error_deg"]),
            row["heading_error_rad"],
        ]
        for row in usable
    ])

    bias = errors.mean(axis=0)
    covariance = np.cov(errors, rowvar=False, ddof=1)
    rmse = np.sqrt(np.mean(errors**2, axis=0))

    print("Mean errors [m, deg, deg]:")
    print([bias[0], *np.degrees(bias[1:])])

    standard_deviations = np.sqrt(np.diag(covariance))
    print("Error standard deviations [m, deg, deg]:")
    print([
        standard_deviations[0],
        *np.degrees(standard_deviations[1:]),
    ])

    print("RMSE [m, deg, deg]:")
    print([rmse[0], *np.degrees(rmse[1:])])

    print("Measured error covariance [metres/radians]:")
    print(covariance)

    np.savez(
        output / "camera_error_statistics.npz",
        bias=bias,
        covariance=covariance,
        rmse=rmse,
        sample_count=len(usable),
    )
else:
    print("Too few usable samples to estimate the covariance.")