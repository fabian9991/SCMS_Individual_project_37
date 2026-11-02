from pathlib import Path
from itertools import permutations

import cv2
import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parent

# Test placement only: never passed to the camera pose estimator.
TEST_X = -6.0
TEST_Y = 0.0
TEST_HEADING_DEG = 180.0

# Higher camera rendering resolution.
WIDTH = 2560
HEIGHT = 1920

# Physical black marker width, excluding its white margin.
MARKER_SIZE = 0.36


def required_id(model, object_type, name):
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise ValueError(f"Could not find '{name}' in the model.")
    return object_id


# ------------------------------------------------------------
# Load the scene and place the robot.
# ------------------------------------------------------------
scene = ROOT / "models/robotis_tb3/scene_assignment.xml"
model = mujoco.MjModel.from_xml_path(str(scene))
data = mujoco.MjData(model)

joint_id = required_id(
    model, mujoco.mjtObj.mjOBJ_JOINT, "base_joint"
)
camera_id = required_id(
    model, mujoco.mjtObj.mjOBJ_CAMERA, "front_camera"
)
marker_geom_id = required_id(
    model, mujoco.mjtObj.mjOBJ_GEOM, "charger_marker"
)

address = model.jnt_qposadr[joint_id]

# Preserve the robot's initial height.
data.qpos[address] = TEST_X
data.qpos[address + 1] = TEST_Y

heading = np.deg2rad(TEST_HEADING_DEG)
data.qpos[address + 3:address + 7] = [
    np.cos(heading / 2),
    0,
    0,
    np.sin(heading / 2),
]

mujoco.mj_forward(model, data)


# ------------------------------------------------------------
# Load calibration and scale it once to the new resolution.
# ------------------------------------------------------------
calibration_path = ROOT / "calibration/camera_calibration.npz"

with np.load(calibration_path) as calibration:
    camera_matrix = calibration["camera_matrix"].astype(float).copy()
    distortion = calibration["distortion"].copy()
    calibrated_size = tuple(calibration["image_size"])

calibration_width, calibration_height = map(int, calibrated_size)

scale_x = WIDTH / calibration_width
scale_y = HEIGHT / calibration_height

if not np.isclose(scale_x, scale_y):
    raise ValueError(
        "This test requires the same aspect ratio as the calibration."
    )

camera_matrix[0, 0] *= scale_x
camera_matrix[1, 1] *= scale_y
camera_matrix[0, 2] = (
    camera_matrix[0, 2] + 0.5
) * scale_x - 0.5
camera_matrix[1, 2] = (
    camera_matrix[1, 2] + 0.5
) * scale_y - 0.5


# ------------------------------------------------------------
# Render a fresh high-resolution image.
# ------------------------------------------------------------
model.vis.global_.offwidth = WIDTH
model.vis.global_.offheight = HEIGHT

with mujoco.Renderer(
    model, height=HEIGHT, width=WIDTH
) as renderer:
    renderer.update_scene(data, camera="front_camera")
    rgb = renderer.render().copy()

gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
display = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

if gray.shape != (HEIGHT, WIDTH):
    raise ValueError(f"Unexpected image shape: {gray.shape}")


# ------------------------------------------------------------
# Detect ArUco markers using the selected settings.
# ------------------------------------------------------------
dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)

parameters = cv2.aruco.DetectorParameters()
parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
parameters.cornerRefinementWinSize = 8
parameters.relativeCornerRefinmentWinSize = 0.5
parameters.cornerRefinementMaxIterations = 30
parameters.cornerRefinementMinAccuracy = 0.001

detector = cv2.aruco.ArucoDetector(dictionary, parameters)
corners, ids, rejected = detector.detectMarkers(gray)

half = MARKER_SIZE / 2

object_points = np.array([
    [-half,  half, 0],
    [ half,  half, 0],
    [ half, -half, 0],
    [-half, -half, 0],
], dtype=np.float64)

detected = None
estimated_range = None

print(f"Rendering resolution: {WIDTH} × {HEIGHT}")

if ids is None:
    print("No ArUco marker detected.")
else:
    print("Detected marker IDs:", ids.flatten().tolist())
    cv2.aruco.drawDetectedMarkers(display, corners, ids)

    matches = np.flatnonzero(ids.flatten() == 0)

    if len(matches) == 0:
        print("Charging-station marker not detected.")
    elif len(matches) > 1:
        print("Multiple detections of ID 0; pose estimation skipped.")
    else:
        print("Charging-station marker detected!")

        detected = corners[matches[0]].reshape(4, 2).astype(np.float64)

        success, rvec, tvec = cv2.solvePnP(
            object_points,
            detected,
            camera_matrix,
            distortion,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        print("Pose solver success:", success)
        print("Detected corners:", detected)
        print("Rotation:", rvec)
        print("Translation:", tvec)
        print(
            "Marker width in pixels:",
            np.linalg.norm(detected[1] - detected[0]),
        )

        valid_pose = (
            success
            and rvec is not None
            and tvec is not None
            and np.all(np.isfinite(rvec))
            and np.all(np.isfinite(tvec))
        )

        if not valid_pose:
            print("Could not estimate marker pose.")
        elif tvec[2, 0] <= 0:
            print("Rejected a pose behind the camera.")
        else:
            # OpenCV axes: right, down, forward.
            x_right, y_down, z_forward = tvec.flatten()

            distance = np.linalg.norm(tvec)
            estimated_range = np.hypot(x_right, z_forward)
            bearing = np.degrees(
                np.arctan2(-x_right, z_forward)
            )

            rotation_matrix, _ = cv2.Rodrigues(rvec)
            normal = rotation_matrix[:, 2]
            toward_camera = -tvec.flatten() / distance

            facing_angle = np.degrees(
                np.arccos(
                    np.clip(np.dot(normal, toward_camera), -1, 1)
                )
            )

            print(f"Camera-to-marker distance: {distance:.5f} m")
            print(f"Horizontal range: {estimated_range:.5f} m")
            print(f"Bearing (+left, -right): {bearing:.3f} degrees")
            print(f"Marker facing angle: {facing_angle:.3f} degrees")

            cv2.drawFrameAxes(
                display,
                camera_matrix,
                distortion,
                rvec,
                tvec,
                MARKER_SIZE / 2,
            )


# ------------------------------------------------------------
# DIAGNOSTIC ONLY: ground-truth geometry for evaluation.
# None of this is passed to solvePnP above.
# ------------------------------------------------------------
if detected is not None:
    panel_rotation = data.geom_xmat[marker_geom_id].reshape(3, 3)
    panel_position = data.geom_xpos[marker_geom_id]

    camera_rotation = data.cam_xmat[camera_id].reshape(3, 3)
    camera_position = data.cam_xpos[camera_id]

    # Black marker occupies 600/800 of the complete panel image.
    half_width = model.geom_size[marker_geom_id, 0] * (600 / 800)
    half_height = model.geom_size[marker_geom_id, 1] * (600 / 800)
    surface_z = model.geom_size[marker_geom_id, 2]

    panel_corners = np.array([
        [-half_width,  half_height, surface_z],
        [ half_width,  half_height, surface_z],
        [ half_width, -half_height, surface_z],
        [-half_width, -half_height, surface_z],
    ])

    world_corners = panel_corners @ panel_rotation.T + panel_position

    camera_corners = (
        world_corners - camera_position
    ) @ camera_rotation

    # MuJoCo right/up/backward -> OpenCV right/down/forward.
    camera_corners *= np.array([1, -1, -1])

    # Reference distance to the visible panel surface.
    surface_position = (
        panel_position + panel_rotation[:, 2] * surface_z
    )
    camera_delta = (
        camera_rotation.T @ (surface_position - camera_position)
    ) * np.array([1, -1, -1])

    true_range = np.hypot(camera_delta[0], camera_delta[2])

    print("\nRANGE DIAGNOSTIC")
    print(f"True horizontal range: {true_range:.5f} m")

    if estimated_range is not None:
        print(
            "Range error: "
            f"{100 * (estimated_range - true_range):+.3f} cm"
        )

    if np.all(camera_corners[:, 2] > 0):
        # Ideal projection from simulator FOV, for scoring only.
        focal = HEIGHT / (
            2 * np.tan(np.deg2rad(model.cam_fovy[camera_id]) / 2)
        )

        expected = (
            focal
            * camera_corners[:, :2]
            / camera_corners[:, 2:3]
            + np.array([(WIDTH - 1) / 2, (HEIGHT - 1) / 2])
        )

        # Match geometric corners to detected corner order.
        order = min(
            permutations(range(4)),
            key=lambda p: np.sum(
                (expected[list(p)] - detected) ** 2
            ),
        )
        expected = expected[list(order)]

        print("\nCORNER DIAGNOSTIC")
        print("Corner offsets: detected minus expected, in pixels")
        print(np.round(detected - expected, 3))

        for label, points in [
            ("Expected", expected),
            ("Detected", detected),
        ]:
            top_width = np.linalg.norm(points[1] - points[0])
            bottom_width = np.linalg.norm(points[2] - points[3])
            mean_width = (top_width + bottom_width) / 2
            print(
                f"{label} mean marker width: "
                f"{mean_width:.3f} pixels"
            )

        # Crop and enlarge only for viewing.
        all_points = np.vstack([expected, detected])
        x0, y0 = np.floor(all_points.min(axis=0) - 12).astype(int)
        x1, y1 = np.ceil(all_points.max(axis=0) + 12).astype(int)

        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(WIDTH, x1), min(HEIGHT, y1)

        zoom = 4
        crop = cv2.cvtColor(
            rgb[y0:y1, x0:x1], cv2.COLOR_RGB2BGR
        )
        crop = cv2.resize(
            crop, None, fx=zoom, fy=zoom,
            interpolation=cv2.INTER_NEAREST,
        )

        for points, colour in [
            (expected, (0, 255, 0)),
            (detected, (0, 0, 255)),
        ]:
            for point in points:
                pixel = np.rint(
                    (point - [x0, y0] + 0.5) * zoom - 0.5
                ).astype(int)

                cv2.drawMarker(
                    crop,
                    tuple(pixel),
                    colour,
                    cv2.MARKER_CROSS,
                    11,
                    1,
                )

        cv2.namedWindow(
            "Corners: green expected, red detected", cv2.WINDOW_NORMAL
        )
        cv2.imshow("Corners: green expected, red detected", crop)


# Resizable display; measurements used the full-resolution image.
cv2.namedWindow("Robot camera - ArUco detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Robot camera - ArUco detection", 1280, 960)
cv2.imshow("Robot camera - ArUco detection", display)

print("\nClick an image window and press any key to close.")
cv2.waitKey(0)
cv2.destroyAllWindows()