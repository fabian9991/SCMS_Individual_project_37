from pathlib import Path

import cv2
import mujoco
import numpy as np

# Load the scene at the robot's starting position.
scene = (
    Path(__file__).resolve().parent
    / "models/robotis_tb3/scene_assignment.xml"
)

model = mujoco.MjModel.from_xml_path(str(scene))

data = mujoco.MjData(model)

# Test placement only—not an input to the camera pose estimator.
TEST_X = -6.0
TEST_Y = 0.0
TEST_HEADING_DEG = 180.0

joint_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_JOINT, "base_joint"
)
address = model.jnt_qposadr[joint_id]

# Place the robot while preserving its initial height.
data.qpos[address] = TEST_X
data.qpos[address + 1] = TEST_Y

# Set its heading around the vertical axis.
heading = np.deg2rad(TEST_HEADING_DEG)
data.qpos[address + 3:address + 7] = [
    np.cos(heading / 2), 0, 0, np.sin(heading / 2)
]

mujoco.mj_forward(model, data)

# Capture one RGB image from the robot's camera.
model.vis.global_.offwidth = 1280
model.vis.global_.offheight = 960

with mujoco.Renderer(model, height=960, width=1280) as renderer:

    renderer.update_scene(data, camera="front_camera")
    rgb = renderer.render().copy()

# Convert the image for OpenCV.
gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
display = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

# Search for markers from the same dictionary used to generate ours.
dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)
parameters = cv2.aruco.DetectorParameters()

# Refine corner positions to fractions of a pixel.
parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
parameters.cornerRefinementWinSize = 8
parameters.relativeCornerRefinmentWinSize = 0.5
parameters.cornerRefinementMaxIterations = 30
parameters.cornerRefinementMinAccuracy = 0.001

detector = cv2.aruco.ArucoDetector(dictionary, parameters)

corners, ids, rejected = detector.detectMarkers(gray)

if ids is None:
    print("No ArUco marker detected.")
else:
    print("Detected marker IDs:", ids.flatten().tolist())
    cv2.aruco.drawDetectedMarkers(display, corners, ids)

    if 0 in ids.flatten():
        print("Charging-station marker detected!")



# Load the camera parameters estimated from checkerboard images.
calibration_path = (
    Path(__file__).resolve().parent
    / "calibration/camera_calibration.npz"
)

with np.load(calibration_path) as calibration:
    camera_matrix = calibration["camera_matrix"].copy()
    distortion = calibration["distortion"].copy()
    calibrated_size = tuple(calibration["image_size"])

height, width = gray.shape
if (width, height) != calibrated_size:
    raise ValueError(
        f"Calibration requires {calibrated_size}, "
        f"but the image is {(width, height)}."
    )

# Physical marker size, excluding the white margin.
marker_size = 0.36  # metres
half = marker_size / 2

# Marker corners in the order required by IPPE_SQUARE.
object_points = np.array([
    [-half,  half, 0],
    [ half,  half, 0],
    [ half, -half, 0],
    [-half, -half, 0],
], dtype=np.float64)

if ids is not None:
    for marker_corners, marker_id in zip(corners, ids.flatten()):
        if marker_id != 0:
            continue

        image_points = marker_corners.reshape(4, 2).astype(np.float64)

        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            distortion,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        print("Pose solver success:", success)
        print("Detected corners:", image_points)
        print("Rotation:", rvec)
        print("Translation:", tvec)
        print(
            "Marker width in pixels:",
            np.linalg.norm(image_points[1] - image_points[0])
        )

        if not success or not np.all(np.isfinite(tvec)):
            print("Could not estimate marker pose.")
            continue

        # OpenCV camera axes: X right, Y down, Z forward.
        x_right, y_down, z_forward = tvec.flatten()

        if z_forward <= 0:
            print("Rejected a pose behind the camera.")
            continue

        distance = np.linalg.norm(tvec)
        horizontal_range = np.hypot(x_right, z_forward)
        bearing = np.degrees(np.arctan2(-x_right, z_forward))

        # Marker orientation, represented as a rotation matrix.
        rotation_matrix, _ = cv2.Rodrigues(rvec)

        # Angle between the marker's outward normal and camera-facing direction.
        normal = rotation_matrix[:, 2]
        toward_camera = -tvec.flatten() / distance
        facing_angle = np.degrees(
            np.arccos(np.clip(np.dot(normal, toward_camera), -1, 1))
        )

        print(f"Camera-to-marker distance: {distance:.2f} m")
        print(f"Horizontal range: {horizontal_range:.2f} m")
        print(f"Bearing (+left, -right): {bearing:.2f} degrees")
        print(f"Marker facing angle: {facing_angle:.2f} degrees")

        # Show the estimated marker coordinate axes.
        cv2.drawFrameAxes(
            display, camera_matrix, distortion,
            rvec, tvec, marker_size / 2
        )

# DIAGNOSTIC ONLY: compare detected corners with simulated geometry.
from itertools import permutations

if ids is not None and 0 in ids.flatten():
    camera_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_CAMERA, "front_camera"
    )
    marker_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "charger_marker"
    )

    # Black marker occupies 600/800 of the panel width and height.
    half_width = model.geom_size[marker_id, 0] * (600 / 800)
    half_height = model.geom_size[marker_id, 1] * (600 / 800)
    surface_z = model.geom_size[marker_id, 2]

    panel_corners = np.array([
        [-half_width,  half_height, surface_z],
        [ half_width,  half_height, surface_z],
        [ half_width, -half_height, surface_z],
        [-half_width, -half_height, surface_z],
    ])

    # Panel coordinates -> world coordinates -> camera coordinates.
    panel_rotation = data.geom_xmat[marker_id].reshape(3, 3)
    world_corners = (
        panel_corners @ panel_rotation.T
        + data.geom_xpos[marker_id]
    )

    camera_rotation = data.cam_xmat[camera_id].reshape(3, 3)
    camera_corners = (
        world_corners - data.cam_xpos[camera_id]
    ) @ camera_rotation

    # Convert MuJoCo right/up/backward to OpenCV right/down/forward.
    camera_corners *= np.array([1, -1, -1])

    if np.any(camera_corners[:, 2] <= 0):
        raise ValueError("Diagnostic corners are behind the camera.")

    # Use the simulator's ideal camera settings for this reference.
    # Do not feed these values into the navigation estimator.
    height, width = gray.shape
    focal = height / (
        2 * np.tan(np.deg2rad(model.cam_fovy[camera_id]) / 2)
    )

    expected = (
        focal * camera_corners[:, :2] / camera_corners[:, 2:3]
        + np.array([(width - 1) / 2, (height - 1) / 2])
    )

    detection_index = np.flatnonzero(ids.flatten() == 0)[0]
    detected = corners[detection_index].reshape(4, 2).astype(float)

    # Match geometric corners to the detector's corner order.
    order = min(
        permutations(range(4)),
        key=lambda p: np.sum((expected[list(p)] - detected) ** 2),
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
        print(f"{label} mean marker width: {mean_width:.3f} pixels")

    # Enlarge a crop so subpixel differences are easier to inspect.
    all_points = np.vstack([expected, detected])
    x0, y0 = np.floor(all_points.min(axis=0) - 12).astype(int)
    x1, y1 = np.ceil(all_points.max(axis=0) + 12).astype(int)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(width, x1), min(height, y1)

    zoom = 10
    crop = cv2.cvtColor(rgb[y0:y1, x0:x1], cv2.COLOR_RGB2BGR)
    crop = cv2.resize(
        crop, None, fx=zoom, fy=zoom,
        interpolation=cv2.INTER_NEAREST,
    )

    for points, colour in [
        (expected, (0, 255, 0)),    # Green: expected
        (detected, (0, 0, 255)),    # Red: detected
    ]:
        for point in points:
            pixel = np.rint(
                (point - [x0, y0] + 0.5) * zoom - 0.5
            ).astype(int)
            cv2.drawMarker(
                crop, tuple(pixel), colour,
                cv2.MARKER_CROSS, 11, 1,
            )

    cv2.imshow("Corners: green expected, red detected", crop)
else:
    print("Corner diagnostic skipped: marker 0 not detected.")




# Show the image until a key is pressed.
cv2.imshow("Robot camera - ArUco detection", display)
print("Click the image window and press any key to close.")
cv2.waitKey(0)
cv2.destroyAllWindows()