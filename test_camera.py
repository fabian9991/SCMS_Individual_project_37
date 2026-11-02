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
parameters.cornerRefinementWinSize = 2
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

# Show the image until a key is pressed.
cv2.imshow("Robot camera - ArUco detection", display)
print("Click the image window and press any key to close.")
cv2.waitKey(0)
cv2.destroyAllWindows()