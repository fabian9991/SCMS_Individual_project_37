from pathlib import Path

import cv2
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "calibration"
OUTPUT.mkdir(exist_ok=True)

WIDTH, HEIGHT = 1280, 960
PATTERN = (9, 6)       # Internal corners: columns, rows
SQUARE_SIZE = 0.10     # Each square is 10 cm

# Read the existing camera's settings.
robot_model = mujoco.MjModel.from_xml_path(
    str(ROOT / "models/robotis_tb3/scene_assignment.xml")
)
camera_id = mujoco.mj_name2id(
    robot_model, mujoco.mjtObj.mjOBJ_CAMERA, "front_camera"
)
if camera_id == -1:
    raise RuntimeError("front_camera was not found.")

fovy = float(robot_model.cam_fovy[camera_id])

# Build a 10-by-7 square checkerboard with a white margin.
squares = []
for row in range(7):
    for col in range(10):
        colour = 0 if (row + col) % 2 == 0 else 1
        x = (col - 4.5) * SQUARE_SIZE
        y = (row - 3.0) * SQUARE_SIZE
        squares.append(
            f'<geom type="box" pos="{x} {y} 0" '
            f'size="0.05 0.05 0.001" '
            f'rgba="{colour} {colour} {colour} 1" '
            f'material="board_material"/>'
        )

xml = f"""
<mujoco>
  <visual>
    <global offwidth="{WIDTH}" offheight="{HEIGHT}"/>
  </visual>
  <asset>
    <material name="board_material" emission="1" specular="0"/>
  </asset>
  <worldbody>
    <camera name="calibration_camera" pos="0 0 0" fovy="{fovy}"/>
    <body name="board" mocap="true" pos="0 0 -1.5">
      <geom type="box" pos="0 0 -0.003"
            size="0.55 0.40 0.001"
            rgba="1 1 1 1" material="board_material"/>
      {''.join(squares)}
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

# Known corner coordinates on the board, not camera poses.
object_points = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
object_points[:, :2] = (
    np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2)
    * SQUARE_SIZE
)

observations = []
rng = np.random.default_rng(37)

with mujoco.Renderer(model, height=HEIGHT, width=WIDTH) as renderer:
    for i in range(24):
        # Move and tilt the board to create different calibration views.
        data.mocap_pos[0] = [
            rng.uniform(-0.25, 0.25),
            rng.uniform(-0.15, 0.15),
            -rng.uniform(1.1, 1.8),
        ]

        rotation_vector = np.array([
            rng.uniform(-0.5, 0.5),
            rng.uniform(-0.5, 0.5),
            rng.uniform(-0.2, 0.2),
        ])
        angle = np.linalg.norm(rotation_vector)
        axis = rotation_vector / angle if angle > 0 else np.array([1, 0, 0])
        data.mocap_quat[0] = np.r_[
            np.cos(angle / 2), axis * np.sin(angle / 2)
        ]

        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera="calibration_camera")
        rgb = renderer.render().copy()
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        found, corners = cv2.findChessboardCornersSB(
            gray, PATTERN, flags=cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(OUTPUT / f"view_{i:02d}.png"), image)

        if found:
            observations.append(corners.astype(np.float32))
            cv2.drawChessboardCorners(image, PATTERN, corners, found)
            cv2.imwrite(str(OUTPUT / f"corners_{i:02d}.png"), image)

        print(f"View {i + 1}/24: {'detected' if found else 'not detected'}")

if len(observations) < 15:
    raise RuntimeError("Too few usable views. Send me the detection results.")

# Reserve five images for a separate validation check.
training = observations[:-5]
validation = observations[-5:]

# MuJoCo's current camera is an ideal pinhole camera.
# Keep distortion at zero; estimate focal lengths and principal point.
flags = (
    cv2.CALIB_ZERO_TANGENT_DIST
    | cv2.CALIB_FIX_K1
    | cv2.CALIB_FIX_K2
    | cv2.CALIB_FIX_K3
)

rms, K, distortion, _, _ = cv2.calibrateCamera(
    [object_points.copy() for _ in training],
    training,
    (WIDTH, HEIGHT),
    None,
    None,
    flags=flags,
)

# Check corner reprojection on images not used to fit K.
errors = []
for corners in validation:
    success, rvec, tvec = cv2.solvePnP(
        object_points, corners, K, distortion,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        raise RuntimeError("Pose estimation failed on a validation image.")

    projected, _ = cv2.projectPoints(
        object_points, rvec, tvec, K, distortion
    )
    residual = corners.reshape(-1, 2) - projected.reshape(-1, 2)
    errors.append(np.sqrt(np.mean(np.sum(residual**2, axis=1))))

np.savez(
    OUTPUT / "camera_calibration.npz",
    camera_matrix=K,
    distortion=distortion,
    image_size=np.array([WIDTH, HEIGHT]),
    training_rms=rms,
    validation_errors=np.array(errors),
)

print("\nEstimated camera matrix:\n", K)
print(f"Training reprojection RMS: {rms:.3f} pixels")
print(f"Mean validation reprojection RMS: {np.mean(errors):.3f} pixels")

# Known simulator settings are used only for this final comparison.
expected_focal = HEIGHT / (2 * np.tan(np.deg2rad(fovy) / 2))
print(f"Expected focal length: {expected_focal:.2f} pixels")
print(f"Estimated focal lengths: fx={K[0, 0]:.2f}, fy={K[1, 1]:.2f}")
print(f"Saved results in: {OUTPUT}")