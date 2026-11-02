from pathlib import Path

import cv2
import mujoco

# Load the scene at the robot's starting position.
scene = (
    Path(__file__).resolve().parent
    / "models/robotis_tb3/scene_assignment.xml"
)

model = mujoco.MjModel.from_xml_path(str(scene))
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

# Capture one RGB image from the robot's camera.
with mujoco.Renderer(model, height=480, width=640) as renderer:
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
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

corners, ids, rejected = detector.detectMarkers(gray)

if ids is None:
    print("No ArUco marker detected.")
else:
    print("Detected marker IDs:", ids.flatten().tolist())
    cv2.aruco.drawDetectedMarkers(display, corners, ids)

    if 0 in ids.flatten():
        print("Charging-station marker detected!")

# Show the image until a key is pressed.
cv2.imshow("Robot camera - ArUco detection", display)
print("Click the image window and press any key to close.")
cv2.waitKey(0)
cv2.destroyAllWindows()