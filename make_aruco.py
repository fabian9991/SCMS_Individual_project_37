from pathlib import Path
import cv2

# Choose the marker family and marker ID.
dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)
marker = cv2.aruco.generateImageMarker(dictionary, 0, 600)

# Add white space around the black border for detection.
image = cv2.copyMakeBorder(
    marker, 100, 100, 100, 100,
    cv2.BORDER_CONSTANT, value=255
)

# Save the image alongside the robot's other assets.
output = (
    Path(__file__).resolve().parent
    / "models/robotis_tb3/assets/charger_aruco.png"
)
output.parent.mkdir(parents=True, exist_ok=True)

if not cv2.imwrite(str(output), image):
    raise RuntimeError(f"Could not save {output}")

print(f"Marker ID 0 saved to: {output}")