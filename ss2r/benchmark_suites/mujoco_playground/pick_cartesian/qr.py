import os

import cv2
import cv2.aruco as aruco

# Parameters
marker_ids = [571, 581, 591, 601, 611, 621]
marker_size_pixels = 600  # Size of each marker in pixels
output_folder = "aruco_markers"
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_1000)
# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)
# Generate and save markers
for marker_id in marker_ids:
    img = aruco.generateImageMarker(aruco_dict, marker_id, marker_size_pixels)
    filepath = os.path.join(output_folder, f"aruco_{marker_id}.png")
    cv2.imwrite(filepath, img)
    print(f"Saved: {filepath}")
