import bpy
import numpy as np
import json

# Filepath for the JSON output
output_filepath = "/Users/rashmikadushan/Desktop/Liveroom/MocapV2/extrinsics/initial_camera_extrinsics.json"

# Function to extract rotation and translation from a camera
def get_camera_extrinsics(camera):
    # Get the world matrix of the camera
    world_matrix = camera.matrix_world
    
    # Extract the rotation (3x3) and translation (3x1)
    rotation_matrix = np.array(world_matrix.to_3x3())
    translation_vector = np.array(world_matrix.to_translation()).reshape(3, 1)
    
    return {
        "R": rotation_matrix.tolist(),
        "t": translation_vector.tolist()
    }

# List to hold camera extrinsics
camera_extrinsics = []

# Loop through all objects in the scene
for obj in bpy.data.objects:
    if obj.type == 'CAMERA':  # Check if the object is a camera
        extrinsics = get_camera_extrinsics(obj)
        camera_extrinsics.append(extrinsics)

# Write the extrinsics to the JSON file
with open(output_filepath, "w") as json_file:
    json.dump(camera_extrinsics, json_file, indent=2)

print(f"Camera extrinsics saved to {output_filepath}")
