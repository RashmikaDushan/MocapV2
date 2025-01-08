import bpy
import json
import numpy as np
import mathutils

# Path to the extrinsics.json file
filename = "/Users/rashmikadushan/Desktop/Liveroom/MocapV2/extrinsics/extrinsics.json"

# Load the extrinsic data
with open(filename, "r") as file:
    extrinsic_data = json.load(file)
    print(extrinsic_data)

# Check if the "calculated" collection exists
if "Calculated" in bpy.data.collections:
    calculated_collection = bpy.data.collections["Calculated"]
    # Delete all objects in the "calculated" collection
    for obj in calculated_collection.objects:
        bpy.data.objects.remove(obj, do_unlink=True)
else:
    # Create the "calculated" collection if it doesn't exist
    calculated_collection = bpy.data.collections.new("Calculated")
    bpy.context.scene.collection.children.link(calculated_collection)

# Create cameras and add them to the "calculated" collection
for i, data in enumerate(extrinsic_data):
    # Create a new camera
    camera_data = bpy.data.cameras.new(name=f"Camera_{i}")
    camera_object = bpy.data.objects.new(name=f"Camera_{i}", object_data=camera_data)
    
    # Link the camera to the "calculated" collection
    calculated_collection.objects.link(camera_object)
    
    # Create a 4x4 transformation matrix
    rotation_matrix = np.array(data["R"])
    translation_vector = np.array(data["t"])
    transformation_matrix = np.eye(4)  # Start with an identity matrix
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = translation_vector
    
    # Convert numpy matrix to Blender's mathutils.Matrix
    transformation_matrix_blender = mathutils.Matrix(transformation_matrix)
    
    # Apply the transformation matrix to the camera
    camera_object.matrix_world = transformation_matrix_blender
