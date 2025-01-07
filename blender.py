import bpy
import json
import numpy as np
import mathutils

filename = "/Users/rashmikadushan/Desktop/Liveroom/MocapV2/extrinsics/extrinsics.json"

with open(filename, "r") as file:
    extrinsic_data = json.load(file)
    print(extrinsic_data)

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

for i, data in enumerate(extrinsic_data):
    camera_data = bpy.data.cameras.new(name=f"Camera_{i}")
    camera_object = bpy.data.objects.new(name=f"Camera_{i}", object_data=camera_data)
    bpy.context.scene.collection.objects.link(camera_object)
    
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
