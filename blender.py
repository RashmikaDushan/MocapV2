import bpy
import json
import numpy as np
import mathutils

# Path to the extrinsics.json file
filename_with_ba = "/Users/rashmikadushan/Desktop/Liveroom/MocapV2/extrinsics/extrinsics.json"
filename_without_ba = "/Users/rashmikadushan/Desktop/Liveroom/MocapV2/extrinsics/before_ba_extrinsics.json"

def generate_setup(filename,collection_name):
    # Load the extrinsic data
    with open(filename, "r") as file:
        extrinsic_data = json.load(file)
        print(extrinsic_data)

    # Check if the "calculated" collection exists
    if collection_name in bpy.data.collections:
        calculated_collection = bpy.data.collections[collection_name]
        # Delete all objects in the "calculated" collection
        for obj in calculated_collection.objects:
            bpy.data.objects.remove(obj, do_unlink=True)
    else:
        # Create the "calculated" collection if it doesn't exist
        calculated_collection = bpy.data.collections.new(collection_name)
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

        camera_object.scale = (0.25, 0.25, 0.25)

generate_setup(filename_without_ba,"Calculated_no_ba")
generate_setup(filename_with_ba,"Calculated_ba")