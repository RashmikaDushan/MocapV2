import bpy
import json

def place_blender_points(json_file_path, collection_name):
    # Load the JSON data
    with open(json_file_path, "r") as file:
        coordinates = json.load(file)

    # Check if the "calculated" collection exists
    if collection_name in bpy.data.collections:
        calculated_collection = bpy.data.collections[collection_name]
        # Delete all objects in the "calculated" collection
        for obj in list(calculated_collection.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
    else:
        # Create the "calculated" collection if it doesn't exist
        calculated_collection = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(calculated_collection)

    # Create spheres at the specified coordinates
    for coord in coordinates:
        x, y, z = coord  # Unpack the x, y, z coordinates
        
        # Create a sphere
        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=0.025,  # Adjust the radius as needed
            location=(x, y, z)
        )
        
        # Get the last created object (active object)
        sphere = bpy.context.object
        
        # Link the sphere to the "calculated" collection
        calculated_collection.objects.link(sphere)
        
        # Unlink from the default collection
        bpy.context.scene.collection.objects.unlink(sphere)

    print("Spheres created successfully at specified locations.")

after_ba_file_path = "/Users/rashmikadushan/Desktop/MocapV2/jsons/after_ba_objects.json"
after_origin_file_path = "/Users/rashmikadushan/Desktop/MocapV2/jsons/after_origin_objects.json"
after_floor_file_path = "/Users/rashmikadushan/Desktop/MocapV2/jsons/after_floor_objects.json"
place_blender_points(after_floor_file_path,"after_floor_objects")
place_blender_points(after_ba_file_path,"after_ba_objects")
place_blender_points(after_origin_file_path,"after_origin_objects")

