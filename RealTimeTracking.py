import json
import numpy as np
import cv2
from scipy import linalg
import bpy

# Configuration class to store all tracking-related parameters
class TrackerConfig:
    def __init__(self):
        self.WORKSPACE_PATH = "/Users/user/liveroom/MocapV2"
        self.EXTRINSICS_PATH = f"{self.WORKSPACE_PATH}/jsons/after_ba_extrinsics.json"
        self.INTRINSICS_PATH = f"{self.WORKSPACE_PATH}/jsons/camera-params-in.json"
        self.RENDER_PATH = "/tmp/temp_render.jpeg"
        self.RENDER_WIDTH = 960
        self.RENDER_HEIGHT = 540

# Main class for real-time 3D tracking in Blender
class BlenderTracker:
    def __init__(self):
        self.config = TrackerConfig()
        self.scene = bpy.context.scene
        # Load camera calibration data
        self.extrinsics = self._load_json(self.config.EXTRINSICS_PATH)
        self.intrinsics = self._load_json(self.config.INTRINSICS_PATH)
        # Get all camera objects from the Blender scene
        self.camera_names = [obj.name for obj in bpy.data.objects if obj.type == 'CAMERA']
        self.tracking_collection = self._setup_collection()
        self.materials = self._setup_materials()
        self.trail_points = []  # Store trail point objects
        self._setup_render_settings()

    def _load_json(self, path):
        with open(path, 'r') as f:
            return json.load(f)

    def _setup_collection(self):
        # Create or clear a collection for tracking visualization
        if "TrackingPoints" not in bpy.data.collections:
            collection = bpy.data.collections.new("TrackingPoints")
            bpy.context.scene.collection.children.link(collection)
        else:
            collection = bpy.data.collections["TrackingPoints"]
            # Clear existing objects
            for obj in collection.objects:
                bpy.data.objects.remove(obj, do_unlink=True)
        return collection

    def _setup_materials(self):
        # Create materials for tracked point (red) and trail points (green)
        materials = {}
        if "TrackedPointMaterial" not in bpy.data.materials:
            mat = bpy.data.materials.new(name="TrackedPointMaterial")
            mat.use_nodes = True
            mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (1, 0, 0, 1)
        materials['main'] = bpy.data.materials["TrackedPointMaterial"]

        if "TrailMaterial" not in bpy.data.materials:
            mat = bpy.data.materials.new(name="TrailMaterial")
            mat.use_nodes = True
            mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0, 1, 0, 1)
        materials['trail'] = bpy.data.materials["TrailMaterial"]

        return materials

    def _setup_render_settings(self):
        # Configure render settings for optimal performance
        self.scene.render.image_settings.file_format = 'JPEG'
        self.scene.render.image_settings.quality = 90
        self.scene.render.resolution_x = self.config.RENDER_WIDTH
        self.scene.render.resolution_y = self.config.RENDER_HEIGHT
        self.scene.render.engine = 'BLENDER_EEVEE_NEXT'
        # Set low sample count for faster rendering
        if hasattr(self.scene.eevee, 'taa_render_samples'):
            self.scene.eevee.taa_render_samples = 1
        elif hasattr(self.scene, 'eevee_next'):
            self.scene.eevee_next.render_samples = 1

    def get_camera_frame(self, camera_name, camera_idx):
        # Render view from specified camera and undistort the image
        self.scene.camera = bpy.data.objects[camera_name]
        self.scene.render.filepath = self.config.RENDER_PATH
        bpy.ops.render.render(write_still=True)

        frame = cv2.imread(self.config.RENDER_PATH)
        if frame is None:
            return None

        # Apply camera undistortion using calibration parameters
        frame = cv2.undistort(
            frame,
            np.array(self.intrinsics[camera_idx]["intrinsic_matrix"]),
            np.array(self.intrinsics[camera_idx]["distortion_coef"])
        )
        frame = cv2.medianBlur(frame, 5)  # Remove noise
        return frame

    def detect_point(self, frame):
        # Detect bright points in the frame using thresholding and contours
        grey = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        grey = cv2.threshold(grey, 255*0.2, 255, cv2.THRESH_BINARY)[1]
        contours, _ = cv2.findContours(grey, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(frame, contours, -1, (0,255,0), 1)

        # Calculate centroid of detected contours
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(frame, f'({cx}, {cy})', (cx, cy - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100,255,100), 1)
                cv2.circle(frame, (cx, cy), 3, (100,255,100), -1)
                return (cx, cy)
        return None

    def triangulate_point(self, image_points, camera_poses):
        """image_points shape = [camera_count,2]"""

        image_points = np.array(image_points)
        none_indicies = np.where(np.all(image_points == None, axis=1))[0]
        image_points = np.delete(image_points, none_indicies, axis=0)
        camera_poses = np.delete(camera_poses, none_indicies, axis=0)

        if len(image_points) <= 1:
            return [None, None, None]

        Ps = [] # projection matrices

        for i, camera_pose in enumerate(camera_poses):
            RT = np.c_[camera_pose["R"], camera_pose["t"]]
            P = self.intrinsics[i]["intrinsic_matrix"] @ RT
            Ps.append(P)

        # https://temugeb.github.io/computer_vision/2021/02/06/direct-linear-transorms.html
        def DLT(Ps, image_points):

            """image_points: [[x_cam1, y_cam1], [x_cam2, y_cam2], ... , [x_cam6, y_cam6]]"""

            A = []

            for P, image_point in zip(Ps, image_points):
                A.append(image_point[1]*P[2,:] - P[1,:])
                A.append(P[0,:] - image_point[0]*P[2,:])

            A = np.array(A).reshape((len(Ps)*2,4))
            B = A.transpose() @ A
            U, s, Vh = linalg.svd(B, full_matrices = False)
            object_point = Vh[3,0:3]/Vh[3,3]

            return object_point

        object_point = DLT(Ps, image_points)

        return object_point

    def triangulate_3d_point(self, image_points):
        # Prepare data for triangulation
        points_2d = []
        camera_poses = []

        for i, ext in enumerate(self.extrinsics):
            if image_points[i] is not None:
                points_2d.append(image_points[i])
                camera_poses.append({
                    "R": np.array(ext["R"], dtype=np.float32).reshape(3, 3),
                    "t": np.array(ext["t"], dtype=np.float32).reshape(3, 1)
                })
            else:
                points_2d.append(None)
                camera_poses.append(None)

        return self.triangulate_point(points_2d, camera_poses)

    def update_tracking_visualization(self, point_3d):
        # Create or update the tracked point sphere
        if "TrackedPoint" not in bpy.data.objects:
            bpy.ops.mesh.primitive_uv_sphere_add(radius=0.05)
            sphere = bpy.context.active_object
            sphere.name = "TrackedPoint"
            sphere.data.materials.append(self.materials['main'])
            if sphere.users_collection:
                for collection in sphere.users_collection:
                    collection.objects.unlink(sphere)
            self.tracking_collection.objects.link(sphere)
        else:
            sphere = bpy.data.objects["TrackedPoint"]

        # Update position and add keyframe
        sphere.location = point_3d
        sphere.keyframe_insert(data_path="location", frame=self.scene.frame_current)

        # Add trail point
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.02)
        trail_point = bpy.context.active_object
        trail_point.location = point_3d
        trail_point.data.materials.append(self.materials['trail'])
        if trail_point.users_collection:
            for collection in trail_point.users_collection:
                collection.objects.unlink(trail_point)
        self.tracking_collection.objects.link(trail_point)
        self.trail_points.append(trail_point)

    def create_combined_view(self, frames, image_points):
        # Add status text to each camera view
        for i, frame in enumerate(frames):
            status = "Detected" if image_points[i] is not None else "No Detection"
            color = (100, 255, 100) if image_points[i] is not None else (100, 100, 255)
            cv2.putText(frame, f'Camera {i+1} - {status}', (10, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Combine frames into a grid layout
        top_row = np.hstack(frames[:3])
        bottom_row = np.hstack(frames[3:])
        combined_view = np.vstack([top_row, bottom_row])

        # Add overall tracking status
        status_text = "Tracking" if all(p is not None for p in image_points) else "Lost Track"
        status_color = (255, 255, 255) if all(p is not None for p in image_points) else (100, 100, 255)
        cv2.putText(combined_view, f'Frame: {self.scene.frame_current}/{self.scene.frame_end} - {status_text}',
                   (10, combined_view.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        return combined_view

    def track(self):
        # Main tracking loop
        if not self.camera_names:
            print("No cameras found in Blender scene.")
            return

        print(f"Scene frame range: {self.scene.frame_start} to {self.scene.frame_end}")

        self.scene.frame_current = self.scene.frame_start
        print(f"Starting tracking from frame {self.scene.frame_current}")

        while True:
            print(f"Processing frame {self.scene.frame_current} of {self.scene.frame_end}")
            self.scene.frame_set(self.scene.frame_current)

            frames = []
            image_points = []
            # Process each camera view
            for i, camera_name in enumerate(self.camera_names):
                frame = self.get_camera_frame(camera_name, i)
                if frame is None:
                    continue
                frames.append(frame)

                point = self.detect_point(frame)
                image_points.append(point)
                print(f"Camera {camera_name} {'detected point at ' + str(point) if point else 'no detection'}")

            # Triangulate 3D position if point detected in all views
            if len(frames) == len(self.camera_names) and all(p is not None for p in image_points):
                point_3d = self.triangulate_3d_point(image_points)
                print(f"Frame {self.scene.frame_current} - 3D point: {point_3d}")
                self.update_tracking_visualization(point_3d)

            # Display combined view of all cameras
            if len(frames) == len(self.camera_names):
                combined_view = self.create_combined_view(frames, image_points)
                cv2.imshow("All Cameras", combined_view)

            # Check for quit command or end of frames
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or self.scene.frame_current >= self.scene.frame_end:
                break

            self.scene.frame_current += 1

        cv2.destroyAllWindows()

def main():
    tracker = BlenderTracker()
    tracker.track()

if __name__ == "__main__":
    main()