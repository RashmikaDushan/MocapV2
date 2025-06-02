import sys
import time
import threading
import PySpin
import cv2
from lib.ImageOperations import _find_dot
from lib.Helpers import find_point_correspondance_and_object_points,get_extrinsics
import queue
import socket
import msgpack


running = threading.Event()
running.set()
camera_poses, camera_count = get_extrinsics("./jsons/after_floor_extrinsics.json")

def track_points(cam, nodemap, nodemap_tldevice,data_queue:queue.Queue,preview=False):
    """
    This function continuously acquires images from a device and displays them using OpenCV.
    
    :param cam: Camera to acquire images from.
    :param nodemap: Device nodemap.
    :param nodemap_tldevice: Transport layer device nodemap.
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    global running
    try:
        # Configure stream buffer handling mode
        sNodemap = cam.GetTLStreamNodeMap()
        node_bufferhandling_mode = PySpin.CEnumerationPtr(sNodemap.GetNode('StreamBufferHandlingMode'))
        if not PySpin.IsReadable(node_bufferhandling_mode) or not PySpin.IsWritable(node_bufferhandling_mode):
            print('Unable to set stream buffer handling mode. Aborting...')
            return False
            
        # Set buffer handling mode to NewestOnly for lowest latency
        node_newestonly = node_bufferhandling_mode.GetEntryByName('NewestOnly')
        if not PySpin.IsReadable(node_newestonly):
            print('Unable to set stream buffer handling mode. Aborting...')
            return False
            
        node_bufferhandling_mode.SetIntValue(node_newestonly.GetValue())
        
        # Configure acquisition mode to continuous
        node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
        if not PySpin.IsReadable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
            print('Unable to set acquisition mode to continuous. Aborting...')
            return False
            
        node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
        if not PySpin.IsReadable(node_acquisition_mode_continuous):
            print('Unable to set acquisition mode to continuous. Aborting...')
            return False
            
        node_acquisition_mode.SetIntValue(node_acquisition_mode_continuous.GetValue())
        print('Acquisition mode set to continuous...')
        
        # Retrieve device serial number for window name
        device_serial_number = ''
        node_device_serial_number = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber'))
        if PySpin.IsReadable(node_device_serial_number):
            device_serial_number = node_device_serial_number.GetValue()
            print(f'Device serial number retrieved as {device_serial_number}...')
        
        window_name = f'Camera Feed - {device_serial_number}'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # Configure exposure
        # try:
        #     # Set exposure auto to off for manual control
        #     node_exposure_auto = PySpin.CEnumerationPtr(nodemap.GetNode('ExposureAuto'))
        #     if PySpin.IsWritable(node_exposure_auto):
        #         node_exposure_auto.SetIntValue(node_exposure_auto.GetEntryByName('Off').GetValue())
                
        #         # Set exposure time manually (in microseconds)
        #         node_exposure_time = PySpin.CFloatPtr(nodemap.GetNode('ExposureTime'))
        #         if PySpin.IsWritable(node_exposure_time):
        #             exposure_time = 5000.0  # 5ms exposure, adjust as needed
        #             node_exposure_time.SetValue(exposure_time)
        #             print(f'Exposure time set to {exposure_time} us')
        # except PySpin.SpinnakerException as ex:
        #     print(f'Error setting exposure: {ex}')
        
        # Begin acquiring images
        cam.BeginAcquisition()
        print('Acquiring images...')
        
        # Frame rate calculation variables
        frame_count = 0
        start_time = time.time()
        fps = 0
        time.sleep(1)  # Allow time for camera to stabilize
        
        # Main acquisition loop
        while running.is_set():
            try:
                # Get next image with shorter timeout for responsiveness
                image_result = cam.GetNextImage(500)
                
                if not image_result.IsIncomplete():
                    # Get image data as numpy array and optimize display
                    image_data = image_result.GetNDArray().copy()
                    image_data = cv2.cvtColor(image_data, cv2.COLOR_BAYER_GR2BGR)
                    image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
                    image_data, detected_points = _find_dot(image_data,print_location=True)

                    try:
                        # Drain the queue to get the most recent data
                        if data_queue.full():
                            data_queue.get_nowait()
                        data_queue.put_nowait(detected_points)
                    except queue.Empty:
                        print("Queue is full")
                    
                    # Calculate and display FPS every second
                    frame_count += 1
                    if frame_count % 30 == 0:
                        end_time = time.time()
                        fps = frame_count / (end_time - start_time)
                        frame_count = 0
                        start_time = end_time
                    
                    # Add FPS text to the image
                    cv2.putText(image_data, f"FPS: {fps:.1f}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                    
                    # Display the image using OpenCV (much faster than matplotlib)
                    cv2.imshow(window_name, image_data)
                    
                    # Process any OpenCV GUI events (like window closing)
                    key = cv2.waitKey(1)
                    if key == 27:  # ESC key
                        print('ESC pressed. Exiting...')
                        running.clear()
                        break

                    
                # Release the image to avoid buffer filling
                image_result.Release()
                
            except PySpin.SpinnakerException as ex:
                print(f'Error: {ex}')
                running.clear()
        
        # End acquisition and clean up
        cam.EndAcquisition()
        cv2.destroyWindow(window_name)
        print("Camera feed stopped")
        
    except PySpin.SpinnakerException as ex:
        print(f'Error: {ex}')
        return False
        
    return True


def track(data_queue1:queue.Queue,data_queue2:queue.Queue,stream=True):
    global camera_poses
    print(camera_poses)
    print("Tracking started")
    
    if stream:
        HOST = "127.0.0.1"
        PORT = 5000
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind((HOST, PORT))
        server.listen(1)
        print("Waiting for Unity to connect...")
        connection, _ = server.accept()
        print("Connected!")
    point = [0,0,0,0,0,0,0,0]
    fps = 0
    old_time = time.time()

    while True:
        fps = time.time() - old_time
        old_time = time.time()
        fps = 1 / fps if fps > 0 else 0
        try:
            if not (data_queue1.empty() or data_queue2.empty()):
                data1 = data_queue1.get_nowait()
                data2 = data_queue2.get_nowait()
                image_points = [data1,data2]
                object_points,image_p = find_point_correspondance_and_object_points(image_points,camera_poses,4)
                if stream:
                    if len(object_points) > 0:
                        point = object_points[0]
                        point = list(point)
                        point = [0,0,0,0] + point
                    data = {"tracker1": point}
                    connection.send(msgpack.packb(data, use_bin_type=True))
                    print(f"Object Points: {point}")
                else:
                    print(f"Object Points: {object_points}")
                print(f"Image Points: {image_p}")
                print(f"FPS: {fps:.2f}")
                # print(f"Data1: {data1}")
                # print(f"Data2: {data2}")
        except queue.Empty:
            print("Queue is empty")
        except ConnectionResetError:
            while True:
                    print("\nUnity disconnected, waiting for reconnection...")
                    connection, _ = server.accept()
                    print("Connected!")
                    break
            
        time.sleep(0.01)
        #send 3d points

def run_single_camera(cam,data_queue):
    """
    Camera initialization and execution function.
    
    :param cam: Camera to run on.
    :type cam: CameraPtr
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    try:
        # Initialize camera
        cam.Init()
        
        # Retrieve nodemap
        nodemap_tldevice = cam.GetTLDeviceNodeMap()
        nodemap = cam.GetNodeMap()
        cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
        exposure_time = 5000  # 1 ms
        cam.ExposureTime.SetValue(exposure_time)
        # Performance optimization: Set packet size to max for GigE cameras
        try:
            # Check if this is a GigE camera
            node_device_type = PySpin.CEnumerationPtr(nodemap_tldevice.GetNode('DeviceType'))
            if (PySpin.IsReadable(node_device_type) and 
                node_device_type.GetCurrentEntry().GetSymbolic() == 'GigEVision'):
                
                # Get GigE specific nodemap
                nodemap_gige = cam.GetTLStreamNodeMap()
                
                # Set stream packet resend
                node_packet_resend = PySpin.CBooleanPtr(nodemap_gige.GetNode('StreamPacketResendEnable'))
                if PySpin.IsWritable(node_packet_resend):
                    node_packet_resend.SetValue(True)
                    print('Packet resend enabled')
                    
                # Set packet size to max
                node_packet_size = PySpin.CIntegerPtr(nodemap_gige.GetNode('StreamPacketSize'))
                if PySpin.IsWritable(node_packet_size):
                    max_packet_size = node_packet_size.GetMax()
                    node_packet_size.SetValue(max_packet_size)
                    print(f'Packet size set to maximum ({max_packet_size})')

                
                # binning_horizontal = PySpin.CIntegerPtr(nodemap.GetNode("Width"))
                # if PySpin.IsAvailable(binning_horizontal) and PySpin.IsWritable(binning_horizontal):
                #     binning_horizontal.SetValue(1224)  # 2x binning

                
                # binning_vertical = PySpin.CIntegerPtr(nodemap.GetNode("Height"))
                # if PySpin.IsAvailable(binning_vertical) and PySpin.IsWritable(binning_vertical):
                #     binning_vertical.SetValue(1024)  # 2x binning
        except PySpin.SpinnakerException as ex:
            print(f'Notice: GigE optimization not applicable - {ex}')
            
        # Run acquisition and display function
        result = track_points(cam, nodemap, nodemap_tldevice,data_queue=data_queue,preview=False)
        
        # Deinitialize camera
        cam.DeInit()
        return result
        
    except PySpin.SpinnakerException as ex:
        print(f'Error: {ex}')
        return False


def main():
    """
    Main function.
    """
    global running
    try:
        # Get system instance
        system = PySpin.System.GetInstance()
        
        # Print system info
        # version = system.GetLibraryVersion()
        print(f'MoCap v2.0')
        
        # Get camera list
        cam_list = system.GetCameras()
        num_cameras = cam_list.GetSize()
        print(f'Number of cameras detected: {num_cameras}')
        
        # Check if cameras are available
        if num_cameras == 0:
            cam_list.Clear()
            system.ReleaseInstance()
            print('No cameras detected!')
            input('Press Enter to exit...')
            sys.exit(0)
            return False
        
        data_queue1 = queue.Queue(maxsize=10)
        data_queue2 = queue.Queue(maxsize=10)

        process_thread = threading.Thread(target=track, args=(data_queue1,data_queue2,))
        process_thread.start()        
        camera1_display = threading.Thread(target=run_single_camera, args=(cam_list[0],data_queue1))
        camera1_display.start()
        camera2_display = threading.Thread(target=run_single_camera, args=(cam_list[1],data_queue2))
        camera2_display.start()

        while running.is_set():
            time.sleep(0.1)
            # print('Running...')
        
        time.sleep(1)
        print('Stopping cameras...')
        camera1_display.join()
        camera2_display.join()

        # Clean up
        # del cam_list[0]  # Important for proper memory management
        cam_list.Clear()
        system.ReleaseInstance()
        
        print('\nDone!')
        return True
        
    except PySpin.SpinnakerException as ex:
        print(f'Error: {ex}')
        return False


if __name__ == '__main__':
    success = main()
    print('Exiting...')
    sys.exit(0 if success else 1)
    quit()