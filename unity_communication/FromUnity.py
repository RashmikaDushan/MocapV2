import socket
import numpy as np
import cv2
import os

# Cleanup previous socket file (if any)
SOCKET_PATH = "/tmp/unity_python.sock"
if os.path.exists(SOCKET_PATH):
    os.remove(SOCKET_PATH)

# Create UDS server
server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
server.bind(SOCKET_PATH)
server.listen(1)

print("Waiting for Unity to connect...")
connection, _ = server.accept()
print("Connected!")

# Frame settings (match Unity's camera resolution)
width = 640
height = 480
buffer_size = width * height * 3  # RGB24: 3 bytes per pixel

try:
    while True:
        # Read raw frame data from Unity
        data = b""
        while len(data) < buffer_size:
            chunk = connection.recv(buffer_size - len(data))
            if not chunk:
                break
            data += chunk
        
        if len(data) == buffer_size:
            # Convert bytes to numpy array (RGB)
            frame = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("Unity Camera Feed", frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
finally:
    connection.close()
    server.close()
    os.remove(SOCKET_PATH)
    cv2.destroyAllWindows()