import socket
import os
import time
import msgpack
import numpy.random as rd

data = {
    "scores": [90.0, 85.1, 95.2,0.22,0.1,0.2,0.5]
}

SOCKET_PATH = "/tmp/unity_python.sock"
if os.path.exists(SOCKET_PATH):
    os.remove(SOCKET_PATH)

server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
server.bind(SOCKET_PATH)
server.listen(1)

print("Waiting for Unity to connect...")
connection, _ = server.accept()
print("Connected!")

try:
    while True:
        print(f"Sending...")
        chunk = connection.send(msgpack.packb(data, use_bin_type=True))
        data["scores"] = rd.random(7).tolist()
        time.sleep(0.01)

finally:
    connection.close()
    server.close()
    os.remove(SOCKET_PATH)