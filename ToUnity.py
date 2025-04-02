import socket
import time
import msgpack
import serial
import serial.tools.list_ports

accepted_pids = [60000]
accepted_vids = [4292]
verified_devices = []
HOST = "127.0.0.1"
PORT = 5000
server = None
connection = None
    
data = {
    "tracker1": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
}


def discover_trackers():
    ports = serial.tools.list_ports.comports()
    if not ports:
        print("No ports found")
        return False
    else:
        for port in ports:
            print(port.device,"|", port.name,"|", port.description,"|", port.hwid,"|", port.vid,"|", port.pid)
            if (port.pid in accepted_pids) and (port.vid in accepted_vids):
                print(f"Found ESP32 with PID {port.pid} and VID {port.vid}")
                verified_devices.append(port.device)
                return True
    

def connect_tracker():
    for device in verified_devices:
        try:
            ser = serial.Serial(device, 115200, timeout=1)
            print(f"Connected to {device}")
            return ser
        except serial.SerialException as e:
            print(f"Failed to connect to {device}: {e}")
    return None

def read_data(ser):
    try:
        raw_data = ser.readline()
        try:
            data = raw_data.decode('ascii').strip().split(",")
        except UnicodeDecodeError:
            print(f"Warning: Received non-ASCII data",end="\r")
            return None

        if data[0] == "data":
            print(f"Received data: {data[1:]}", end="")
            return data[1:]
        print("Invalid data format", end="\r")
        return None
    except serial.SerialException as e:
        print(f"Error reading data: {e}")
        return None


def connect_unity():
    global server
    global connection
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, PORT))
    server.listen(1)

    print("Waiting for Unity to connect...")
    connection, _ = server.accept()
    print("Connected!")

if __name__ == "__main__":
    connect_unity()
    found = discover_trackers()
    if found:
        ser = connect_tracker()
    try:
        while True:
            try:
                if found:
                    data["tracker1"] = read_data(ser)  
                if data["tracker1"] is not None:
                    chunk = connection.send(msgpack.packb(data, use_bin_type=True))
                    print(f" | Sent data: {data['tracker1']}",end="\r")
                time.sleep(0.01)
            except ConnectionResetError:
                while True:
                    print("\nUnity disconnected, waiting for reconnection...")
                    connection, _ = server.accept()
                    print("Connected!")
                    break
    finally:
        connection.close()
        server.close()