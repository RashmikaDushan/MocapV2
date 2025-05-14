import serial
import serial.tools.list_ports
import time
import queue

accepted_pids = [60000]
accepted_vids = [4292]  
verified_device = None
ser = None

def discover_trackers():
    ports = serial.tools.list_ports.comports()
    global verified_device
    if not ports:
        print("No ports found")
    else:
        for port in ports:
            print(port.device,"|", port.name,"|", port.description,"|", port.hwid,"|", port.vid,"|", port.pid)
            if (port.pid in accepted_pids) and (port.vid in accepted_vids):
                print(f"Found ESP32 with PID {port.pid} and VID {port.vid}")
                verified_device.append(port.device)
                return True
    return False

def connect_tracker():
    global ser
    global verified_device
    try:
        ser = serial.Serial(verified_device, 115200, timeout=0, writeTimeout=0)
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        print(f"Connected to {verified_device}")
        return True
    except serial.SerialException as e:
        print(f"Failed to connect to {verified_device}: {e}")
    return False


def parse_data(data_str):
    try:
        if data_str.startswith("data"):
            parts = data_str.split(",")
            if len(parts) > 1:
                data = [float(x) for x in parts[1:]]
                data.extend([0.0,0.0,0.0])
                return data
    except ValueError:
        pass
    return None


def serial_reader_thread(data_queue):
    buffer = ""
    last_read_time = time.time()
    global ser
    
    while True:
        if ser is None or ser.closed:
            discover_trackers()
            connect_tracker()
            time.sleep(0.001)
            continue
            
        try:
            # Read whatever is available without waiting
            if ser.in_waiting > 0:
                chunk = ser.read(ser.in_waiting).decode('ascii', errors='ignore')
                buffer += chunk
                last_read_time = time.time()
                
                # Process any complete lines in the buffer
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    data = parse_data(line.strip())
                    if data:
                        # Use put_nowait to avoid blocking if queue is full
                        try:
                            data_queue.put_nowait(data)
                        except queue.Full:
                            # If queue is full, get the oldest item and replace it
                            _ = data_queue.get()
                            data_queue.put_nowait(data)
            
            # If no data received for a while, flush the buffer
            elif time.time() - last_read_time > 0.5:
                buffer = ""
                
            # Minimal sleep to prevent CPU hogging
            time.sleep(0.0005)  # 0.5ms, to allow for 200Hz (5ms) data rate
        except Exception as e:
            print(f"Error in reader thread: {e}")
            time.sleep(0.1)