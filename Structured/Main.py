import time
import threading
import queue
from SerialConnection import discover_trackers, connect_tracker, serial_reader_thread
from Process import process_data_thread

data_queue = queue.Queue(maxsize=10)  # Limited queue to prevent memory issues
display_queue = queue.Queue(maxsize=10)  # Queue for display thread

def display_data_thread():
    last_display_time = time.time()
    last_data = None
    
    while True:
        try:
            # Try to get latest data without blocking
            try:
                # Drain the queue to get the most recent data
                while not data_queue.empty():
                    last_data = data_queue.get_nowait()
            except queue.Empty:
                pass
            
            # Display at a consistent rate regardless of data arrival
            current_time = time.time()
            if current_time - last_display_time >= 0.01:  # 100Hz display update (adjust as needed)
                if last_data is not None:
                    print(f"Data: {[f'{x:.3f}' for x in last_data]}", end="\r")
                last_display_time = current_time
                
            time.sleep(0.001)  # Small sleep to prevent CPU hogging
        except Exception as e:
            print(f"Error in display thread: {e}")

if __name__ == "__main__":
    found = discover_trackers()
    connected = False
    
    if found:
        connected = connect_tracker()
        
    if connected:
        # Start the reader thread
        reader_thread = threading.Thread(target=serial_reader_thread, daemon=True,args=(data_queue,))
        reader_thread.start()

        # Start the process thread
        process_thread = threading.Thread(target=process_data_thread, daemon=True,args=(data_queue,display_queue,))
        process_thread.start()
        
        # Start the display thread
        # display_thread = threading.Thread(target=display_data_thread, daemon=True)
        # display_thread.start()
    
    # Keep the main thread alive
    while True:
        time.sleep(0.1)