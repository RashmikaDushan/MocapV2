import queue
import time
from filterpy.kalman import ExtendedKalmanFilter
import numpy as np
from MathLib import quaternion_multiply_normalized
import quaternion

Q = np.array([[0.0061591104,0,0, 0,0,0, 0,0,0],
                [0,0.0061591104,0, 0,0,0, 0,0,0],
                [0,0,0.0061591104, 0,0,0, 0,0,0],
                [0,0,0, 0.01,0,0, 0,0,0],
                [0,0,0, 0,0.01,0, 0,0,0],
                [0,0,0, 0,0,0.01, 0,0,0],
                [0,0,0, 0,0,0, 69.4,0,0],
                [0,0,0, 0,0,0, 0,69.4,0],
                [0,0,0, 0,0,0, 0,0,69.4]
                 ])
L = np.array([[0,0,0,1,1,1,0,0,0],
              []
              ])

a_w = None

def f(x, dt, u):
    """
    State transition function for the EKF.

    1. Compute orientation update:
        q_{k+1} = q_k + 0.5 * Δt * q_k ⊗ [0; ω_m]
        q_{k+1} = normalize(q_{k+1})

    2. Compute rotation matrix R(q_k)

    3. Compute world acceleration:
        a_w = R(q_k) * a_m + g

    4. Update velocity:
        v_{k+1} = v_k + Δt * a_w

    5. Update position:
        p_{k+1} = p_k + Δt * v_k + 0.5 * Δt^2 * a_w
    """
    # Extract states
    # x_k = [q0, q1, q2, q3, x, y, z, vx, vy, vz, bwx, bwy, bwz, bax, bay, baz]
    # u = [ax, ay, az, wx, wy, wz, mx, my, mz]
    global a_w
    q = x[0:4]
    pos = x[4:7]
    vel = x[7:10]

    a = u[0:3] - x[10:13]  # Accelerometer measurement minus bias
    w = u[3:6] - x[13:16]  # Gyroscope measurement minus bias
    m = u[6:9]  # Magnetometer measurement
    
    # Initialize new state
    x_new = np.zeros_like(x)
    
    # 1. Compute orientation update:
    # q_{k+1} = q_k + 0.5 * Δt * q_k ⊗ [0; ω_m]
    x_new[0:4] = q + 0.5 * dt * quaternion_multiply_normalized(q, np.array([0, w[0], w[1], w[2]]))
    

    # 2. Compute rotation matrix R(q_k)
    R = quaternion.as_rotation_matrix(quaternion.from_float_array(x_new[0:4]))

    # 3. Compute world acceleration:
    # a_w = R(q_k) * a_m + g
    a_w = np.dot(R, a) + np.array([0, 0, -9.81])  # Gravity vector in world frame

    # 4. Update velocity:
    # v_{k+1} = v_k + Δt * a_w
    x_new[7:10] = vel + dt * a_w

    # 5. Update position:
    # p_{k+1} = p_k + Δt * v_k + 0.5 * Δt^2 * a_w
    x_new[4:7] = pos + dt * vel + 0.5 * dt**2 * a_w

    x_new[10:16] = x[10:16]  # Biases remain unchanged in this model
    
    return x_new

def F_jacobian(self, x, dt, u):
    """
    Jacobian of the state transition function.
    """
    # x_k = [q0, q1, q2, q3, x, y, z, vx, vy, vz, bwx, bwy, bwz, bax, bay, baz]
    # u = [ax, ay, az, wx, wy, wz, mx, my, mz]
    global a_w
    a = u[0:3] # Accelerometer measurement
    w = u[3:6] # Gyroscope measurement
    m = u[6:9]  # Magnetometer measurement
    bw = x[13:16]  # Gyroscope bias
    ba = x[10:13]  # Accelerometer bias

    J = np.zeros((self.n_states, self.n_states)) # q -> w
    J[0:4, 0:4] = np.eye(4) + np.array([
        [0, -w[1]+bw[1], -w[2]+bw[2], -w[3]+bw[3]],
        [w[1]-bw[1], 0, w[3]-bw[3], -w[2]+bw[2]],
        [w[2]-bw[2], -w[3]+bw[3], 0, w[1]-bw[1]],
        [w[3]-bw[3], -w[2]+bw[2], -w[1]+bw[1], 0]
        ]) * (dt/2)
    
    J[0:4, 10:13] = np.eye(4) - np.array([ # q -> bw
        [0, -x[1], -x[2], -x[3]],
        [x[0], 0, x[2], -x[3]],
        [x[0], -x[1], 0, x[1]],
        [x[3], -x[2], -x[1], 0]
        ]) * (dt/2)
    
    J[7:10,7:10] = np.eye(3)

    J[7:10,0:4] = a_w*dt * np.array([
        [4*x[0]-2*x[3]+2*x[2], 4*x[1]+2*x[2]+2*x[3], 2*x[1]+2*x[0], -2*x[0]+2*x[1]],
        [2*x[3]+4*x[0]-2*x[1], 2*x[2]-2*x[0], 2*x[1]+4*x[2]+2*x[3], 2*x[0]+2*x[2]],
        [-2*x[2]+2*x[1]+4*x[0], 2*x[3]+2*x[0], -2*x[0]+2*x[3], 4*x[3]+2*x[1]+2*x[2]]
    ])

    # This is a complex function for quaternions. We'll use a simplified approach
    F = np.eye(self.n_states)
    
    # Position derivative is velocity 
    F[0, 3] = dt
    F[1, 4] = dt
    F[2, 5] = dt
    
    # Quaternion derivative depends on angular velocity (this is approximate)
    q = x[6:10]
    omega = x[10:13]
    
    # Simplified Jacobian for quaternion update
    # In practice, you would compute the exact partial derivatives
    F[6:10, 10:13] = np.array([
        [0, -q[1]/2, -q[2]/2, -q[3]/2],
        [q[0]/2, 0, q[3]/2, -q[2]/2],
        [-q[3]/2, q[0]/2, 0, q[1]/2],
        [q[2]/2, -q[1]/2, q[0]/2, 0]
    ]).T * dt
    
    return F

def process_data_thread(data_queue, display_queue):
    # x_k = [q0, q1, q2, q3, x, y, z, vx, vy, vz, bwx, bwy, bwz, bax, bay, baz]
    n_states = 13  # Number of states
    n_measurements = 9  # Number of measurements

    ekf = ExtendedKalmanFilter(dim_x=n_states, dim_z=n_measurements)
    ekf.x = np.zeros((n_states, 1))  # State vector
    ekf.x[6] = 1.0  # Initial quaternion w component

    ekf.P *= np.eye(n_states) * 100  # Initial covariance matrix
    ekf.P[6:10, 6:10] = np.eye(4) * 0.1
   
    ekf.Q = np.eye(n_states) # Measurement noise covariance
    # Position and velocity process noise
    for i in range(6):
        ekf.Q[i, i] = 0.01
    # Quaternion process noise
    for i in range(6, 10):
        ekf.Q[i, i] = 0.01
    # Angular velocity process noise
    for i in range(10, 13):
        ekf.Q[i, i] = 0.1
    
    # Initialize measurement noise covariance
    ekf.R = np.eye(n_measurements)
    # Accelerometer noise
    for i in range(3):
        ekf.R[i, i] = 0.5
    # Gyroscope noise
    for i in range(3, 6):
        ekf.R[i, i] = 0.1
    # Magnetometer noise
    for i in range(6, 9):
        ekf.R[i, i] = 0.8
    
    # Gravity constant
    g = 9.81
    
    # Earth's magnetic field reference (normalized)
    mag_ref = np.array([1.0, 0.0, 0.0])
    
    while True:
        try:
            # Try to get latest data without blocking
            try:
                # Drain the queue to get the most recent data
                while not data_queue.empty():
                    last_data = data_queue.get_nowait()
            except queue.Empty:
                pass
            
            # Measure time duration
            current_time = time.time()
            dt = current_time - prev_time
            prev_time = current_time

            
                
            time.sleep(0.001)  # Small sleep to prevent CPU hogging
        except Exception as e:
            print(f"Error in process thread: {e}")