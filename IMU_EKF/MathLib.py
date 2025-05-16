import numpy as np

def quaternion_multiply_normalized(q1, q2):
    """
    Multiply two quaternions.
    
    Parameters:
    - q1, q2: quaternions in format [q0, q1, q2, q3] where q0 is scalar part
    
    Returns:
    - q1*q2: quaternion product
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    q = np.zeros(4)
    
    q[0] = w1*w2 - x1*x2 - y1*y2 - z1*z2
    q[1] = w1*x2 + x1*w2 + y1*z2 - z1*y2
    q[2] = w1*y2 - x1*z2 + y1*w2 + z1*x2
    q[3] = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    q = q / np.linalg.norm(q)

    return q