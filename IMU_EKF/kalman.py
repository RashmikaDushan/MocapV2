from matrix import Matrix3x3

class KalmanFilter:
    def __init__(self, initial_state):
        self.state_error_covarience = Matrix3x3()
        self.state = initial_state
        self.covariance = 1.0  # Initial covariance

    def predict(self):
        # Predict the next state and covariance
        self.state += self.process_noise
        self.covariance += self.process_noise

    def update(self, measurement):
        # Update the state with the new measurement
        kalman_gain = self.covariance / (self.covariance + self.measurement_noise)
        self.state += kalman_gain * (measurement - self.state)
        self.covariance *= (1 - kalman_gain)