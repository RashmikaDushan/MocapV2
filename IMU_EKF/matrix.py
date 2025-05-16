class Matrix3x3:
    def __init__(self, data=None):
        if data:
            if len(data) != 3 or any(len(row) != 3 for row in data):
                raise ValueError("Data must be a 3x3 matrix.")
            else:
                self.data = data
        else:
            self.data = [[0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0]]
    
    def update_matrix(self, data):
        if len(data) != 3 or any(len(row) != 3 for row in data):
            raise ValueError("Data must be a 3x3 matrix.")
        else:
            self.data = data