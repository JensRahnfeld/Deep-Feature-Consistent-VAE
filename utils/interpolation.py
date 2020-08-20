def linear_interpolation(z_left, z_right):
    def _linear_interpolation(alpha):
        x = (1 - alpha) * z_left + alpha * z_right
        return x
    
    return _linear_interpolation