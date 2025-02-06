import numpy as np
def calculate_parameter(t, pressure, temperature):
    """
    Example external function that calculates some parameter
    In reality, this could be a complex calculation from another module
    """
    return pressure * np.exp(-t/temperature) + 5