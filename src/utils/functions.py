import math


def constant_decay_function(variable, rate):
    result = variable * rate
    return result


def exponential_function(a, x, k, b, exp):
    """
    Exponential function where
    
        y = a * e^(-k * (x / b)^exp)
    """
    result = a * math.exp(-k * (x / b) ** exp)
    return result


def casted_exponential_function(a, x, k, b, exp):
    """
    Exponential function where x is casted to int

        y = a * e^(-k * int((x / b)^exp))
    """
    result = a * math.exp(-k * int((x / b) ** exp))
    return result
