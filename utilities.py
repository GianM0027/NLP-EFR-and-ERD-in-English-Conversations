import numpy as np


def replace_nan_with_zero(lst: list) -> list:
    """
    Takes a list with NaN values and convert them to zero

    :param lst: original list
    :return: the list with all the NaNs converted to zero
    """
    return [0 if (x is not float or np.isnan(x)) else x for x in lst]
