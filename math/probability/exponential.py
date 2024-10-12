#!/usr/bin/env python3
"""creates a class for exponential distribution"""


class Exponential:
    """
    Class that calculates the exponential distribution
    """
    def __init__(self, data=None, lambtha=1.):
        """
        Constructor method

        Args:
        data: list of the data to be used to estimate the distribution
        lambtha: expected number of occurrences in a given time frame
        """

        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = 1 / (sum(data) / len(data))
