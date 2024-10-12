#!/usr/bin/env python3
"""Creates a class that represents a poisson distribution"""
import math


class Poisson:
    def __init__(self, data=None, lambtha=1.):
        """
        Initializes the poisson distribution
        Args:
        class constructor

        parameters:
            data [list]: data to be used to estimate the distibution
            lambtha [float]: the expected number of occurances on a given time

        Sets the instance attribute lambtha as a float
        If data is not given:
            Use the given lambtha or
            raise ValueError if lambtha is not positive value
        If data is given:
            Calculate the lambtha of data
            Raise TypeError if data is not a list
            Raise ValueError if data does not contain at least two data points
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
            # Calculate lambtha as the mean of the data
            self.data = data
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """
        The instance method PMF to calculate given number of successes k

        Args:
        k [int]: the number of successes

        Returns:
        PMF: the probability of successes 'k'
        """
        if type(k) is not int:
            k = int(k)
        # PMF defined for non-negative integers
        if k < 0:
            return 0
        # calculates PMF using the formula: (lambtha^k * e^-lambtha) / k!
        lambtha = self.lambtha
        PMF = (lambtha ** k) * math.exp(-lambtha) / math.factorial(k)

        return PMF
