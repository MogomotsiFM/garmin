import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from numpy.polynomial import Polynomial as Poly


def transformToCartesianCoordinates(dist1, dist2):
    # Computing the angle between sensor2 vector and the floor using cosine rule
    nume = L*L + dist2*dist2 - dist1*dist1
    denom = 2 * L * dist2
    # The angle of interest
    alpha = np.acos(nume / denom)

    # Convert to x and y coordinates
    x = dist2 * np.cos(alpha)
    y = dist2 * np.sin(alpha)

    xs = x.reshape((x.size,))
    ys = y.reshape((y.size,))

    return xs, ys


def readDistances(data, header):
    dist = np.array(data[header], ndmin=2)
    dist = dist.transpose()
    
    # Take into account the radius of the ball
    dist = dist + R

    return dist


def assessOutcome(poly):
    if poly(RR) > (HR + R):
        # Ball passes above the rim
        if poly(-RR) < (HR - R):
            # This is a goal
            outcome = "Score"
        elif poly(-RR) < (HR + R):
            # Ball collides with the  last edge of the rim. We need to model this.
            outcome = "Collision with the last edge of the rim."
        elif poly(-RR - DB) < (HR + HB):
            # Ball bounces of the backboard. We must model this.
            outcome = "Collision with backbboard."
        else:
            # Overshot
            outcome = "Overshot."
    elif poly(RR) > (HR - R):
        # Ball collides with the rim
        # We model the collision
        outcome = "Collision with the first edge of the rim. Need to model this."
    else:
        # Undershot
        outcome = "Undershot"

    return outcome


# Distance between sensors in meters
L = 12.4

# Radius of the ball in meters
R = 0.24/2

# Height of the rim in meters
HR = 3.05

# Radius of the rim in meters
RR = 0.4572/2

# Height of the backboard above the rim in meters
HB = 1.1

# Distance of the backboard from the rim in meters
DB = 0.1016


df = pd.read_csv("basketball.csv", sep=', ', engine="python")

for i in range(1, 7):
    s1_label = f"b{i}_s1"
    dist1 = readDistances(df, s1_label)

    s2_label = f"b{i}_s2"
    dist2 = readDistances(df, s2_label)

    # (Least squares) Fit the data to a second order polynomial
    x, y = transformToCartesianCoordinates(dist1, dist2)

    poly = Poly.fit(x, y, deg=2)

    outcome = assessOutcome(poly)
    print("Assessment: ", outcome)

    xs = np.linspace(-1, 10, 100)
    plt.plot(xs, poly(xs))
    plt.plot(x, y)
    plt.plot([-RR, RR], [3.05, 3.05])
    plt.show()

