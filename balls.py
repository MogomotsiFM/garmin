import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from numpy.polynomial import Polynomial as Poly

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
    dist1 = np.array(df[s1_label], ndmin=2)
    dist1 = dist1.transpose()
    # Take into account the radius of the ball
    dist1 = dist1 + R

    s2_label = f"b{i}_s2"
    dist2 = np.array(df[s2_label], ndmin=2)
    dist2 = dist2.transpose()
    # Take into account the radius of the ball
    dist2 = dist2 + R

    # Computing the angle between sensor2 vector and the floor using cosine rule
    nume = L*L + dist2*dist2 - dist1*dist1
    denom = 2 * L * dist2
    # The angle of interest
    alpha = np.acos(nume / denom)

    # Convert to x and y coordinates
    x = dist2 * np.cos(alpha)
    y = dist2 * np.sin(alpha)

    # (Least squares) Fit the data to a second order polynomial
    xs = x.reshape((x.size,))
    ys = y.reshape((y.size,))
    poly = Poly.fit(xs, ys, deg=2)

    if poly(RR) > (HR + R):
        # Ball passes above the rim
        if poly(-RR) < (HR - R):
            # This is a goal
            print("Outcome: Score.")
        elif poly(-RR) < (HR + R):
            # Ball collides with the rim
            print("Collision with the last edge of the rim. Need to model this.")
        elif poly(-RR - DB) < (HR + HB):
            # Ball bounces of the backboard. We must model this
            print("Collision with backbboard.")
        else:
            # Overshot
            print("Overshot.")
    elif poly(RR) > (HR - R):
        # Ball collides with the rim
        # We model the collision
        print("Collision with the first edge of the rim. Need to model this.")
    else:
        # Undershot
        print("Undershot") 

    xs = np.linspace(-1, 10, 100)
    plt.plot(xs, poly(xs))
    plt.plot(x, y)
    plt.plot([-RR, RR], [3.05, 3.05])
    plt.show()
    