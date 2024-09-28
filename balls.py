import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
            # We may need to backtrack to identify the point of collision.
            outcome = "Collision with the last edge of the rim."

            x, y = findPositionOfBallCollidingWithRimEdge(poly, -RR, HR)

            return outcome, x, y
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

        x, y = findPositionOfBallCollidingWithRimEdge(poly, RR, HR)
        
        return outcome, x, y
    else:
        # Undershot
        outcome = "Undershot"

    return outcome, None, None


def findPositionOfBallCollidingWithRimEdge(poly: Poly, x_rim, y_rim):
    '''
        Input:
            poly: The polynomial that models the path followed by a ball
            (x_rim, y_rim): The cartesian coordinate of an edge of the rim
    '''

    # Let the Polynomial package do the heavy lifting
    #   Construct the equation of a circle using the Polynomial package
    #       (x - x_c)**2 + (y - y_c)**2 = R*R
    #       where x = x_rim , y = y_rim and R is the radius of the ball
    #       x_c is unknown and y_c = poly 

    p1 = Poly([x_rim, -1], domain=poly.domain, window=poly.window)
    p1 = p1 * p1
    
    p2 = poly - y_rim
    p2 = p2 * p2
    
    p = p1 + p2 - R*R
    
    roots = p.roots()
    real_roots = np.extract(np.logical_not(np.iscomplex(roots)), roots)
    correct_root = np.extract(np.real(real_roots) >= x_rim, real_roots.real)
    
    return correct_root[0], poly(correct_root[0])



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

    poly = Poly.fit(x, y, deg=2, domain=[-15, 15], window=[-15, 15])

    # If x_c and y_c exist then they represent the center of the ball that 
    # collides with either the ring or backboarc
    outcome, x_c, y_c = assessOutcome(poly)
    print("Assessment: ", outcome)

    xs = np.linspace(-1, 10, 100)

    figure, axes = plt.subplots()

    axes.plot(xs, poly(xs))
    axes.plot(x, y)
    # Plot a line that represents the rim
    axes.plot([-RR, RR], [3.05, 3.05])

    if x_c is not None:
        print("Drawing a circle.")
        circle = plt.Circle((x_c, y_c), R, fill=False)

        axes.set_aspect( 1 )
        axes.add_artist( circle )

    plt.show()

