import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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

            # Model the collision
            outcome = ballCollisionWithEdgeModel(poly, (x, y), (-RR, HR))

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


def ballCollisionWithEdgeModel(poly: Poly, center, collision_pnt):
    '''
        Input:
            poly: The polynomial that models the path followed by a ball,
            center = (x_c, y_c): The center of the ball when the collision is predicted to happen
            collision_pnt = (x_r, y_r): The edge of the rim against which the ball is colliding
        Output:
            The direction of the ball after the collision
    '''
    x_center, y_center = center
    x_r, y_r = collision_pnt
    
    # Gradient at the point of collision
    m = - (x_r - x_center) / (y_r - y_center + 0.000000001)
    print("Gradient on a circle: ", m)

    theta = np.atan(m)
    print("Theta: ", theta)

    # Rotation matrix
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    # Translation: We rotate about the point of collision
    trans = np.array(center, ndmin=2)
    trans = trans.transpose()

    # Sample the path after the point of collision
    # This simulates tracking the ball past the collision point
    xs = np.linspace(x_center, -5, 10)
    ys = poly(xs)

    figure, axes = plt.subplots()
    axes.set_aspect(1)

    x_ = np.linspace(-5, 15, 50)
    axes.plot(x_, poly(x_))

    print(rot)
    print("YS: ", trans)

    coords = np.vstack([xs, ys]) - trans
    print("Coords: ", coords)
    split = np.hsplit(coords, xs.size)
    print("HSplit: ", split)
    transformed = rot@split #+ trans
    trans_coords = np.hstack(transformed)
    print("Rotated: ", trans_coords)

    trans_xs, trans_ys = np.vsplit(trans_coords, 2)
    trans_xs = trans_xs[0]
    trans_ys = trans_ys[0]
    axes.plot(trans_xs, trans_ys)
    print("Trans")
    print(trans_xs)
    print(trans_ys)

    trans_xs = np.abs(trans_xs)
    axes.plot(trans_xs, trans_ys)

    reflected_coords = np.vstack([trans_xs, trans_ys])
    reflected_coords = np.hsplit(reflected_coords, xs.size)
    print("Reflected coords hsplit: ", reflected_coords)

    reverse_rot = np.array([[np.cos(-theta), -np.sin(-theta)], [np.sin(-theta), np.cos(-theta)]])
    coords_post_collision = reverse_rot@reflected_coords #+ trans
    print("New coords post collision: ", coords_post_collision)

    new_coords = np.hstack(coords_post_collision)
    print("\nStacked: ", new_coords)
    print("Old coords: ", xs, ys)

    new_xs, new_ys = np.vsplit(new_coords, 2)
    new_xs = new_xs[0]
    new_ys = new_ys[0]

    axes.plot(new_xs, new_ys)

    # We need to tamper these with the direction of flight of the ball at time of collision
    # Hence we compute the gradient of the path at that point
    deriv = poly.deriv()
    m2 = deriv(x_center)

    alpha = np.atan(m2)

    gamma = 0

    print("Alpha: ", alpha*180/np.pi)
    if alpha < 0:
        alpha = np.pi + alpha
    print("Alpha: ", alpha*180/np.pi)


    if m < 0:
        print("Theta: ", theta*180/np.pi)
        theta = np.pi + theta # Automatically the angle this line makes with the x-axis is less than zero
        print("Theta: ", theta*180/np.pi)

        # Angle between the two lines
        beta = theta - alpha
        print("Beta: ", beta*180/np.pi)

        if beta >= np.pi:
            tmp = np.pi - beta

            gamma = np.pi - 2 * tmp
        else: # beta < np.pi
            # The negative sign indicates that we rotate clockwise
            gamma = -(np.pi - 2*beta) 

    else: # m >= 0 
        # We expect alpha to be greater than zero
        beta = theta - alpha

        if beta >= 0:
            # The negative sign indicates that we rotate clockwise
            gamma = -(np.pi - 2*beta)
        else: # beta < 0
            beta = -beta

            gamma = np.pi - 2*beta
    
    print("Gamma: ", gamma*180/np.pi)
    adjusted_coords = np.vstack([new_xs, new_ys])
    adjusted_coords = np.hsplit(adjusted_coords, xs.size)
    print("Reflected coords hsplit: ", adjusted_coords)

    adjust_rot = np.array([[np.cos(gamma), -np.sin(gamma)], [np.sin(gamma), np.cos(gamma)]])
    coords_post_collision = adjust_rot@adjusted_coords + trans
    print("New coords post collision: ", coords_post_collision)

    new_coords = np.hstack(coords_post_collision)
    print("\nStacked: ", new_coords)
    print("Old coords: ", xs, ys)

    new_xs, new_ys = np.vsplit(new_coords, 2)
    new_xs = new_xs[0]
    new_ys = new_ys[0]

    axes.plot(new_xs, new_ys)

    # The line tangent to the ball at the point of colllision
    c = y_r - m*x_r
    axes.plot([-1.0, 1.0], [-m+c, m+c])

    # A line that esitmates the path followed by the ball at the time of collision
    c = y_center - x_center*m2
    axes.plot([-2, 2], [-2*m2+c, 2*m2+c])

    # The line that is perpendicular to the tangent
    c = y_r - x_r * (-1/m)
    axes.plot([-2, 2], [-2*(-1/m)+c, 2*(-1/m)+c])

    circle = patches.Circle(center, R, fill=False)
    axes.add_artist(circle)

    plt.show()

    # reflected_poly = poly.fit(new_vals[0], new_vals[1])

    if m > 0 and x_r < 0:
        return "Score."
    elif m > 0 and x_r > 0:
        return "Undershot."
    elif x_r < 0:
        return "Collision with the far edge of the rim still to be modelled."
    else:
        return "Collision with the near edge of the rim still to be modelled."


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

    x, y = transformToCartesianCoordinates(dist1, dist2)

    # (Least squares) Fit the data to a second order polynomial
    poly = Poly.fit(x, y, deg=2, domain=[-15, 15], window=[-15, 15])

    # If x_c and y_c exist then they represent the center of the ball that 
    # collides with either the ring or backboarc
    outcome, x_c, y_c = assessOutcome(poly)
    print("Assessment: ", outcome)

    xs = np.linspace(-1, 10, 100)

    figure, axes = plt.subplots()
    axes.set_aspect(1)

    axes.plot(xs, poly(xs))
    axes.plot(x, y)
    # Plot a line that represents the rim
    axes.plot([-RR, RR], [3.05, 3.05])

    circle_l = plt.Circle( (RR, poly(RR)), R, fill=False)
    axes.add_artist(circle_l)

    circle_2 = plt.Circle( (-RR, poly(-RR)), R, fill=False)
    axes.add_artist(circle_2)

    if x_c is not None:
        print("Drawing a circle.")
        circle = plt.Circle((x_c, y_c), R, fill=False)

        axes.add_artist( circle )

    plt.show()

