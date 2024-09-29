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


def ballCollisionWithEdgeModel(path: Poly, center, collision_pnt):
    '''
        Input:
            path: The polynomial that models the path followed by a ball,
            center = (x_c, y_c): The center of the ball when the collision is predicted to happen
            collision_pnt = (x_r, y_r): The edge of the rim against which the ball is colliding
        Output:
            A set of points that model the path followed by the ball after the collision
    '''
    x_center, y_center = center
    x_r, y_r = collision_pnt
    
    # Gradient at the point of collision
    ball_tangent_grad = - (x_r - x_center) / (y_r - y_center + 0.000000001)
    print("Gradient on a circle: ", ball_tangent_grad)

    theta = np.atan(ball_tangent_grad)

    # Translation: We rotate about the point of collision
    trans = np.array(center, ndmin=2)
    trans = trans.transpose()

    # Sample the path after the point of collision
    # This simulates tracking the ball past the collision point
    xs = np.linspace(x_center, -5, 20)
    ys = path(xs)

    # Translate the points past the predicted collision point so that they can be rotated about the axis
    xs, ys = translate(xs, ys, -trans)

    # Rotate the points such that the tangent of the ball at the collision point is parallel to the y-axis
    trans_xs, trans_ys = rotate(xs, ys, theta)

    # Reflect the path about the y-axis to model the ball bouncing off a wall
    # This is really the what we wanted to get at!!
    reflected_xs = np.abs(trans_xs)
    reflected_ys = trans_ys

    # Undo the first rotation
    new_xs, new_ys = rotate(reflected_xs, reflected_ys, -theta)

    # We need to tamper these with the direction of flight of the ball at time of collision
    # Hence we compute the gradient of the path at that point
    deriv = path.deriv()
    path_tangent_grad = deriv(x_center)

    post_collision_xs, post_collision_ys = adjustPostCollisionPathWithFlightDirection(path_tangent_grad, 
                                                                                      ball_tangent_grad, 
                                                                                      new_xs, 
                                                                                      new_ys)
    
    # Finally, undo the initial translation
    post_collision_xs, post_collision_ys = translate(post_collision_xs, post_collision_ys, trans)

    plotDebugGraphs(path,
                    trans_xs, trans_ys,
                    reflected_xs, reflected_ys,
                    new_xs, new_ys,
                    post_collision_xs, post_collision_ys,
                    collision_pnt, center,
                    ball_tangent_grad, path_tangent_grad)

    return post_collision_xs, post_collision_ys


def translate(xs, ys, trans):
    return xs + trans[0], ys + trans[1]


def adjustPostCollisionPathWithFlightDirection(path_tangent_grad, ball_tangent_grad, xs, ys):
    '''
        The direction of flight relative to a surface can change the direction of flight after
        the collision. The surface in this case is a tangent to the ball at the point of collision.
        The direction of flight is estimated using the tangent of the flight path at the time of 
        collision.
    '''
    gamma = computeAngleBetweenTwoLines(ball_tangent_grad, path_tangent_grad)
    
    # Actually adjust the flight path to take into account the direction of the ball
    return rotate(xs, ys, gamma)


def plotDebugGraphs(path: Poly,
                    trans_xs, trans_ys,
                    reflected_xs, reflected_ys,
                    new_xs, new_ys,
                    post_collision_xs, post_collision_ys,
                    collision_point, center,
                    ball_tangent_grad, path_tangent_grad):
    figure, axes = plt.subplots()
    axes.set_aspect(1)

    # We need this in order to display the entire flight path of the ball
    x_ = np.linspace(-5, 15, 50)
    axes.plot(x_, path(x_))

    axes.plot(trans_xs, trans_ys)

    axes.plot(reflected_xs, reflected_ys)

    axes.plot(new_xs, new_ys)

    axes.plot(post_collision_xs, post_collision_ys)

    # The line tangent to the ball at the point of colllision
    x_r, y_r = collision_point
    c = y_r - ball_tangent_grad*x_r
    axes.plot([-1.0, 1.0], [-ball_tangent_grad + c, ball_tangent_grad + c])

    # A line that esitmates the path followed by the ball at the time of collision
    x_center, y_center = center
    c = y_center - x_center*path_tangent_grad
    axes.plot([-2, 2], [-2*path_tangent_grad + c, 2*path_tangent_grad + c])

    # The line that is perpendicular to the tangent
    c = y_r - x_r * (-1/ball_tangent_grad)
    axes.plot([-2, 2], [-2*(-1/ball_tangent_grad) + c, 2*(-1/ball_tangent_grad) + c])

    # Show the position of the ball at the time of the collision
    circle = patches.Circle(center, R, fill=False)
    axes.add_artist(circle)

    plt.show()


def rotate(xs, ys, theta):
    reflected_coords = np.vstack([xs, ys])
    reflected_coords = np.hsplit(reflected_coords, xs.size)
    print("Reflected coords hsplit: ", reflected_coords)

    reverse_rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    coords_post_collision = reverse_rot@reflected_coords 
    print("New coords post collision: ", coords_post_collision)

    new_coords = np.hstack(coords_post_collision)
    print("\nStacked: ", new_coords)
    print("Old coords: ", xs, ys)

    new_xs, new_ys = np.vsplit(new_coords, 2)

    return new_xs[0], new_ys[0]


def computeAngleBetweenTwoLines(ball_tangent_gradient, path_tangent_gradient):
    '''
        This function computes the angle between these two straight lines.
    '''

    alpha = np.atan(path_tangent_gradient)
    theta = np.atan(ball_tangent_gradient)

    gamma = 0

    print("Alpha: ", alpha*180/np.pi)
    if alpha < 0:
        alpha = np.pi + alpha
    print("Alpha: ", alpha*180/np.pi)


    if ball_tangent_gradient < 0:
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

    return gamma



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

