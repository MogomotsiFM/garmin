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


def predictOutcome(path):
    '''
        Output:
            outcome: "Score" | "Undershot" | "Flies off the rim" | "Flies off the backboard" | "Overshot"
            x_center: x coordinate of the center of the ball if there is a collision
            y_center: y coordinate of the center of the ball if there is a collision
            post_collision_path: The flight path of the ball after a collision.
    '''
    if path(RR) < (HR - R):
        return "Undershot", None, None, None
    
    outcome, x_collision, y_collision = doesBallCollideWithRim(path)
    collision_pnt = (x_collision, y_collision)

    if outcome == "score":
        return "Score", None, None, None
    elif outcome == "miss":
        outcome, x_center, y_center = findPositionOfBallCollidingWithBackboard(path)
        collision_pnt = (-RR-DB, y_center)
        if outcome == "overshot":
            return "Overshot", None, None, None
    else: # outcome == "collision"
       x_center, y_center = findPositionOfBallCollidingWithRimEdge(path, x_collision, y_collision) 

    xs, ys = predictFlightPathPostCollision(path, (x_center, y_center), collision_pnt)

    outcome = predictOutcomePostCollision(xs, ys)

    return outcome, x_center, y_center, (xs, ys)


def predictOutcomePostCollision(xs, ys):
    '''
        Our model for the behaviour of the ball post-collision is fairly good in the first few centi-meters.
        Anything beyond that it break down. To improve it we would have to keep track of the time in flight
        and then use it to estimate the velocity. This statement is to justify our assumption that 
        the path after the collision is monotonic.

        Input:
            (xs, ys): Path followed by the ball after the collision.
    '''
    path = Poly.fit(xs, ys, deg=2)

    # Compute the gradient at the center of the rim
    deriv = path.deriv()
    grad = deriv(0)

    if grad > 0:
        return "Ball bounced of the rim."
    
    else:
        left_y = path(RR)
        right_y = path(-RR)

        # Because the post collision path is monotonic it is enough to check at the edges if the center of the 
        # ball is less than the height of the rim.
        if left_y <= HR or right_y <= HR:
            return "Score."
        else:
            return "Ball bounced of the rim."


def doesBallCollideWithRim(path: Poly):
    '''
        We want to the point of intersection of two equations
           y = HR (The height of the rim)
           y = path

        Output:
            outcome: "score" | "collision" | "miss",
            x_rim, y_rim: An edge of the rim that the ball collides with.
    '''
    obj = path - HR

    roots = obj.roots()
    real_roots = np.extract(np.isreal(roots), roots)

    if real_roots.size == 0:
        return "miss", None, None

    # Distance from the right edge of the rim to the center of the ball
    d1 = (real_roots <= ( RR - R))
    # Distance from the left edge of the rim to the center of the ball
    d2 = (real_roots >= (-RR + R))
    condition = np.all([d1, d2], axis=0)
    pnts = np.extract(condition , roots)

    if pnts.size:
        return "score", None, None
    else:
        # Notice how we include points that are just outside the rim by extending the search window by the 
        # radius of the ball R
        condition = np.all([real_roots >= -RR, real_roots <= (RR + R)], axis=0)
        collision_pnts = np.extract(condition , real_roots)
        if collision_pnts.size:
            # We expect only one value
            if collision_pnts[0] >= 0: 
                return "collision", RR, HR
            else:
                return "collision", -RR, HR
        else:
            # This should not happen    
            return "miss", None, None


def findPositionOfBallCollidingWithRimEdge(path: Poly, x_rim, y_rim):
    '''
        Input:
            poly: The polynomial that models the path followed by a ball
            (x_rim, y_rim): The cartesian coordinate of an edge of the rim
    '''

    # Let the Polynomial package do the heavy lifting
    #   Construct the equation of a circle using the Polynomial package
    #       (x - x_c)**2 + (y - y_c)**2 = R*R
    #       where x = x_rim , y = y_rim and R is the radius of the ball
    #       x_c is unknown and y_c = path 

    p1 = Poly([x_rim, -1], domain=path.domain, window=path.window)
    p1 = p1 * p1
    
    p2 = path - y_rim
    p2 = p2 * p2
    
    p = p1 + p2 - R*R
    
    roots = p.roots()
    real_roots = np.extract(np.logical_not(np.iscomplex(roots)), roots)

    y_c = path( np.real(real_roots))
    if y_c.size == 1:
        return np.real(real_roots[0]), y_c[0]
    else:
        root = np.max(real_roots)
        root = np.real(root)
        return root, path(root)


def findPositionOfBallCollidingWithBackboard(path: Poly):
    '''
        Find the point of contact of a circle and a straight line (x=x0)
        Also, we know that the y value of the center of the ball is modelled by the given polynomial (path)

        Input: 
            path: The polynomial that models the path of the center of the ball

        Output:
            outcome: "overshot" | "collision"
            x_c: The x coordinate of the ball when it first makes contact with the backboard
            y_c: The y coordinate of the ball when it first makes contact with the backboard
    '''
    # Because the straight line is the y-axis, the equation of the circle is slightly simpler:
    #   (x0 - x_c)**2 + (y - y_c) = R*R
    # (x_c, y_c) is the center of the ball (circle),
    # y_c = path,
    # R is the radius of the circle,
    # y = y_c (By definition of a tangent to a circle)
    # Therefore:
    x0 = -RR - DB
    x_c = x0 + R
    if x_c < x0:
        x_c = x0 - R
    y_c = path(x_c)

    if y_c > (HR + HB):
        return "overshot", None, None
    else:
        return "collision", x_c, y_c


def predictFlightPathPostCollision(path: Poly, center, collision_pnt):
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
    ball_tangent_grad = - (x_r - x_center) / (y_r - y_center)

    # Explanation?
    nx_circle, ny_circle = normal(ball_tangent_grad, collision_pnt[0], collision_pnt[1])
    deriv = path.deriv()
    path_tangent_grad = deriv(x_center)
    nx_path, ny_path = vector(path_tangent_grad, center[0], center[1])

    dot = np.dot([nx_circle, ny_circle], [nx_path, ny_path])

    theta = np.atan(ball_tangent_grad)
    if dot < 0:
        rotation_angle = np.pi/2 - np.abs(theta)
        if theta < 0:
            rotation_angle = -1*rotation_angle
    else:
        rotation_angle = -theta

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
    trans_xs, trans_ys = rotate(xs, ys, rotation_angle)        
    
    # Reflect the path about the y-axis to model the ball bouncing off a wall
    # This is really the what we wanted to get at!!
    if dot < 0:
        reflected_xs = np.abs(trans_xs)
        reflected_ys = trans_ys
    else:
        reflected_ys = np.abs(trans_ys)
        reflected_xs = trans_xs

    # Undo the first rotation
    new_xs, new_ys = rotate(reflected_xs, reflected_ys, -rotation_angle)

    # Finally, undo the initial translation
    post_collision_xs, post_collision_ys = translate(new_xs, new_ys, trans)

    plotDebugGraphs(path,
                    trans_xs, trans_ys,
                    reflected_xs, reflected_ys,
                    new_xs, new_ys,
                    post_collision_xs, post_collision_ys,
                    collision_pnt, center,
                    ball_tangent_grad)

    return post_collision_xs, post_collision_ys


def normal(grad, x, y):
    grad = -1 / grad
    
    return vector(grad, x, y, False)


def vector(grad, x_before, y_before, flight_path=True):
    # y = mx + c
    c = y_before - grad*x_before

    # We need two point to compute a normal
    if flight_path:
        # Note that we are subracting a positive number because the ball flies from positive inf to negative inf
        # If we use parametric equations to model the flight path we would no have to choose a random point.
        x_after = x_before - 10
    else:
        # The normal of the wall should always point (relatively) towards +x-axis
        x_after = x_before + 10
    
    y_after = grad*x_after + c

    dx = x_after-x_before
    dy = y_after-y_before
    dist = np.sqrt( dx*dx + dy*dy )

    return dx/dist, dy/dist


def translate(xs, ys, trans):
    return xs + trans[0], ys + trans[1]


def plotDebugGraphs(path: Poly,
                    trans_xs, trans_ys,
                    reflected_xs, reflected_ys,
                    new_xs, new_ys,
                    post_collision_xs, post_collision_ys,
                    collision_point, center,
                    ball_tangent_grad):
    figure, axes = plt.subplots()
    axes.set_aspect(1)

    # We need this in order to display the entire flight path of the ball
    x_ = np.linspace(-5, 15, 50)
    axes.plot(x_, path(x_), label="Estimated flight path")

    axes.plot(trans_xs, trans_ys, '+', label="Translated and rotated")

    axes.plot(reflected_xs, reflected_ys, label="Reflection")

    axes.plot(new_xs, new_ys, 'o', label="Undo rotation")

    axes.plot(post_collision_xs, post_collision_ys, label="Post-collision path")

    # The tangent of the ball at the point of colllision
    x_r, y_r = collision_point
    if np.isfinite(ball_tangent_grad):
        c = y_r - ball_tangent_grad*x_r
        axes.plot([-1.0, 1.0], [-ball_tangent_grad + c, ball_tangent_grad + c], label="Tanget of ball (Wall)")
    else:
        axes.plot([-RR-DB, -RR-DB], [3.05, 4.2], label="Tanget of ball")

    # We want to plot the line that estimates the path followed by the ball at the time of collision
    x_center, y_center = center
    
    deriv = path.deriv()
    path_tangent_grad = deriv(x_center)
    
    c = y_center - x_center*path_tangent_grad
    # This plots the line that estimates the path followed by the ball at the time of collision
    axes.plot([-2, 2], [-2*path_tangent_grad + c, 2*path_tangent_grad + c], label="Flight path tangent")

    # The line that is perpendicular to the tangent of the ball
    # It just makes it easier to see that the angle of incident is equal to the angle of reflection.
    c = y_r - x_r * (-1/ball_tangent_grad)
    axes.plot([-2, 2], [-2*(-1/ball_tangent_grad) + c, 2*(-1/ball_tangent_grad) + c], label="Perpendicular to ball tangent")

    # Show the position of the ball at the time of the collision
    circle = patches.Circle(center, R, fill=False, label="Ball")
    axes.add_artist(circle)

    axes.legend(shadow=True)
    
    plt.show()


def rotate(xs, ys, theta):
    reflected_coords = np.vstack([xs, ys])
    reflected_coords = np.hsplit(reflected_coords, xs.size)

    reverse_rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    coords_post_collision = reverse_rot@reflected_coords 

    new_coords = np.hstack(coords_post_collision)

    new_xs, new_ys = np.vsplit(new_coords, 2)

    return new_xs[0], new_ys[0]


debug = True

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
    start = x[0] + 1
    poly = Poly.fit(x, y, deg=2, domain=[-15, start], window=[-15, start])

    # If x_c and y_c exist then they represent the center of the ball that 
    # collides with either the ring or backboard
    outcome, x_c, y_c, post_collision_path = predictOutcome(poly)

    xs = np.linspace(-1, y[0], 50)
    # However, if there is a collision, then
    if x_c is not None:
        xs = np.linspace(x_c, y[0], 50)

    figure, axes = plt.subplots()
    axes.set_aspect(1)
    axes.legend(shadow=True)

    axes.plot(xs, poly(xs), label="Estimate flight path")
    axes.plot(x, y, label="Measure fligh path")
    # Plot a line that represents the rim
    axes.plot([-RR, RR], [3.05, 3.05], label="Rim")

    # Plot the line that represents the backboard
    axes.plot([-RR-DB, -RR-DB], [HR, HR+HB], label="Backboard")

    if debug:
        circle_l = plt.Circle( (RR, poly(RR)), R, fill=False, label="Ball(First rim edge)", color="green")
        axes.add_artist(circle_l)

        circle_2 = plt.Circle( (-RR, poly(-RR)), R, fill=False, label="Second rim edge", color="blue")
        axes.add_artist(circle_2)

    if x_c is not None:
        circle = plt.Circle((x_c, y_c), R, fill=False, label="Ball at collision point", color="red")
        axes.add_artist(circle)

        xs, ys = post_collision_path
        condition = xs >= -1
        xs = xs[condition]
        ys = ys[condition]
        axes.plot(xs, ys, label="Post collision path")
    plt.show()

