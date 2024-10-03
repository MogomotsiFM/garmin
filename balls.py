import os
import sys
import time

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


def evaluate(y_path: Poly, t_path: Poly, x):
    """
        Given the value of x, what is the corresponding value of y?
    """
    t = t_path(x)

    return y_path(t)


def predictOutcome(x_path: Poly, y_path: Poly, t_path: Poly, start_time):
    '''
        Output:
            outcome: "Score" | "Undershot" | "Flies off the rim" | "Flies off the backboard" | "Overshot"
            x_center: x coordinate of the center of the ball if there is a collision
            y_center: y coordinate of the center of the ball if there is a collision
            post_collision_path: The flight path of the ball after a collision.
    '''
    t = t_path(RR)
    y = evaluate(y_path, t_path, RR)
    if y < (HR - R) and t > start_time:
        return "undershot", None, None
    
    outcome, x_collision, y_collision, t_collision = doesBallCollideWithRim(x_path, y_path, t_path, start_time)

    if outcome == "score":
        return "score", None, None
    elif outcome == "miss":
        outcome, x_center, y_center, t_collision = findPositionOfBallCollidingWithBackboard(x_path, y_path, t_path, start_time)
        if outcome == "overshot":
            return "overshot", None, None
        elif outcome == "collision":
            collision_pnt = (-RR-DB, y_center, t_collision)
            #obstraction = "backboard"
        else: # outcome = "miss"
            return "miss", None, None
    else: # outcome == "collision"
        x_center, y_center, t_collision = findPositionOfBallCollidingWithRimEdge(x_path, y_path, t_path, x_collision, y_collision, start_time)
        collision_pnt = (x_collision, y_collision, t_collision)
        
    xs, ys, ts = predictFlightPathPostCollision(x_path, y_path, t_path, (x_center, y_center), collision_pnt)

    collision_pnts = []
    collision_pnts.append((x_center, y_center))

    if outcome == "collision":
        N = 10 # Number of samples
        x_path = Poly.fit(ts[0:N], xs[0:N], deg=1, domain=x_path.domain, window=x_path.window)
        t_path = Poly.fit(xs[0:N], ts[0:N], deg=1, domain=t_path.domain, window=t_path.window)
        # Simplification: Otherwise we have to solve a constrained optimization problem
        y_path = Poly.fit(ts[0:N], ys[0:N], deg=1, domain=y_path.domain, window=y_path.window)

        out_, collision_pnts_, paths = predictOutcome(x_path, y_path, t_path, (ts[0] + ts[1])/2)
                
        if paths:
            collision_pnts.extend(collision_pnts_)

            x_, y_, t_ = paths

            condition = ts <= t_[0]            
            xs = xs[condition]
            ys = ys[condition]
            ts = ts[condition]
            
            xs = np.hstack([xs, x_])
            ys = np.hstack([ys, y_])
            ts = np.hstack([ts, t_])

        return out_, collision_pnts, (xs, ys, ts)

    return outcome, collision_pnts, (xs, ys, ts)


def doesBallCollideWithRim(x_path: Poly, y_path: Poly, t_path: Poly, start_time):
    '''
        We want find to the point of intersection of two equations
           y = HR (The height of the rim)
           y = path

        Output:
            outcome: "score" | "collision" | "miss",
            x_rim, y_rim: An edge of the rim that the ball collides with,
            t: The time at which the collision takes place.
    '''
    obj = y_path - HR

    roots_t = obj.roots()

    condition = np.all([np.isreal(roots_t), roots_t>start_time], axis=0)
    real_roots_t = np.extract(condition, roots_t)

    if real_roots_t.size == 0:
        return "miss", None, None, None

    real_roots_x = x_path(real_roots_t)
    # Distance from the right edge of the rim to the center of the ball
    d1 = (real_roots_x <= ( RR - R))
    # Distance from the left edge of the rim to the center of the ball
    d2 = (real_roots_x >= (-RR + R))
    condition = np.all([d1, d2], axis=0)
    pnts = np.extract(condition , real_roots_x)

    if pnts.size:
        return "score", None, None, None
    else:
        # Notice how we include points that are just outside the rim by extending the search window by the 
        # radius of the ball R
        condition = np.all([real_roots_x >= -RR, real_roots_x <= (RR + R)], axis=0)
        collision_pnts = np.extract(condition , real_roots_x)
        if collision_pnts.size:
            # If we have two values, we select the one that results in the first collision
            real_roots_t = t_path(collision_pnts)
            
            pnt_t = np.min(real_roots_t)

            if x_path(pnt_t) >= 0: 
                return "collision", RR, HR, pnt_t
            else:
                return "collision", -RR, HR, pnt_t
        else:
            # This should not happen    
            return "miss", None, None, None


def findPositionOfBallCollidingWithRimEdge(x_path: Poly, y_path: Poly, t_path: Poly, x_rim, y_rim, start_time):
    '''
        Input:
            x_path: 
            y_path:
            t_path:
            (x_rim, y_rim): The cartesian coordinate of an edge of the rim
            start_time:
    '''
    # Let the Polynomial package do the heavy lifting
    #   Construct the equation of a circle using the Polynomial package
    #       (x - x_c)**2 + (y - y_c)**2 = R*R
    #       where x = x_rim , y = y_rim and R is the radius of the ball
    #       x_c is unknown and y_c = path 

    p1 = x_path - x_rim
    p1 = p1 * p1
    
    p2 = y_path - y_rim
    p2 = p2 * p2
    
    p = p1 + p2 - R*R
    
    roots_t = p.roots()
    condition = np.all([np.isreal(roots_t), roots_t>start_time], axis=0)
    real_roots_t = np.extract(condition, roots_t)

    assert real_roots_t.size>0, "We should always have a root at this point."

    # We can have at most two collision points at a time
    # We are interested in the collision that takes place first
    root_t = np.min(real_roots_t)
    root_t = np.real(root_t)

    return x_path(root_t), y_path(root_t), root_t


def findPositionOfBallCollidingWithBackboard(x_path: Poly, y_path: Poly, t_path: Poly, start_time):
    '''
        Find the point of contact of a circle and a straight line (x=x0)
        Also, we know that the y value of the center of the ball is modelled by the given polynomial (path)

        Input: 
            x_path: The polynomial that models the height of the center of the ball in time
            x_path: The poly that models the horizontal displacement of the center of the ball in time
            t_path: Time as a function of horizontal displacement. The inverse of x_path
            start_time: 

        Output:
            outcome: "overshot" | "collision" | "miss"
            x_c: The x coordinate of the ball when it first makes contact with the backboard
            y_c: The y coordinate of the ball when it first makes contact with the backboard
            t_c: The time the collision took place
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

    t_collision = t_path(x_c)

    if t_collision <= start_time:
        return "miss", None, None, None

    y_c = y_path(t_collision)

    if y_c > (HR + HB):
        return "overshot", None, None, None
    else:
        return "collision", x_c, y_c, t_path(x_c)


def predictFlightPathPostCollision(x_path: Poly, y_path: Poly, t_path: Poly, center, collision_pnt):
    '''
        Input:
            path: The polynomial that models the path followed by a ball,
            center = (x_c, y_c): The center of the ball when the collision is predicted to happen
            collision_pnt = (x_r, y_r): The edge of the rim against which the ball is colliding
        Output:
            xs, ys, ts: A set of points that model the path followed by the ball after the collision
    '''
    x_center, y_center = center
    x_r, y_r, t_collision = collision_pnt
    
    # Gradient at the point of collision
    ball_tangent_grad = - (x_r - x_center) / (y_r - y_center)

    # Explanation?
    # Compute the normal of the collision surface
    nx_circle, ny_circle = normal(y_path, t_path, ball_tangent_grad, x_r, y_r)
    deriv = y_path.deriv()
    deriv.window =  y_path.window
    deriv.domain =  y_path.domain
    path_tangent_grad = evaluate(deriv, t_path, x_center)
    nx_path, ny_path = vector(x_path, y_path, path_tangent_grad, x_center, y_center, t_collision)

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
    # The end time of our simulation is not easy to estimate because we do not even know the unit of time!
    start = t_collision
    end = 5*t_collision
    num_points = np.ceil( (end - start) / (DELTA_T/10) + 1)
    ts, dt = np.linspace(start, end, int(num_points), retstep=True)
    xs = x_path(ts)
    ys = y_path(ts)

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

    if debug:
        plotDebugGraphs(
            x_path,
            y_path,
            t_path,
            ts,
            trans_xs, trans_ys,
            reflected_xs, reflected_ys,
            new_xs, new_ys,
            post_collision_xs, post_collision_ys,
            collision_pnt, center,
            ball_tangent_grad)

    return post_collision_xs, post_collision_ys, ts


def normal(y_path: Poly, t_path: Poly, grad, x, y):
    grad = -1 / grad
    
    return _vector(grad, x, y, flight_path=False, y_path=y_path, t_path=t_path)


def vector(x_path: Poly, y_path: Poly, grad, x_before, y_before, t_collision):
    return _vector(grad,
                   x_before, 
                   y_before, 
                   flight_path=True, 
                   x_path=x_path, 
                   y_path=y_path, 
                   t_collision=t_collision)


def _vector(grad, x_before, y_before, flight_path=True, x_path=None, y_path=None, t_path=None, t_collision=None):
    # y = mx + c
    c = y_before - grad*x_before

    # We need two point to compute a normal
    if flight_path:
        # Take a step forward in time...
        t = t_collision + 10

        x_after = x_path(t)
        y_after = y_path(t)
    else:
        # The normal of the wall should always point (relatively) towards +x-axis
        x_after = x_before + 5

        y_after = evaluate(y_path, t_path, x_after)
    
    dx = x_after - x_before
    dy = y_after - y_before
    dist = np.sqrt(dx*dx + dy*dy)

    return dx/dist, dy/dist


def translate(xs, ys, trans):
    return xs + trans[0], ys + trans[1]


def plotDebugGraphs(x_path: Poly,
                    y_path: Poly,
                    t_path: Poly,
                    ts,
                    trans_xs, trans_ys,
                    reflected_xs, reflected_ys,
                    new_xs, new_ys,
                    post_collision_xs, post_collision_ys,
                    collision_point, center,
                    ball_tangent_grad):
    figure, axes = plt.subplots()
    axes.set_aspect(1)

    # We need this in order to display the entire flight path of the ball
    axes.plot(x_path(ts), y_path(ts), label="1. Path extrapolation beyond collision")

    axes.plot(trans_xs, trans_ys, '+', label="2. Translated and rotated")

    axes.plot(reflected_xs, reflected_ys, label="3. Reflection")

    axes.plot(new_xs, new_ys, 'o', label="4. Undo rotation")

    axes.plot(post_collision_xs, post_collision_ys, label="5. Post-collision path")

    # The tangent of the ball at the point of colllision
    x_r, y_r, _ = collision_point
    if np.isfinite(ball_tangent_grad):
        c = y_r - ball_tangent_grad*x_r
        axes.plot([-1.0, 1.0], [-ball_tangent_grad + c, ball_tangent_grad + c], label="Tanget of ball (Wall)")
    else:
        axes.plot([-RR-DB, -RR-DB], [3.05, 4.2], label="Tanget of ball")

    # We want to plot the line that estimates the path followed by the ball at the time of collision
    x_center, y_center = center
    
    deriv = y_path.deriv()
    deriv.window = y_path.window
    deriv.domain = y_path.domain
    path_tangent_grad = evaluate(deriv, t_path, x_center)
    
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
    axes.set_title("Generating post-collision path")

    filename = os.path.join(images_folder, debug_folder, f"image-{time.time()}.png")
    plt.savefig(filename)
    plt.show()


def rotate(xs, ys, theta):
    reflected_coords = np.vstack([xs, ys])
    reflected_coords = np.hsplit(reflected_coords, xs.size)

    reverse_rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    coords_post_collision = reverse_rot@reflected_coords 

    new_coords = np.hstack(coords_post_collision)

    new_xs, new_ys = np.vsplit(new_coords, 2)

    return new_xs[0], new_ys[0]


def createImagesDirectory(images_folder):
    try:
        os.mkdir(images_folder)
    except Exception as exp:
        print(exp)

    if debug:
        # Create a folder for each of the basketballs
        for i in range(1, B+1):
            debug_folder = f"ball-{i}"
            
            try:
                foldername = os.path.join(images_folder, debug_folder)
                os.mkdir(foldername)
            except Exception as exp:
                print(exp)


def isDebugModeEnabled(argv):
    debug = False
    
    if len(argv) > 1:
        if argv[1] == "-d":
            debug = True
        else:
            print()
            print("Help")
            print("To run in debug mode: python balls.py -d")
            print("To run in release mode: python balls.py")

            exit()

    return debug


def controller(s1_label, s2_label):
    dist1 = readDistances(df, s1_label)

    dist2 = readDistances(df, s2_label)

    x, y = transformToCartesianCoordinates(dist1, dist2)

    # We need parametric models of the flight path
    global DELTA_T
    t, DELTA_T = np.linspace(0, x.size/2, num=x.size, retstep=True)

    # (Least squares) Fit the data to a second order polynomial
    start = x[0] - 1
    x_poly = Poly.fit(t, x, deg=1, domain=[0, 10000], window=[0, 10000])
    # As much as we want to use parametric representation throughout, some of our data is 
    # in (x, y) coordinate frame. We want this intermidiate representation to make it easy to 
    # link the two spaces. 
    # That is, if we have x then what is the corresponding value of y?

    t_poly = Poly.fit(x, t, deg=1, domain=[-10*start, 10*start], window=[-10*start, 10*start])
    y_ = Poly.fit(x, y, deg=2, domain=[-10*start, 10*start], window=[-10*start, 10*start])
    y_poly = y_(x_poly)

    # If x_c and y_c exist then they represent the center of the ball that 
    # collides with either the ring or backboard
    outcome, collision_pnts, post_collision_path = predictOutcome(x_poly, y_poly, t_poly, start_time=0)

    return outcome, collision_pnts, post_collision_path, (x_poly, t_poly, y_poly), (x, y)


def displayCompletFlightPath(original_data, outcome, pre_collision_path, collision_pnts, post_collision_path):
    x, y = original_data

    x_poly, t_poly, y_poly = pre_collision_path
    
    if post_collision_path:
        xs, ys, ts = post_collision_path

    xs = np.linspace(-1, x[0], num=50)
    # However, if there is a collision, then set want to clip the predicted path at the point of collision.
    if collision_pnts:
        pnt = collision_pnts[0]
        xs = np.linspace(pnt[0], x_poly(0), num=50)

    figure, axes = plt.subplots()
    axes.set_aspect(1)

    ys = evaluate(y_poly, t_poly, xs)
    axes.plot(xs, ys, label="Estimate flight path")
    axes.plot(x, y, label="Measured fligh path")
    # Plot a line that represents the rim
    axes.plot([-RR, RR], [3.05, 3.05], label="Rim")

    # Plot the line that represents the backboard
    axes.plot([-RR-DB, -RR-DB], [HR, HR+HB], label="Backboard")

    if debug:
        y_rr = evaluate(y_poly, t_poly, RR)
        circle_l = patches.Circle( (RR, y_rr), R, fill=False, label="Ball(First rim edge)", color="green")
        axes.add_artist(circle_l)

        y_rr = evaluate(y_poly, t_poly, -RR)
        circle_2 = patches.Circle( (-RR, y_rr), R, fill=False, label="Second rim edge", color="blue")
        axes.add_artist(circle_2)

    if collision_pnts:
        for x_c, y_c in collision_pnts:
            circle = patches.Circle((x_c, y_c), R, fill=False, label="Ball(s) at collision point", color="red")
            axes.add_artist(circle)

        xs, ys, ts = post_collision_path
        condition = np.all([xs >= -1, xs <= x[0], ys >= 0], axis=0)
        xs = xs[condition]
        ys = ys[condition]

        axes.plot(xs, ys, '*', label="Post collision path")

    axes.legend(shadow=True)
    
    axes.set_title(f"Ball {i} {outcome}")
    filename = os.path.join(images_folder, f"Ball {i} Flight Path")
    plt.savefig(filename)
    plt.show()





debug = isDebugModeEnabled(sys.argv)
print("Debug mode: ", debug)

# Number of balls
B = 6

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

# Time between measurements
# It is set when we first model the given data
DELTA_T = 0

df = pd.read_csv("basketball.csv", sep=', ', engine="python")

# Keep a list of assessments and write them to file when done.
assessments = []

# A global folder name to store debug figures for a given ball.
debug_folder = ""

# A folder for all images
images_folder = "images"
createImagesDirectory(images_folder)

for i in range(1, B+1):
    debug_folder = f"ball-{i}"

    outcome, collision_pnts, post_collision_path, pre_collision_path, original_data = controller(f"b{i}_s1", f"b{i}_s2")

    print("\n\nBall: ", i)
    print("               ", outcome)
    assessments.append(outcome)

    displayCompletFlightPath(original_data, outcome, pre_collision_path, collision_pnts, post_collision_path)


print("Write assessments to file")
with open("assessment.txt", "w") as file:
    for idx, outcome in enumerate(assessments, start=1):
        print(f"{idx} {outcome}", file=file)

