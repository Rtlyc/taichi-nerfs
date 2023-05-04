import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

def normalize(vector):
    """Normalize a 3D vector."""
    magnitude = np.linalg.norm(vector)
    normalized_vector = vector / magnitude
    return normalized_vector

def interpolate_arc(u, v, num_points=100):
    """Interpolate points along the circular arc between two unit vectors."""
    # Calculate the angle between u and v
    cos_angle = np.dot(u, v)
    angle = np.arccos(cos_angle)

    # Calculate the rotation axis (cross product of u and v)
    rotation_axis = normalize(np.cross(u, v))

    # Interpolate points along the arc
    points = []
    for t in np.linspace(0, angle, num_points):
        rot = R.from_rotvec(t * rotation_axis)
        point = rot.apply(u)
        points.append(point)

    return np.array(points)

if __name__ == '__main__':
    # Given unit vectors
    u = normalize(np.array([5, -2, 3]))
    v = normalize(np.array([-4, 1, 0]))

    # Generate interpolation points
    num_points = 3
    interpolate_points = interpolate_arc(u, v, num_points=num_points+2)

    # Generate arc points between u and v
    arc_points = interpolate_arc(u, v)

    # Plot the vectors and arc
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for vector, color, label in zip(interpolate_points[1:-1], ['grey']*num_points, ['p{}'.format(i) for i in range(num_points)]):
        ax.quiver(0, 0, 0, vector[0], vector[1], vector[2], color=color, label=label)

    for vector, color, label in zip([u, v], ['r', 'g'], ['u', 'v']):
        ax.quiver(0, 0, 0, vector[0], vector[1], vector[2], color=color, label=label)

    ax.plot(arc_points[:, 0], arc_points[:, 1], arc_points[:, 2], 'k--', label='Circular Trajectory')

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    plt.show()