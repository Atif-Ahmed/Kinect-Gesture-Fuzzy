from pykinect.nui import JointId as Joint_Id
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import skfuzzy as fuzz

arm_joint_index = {'left_shoulder': 0,
                   'right_shoulder': 1,
                   'left_elbow': 2,
                   'right_elbow': 3,
                   'left_wrist': 4,
                   'right_wrist': 5}

arm_spherical = []


# fetch the arm position of the system.
def get_arm_locations(_skeleton):
    arm_pos_cartesian = []

    arm_pos_cartesian.append((_skeleton[Joint_Id.shoulder_left].x, _skeleton[Joint_Id.shoulder_left].y, _skeleton[Joint_Id.shoulder_left].z))
    arm_pos_cartesian.append((_skeleton[Joint_Id.shoulder_right].x, _skeleton[Joint_Id.shoulder_right].y, _skeleton[Joint_Id.shoulder_right].z))
    arm_pos_cartesian.append((_skeleton[Joint_Id.elbow_left].x, _skeleton[Joint_Id.elbow_left].y, _skeleton[Joint_Id.elbow_left].z))
    arm_pos_cartesian.append((_skeleton[Joint_Id.elbow_right].x, _skeleton[Joint_Id.elbow_right].y, _skeleton[Joint_Id.elbow_right].z))
    arm_pos_cartesian.append((_skeleton[Joint_Id.wrist_left].x, _skeleton[Joint_Id.wrist_left].y, _skeleton[Joint_Id.wrist_left].z))
    arm_pos_cartesian.append((_skeleton[Joint_Id.wrist_right].x, _skeleton[Joint_Id.wrist_right].y, _skeleton[Joint_Id.wrist_right].z))

    return arm_pos_cartesian


def cartesian2spherical(xyz):
    ptsnew = np.zeros(xyz.shape)
    xy = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
    ptsnew[:, 0] = np.sqrt(xy + xyz[:, 2] ** 2)
    ptsnew[:, 1] = np.arctan2(np.sqrt(xy), xyz[:, 2])  # for elevation angle defined from Z-axis distance from Kinect
    ptsnew[:, 2] = np.arctan2(xyz[:, 1], xyz[:, 0])
    return ptsnew


def apply_fuzzy(skeleton):
    pass


def test_fuzzy():
    colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen']

    # Define three cluster centers
    centers = [[4, 2],
               [1, 7],
               [5, 6]]

    # Define three cluster sigmas in x and y, respectively
    sigmas = [[0.8, 0.3],
              [0.3, 0.5],
              [1.1, 0.7]]

    # Generate test data
    np.random.seed(42)  # Set seed for reproducibility
    xpts = np.zeros(1)
    ypts = np.zeros(1)
    labels = np.zeros(1)
    for i, ((xmu, ymu), (xsigma, ysigma)) in enumerate(zip(centers, sigmas)):
        xpts = np.hstack((xpts, np.random.standard_normal(200) * xsigma + xmu))
        ypts = np.hstack((ypts, np.random.standard_normal(200) * ysigma + ymu))
        labels = np.hstack((labels, np.ones(200) * i))

    # Visualize the test data
    fig0, ax0 = plt.subplots()
    for label in range(3):
        ax0.plot(xpts[labels == label], ypts[labels == label], '.',
                 color=colors[label])
    ax0.set_title('Test data: 200 points x3 clusters.')
    plt.show()


def mountain_clustering_test():
    # experiment parameters...
    sigmas = [[0.8, 0.3], [0.3, 0.5], [1.1, 0.7]]
    centers = [[4, 2], [1, 7], [5, 16]]
    grid_size = 15
    pts_per_cluster = 200
    alpha = 20
    beta = 0.25

    # generation of rand clusters....
    x_pts = np.zeros(1)
    y_pts = np.zeros(1)

    for i, ((xmu, ymu), (x_sigma, y_sigma)) in enumerate(zip(centers, sigmas)):
        x_pts = np.hstack((x_pts, np.random.standard_normal(pts_per_cluster) * x_sigma + xmu))
        y_pts = np.hstack((y_pts, np.random.standard_normal(pts_per_cluster) * y_sigma + ymu))

    cluster_centers = mountain_clustering_2d(x_pts, y_pts, grid_size, alpha, beta, True)

    cluster_x =[]
    cluster_y =[]
    for (x,y) in cluster_centers:
        cluster_x.append(x)
        cluster_y.append(y)

    ax = plt.figure().add_subplot(111)
    ax.plot(y_pts, x_pts, '.', color='g')
    ax.plot(cluster_x,cluster_y, '.', color='r')
    plt.show()






# not a very efficient implementation.. not using threading or anything.
def mountain_clustering_2d(x_pts_original, y_pts_original, grid_size, alpha, beta, show_plot=False):
    cluster_center_x = []
    cluster_center_y = []

    x_pts = normalize(x_pts_original)
    y_pts = normalize(y_pts_original)

    #################################################################
    ################# STEP 1  CREATE GRID    ########################
    #################################################################

    x_intervals = np.linspace(x_pts.min(), x_pts.max(), grid_size)
    y_intervals = np.linspace(y_pts.min(), y_pts.max(), grid_size)

    mountain_graph = np.zeros((grid_size, grid_size))

    #################################################################
    ################# STEP 2  CREATE MOUNTAIN  ######################
    #################################################################
    for i, x_node in enumerate(x_intervals):
        for j, y_node in enumerate(y_intervals):
            for (x, y) in zip(x_pts, y_pts):
                distance = np.sqrt((x_node - x) ** 2 + (y_node - y) ** 2)
                mountain_graph[i][j] = mountain_graph[i][j] + np.exp(-alpha * distance)

    if show_plot:
        plot_mountain(mountain_graph, x_intervals, x_pts, y_intervals, y_pts)

    #################################################################
    ################# STEP 2  CRUSH MOUNTAIN  #######################
    #################################################################
    while (mountain_graph.max() > 0):
        center_x, center_y = np.unravel_index(mountain_graph.argmax(), mountain_graph.shape)
        max_value = mountain_graph[center_x][center_y]

        cluster_center_x.append(x_intervals[center_y])
        cluster_center_y.append(x_intervals[center_x])

        for i, x_node in enumerate(x_intervals):
            for j, y_node in enumerate(y_intervals):
                distance = np.sqrt((center_x - i) ** 2 + (center_y - j) ** 2)
                mountain_graph[i][j] = (mountain_graph[i][j] - max_value * np.exp(-beta * distance))
        # clip all negative values
        mountain_graph[mountain_graph < 0] = 0
        if show_plot:
            plot_mountain(mountain_graph, x_intervals, x_pts, y_intervals, y_pts)

    if show_plot:
        ax = plt.figure().add_subplot(111)
        ax.plot(y_pts, x_pts, '.', color='g')
        ax.plot(cluster_center_x, cluster_center_y, '.', color='r')
        # plt.show()

    ######### de-normalize center points.....
    final_cluster_x =[]
    final_cluster_y = []
    for (y, x) in zip(cluster_center_x, cluster_center_y):
        final_cluster_y.append(x*(x_pts_original.max() - x_pts_original.min())+ x_pts_original.min())
        final_cluster_x.append(y*(y_pts_original.max() - y_pts_original.min()) + y_pts_original.min())

    return zip(final_cluster_x, final_cluster_y)


def plot_mountain(mountain_graph, x_intervals, x_pts, y_intervals, y_pts):
    ax = plt.figure().add_subplot(111, projection='3d')
    X, Y = np.meshgrid(x_intervals, y_intervals)
    ax.plot_surface(X, Y, mountain_graph, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.plot(y_pts, x_pts, '.', color='g')  ## x and y axis are swapped.. in the 3D plot.. don't know why


def normalize(v):
    v_new = np.zeros(0)
    for value in v:
        temp = (value - v.min()) / (v.max() - v.min())
        v_new = np.hstack((v_new, temp))
    return v_new
