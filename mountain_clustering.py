import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d


# not a very efficient implementation.. not using threading or anything.
def mountain_clustering_2d(x_pts_original, y_pts_original, grid_size, radius, show_plot=False):
    ra = radius
    rb = 1.5 * radius
    alpha = 4 / (ra ** 2)
    beta = 4 / (rb ** 2)

    accept_threshold = 0.5
    reject_threshold = 0.15

    cluster_center_x = []
    cluster_center_y = []

    x_pts = normalize(x_pts_original)
    y_pts = normalize(y_pts_original)

    #################################################################
    ################# STEP 1  CREATE GRID    ########################
    #################################################################

    intervals = np.linspace(0, 1, grid_size)

    mountain_graph = np.zeros((grid_size, grid_size))

    #################################################################
    ################# STEP 2  CREATE MOUNTAIN  ######################
    #################################################################
    for i, x_node in enumerate(intervals):
        for j, y_node in enumerate(intervals):
            for (x, y) in zip(x_pts, y_pts):
                distance = np.sqrt((x_node - x) ** 2 + (y_node - y) ** 2)
                mountain_graph[i][j] = mountain_graph[i][j] + np.exp(-alpha * distance)

    if show_plot:
        plot_mountain(mountain_graph, intervals, x_pts, intervals, y_pts)
    #################################################################
    ################# STEP 2  CRUSH MOUNTAIN  #######################
    #################################################################
    first_cluster_max = 0
    first_cluster = False
    while mountain_graph.max() > 0:
        row_center, column_center = np.unravel_index(mountain_graph.argmax(), mountain_graph.shape)
        max_value = mountain_graph[row_center][column_center]

        if not first_cluster:
            first_cluster_max = max_value
            first_cluster = True

        if max_value / first_cluster_max > accept_threshold:
            cluster_center_x.append(intervals[column_center])
            cluster_center_y.append(intervals[row_center])
            destroy_mountain(alpha, beta, column_center, first_cluster_max, intervals, max_value, mountain_graph, reject_threshold, row_center)

        else:
        ## Compute the minimum distance.
            dis = np.array(np.sqrt(
                np.power(np.subtract(cluster_center_x, intervals[column_center]), 2) +
                np.power(np.subtract(cluster_center_y, intervals[row_center]), 2)))
            if (dis.min() / ra + max_value / first_cluster_max) >= 1:
                cluster_center_x.append(intervals[column_center])
                cluster_center_y.append(intervals[row_center])
                destroy_mountain(alpha, beta, column_center, first_cluster_max, intervals, max_value, mountain_graph, reject_threshold, row_center)
            else:
                destroy_mountain(alpha, beta, column_center, first_cluster_max, intervals, max_value, mountain_graph, reject_threshold, row_center)

        if show_plot:
            plot_mountain(mountain_graph, intervals, x_pts, intervals, y_pts)

    if show_plot:
        ax = plt.figure().add_subplot(111)
        ax.plot(y_pts, x_pts, '.', color='g')
        ax.plot(cluster_center_x, cluster_center_y, '.', color='r')
        plt.show()

    ######### de-normalize center points.....
    final_cluster_x = []
    final_cluster_y = []
    for (y, x) in zip(cluster_center_x, cluster_center_y):
        final_cluster_y.append(x * (x_pts_original.max() - x_pts_original.min()) + x_pts_original.min())
        final_cluster_x.append(y * (y_pts_original.max() - y_pts_original.min()) + y_pts_original.min())

    return zip(final_cluster_x, final_cluster_y)


def destroy_mountain(alpha, beta, column_center, first_cluster_max, intervals, max_value, mountain_graph, reject_threshold, row_center):
    for i, row in enumerate(intervals):
        for j, column in enumerate(intervals):
            distance = np.sqrt((intervals[row_center] - row) ** 2 + (intervals[column_center] - column) ** 2)
            mountain_graph[i][j] = mountain_graph[i][j] - max_value * np.exp(-(beta * distance) ** 4 / alpha)
    mountain_graph[mountain_graph < reject_threshold * first_cluster_max] = 0


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
