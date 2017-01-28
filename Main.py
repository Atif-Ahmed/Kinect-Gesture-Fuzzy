from threading import Thread
from sklearn import datasets
import kinect_interface as kinect
import fuzzy_logic
import matplotlib.pyplot as plt
import numpy as np
import mountain_clustering as mount_cluster
import subtractive_clustering as subtractive_cluster
import DataCollection.data_collection as data_collection

data_collection_process = False
mountain_testing = False
subtractive_clustering = True
training = False


def start_kinect():
    loop = True
    kinect.init()
    while loop:
        events = kinect.window.event.get()
        for event in events:
            if event.type == kinect.window.QUIT:
                loop = False
            if event.type == kinect.window.KEYDOWN:
                if event.key == kinect.window.K_UP:
                    kinect.device_angle_up()
                if event.key == kinect.window.K_DOWN:
                    kinect.device_angle_down()


def mountain_clustering_test():
    # experiment parameters...
    sigmas = [[1.5, 0.2], [0.3, 0.5], [0.2, 0.9], [0.3, 0.5], [0.5, 0.8]]
    centers = [[4, 2], [1, 7], [5, 16], [3, 16], [10, 1]]
    grid_size = 30
    pts_per_cluster = 200
    radius = 0.30

    # generation of rand clusters....
    x_pts = np.zeros(1)
    y_pts = np.zeros(1)

    for i, ((xmu, ymu), (x_sigma, y_sigma)) in enumerate(zip(centers, sigmas)):
        x_pts = np.hstack((x_pts, np.random.standard_normal(pts_per_cluster) * x_sigma + xmu))
        y_pts = np.hstack((y_pts, np.random.standard_normal(pts_per_cluster) * y_sigma + ymu))

    cluster_centers = mount_cluster.mountain_clustering_2d(x_pts, y_pts, grid_size, radius)

    cluster_x = []
    cluster_y = []
    for (x, y) in cluster_centers:
        cluster_x.append(x)
        cluster_y.append(y)

    ax = plt.figure().add_subplot(111)
    ax.plot(y_pts, x_pts, '.', color='g')
    ax.plot(cluster_x, cluster_y, '.', color='r')
    plt.show()


def subtractive_clustering_test():
    # experiment parameters...
    sigmas = [[1.5, 0.2], [0.3, 0.5], [0.2, 0.9], [0.3, 0.5], [0.5, 0.8]]
    centers = [[4, 2], [1, 7], [5, 16], [3, 16], [10, 1]]
    grid_size = 30
    pts_per_cluster = 200
    radius = 0.5

    # generation of rand clusters....
    x_pts = np.zeros(1)
    y_pts = np.zeros(1)

    for i, ((xmu, ymu), (x_sigma, y_sigma)) in enumerate(zip(centers, sigmas)):
        x_pts = np.hstack((x_pts, np.random.standard_normal(pts_per_cluster) * x_sigma + xmu))
        y_pts = np.hstack((y_pts, np.random.standard_normal(pts_per_cluster) * y_sigma + ymu))

    cluster_centers = subtractive_cluster.subtractive_clustering (zip(x_pts, y_pts), radius)

    cluster_x = []
    cluster_y = []
    for (x, y) in cluster_centers:
        cluster_x.append(x)
        cluster_y.append(y)

    ax = plt.figure().add_subplot(111)
    ax.plot(y_pts, x_pts, '.', color='g')
    ax.plot(cluster_y, cluster_x, '.', color='r')
    plt.show()


def main():
    if data_collection_process:
        thread_kinect = Thread(target=start_kinect)
        thread_kinect.start()
        data_collection.start()

    if mountain_testing:
        mountain_clustering_test()

    if training:
        pass

    if subtractive_clustering:
        subtractive_clustering_test()


if __name__ == '__main__':
    main()
