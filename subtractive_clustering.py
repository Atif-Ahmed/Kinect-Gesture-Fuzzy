import numpy as np
import matplotlib.pyplot as plt

def subtractive_clustering(points_original, radius):
    ra = radius
    rb = 1.25 * radius
    alpha = 4 / (ra ** 2)
    beta = 4 / (rb ** 2)

    accept_threshold = 0.5
    reject_threshold = 0.15

    cluster_center_indexes = []

    (length, dim) = np.shape(points_original)
    points_array = np.array(points_original)
    points_normalized = np.zeros(np.shape(points_original))

    for i in range(0, dim, 1):
        column = normalize(points_array[:, i])
        points_normalized[:, i] = column

    mountain_map = np.zeros(length)
    #################################################################
    ################# STEP 1  CREATE MOUNTAIN  ######################
    counter = 0
    for ref in points_normalized:
        for point in points_normalized:
            distance = np.linalg.norm(ref - point)
            mountain_map[counter] = mountain_map[counter] + np.exp(-alpha * distance)
        counter += 1

    #################################################################
    ################# STEP 2  CRUSH MOUNTAIN  #######################
    first_cluster_value = 0
    first_cluster = False

    while mountain_map.max() > 0:
        current_cluster_center_index = mountain_map.argmax()
        current_cluster_value = mountain_map.max()
        if not first_cluster:
            first_cluster_value = current_cluster_value
            first_cluster = True
        if current_cluster_value / first_cluster_value > accept_threshold:  # New cluster...
            cluster_center_indexes.append(current_cluster_center_index)

            for i, point in enumerate(points_normalized):
                distance = np.linalg.norm(points_normalized[current_cluster_center_index] - point)
                mountain_map[i] = mountain_map[i] - current_cluster_value * np.exp(-beta * distance)
                mountain_map[mountain_map < reject_threshold * first_cluster_value] = 0

        else:
            # compute distance from already created clusters.
            nearest_cluster_distance = 1e50
            for center in cluster_center_indexes:
                distance = np.linalg.norm(points_normalized[current_cluster_center_index] - points_normalized[center])
                if distance < nearest_cluster_distance:
                    nearest_cluster_distance = distance

            if (nearest_cluster_distance / ra + current_cluster_value / first_cluster_value) >= 1:  # New Cluster....
                cluster_center_indexes.append(current_cluster_center_index)
                for i, point in enumerate(points_normalized):
                    distance = np.linalg.norm(points_normalized[current_cluster_center_index] - point)
                    mountain_map[i] = mountain_map[i] - current_cluster_value * np.exp(-beta * distance)
                    mountain_map[mountain_map < reject_threshold * first_cluster_value] = 0
            else:
                for i, point in enumerate(points_normalized):
                    distance = np.linalg.norm(points_normalized[current_cluster_center_index] - point)
                    mountain_map[i] = mountain_map[i] - current_cluster_value * np.exp(-beta * distance)
                    mountain_map[mountain_map < reject_threshold * first_cluster_value] = 0

    # ######### de-normalize center points ##################
    final_center_indexes = []
    for center in cluster_center_indexes:
        final_center_indexes.append(points_original[center])
    return final_center_indexes

def normalize(v):
    v_new = np.zeros(0)
    for value in v:
        temp = (value - v.min()) / (v.max() - v.min())
        v_new = np.hstack((v_new, temp))
    return v_new
