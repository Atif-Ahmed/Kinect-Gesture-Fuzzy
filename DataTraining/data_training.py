import csv
import subtractive_clustering
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

left_arm = []
right_arm = []

left_arm_fuzzy_set =[]
right_arm_fuzzy_set =[]

poses = 6

def start():

    read_csv("DataCollection/Data/Side/data.csv")
    read_csv("DataCollection/Data/Up/data.csv")
    read_csv("DataCollection/Data/Front/data.csv")
    read_csv("DataCollection/Data/Down/data.csv")
    read_csv("DataCollection/Data/Bend Up/data.csv")
    read_csv("DataCollection/Data/Bend Front/data.csv")

    # Finding optimum number of clusters for each dimension.. for left and right hand.....
    print "Starting subtractive Clustering Process"
    left_arm_array = np.array(left_arm)
    right_arm_array = np.array(right_arm)
    radius = 0.5

    ########################################################################################################

    # Left Hand Clustering
    cluster_left_1, sigma_left_1 = getCluster(np.ravel(left_arm_array[:, :, 0]), radius)
    cluster_left_2, sigma_left_2 = getCluster(np.ravel(left_arm_array[:, :, 1]), radius)
    cluster_left_3, sigma_left_3 = getCluster(np.ravel(left_arm_array[:, :, 2]), radius)
    cluster_left_4, sigma_left_4 = getCluster(np.ravel(left_arm_array[:, :, 3]), radius)

    # Right Hand Clustering
    cluster_right_1, sigma_right_1 = getCluster(np.ravel(right_arm_array[:, :, 0]), radius)
    cluster_right_2, sigma_right_2 = getCluster(np.ravel(right_arm_array[:, :, 1]), radius)
    cluster_right_3, sigma_right_3 = getCluster(np.ravel(right_arm_array[:, :, 2]), radius)
    cluster_right_4, sigma_right_4 = getCluster(np.ravel(right_arm_array[:, :, 3]), radius)

    ########################################################################################################

    # Left Hand membership
    global left_arm_fuzzy_set
    left_arm_fuzzy_set.append(getFuzzyMembership(cluster_left_1, sigma_left_1, left_arm_array[:, :, 0]))
    left_arm_fuzzy_set.append(getFuzzyMembership(cluster_left_2, sigma_left_2, left_arm_array[:, :, 1]))
    left_arm_fuzzy_set.append(getFuzzyMembership(cluster_left_3, sigma_left_3, left_arm_array[:, :, 2]))
    left_arm_fuzzy_set.append(getFuzzyMembership(cluster_left_4, sigma_left_4, left_arm_array[:, :, 3]))

    # Right Hand membership
    global right_arm_fuzzy_set
    right_arm_fuzzy_set.append(getFuzzyMembership(cluster_right_1, sigma_right_1, right_arm_array[:, :, 0]))
    right_arm_fuzzy_set.append(getFuzzyMembership(cluster_right_2, sigma_right_2, right_arm_array[:, :, 1]))
    right_arm_fuzzy_set.append(getFuzzyMembership(cluster_right_3, sigma_right_3, right_arm_array[:, :, 2]))
    right_arm_fuzzy_set.append(getFuzzyMembership(cluster_right_4, sigma_right_4, right_arm_array[:, :, 3]))









def getCluster(arr, radius):
    cluster = subtractive_clustering.subtractive_clustering(arr, radius)
    sigma = (radius * arr.max() - arr.min()) / np.sqrt(8)
    print "Done Cluster Centers found are... ", len(cluster)
    print "Done Sigma Value... ", sigma

    return cluster, sigma


def read_csv(path):
    global left_arm
    global right_arm
    data = []
    with open(path, "rb") as f:
        reader = csv.reader(f)
        for data in reader:
            pass

    temp_left_arm = []
    temp_right_arm = []

    for entry in data:
        for ch in ["(", ")", "[", "]"]:
            entry = entry.replace(ch, "")
        entries = entry.split(", ")
        left = [float(entries[0]), float(entries[1]), float(entries[2]), float(entries[3])]
        right = [float(entries[4]), float(entries[5]), float(entries[6]), float(entries[7])]
        temp_left_arm.append(left)
        temp_right_arm.append(right)

    left_arm.append(temp_left_arm)
    right_arm.append(temp_right_arm)


def getFuzzyMembership(centers, sigma, array):
    membership = []
    x = np.linspace((array.min() - 10), (array.max() + 10), 500)
    for center in centers:
        rule = fuzz.gaussmf(x, center, sigma)
        membership.append(rule)

    return membership


def plotMembership(membership):
    ax = plt.figure().add_subplot(111)
    for rule in membership:
        ax.plot(rule, '-')
    plt.show()
