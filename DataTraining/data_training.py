import csv
import subtractive_clustering
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from pybrain.tools.shortcuts import buildNetwork
from pybrain import structure as structure
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

left_arm = []
right_arm = []
colors =['b','g','r','c']

dimension = 4
input_neurons = 0
output_neurons = 0
hidden_neurons = 1

show_plot = False

epochs = 10
sigma_increment_factor = 0.01

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
    radius = 0.45

    train_neuro_fuzzy(left_arm_array, radius,"DataTraining/Trained/Left_Arm_NN.pickle")
    train_neuro_fuzzy(right_arm_array, radius,"DataTraining/Trained/Right_Arm_NN.pickle")

def train_neuro_fuzzy(arm_data, radius,NeuralNetworkName):
    #  Fuzzy
    clusters=[]
    sigmas=[]
    fuzzy_sets = []
    fuzzy_set_range = []
    for i in range(0,dimension,1):
        cluster, sigma = get_cluster(np.ravel(arm_data[:, :, i]), radius)
        sets, x = get_fuzzy_membership(cluster, sigma, np.ravel(arm_data[:, :, i]))
        fuzzy_sets.append(sets)
        fuzzy_set_range.append(x)
        clusters.append(cluster)
        sigmas.append(sigma)

    #adjustment of sigma and cluster center.
    adust_sigma(arm_data, clusters, fuzzy_set_range, fuzzy_sets, sigmas)


    # Neural Network
    global output_neurons, input_neurons,hidden_neurons
    input_neurons = 0
    output_neurons = 0
    hidden_neurons = 1
    output_neurons = len(arm_data)
    for fuzzy_set in fuzzy_sets:
        hidden =0
        for membership in fuzzy_set:
            input_neurons += 1
            hidden += 1
        hidden_neurons *= hidden
    neural_net, data_set = build_neural_network()

    # Generate Neural Network Training Data
    input_data, output_data = generate_data(arm_data, fuzzy_set_range, fuzzy_sets)
    for i in np.arange(len(output_data)):
        data_set.addSample(input_data[i], output_data[i])
    # Training the neural network
    trainer = BackpropTrainer(neural_net, data_set)
    for i in range(epochs):
        error = trainer.train()
        print "epoch ", i, " error : " , error

    import pickle
    fileObject = open(NeuralNetworkName, 'w')
    pickle.dump(neural_net, fileObject)
    fileObject.close()


def adust_sigma(arm_data, clusters, fuzzy_set_range, fuzzy_sets, sigmas):
    global show_plot
    for x in range(10):
        for pose in arm_data:
            for j in range(0, dimension, 1):
                dim = pose[:, j]
                fuzzy_rules = fuzzy_sets[j]
                frequency = np.zeros(len(fuzzy_rules))

                if show_plot:
                    ax = plt.figure().add_subplot(111)
                    for rule in fuzzy_rules:
                        ax.plot(fuzzy_set_range[j], rule, '-')

                for entry in dim:
                    result = []
                    for rule in fuzzy_rules:
                        temp = fuzz.interp_membership(fuzzy_set_range[j], rule, entry)
                        result.append(temp)
                    max_index = np.array(result).argmax()
                    frequency[max_index] += 1

                    if result[max_index] < 0.1:  # sigma is really thin...
                        sigmas[j][max_index] += sigma_increment_factor
                        # update fuzzy rule
                        fuzzy_sets[j][max_index] = fuzz.gaussmf(fuzzy_set_range[j], clusters[j][max_index], sigmas[j][max_index])
                    if show_plot:
                        ax.plot(entry, 0.5, '.', color=colors[max_index])

                        # plot the data
                        # update sigma and centroids of cluster
                total = frequency.sum()
                percent_max = frequency.max() / total
                index_max = frequency.argmax()
                print frequency
                if percent_max != 1:
                    pass
        if show_plot:
            plt.show()


def generate_data(arm_data, fuzzy_set_range, fuzzy_sets):
    global output_neurons, input_neurons

    output_data = []
    input_data = []
    for i, pose in enumerate(arm_data):
        output = np.zeros(len(arm_data))
        output[i] = 1  # target matrix...

        for k in range(0, len(pose.tolist()), 1):
            output_data.append(output)

        # create input matrix for each pose
        for j in range(0, len(pose), 1):
            entry = pose[i, :]
            rule_data = []
            for k, dim in enumerate(entry):
                fuzzy_rules = fuzzy_sets[k]
                for rule in fuzzy_rules:
                    temp = fuzz.interp_membership(fuzzy_set_range[k], rule, dim)
                    rule_data.append(temp)
            input_data.append(rule_data)

    # reformat the input data


    return input_data, output_data

def build_neural_network():
    global input_neurons, output_neurons,hidden_neurons
    net = buildNetwork(input_neurons, hidden_neurons, output_neurons, hiddenclass=structure.TanhLayer, outclass=structure.SoftmaxLayer)
    ds = SupervisedDataSet(input_neurons, output_neurons)
    return net, ds

def get_cluster(arr, radius):
    cluster = subtractive_clustering.subtractive_clustering(arr, radius)
    sigma = np.absolute((radius * arr.max() - arr.min()) / np.sqrt(8))
    print "Done Cluster Centers found are... ", len(cluster)
    print "Done Sigma Value... ", sigma
    sigma_list =[]
    for i in range(len(cluster)):
        sigma_list.append(sigma)
    return cluster, sigma_list

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

def get_fuzzy_membership(centers, sigma, array):
    membership = []
    x = np.linspace((array.min() - 10), (array.max() + 10), 500)
    for i, center in enumerate(centers):
        rule = fuzz.gaussmf(x, center, sigma[i])
        membership.append(rule)

    return membership, x

def plot_membership(membership):
    ax = plt.figure().add_subplot(111)
    for rule in membership:
        ax.plot(rule, '-')
    plt.show()
