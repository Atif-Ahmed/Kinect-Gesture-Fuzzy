import csv
import subtractive_clustering
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from pybrain.tools.shortcuts import buildNetwork
from pybrain import structure as structure
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
import collections

left_arm = []
right_arm = []
colors = ['b', 'g', 'r', 'c', 'm']

dimension = 4
input_neurons = 0
output_neurons = 0
hidden_neurons = 1
show_plot = 1
epochs = 10
gain = 25
radius = 0.50

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


    train_neuro_fuzzy(left_arm_array, radius, "DataTraining/Trained/Left_Arm_NN.pickle")
    train_neuro_fuzzy(right_arm_array, radius, "DataTraining/Trained/Right_Arm_NN.pickle")


def get_direction_clusters(param):
    counter = collections.Counter(param)
    sigma = np.ones(len(counter.keys()))
    return counter.keys(), sigma


def train_neuro_fuzzy(arm_data, radius, NeuralNetworkName):
    #  Fuzzy
    clusters = []
    sigmas = []
    fuzzy_sets = []
    fuzzy_set_range = []
    for i in range(0, dimension, 1):
        if(i%2 == 0):
            cluster, sigma = get_cluster(np.ravel(arm_data[:, :, i]), radius)
        else:
            cluster, sigma = get_direction_clusters(np.ravel(arm_data[:, :, i]))

        sets, x = get_fuzzy_membership(cluster, sigma, np.ravel(arm_data[:, :, i]))
        fuzzy_sets.append(sets)
        fuzzy_set_range.append(x)
        clusters.append(cluster)
        sigmas.append(sigma)

    # adjustment of sigma and cluster center.
    adjust_sigma(arm_data, clusters, fuzzy_set_range, fuzzy_sets, sigmas)

    # Neural Network
    global output_neurons, input_neurons, hidden_neurons
    input_neurons  = 0
    output_neurons = 0
    hidden_neurons = 1
    output_neurons = len(arm_data)
    for fuzzy_set in fuzzy_sets:
        hidden = 0
        for membership in fuzzy_set:
            input_neurons += 1
            hidden += 1
        hidden_neurons  *= hidden
    neural_net, data_set = build_neural_network()

    # Generate Neural Network Training Data
    input_data, output_data = generate_data(arm_data, fuzzy_set_range, fuzzy_sets)
    for i in np.arange(len(output_data)):
        data_set.addSample(input_data[i], output_data[i])
    # Training the neural network
    trainer = BackpropTrainer(neural_net, data_set)
    for i in range(epochs):
        error = trainer.train()
        print "epoch ", i, " error : ", error

    import pickle

    with open(NeuralNetworkName, 'w') as f:  # Python 3: open(..., 'wb')
        pickle.dump([neural_net, fuzzy_sets, fuzzy_set_range], f)
        f.close()


def adjust_sigma(arm_data, clusters, fuzzy_set_range, fuzzy_sets, sigmas):
    global show_plot
    global gain
    global iteration_count

    plot_entry =[[],[],[],[]]
    plot_index =[[],[],[],[]]

    for pose in arm_data:
        for j in range(0, dimension, 1):
            dim = pose[:, j]
            fuzzy_rules = fuzzy_sets[j]
            frequency = np.zeros(len(fuzzy_rules))

            for entry in dim:
                result = []
                for rule in fuzzy_rules:
                    temp = fuzz.interp_membership(fuzzy_set_range[j], rule, entry)
                    result.append(temp)
                max_index = np.array(result).argmax()
                frequency[max_index] += 1
            total = frequency.sum()
            percent_max = frequency.max() / total
            print frequency
            if percent_max != 1:
                # not all falling in same cluster.
                #sigmas[j][max_index] = sigmas[j][max_index] + gain * result[max_index]
                # update fuzzy rule
                fuzzy_sets[j][max_index] = fuzz.gaussmf(fuzzy_set_range[j], clusters[j][max_index], sigmas[j][max_index])

    if show_plot:
        for pose in arm_data:
            for j in range(0, dimension, 1):
                dim = pose[:, j]
                fuzzy_rules = fuzzy_sets[j]
                frequency = np.zeros(len(fuzzy_rules))
                for entry in dim:
                    result = []
                    for rule in fuzzy_rules:
                        temp = fuzz.interp_membership(fuzzy_set_range[j], rule, entry)
                        result.append(temp)
                    max_index = np.array(result).argmax()
                    frequency[max_index] += 1
                    plot_entry[j].append(entry)
                    plot_index[j].append(max_index)

        for j in range(0, dimension, 1):
            fuzzy_rules = fuzzy_sets[j]
            ax = plt.figure().add_subplot(111)
            for rule in fuzzy_rules:
                ax.plot(fuzzy_set_range[j], rule, '-')
                for i in range(len(plot_entry[j])):
                    ax.plot(plot_entry[j][i], 0.5, '.', color=colors[plot_index[j][i]])
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
    global input_neurons, output_neurons, hidden_neurons
    net = buildNetwork(input_neurons, hidden_neurons, output_neurons, hiddenclass=structure.TanhLayer, outclass=structure.SoftmaxLayer)
    ds = SupervisedDataSet(input_neurons, output_neurons)
    return net, ds


def get_cluster(arr, radius):
    cluster = subtractive_clustering.subtractive_clustering(arr, radius)
    sigma = np.absolute(radius * (arr.max() - arr.min()) / np.sqrt(8))
    print "Done Cluster Centers found are... ", len(cluster)
    print "Done Sigma Value... ", sigma
    sigma_list = []
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