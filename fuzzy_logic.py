import matplotlib.pyplot as plt
import numpy as np
from pykinect.nui import JointId as Joint_Id

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
