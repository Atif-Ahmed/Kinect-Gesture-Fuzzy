#import the pre trained data files

import pickle
import time
import numpy as np
import dialog as dialog
import speech as speech
import skfuzzy as fuzz
from pykinect.nui import JointId as Joint_Id

left_neural_net = []
left_fuzzy_sets = []
left_fuzzy_sets_range = []
right_neural_net = []
right_fuzzy_sets = []
right_fuzzy_sets_range = []
arm_pos_cartesian =[]

is_identify = False
def init():
    loadfile()
    print("file loaded")
    speech.speak(dialog.testing_intro)

    while(True):
        speech.speak(dialog.testing_make_pose)
        time.sleep(1)
        identify_pose()
        time.sleep(2)

def loadfile():
    global right_neural_net, right_fuzzy_sets, right_fuzzy_sets_range
    global left_neural_net, left_fuzzy_sets, left_fuzzy_sets_range
    with open('DataTraining/Trained/Left_Arm_NN.pickle', 'r') as f:
        [left_neural_net, left_fuzzy_sets, left_fuzzy_sets_range]= pickle.load(f)
        f.close()

    with open('DataTraining/Trained/Right_Arm_NN.pickle', 'r') as f:
        [right_neural_net, right_fuzzy_sets, right_fuzzy_sets_range]= pickle.load(f)
        f.close()


def announce_pose(pose):
    if (pose == 0):
        speech.speak(dialog.pose_side)
    if (pose == 1):
        speech.speak(dialog.pose_up)
    if (pose == 2):
        speech.speak(dialog.pose_front)
    if (pose == 3):
        speech.speak(dialog.pose_down)
    if (pose == 4):
        speech.speak(dialog.pose_bend_up)
    if (pose == 5):
        speech.speak(dialog.pose_bend_front)


def identify_pose():
    global arm_pos_cartesian
    arm_pos_cartesian =[]
    global is_identify
    global right_neural_net, right_fuzzy_sets, right_fuzzy_sets_range
    global left_neural_net, left_fuzzy_sets, left_fuzzy_sets_range
    is_identify = True
    while len(arm_pos_cartesian) < 5:
        is_identify = True
    is_identify = False

    # taking average of all the poses...
    scanned = np.array(arm_pos_cartesian)
    pose_left = [scanned[:, 0, 0].mean(),scanned[:, 0, 1].mean(),scanned[:, 1, 0].mean(),scanned[:, 1, 1].mean()]
    pose_right= [scanned[:, 2, 0].mean(),scanned[:, 2, 1].mean(),scanned[:, 3, 0].mean(),scanned[:, 3, 1].mean()]

    left_rule_data = []
    for k, dim in enumerate(pose_left):
        fuzzy_rules = left_fuzzy_sets[k]
        for rule in fuzzy_rules:
            temp = fuzz.interp_membership(left_fuzzy_sets_range[k], rule, dim)
            left_rule_data.append(temp)
    left_detected_pose = np.argmax(left_neural_net.activate(left_rule_data))

    right_rule_data = []
    for k, dim in enumerate(pose_right):
        fuzzy_rules = right_fuzzy_sets[k]
        for rule in fuzzy_rules:
            temp = fuzz.interp_membership(right_fuzzy_sets_range[k], rule, dim)
            right_rule_data.append(temp)
    right_detected_pose = np.argmax(right_neural_net.activate(right_rule_data))


    speech.speak(dialog.left_pose_guess)
    announce_pose(left_detected_pose)
    speech.speak(dialog.right_pose_guess)
    announce_pose(right_detected_pose)

def get_arm_locations(_skeleton):
    global arm_pos_cartesian
    global is_identify
    is_identify = False
    arm_angles = []
    shoulder_left = np.array([_skeleton[Joint_Id.shoulder_left].x, _skeleton[Joint_Id.shoulder_left].y, _skeleton[Joint_Id.shoulder_left].z])
    shoulder_right = np.array([_skeleton[Joint_Id.shoulder_right].x, _skeleton[Joint_Id.shoulder_right].y, _skeleton[Joint_Id.shoulder_right].z])
    elbow_left = np.array([_skeleton[Joint_Id.elbow_left].x, _skeleton[Joint_Id.elbow_left].y, _skeleton[Joint_Id.elbow_left].z])
    elbow_right = np.array([_skeleton[Joint_Id.elbow_right].x, _skeleton[Joint_Id.elbow_right].y, _skeleton[Joint_Id.elbow_right].z])
    wrist_left = np.array([_skeleton[Joint_Id.wrist_left].x, _skeleton[Joint_Id.wrist_left].y, _skeleton[Joint_Id.wrist_left].z])
    wrist_right = np.array([_skeleton[Joint_Id.wrist_right].x, _skeleton[Joint_Id.wrist_right].y, _skeleton[Joint_Id.wrist_right].z])

    arm_angles.append(translate_to_spherical(wrist_left, elbow_left, shoulder_left))
    arm_angles.append(translate_to_spherical(elbow_left, shoulder_left, shoulder_right))

    arm_angles.append(translate_to_spherical(wrist_right, elbow_right, shoulder_right))
    arm_angles.append(translate_to_spherical(elbow_right, shoulder_right, shoulder_left))

    arm_pos_cartesian.append(arm_angles)

def translate_to_spherical(begin, center, end):
    # get single angle.....
    angle = np.rint(compute_joint_angle(np.subtract(center, begin), np.subtract(center, end)))

    # computing angle direction
    center_xy = (center[0], center[1])
    start_xy = (begin[0], begin[1])
    end_xy = (end[0], end[1])

    direction = angle_cross_2d(np.subtract(center_xy, start_xy), np.subtract(center_xy, end_xy))
    if direction < 0:
        angle = - angle

        # Normal vectors are accurate only when the angle are close to 90 degree
    v1_u = unit_vector(np.subtract(center, begin))
    v2_u = unit_vector(np.subtract(center, end))

    normal = [i * 1000 for i in (np.cross(v1_u, v2_u).tolist())]
    normal_abs = np.absolute(normal).tolist()
    index = normal_abs.index(max(normal_abs))
    if (normal[index] >= 0):
        normal_idx = index*10
    else:
        normal_idx = -index*10

    if (np.absolute(angle) >= 150.0):
        return [np.absolute(angle), 30.0]
    else:
        return [np.absolute(angle), normal_idx]

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def compute_joint_angle(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))

def angle_cross_2d(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.cross(v1_u, v2_u)
