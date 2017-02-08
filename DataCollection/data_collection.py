import DataCollection.dialog as dialog
import DataCollection.speech as speech

from pykinect.nui import JointId as Joint_Id
import time
import csv
import numpy as np

is_capture = False
arm_pos_cartesian = []
arm_joint_index = {'left_shoulder': 0,
                   'right_shoulder': 1,
                   'left_elbow': 2,
                   'right_elbow': 3,
                   'left_wrist': 4,
                   'right_wrist': 5}


def start():
    speech.speak(dialog.welcome)

    # ########## Side Pose ##############################
    capture_data(dialog.side_1, "DataCollection/Data/Side/data.csv")

    # ########## Up Pose ################################
    capture_data(dialog.up_1, "DataCollection/Data/Up/data.csv")

    # ########## Front Pose ################################
    capture_data(dialog.front_1, "DataCollection/Data/Front/data.csv")

    # ########## Down Pose ################################
    capture_data(dialog.down_1, "DataCollection/Data/Down/data.csv")

    # ########## BendUp Pose ################################
    capture_data(dialog.bend_up_1, "DataCollection/Data/Bend Up/data.csv")

    # ########## Bend Front Pose ################################
    capture_data(dialog.bend_front_1, "DataCollection/Data/Bend Front/data.csv")


def capture_data(phrase, path):
    global arm_pos_cartesian
    arm_pos_cartesian = []
    speech.speak(phrase)
    time.sleep(1)
    speech.speak(dialog.start)
    # start recording data
    global is_capture
    is_capture = True
    while len(arm_pos_cartesian) <= 500:
        print len(arm_pos_cartesian)
    is_capture = False
    speech.speak(dialog.stop)
    with open(path, 'a') as resultFile:
        wr = csv.writer(resultFile)
        wr.writerow(arm_pos_cartesian)
        resultFile.close()

    speech.speak(dialog.relax)
    time.sleep(3)
    # ########## Side Pose End ##########################


def get_arm_locations(_skeleton):
    global arm_pos_cartesian
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
    normal = np.absolute(np.rint(normal)).tolist()

    return [np.absolute(angle), normal.index(max(normal))]


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
