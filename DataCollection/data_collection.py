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
    captureData(dialog.side_1, "DataCollection/Data/Side/data.csv")

    # ########## Up Pose ################################
    captureData(dialog.up_1, "DataCollection/Data/Up/data.csv")

    # ########## Front Pose ################################
    captureData(dialog.front_1, "DataCollection/Data/Front/data.csv")

    # ########## Down Pose ################################
    captureData(dialog.down_1, "DataCollection/Data/Down/data.csv")

    # ########## BendUp Pose ################################
    captureData(dialog.bend_up_1, "DataCollection/Data/Bend Up/data.csv")

    # ########## Bend Front Pose ################################
    captureData(dialog.bend_front_1, "DataCollection/Data/Bend Front/data.csv")


def captureData(phrase, path):
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

    arm_angles.append(translate_to_spherical(shoulder_left, elbow_left))
    arm_angles.append(translate_to_spherical(elbow_left, wrist_left))
    arm_angles.append(translate_to_spherical(shoulder_right, elbow_right))
    arm_angles.append(translate_to_spherical(elbow_right, wrist_right))

    arm_pos_cartesian.append(arm_angles)


def translate_to_spherical(point_ref, point):
    translated = point_ref - point
    az = np.rad2deg(np.arctan2(translated[2], np.sqrt(translated[0] ** 2 + translated[1] ** 2)))  # theta
    elev = np.rad2deg(np.arctan2(translated[1], translated[0]))  # phi
    return [elev, az]
