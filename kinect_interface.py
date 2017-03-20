import pygame as window
from pykinect import nui
from pykinect.nui import JointId as Joint_Id
from pykinect.nui.structs import TransformSmoothParameters
import DataCollection.data_collection as dataCollection
import DataTesting.testing_exercise as testing_exercise

WINDOW_SIZE = 640, 480
red = window.color.THECOLORS['red']

# Global variables
screen = None  # handle to Kinect video window (pygame Window)
device = None  # handle to Kinect device
skeleton = None  # Tuple of skeleton data 3D
arm_length = None  # data to store information of _right_arm

enable_video = True  # enable or disable video feed


# Filtering parameters.... of skeletal data..
SMOOTH_PARAMS_SMOOTHING = 0.5
SMOOTH_PARAMS_CORRECTION = 0.1
SMOOTH_PARAMS_PREDICTION = 0.5
SMOOTH_PARAMS_JITTER_RADIUS = 0.1
SMOOTH_PARAMS_MAX_DEVIATION_RADIUS = 0.1
SMOOTH_PARAMS = TransformSmoothParameters(SMOOTH_PARAMS_SMOOTHING,
                                          SMOOTH_PARAMS_CORRECTION,
                                          SMOOTH_PARAMS_PREDICTION,
                                          SMOOTH_PARAMS_JITTER_RADIUS,
                                          SMOOTH_PARAMS_MAX_DEVIATION_RADIUS)

# initialize Kinect device
# noinspection PyUnresolvedReferences,PyUnboundLocalVariable
def init():
    # Initialize variable at global level
    global screen, device
    # Initialize PyGame
    window.init()
    screen = window.display.set_mode(WINDOW_SIZE, 0, 32)
    window.display.set_caption('PyKinect Skeleton Example')
    screen.fill(window.color.THECOLORS["black"])
    device = nui.Runtime()
    device.skeleton_engine.enabled = True
    device.video_frame_ready += video_frame_ready
    device.video_stream.open(nui.ImageStreamType.Video, 2, nui.ImageResolution.Resolution640x480, nui.ImageType.Color)

# noinspection PyProtectedMember
def video_frame_ready(frame):
    skeleton_2_d = get_skeleton()
    if enable_video:
        frame.image.copy_bits(screen._pixels_address)
    else:
        screen.fill((0, 0, 0))
    if skeleton_2_d is not None:
        draw_skeleton(skeleton_2_d)
        if dataCollection.is_capture:
            dataCollection.get_arm_locations(skeleton)
        if testing_exercise.is_identify:
            testing_exercise.get_arm_locations(skeleton)
    window.display.update()

def get_skeleton():
    global skeleton
    skeleton_2_d = []

    skeleton_frame = device.skeleton_engine.get_next_frame()

    device._nui.NuiTransformSmooth(skeleton_frame, SMOOTH_PARAMS)

    for index, _skeleton in enumerate(skeleton_frame.skeleton_data):
        if _skeleton.eTrackingState == nui.SkeletonTrackingState.TRACKED:

            skeleton = _skeleton.SkeletonPositions
            for joints in _skeleton.SkeletonPositions:
                skeleton_2_d.append(
                    nui.SkeletonEngine.skeleton_to_depth_image(joints, screen.get_width(), screen.get_height()))
            return skeleton_2_d

def draw_skeleton(_skeleton):
    # draw Circles for joints

    window.draw.circle(screen, red, map(int, _skeleton[Joint_Id.head]), 10)
    window.draw.circle(screen, red, map(int, _skeleton[Joint_Id.spine]), 10)
    window.draw.circle(screen, red, map(int, _skeleton[Joint_Id.shoulder_center]), 10)
    window.draw.circle(screen, red, map(int, _skeleton[Joint_Id.shoulder_right]), 10)
    window.draw.circle(screen, red, map(int, _skeleton[Joint_Id.elbow_right]), 10)
    window.draw.circle(screen, red, map(int, _skeleton[Joint_Id.wrist_right]), 10)
    window.draw.circle(screen, red, map(int, _skeleton[Joint_Id.hand_right]), 10)
    window.draw.circle(screen, red, map(int, _skeleton[Joint_Id.shoulder_left]), 10)
    window.draw.circle(screen, red, map(int, _skeleton[Joint_Id.elbow_left]), 10)
    window.draw.circle(screen, red, map(int, _skeleton[Joint_Id.wrist_left]), 10)
    window.draw.circle(screen, red, map(int, _skeleton[Joint_Id.hand_left]), 10)
    # Draw lines on All connections
    window.draw.line(screen, red, map(int, _skeleton[Joint_Id.head]), map(int, _skeleton[Joint_Id.shoulder_center]), 2)
    window.draw.line(screen, red, map(int, _skeleton[Joint_Id.spine]), map(int, _skeleton[Joint_Id.shoulder_center]), 2)
    window.draw.line(screen, red, map(int, _skeleton[Joint_Id.spine]), map(int, _skeleton[Joint_Id.hip_center]), 2)
    # Right Arm
    window.draw.line(screen, red, map(int, _skeleton[Joint_Id.shoulder_center]), map(int, _skeleton[Joint_Id.shoulder_right]), 2)
    window.draw.line(screen, red, map(int, _skeleton[Joint_Id.shoulder_right]), map(int, _skeleton[Joint_Id.elbow_right]), 2)
    window.draw.line(screen, red, map(int, _skeleton[Joint_Id.elbow_right]), map(int, _skeleton[Joint_Id.wrist_right]), 2)
    window.draw.line(screen, red, map(int, _skeleton[Joint_Id.wrist_right]), map(int, _skeleton[Joint_Id.hand_right]), 2)
    # Left Arm
    window.draw.line(screen, red, map(int, _skeleton[Joint_Id.shoulder_center]), map(int, _skeleton[Joint_Id.shoulder_left]), 2)
    window.draw.line(screen, red, map(int, _skeleton[Joint_Id.shoulder_left]), map(int, _skeleton[Joint_Id.elbow_left]), 2)
    window.draw.line(screen, red, map(int, _skeleton[Joint_Id.elbow_left]), map(int, _skeleton[Joint_Id.wrist_left]), 2)
    window.draw.line(screen, red, map(int, _skeleton[Joint_Id.wrist_left]), map(int, _skeleton[Joint_Id.hand_left]), 2)
    # Shoulder to Shoulder
    window.draw.line(screen, red, map(int, _skeleton[Joint_Id.shoulder_right]), map(int, _skeleton[Joint_Id.shoulder_left]), 2)

def device_angle_up():
    device.camera.elevation_angle += 2
    print "Device Camera Angle = ", device.camera.elevation_angle

def device_angle_down():
    device.camera.elevation_angle -= 2
    print "Device Camera Angle = ", device.camera.elevation_angle


