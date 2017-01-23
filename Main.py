import kinect_interface as kinect
import fuzzy_logic

def startKinect():
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


fuzzy_logic.mountain_clustering_test()