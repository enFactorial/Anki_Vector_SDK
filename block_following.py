import cv2
import anki_vector
import math
import numpy as np
import time


def initialize_robot(robot):
    '''
    Sets the robot in a known state:
        1) head is horizontal (0 deg)
        2) lifter is in down position

    Parameters
    ----------

    robot : anki_vector.Robot
      handle of robot to be initialized
    '''
    robot.say_text("Initializing.")

    # Level head
    HEAD_SPEED_rad = 1.0
    while (robot.head_angle_rad > 0.1 or robot.head_angle_rad < -0.1):
        print("Head: {}".format(robot.head_angle_rad))
        robot.motors.set_head_motor(0)
        if robot.head_angle_rad > 0.1:
            robot.motors.set_head_motor(-HEAD_SPEED_rad)
        elif robot.head_angle_rad < -0.1:
            robot.motors.set_head_motor(HEAD_SPEED_rad)

    robot.motors.set_head_motor(0.0)

    # Lower lifter
    while (robot.lift_height_mm > 35):
        print("Lift: {}.".format(robot.lift_height_mm))
        robot.motors.set_lift_motor(0)
        if (robot.lift_height_mm > 0):
            robot.motors.set_lift_motor(-5.0)

    robot.motors.set_lift_motor(0)
    robot.say_text("Init complete")

def pivot_to_point(robot, point):
    '''
    Sets the wheel motors to rotate so that the point x coordinate is 
    within a tracking window.

    Parameters
    ----------

    robot : anki_vector.Robot
      handle of robot to be controlled

    point : [int, int]
      x, y coordinates to be tracked
    '''
    TOP_SPEED_mmps = 25
    DELAY_s = 0.5
    P = 0.01

    CENTER_POS_pix = 350
    DEAD_BAND_pix = 30
    RIGHT_LIMIT = 270
    LEFT_LIMIT = 300
    
    delta_pix = point[0] - CENTER_POS_pix

    if (math.fabs(delta_pix) > DEAD_BAND_pix):
        robot.motors.set_wheel_motors(0, 0)
        speed_mmps = np.int32(P * math.fabs(delta_pix) * TOP_SPEED_mmps)
        print("Delta: {} | Speed: {}".format(delta_pix, speed_mmps));
        
        if delta_pix < 0:
            # Turn left
            robot.motors.set_wheel_motors(-speed_mmps, speed_mmps)
        else:
            #Turn right
            robot.motors.set_wheel_motors(speed_mmps, -speed_mmps)

        # TODO: Test keeping speed constant and varying the duration motors are
        # on.
        time.sleep(DELAY_s)
        
    robot.motors.set_wheel_motors(0, 0)
    time.sleep(0.1)

def follow_cube(robot, size):
    '''
    Moves the robot forward or backwards so that it is at a set distance from
    a block.

    Parameters
    ----------
    
    robot : anki_vector.Robot
      handle of robot to be controlled

    size : int32

      dimension of the block in pixels (largest diagonal). This is used to infer
      the distance to the block.
    '''
    MIN_SIZE = 120
    MAX_SIZE = 160

    SPEED_mmps = 25
    
    robot.motors.set_wheel_motors(0,0)

    if size < MIN_SIZE:
        print("Move forward {}".format(size))
        robot.motors.set_wheel_motors(SPEED_mmps, SPEED_mmps)
        time.sleep(.5)
    elif size > MAX_SIZE:
        print("Move backward {}".format(size))
        robot.motors.set_wheel_motors(-SPEED_mmps, -SPEED_mmps)
        time.sleep(.5)

    robot.motors.set_wheel_motors(0, 0)

img_template = cv2.imread("./block_pattern.jpg", cv2.IMREAD_GRAYSCALE)
sift = cv2.xfeatures2d.SIFT_create()
surf = cv2.xfeatures2d.SURF_create()
orb  = cv2.ORB_create(nfeatures=100)

kp_template, desc_template = sift.detectAndCompute(img_template, None)
# img_template_key = cv2.drawKeypoints(img_template, kp_template, None)

index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)

with anki_vector.Robot(enable_camera_feed=True,
        ip='10.42.0.168') as robot:
    robot.say_text("Battery level " + str(robot.get_battery_state().battery_level))
    initialize_robot(robot)
    robot.say_text("Starting program block following.")
    while True:
        # Acquire an image
        img_pil = robot.camera.latest_image
        img_live = cv2.cvtColor(np.array(img_pil), cv2.COLOR_BGR2GRAY)

        kp_live, desc_live = sift.detectAndCompute(img_live, None)

        matches = flann.knnMatch(desc_template, desc_live, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.6*n.distance:
                good_matches.append(m)

        img3 = cv2.drawMatches(img_template, kp_template, img_live, kp_live, good_matches, img_live)

        # If we have at least 10 good matches show the homography
        if len(good_matches) > 5:
            query_pts = np.float32([kp_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            train_pts = np.float32([kp_live[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
            
            # If findHomography returns an empty matrix don't run throughe rest of the function.
            if matrix is None or matrix.size == 0:
                continue
            
            matches_mask = mask.ravel().tolist()

            # Perspective transform
            h, w = img_template.shape
            pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)

            dst = cv2.perspectiveTransform(pts, matrix)
            
            # calculate the instantaneous center of the paralelogram
            dim = { 'A': {'x': dst[0][0][0], 'y': dst[0][0][1]},
                    'B': {'x': dst[1][0][0], 'y': dst[1][0][1]},
                    'C': {'x': dst[2][0][0], 'y': dst[2][0][1]},
                    'D': {'x': dst[3][0][0], 'y': dst[3][0][1]}
                  }
            
            cm_inst = [(dim['A']['x'] + dim['C']['x']) / 2, (dim['A']['y'] + dim['C']['y'])/2]
            # impose sane values
            cm_inst[0] = cm_inst[0] if cm_inst[0] < 640 else 640
            cm_inst[0] = cm_inst[0] if cm_inst[0] > 0 else 0

            cm_inst[1] = cm_inst[1] if cm_inst[1] < 360 else 360
            cm_inst[1] = cm_inst[1] if cm_inst[1] > 0 else 0

            # Use diagonals length as an indicator of cube size:
            diag_A = math.sqrt(math.pow((dim['A']['x'] - dim['C']['x']), 2) +
                               math.pow((dim['A']['y'] - dim['C']['y']), 2)
                              )
            diag_B = math.sqrt(math.pow((dim['B']['x'] - dim['D']['x']), 2) +
                               math.pow((dim['B']['y'] - dim['D']['y']), 2)
                              )
            cube_size = max(diag_A, diag_B, 0) #minimum size is 0
            
            print("CM: {} | Size: {}".format(cm_inst, cube_size))

            # We need to pass the dst points as integer as they are pixel values
            homography = cv2.polylines(img_live, [np.int32(dst)], True, (255, 0, 0), 3)

            pivot_to_point(robot, cm_inst)
            follow_cube(robot, cube_size)

            cv2.imshow("Homography", homography)
        else:
            cv2.imshow("Homography", img3)

        #cv2.imshow("Image", img3)

        # Exit on pressing 'Esc'
        if cv2.waitKey(1) == 27:
            break 
    cv2.destroyAllWindows()
        #image.show()
    robot.say_text("Good bye.")
