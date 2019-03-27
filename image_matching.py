import cv2
import anki_vector
import numpy as np


img_template = cv2.imread("./block_pattern.jpg", cv2.IMREAD_GRAYSCALE)
sift = cv2.xfeatures2d.SIFT_create()
surf = cv2.xfeatures2d.SURF_create()
orb  = cv2.ORB_create(nfeatures=100)

index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)

keypoints, descriptors = sift.detectAndCompute(img_template, None)

img_template_key = cv2.drawKeypoints(img_template, keypoints, None)

with anki_vector.Robot(enable_camera_feed=True,
        ip='10.42.0.168') as robot:
    robot.say_text("Starting program block tracking.")
    while True:
        img_pil = robot.camera.latest_image
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_BGR2GRAY)
        cv2.imshow("Image", img_cv)
        if cv2.waitKey(1) == 27:
            break # 'Esc' to quit
    cv2.destroyAllWindows()
        #image.show()
    robot.say_text("Good bye and good luck.")
