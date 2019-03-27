import cv2
import anki_vector
import numpy as np


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
    robot.say_text("Starting program block tracking.")
    robot.say_text("Battery level " + str(robot.get_battery_state().battery_level))
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
            matches_mask = mask.ravel().tolist()

            # Perspective transform
            h, w = img_template.shape
            pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)

            dst = cv2.perspectiveTransform(pts, matrix)

            # We need to pass the dst points as integer as they are pixel values
            homography = cv2.polylines(img_live, [np.int32(dst)], True, (255, 0, 0), 3)

            cv2.imshow("Homography", homography)
        else:
            cv2.imshow("Homography", img3)

        #cv2.imshow("Image", img3)

        # Exit on pressing 'Esc'
        if cv2.waitKey(1) == 27:
            break 
    cv2.destroyAllWindows()
        #image.show()
    robot.say_text("Good bye and good luck.")
