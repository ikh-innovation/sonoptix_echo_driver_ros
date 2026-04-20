#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class ContrastEnhancer:
    def __init__(self):
        rospy.init_node("contrast_enhancer_node", anonymous=True)

        self.bridge = CvBridge()

        self.sub = rospy.Subscriber(
            "/sonar_cone_view",
            Image,
            self.image_callback,
            queue_size=1
        )

        self.pub = rospy.Publisher(
            "/sonar_image_enhanced",
            Image,
            queue_size=1
        )

        rospy.loginfo("Contrast Enhancer Node Started")

    def adjust_gamma(self, image, gamma=1.5):
        inv_gamma = 1.0 / gamma
        table = np.array([
            ((i / 255.0) ** inv_gamma) * 255
            for i in np.arange(256)
        ]).astype("uint8")

        return cv2.LUT(image, table)

    def enhance_contrast(self, img_bgr):
        # Convert to LAB color space (better for lighting adjustments)
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to L-channel (lightness)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l2 = clahe.apply(l)

        lab = cv2.merge((l2, a, b))
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # Apply gamma correction to brighten dark images
        enhanced = self.adjust_gamma(enhanced, gamma=1.3)

        return enhanced

    def image_callback(self, msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

            enhanced_img = self.enhance_contrast(cv_img)

            out_msg = self.bridge.cv2_to_imgmsg(enhanced_img, encoding="bgr8")
            out_msg.header = msg.header

            self.pub.publish(out_msg)

        except CvBridgeError as e:
            rospy.logerr(f"CvBridge error: {e}")


if __name__ == "__main__":
    node = ContrastEnhancer()
    rospy.spin()