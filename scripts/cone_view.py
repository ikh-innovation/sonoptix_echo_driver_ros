#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class SonarConeVisualizer:
    def __init__(self):
        rospy.init_node("sonar_cone_visualizer", anonymous=True)

        self.bridge = CvBridge()

        self.sub = rospy.Subscriber(
            "/sonar_image",
            Image,
            self.callback,
            queue_size=1
        )

        self.pub = rospy.Publisher(
            "/sonar_cone_view",
            Image,
            queue_size=1
        )

        rospy.loginfo("Sonar Cone Visualizer Started")

    def callback(self, msg):
        try:
            # Convert ROS Image → OpenCV image
            sonar = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")

            cone = self.polar_to_cone(sonar)

            out_msg = self.bridge.cv2_to_imgmsg(cone, encoding="bgr8")
            out_msg.header = msg.header

            self.pub.publish(out_msg)

        except CvBridgeError as e:
            rospy.logerr(f"CvBridge error: {e}")

    def polar_to_cone(self, polar_img):
        """
        polar_img: (r, θ) image
        output: fan-shaped Cartesian visualization
        """

        h, w = polar_img.shape  # h = range, w = angle

        # Output canvas size
        size = h * 2
        center = size // 2

        output = np.zeros((size, size, 3), dtype=np.uint8)

        # Field of view (assume full width is sonar FOV)
        fov = np.deg2rad(120.0)  # adjust if needed
        angle_min = -fov / 2
        angle_max = fov / 2

        # Precompute angles for each column
        angles = np.linspace(angle_min, angle_max, w)

        for r in range(h):
            for c in range(w):

                intensity = int(polar_img[r, c])

                if intensity == 0:
                    continue

                theta = angles[c]

                # polar → Cartesian
                x = int(center + r * np.sin(theta))
                y = int(center - r * np.cos(theta))

                if 0 <= x < size and 0 <= y < size:
                    color = (intensity, intensity, intensity)
                    output[y, x] = color

        # Optional: draw sonar origin
        cv2.circle(output, (center, center), 3, (0, 0, 255), -1)

        return output


if __name__ == "__main__":
    SonarConeVisualizer()
    rospy.spin()