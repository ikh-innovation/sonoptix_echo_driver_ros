#!/usr/bin/python3
import cv2
import rospy
import requests
import threading
import numpy as np
from cv_bridge import CvBridge
from std_srvs.srv import SetBool
from sensor_msgs.msg import LaserScan, Image
from dynamic_reconfigure.server import Server
from sonoptix_echo_driver_ros.cfg import SonoptixEchoConfig


class SonoptixEchoDriver:
    def __del__(self):
        self.pub_timer.shutdown()
        if self.sonar_capture:
            self.sonar_capture.release()

    def __init__(self):
        rospy.init_node("sonoptix_echo_driver_node")
        self.enable_sonar_service = rospy.get_param(
            "~enable_sonar_service", "enable_sonar"
        )
        self.srv = rospy.Service(
            self.enable_sonar_service, SetBool, self.enable_sonar_callback
        )
        self.ip = rospy.get_param("~ip", "192.168.2.42")
        self.rtsp_port = rospy.get_param("~rtsp_port", 8554)
        self.api_port = rospy.get_param("~api_port", 8000)
        self.laserscan_topic = rospy.get_param("~laserscan_topic", "sonar_scan")
        self.sonar_image_topic = rospy.get_param("~sonar_image_topic", "sonar_image")
        self.pub_hz = rospy.get_param("~publish_hz", 25)

        # NOTE: The following parameters are included in dynamic reconfigure
        self.sonar_frame = rospy.get_param("~sonar_frame", "sonar_link")
        self.sonar_range = rospy.get_param("~sonar_range", 9)
        self.sonar_min_range = rospy.get_param("~sonar_min_range", 0.2)
        self.default_distance_value = rospy.get_param("~default_distance_value", 999)
        self.publish_sonar_image = rospy.get_param("~publish_sonar_image", True)
        self.mock_hardware = rospy.get_param("~mock_hardware", True)
        self.minimum_obstacle_value = rospy.get_param("~minimum_obstacle_value", 180)
        self.sonar_enabled = rospy.get_param("~enabled", False)
        # ------------------------------------------------------------------
        self.laserscan_pub = rospy.Publisher(
            self.laserscan_topic, LaserScan, queue_size=10
        )
        self.image_pub = rospy.Publisher(self.sonar_image_topic, Image, queue_size=10)
        self.rtsp_url = "rtsp://" + self.ip + ":" + str(self.rtsp_port) + "/raw"
        self.api_url = "http://" + self.ip + ":" + str(self.api_port) + "/api/v1"

        self.pub_timer = rospy.Timer(
            rospy.Duration(1.0 / self.pub_hz), self.sonar_read_laserscan_pub
        )
        self.mutex = threading.Lock()
        self.cv_bridge = CvBridge()
        self.sonar_capture = None

        self.pi_deg_ratio = np.pi / 180

        self.sonar_enabled = False
        self.sonar_fov_rad = 0.0

        self.dyn_reconf_server = Server(SonoptixEchoConfig, self.dyn_reconf_callback)

        self.setup_sonar_capture()

    def dyn_reconf_callback(self, config, _):
        rospy.loginfo(config)
        with self.mutex:
            self.sonar_frame = config.sonar_frame
            self.default_distance_value = config.default_distance_value
            self.publish_sonar_image = config.publish_sonar_image
            self.mock_hardware = config.mock_hardware
            self.minimum_obstacle_value = config.minimum_obstacle_value
            if config.enabled != self.sonar_enabled:
                self.set_sonar_state(config.enabled)
            if config.sonar_range != self.sonar_range:
                self.set_sonar_range(config.sonar_range)
        return config

    def enable_sonar_callback(self, srv):
        with self.mutex:
            self.set_sonar_state(srv.data)
        return True, ""

    def sonar_read_laserscan_pub(self, _):
        try:
            with self.mutex:
                if not self.sonar_enabled:
                    return
                sonar_image = None
                if self.mock_hardware:
                    height = 268
                    if self.sonar_range >= 30:
                        height = 1024
                    elif self.sonar_range >= 5:
                        height = 608
                    sonar_image = np.random.randint(
                        0, 256, size=(height, 256, 3), dtype=np.uint8
                    )
                else:
                    if self.sonar_capture is None or not self.sonar_capture.isOpened():
                        self.setup_sonar_capture()
                    success, sonar_image = self.sonar_capture.read()
                    if not success:
                        self.sonar_capture.release()
                        self.sonar_capture = None
                        rospy.logerr(
                            "Error reading from sonar image stream... Skipping publication!"
                        )
                        return
                    print(sonar_image.shape)
                laserscan_msg = LaserScan()
                laserscan_msg.header.frame_id = self.sonar_frame
                laserscan_msg.header.stamp = rospy.Time.now()
                laserscan_msg.angle_min = -self.sonar_fov_rad / 2.0
                laserscan_msg.angle_max = self.sonar_fov_rad / 2.0
                laserscan_msg.angle_increment = self.sonar_fov_rad / 256
                laserscan_msg.scan_time = 1.0 / self.pub_hz
                laserscan_msg.range_min = self.sonar_min_range
                laserscan_msg.range_max = self.sonar_range
                laserscan_msg.ranges = self.sonar_image_to_ranges(sonar_image)
                self.laserscan_pub.publish(laserscan_msg)
                if self.publish_sonar_image:
                    image_msg = self.cv_bridge.cv2_to_imgmsg(
                        sonar_image, encoding="bgr8"
                    )
                    image_msg.header.stamp = rospy.Time.now()
                    image_msg.header.frame_id = self.sonar_frame
                    self.image_pub.publish(image_msg)
        except Exception as e:
            self.sonar_capture.release()
            self.sonar_capture = None
            rospy.logerr(e)

    def set_sonar_state(self, state):
        self.sonar_enabled = state
        if self.mock_hardware:
            return
        # BUG: Setting sensor state and range independently does not seem to work
        # requests.patch(
        #     self.api_url + "/transponder",
        #     json={
        #         "enable": state,
        #     },
        # )

    def set_sonar_range(self, r):
        self.set_sonar_state(False)
        self.sonar_range = r
        self.sonar_fov_rad = (
            120 * self.pi_deg_ratio
            if self.sonar_range <= 30
            else 90 * self.pi_deg_ratio
        )
        if self.mock_hardware:
            return
        requests.patch(
            self.api_url + "/transponder",
            json={
                # BUG:
                # NOTE: Currently force enabling here
                # but this should be removed when the bug is investigated
                "enable": True,
                "sonar_range": r,
            },
        )
        if self.sonar_enabled:
            self.set_sonar_state(True)

    def set_datastream_to_rtsp(self):
        if self.mock_hardware:
            return
        requests.put(
            self.api_url + "/streamtype",
            json={
                "value": 2,
            },
        )

    def setup_sonar_capture(self):
        if self.mock_hardware:
            return
        self.set_sonar_range(self.sonar_range)
        self.set_sonar_state(self.sonar_enabled)
        self.set_datastream_to_rtsp()
        self.sonar_capture = cv2.VideoCapture(self.rtsp_url)

    def sonar_image_to_ranges(self, sonar_image):
        ranges = []
        height, width, _ = sonar_image.shape
        for w in range(width):
            beam = sonar_image[:, w, 0]
            distance = self.default_distance_value
            for h in range(height):
                if beam[h] >= self.minimum_obstacle_value:
                    distance = self.sonar_min_range + h / height * (
                        self.sonar_range - self.sonar_min_range
                    )
                    break

            ranges.append(distance)
        return ranges


if __name__ == "__main__":
    sonar = SonoptixEchoDriver()
    rospy.spin()
