#!/usr/bin/python3
import cv2
import time
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
        # NOTE: This debugging image topic is used to experiment with
        # laserscan generation using image messages from rosbags since the sensor
        # cannot be used
        self.debugging_sonar_image_sub_topic = rospy.get_param(
            "~debugging_sonar_image_sub_topic", ""
        )

        # NOTE: The following parameters are included in dynamic reconfigure
        self.sonar_frame = rospy.get_param("~sonar_frame", "sonar_link")
        self.sonar_range = rospy.get_param("~sonar_range", 9)
        self.sonar_min_range = rospy.get_param("~sonar_min_range", 0.2)
        self.image_data_factor = rospy.get_param("~image_data_factor", 1)
        self.default_distance_value = rospy.get_param("~default_distance_value", 999)
        self.publish_sonar_image = rospy.get_param("~publish_sonar_image", True)
        self.flip_input_x_sonar_image = rospy.get_param(
            "~flip_input_x_sonar_image", True
        )
        self.flip_input_y_sonar_image = rospy.get_param(
            "~flip_input_y_sonar_image", True
        )
        self.flip_output_x_sonar_image = rospy.get_param(
            "~flip_output_x_sonar_image", True
        )
        self.flip_output_y_sonar_image = rospy.get_param(
            "~flip_output_y_sonar_image", True
        )
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
        self.debugging_image_sub = None
        if self.debugging_sonar_image_sub_topic:
            self.debugging_image_sub = rospy.Subscriber(
                self.debugging_sonar_image_sub_topic,
                Image,
                self.debugging_image_callback,
            )

    def debugging_image_callback(self, image_msg):
        sonar_image = self.cv_bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
        flip_opt = None
        if self.flip_input_x_sonar_image and self.flip_input_y_sonar_image:
            flip_opt = -1
        elif self.flip_input_y_sonar_image:
            flip_opt = 1
        elif self.flip_input_x_sonar_image:
            flip_opt = 0
        if flip_opt is not None:
            sonar_image = cv2.flip(sonar_image, flip_opt)
        self.publish_sonar_image_to_laserscan(sonar_image)
        self.publish_sonar_image_to_image(sonar_image)

    def dyn_reconf_callback(self, config, _):
        rospy.loginfo(config)
        with self.mutex:
            self.sonar_frame = config.sonar_frame
            self.image_data_factor = config.image_data_factor
            self.default_distance_value = config.default_distance_value
            self.publish_sonar_image = config.publish_sonar_image
            self.flip_input_x_sonar_image = config.flip_input_x_sonar_image
            self.flip_input_y_sonar_image = config.flip_input_y_sonar_image
            self.flip_output_x_sonar_image = config.flip_output_x_sonar_image
            self.flip_output_y_sonar_image = config.flip_output_y_sonar_image
            self.sonar_min_range = config.sonar_min_range
            self.mock_hardware = config.mock_hardware
            self.minimum_obstacle_value = config.minimum_obstacle_value
            if config.sonar_range != self.sonar_range:
                self.set_sonar_range(config.sonar_range)
            if config.enabled != self.sonar_enabled:
                self.set_sonar_state(config.enabled)
        return config

    def enable_sonar_callback(self, srv):
        with self.mutex:
            self.set_sonar_state(srv.data)
        return True, ""

    def sonar_read_laserscan_pub(self, _):
        with self.mutex:
            try:
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
                flip_opt = None
                if self.flip_input_x_sonar_image and self.flip_input_y_sonar_image:
                    flip_opt = -1
                elif self.flip_input_y_sonar_image:
                    flip_opt = 1
                elif self.flip_input_x_sonar_image:
                    flip_opt = 0
                if flip_opt is not None:
                    sonar_image = cv2.flip(sonar_image, flip_opt)
                self.publish_sonar_image_to_laserscan(sonar_image)
                self.publish_sonar_image_to_image(sonar_image)
            except Exception as e:
                self.sonar_capture.release()
                self.sonar_capture = None
                rospy.logerr(e)

    def set_sonar_state(self, state):
        self.sonar_enabled = state
        if self.mock_hardware:
            return
        try:
            requests.patch(
                self.api_url + "/transponder",
                json={
                    "enable": self.sonar_enabled,
                    "sonar_range": self.sonar_range,
                },
            )
        except Exception as e:
            rospy.logerr("Failed while enabling sonar:" + str(e))

    def set_sonar_range(self, r):
        previous_sonar_state = self.sonar_enabled
        self.sonar_range = r
        self.sonar_fov_rad = (
            120 * self.pi_deg_ratio
            if self.sonar_range <= 30
            else 90 * self.pi_deg_ratio
        )
        if self.mock_hardware:
            return
        self.set_sonar_state(False)
        if previous_sonar_state:
            time.sleep(1)
            self.set_sonar_state(True)

    def set_datastream_to_rtsp(self):
        if self.mock_hardware:
            return
        try:
            requests.put(
                self.api_url + "/streamtype",
                json={
                    "value": 2,
                },
            )
        except Exception as e:
            rospy.logerr("Failed while setting sonar datastream to rtsp:" + str(e))

    def setup_sonar_capture(self):
        if self.mock_hardware:
            return
        self.set_sonar_range(self.sonar_range)
        self.set_sonar_state(self.sonar_enabled)
        self.set_datastream_to_rtsp()
        try:
            self.sonar_capture = cv2.VideoCapture(self.rtsp_url)
        except Exception as e:
            rospy.logerr("Failed to create sonar image capture:" + str(e))

    def sonar_image_to_ranges(self, sonar_image):
        ranges = []
        height, width, _ = sonar_image.shape
        range_height_ratio = self.sonar_range / height
        for w in range(width):
            beam = sonar_image[:, w, 0]
            distance = self.default_distance_value
            for h in range(height):
                if beam[h] >= self.minimum_obstacle_value:
                    # distance = self.sonar_min_range + h / height * (
                    #     self.sonar_range - self.sonar_min_range
                    # )
                    distance_candidate = h * range_height_ratio
                    if distance_candidate >= self.sonar_min_range:
                        distance = distance_candidate
                        break

            ranges.append(distance)
        return ranges

    def publish_sonar_image_to_laserscan(self, sonar_image):
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

    def publish_sonar_image_to_image(self, sonar_image):
        if self.publish_sonar_image:
            if self.image_data_factor != 1.0:
                sonar_image = np.clip(
                    sonar_image.astype(np.float32) * self.image_data_factor, 0, 255
                ).astype(np.uint8)
            flip_opt = None
            if self.flip_output_x_sonar_image and self.flip_output_y_sonar_image:
                flip_opt = -1
            elif self.flip_output_y_sonar_image:
                flip_opt = 1
            elif self.flip_output_x_sonar_image:
                flip_opt = 0
            if flip_opt is not None:
                sonar_image = cv2.flip(sonar_image, flip_opt)
            image_msg = self.cv_bridge.cv2_to_imgmsg(sonar_image, encoding="bgr8")
            image_msg.header.stamp = rospy.Time.now()
            image_msg.header.frame_id = self.sonar_frame
            self.image_pub.publish(image_msg)


if __name__ == "__main__":
    sonar = SonoptixEchoDriver()
    rospy.spin()
