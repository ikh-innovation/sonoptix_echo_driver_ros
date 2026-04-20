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
        
        self.draw_distance_lines = rospy.get_param("~draw_distance_lines", True)

        # ------------------------------------------------------------------
        self.laserscan_pub = rospy.Publisher(
            self.laserscan_topic, LaserScan, queue_size=10
        )
        self.image_pub = rospy.Publisher(self.sonar_image_topic, Image, queue_size=10)
        self.cone_projection_pub = rospy.Publisher(
            self.sonar_image_topic + "_cone_projection", Image, queue_size=10
        )
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
        # if flip_opt is not None:
        #     sonar_image = cv2.flip(sonar_image, flip_opt)
        flip_opt = 0
        sonar_image = cv2.flip(sonar_image, flip_opt)

        self.publish_sonar_image_to_laserscan(sonar_image)
        self.publish_sonar_image_to_image(sonar_image)

    def dyn_reconf_callback(self, config, _):
        rospy.logwarn(config)
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
            self.draw_distance_lines = config.draw_distance_lines
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

    def sonar_image_to_cone_projection(self, sonar_image):
        """
        Project sonar image from rectangular beam format to polar cone projection.
        Each column of the input image represents a beam, emanating from the sonar position.

        Args:
            sonar_image: Input image where columns are beams and rows are ranges

        Returns:
            Projected image with polar cone layout, black outside FOV
        """
        height, width, channels = sonar_image.shape
        rospy.logwarn(f"[CONE] Input image shape: {sonar_image.shape}")
        rospy.logwarn(f"[CONE] Input pixel range: min={sonar_image.min()}, max={sonar_image.max()}, mean={sonar_image.mean():.1f}")
        rospy.logwarn(f"[CONE] Sonar range: {self.sonar_range}, FOV rad: {self.sonar_fov_rad}")

        range_height_ratio = self.sonar_range / height
        rospy.logwarn(f"[CONE] Range height ratio: {range_height_ratio}")

        # Calculate output image size based on actual sonar geometry
        # At max range, cone width is: 2 * sonar_range * sin(FOV/2)
        # We need height = sonar_range (in pixels matching input height), width = cone_width
        # But to preserve aspect ratio and have room, use square based on cone width
        if self.sonar_fov_rad > 0:
            cone_width_meters = 2.0 * self.sonar_range * np.sin(self.sonar_fov_rad / 2.0)
            # Pixels per meter (match input resolution)
            pixels_per_meter = height / self.sonar_range
            output_width_pixels = int(cone_width_meters * pixels_per_meter)
            output_height_pixels = height
            # Make output square to fit the cone nicely
            output_size = max(output_width_pixels, output_height_pixels)
        else:
            output_size = height

        cone_image = np.zeros((output_size, output_size, channels), dtype=np.uint8)
        rospy.logwarn(f"[CONE] Output image size: {output_size}x{output_size} (cone_width_m={cone_width_meters:.1f}, pixels_per_m={pixels_per_meter:.1f})")

        # Sonar position at center-bottom
        sonar_x = output_size / 2.0
        sonar_y = output_size - 1.0
        rospy.logwarn(f"[CONE] Sonar position: ({sonar_x}, {sonar_y})")

        # Calculate angle increment
        if self.sonar_fov_rad > 0:
            angle_increment = self.sonar_fov_rad / (width - 1) if width > 1 else 0
            angle_start = -self.sonar_fov_rad / 2.0
            rospy.logwarn(f"[CONE] Angle start: {angle_start}, increment: {angle_increment}")
        else:
            rospy.logwarn("[CONE] FOV is 0, returning black image")
            return cone_image  # Return black image if FOV not set

        

        for beam_idx in range(width):
            # Calculate angle for this beam
            beam_angle = angle_start + beam_idx * angle_increment

            # Process each range bin in this beam
            for range_idx in range(height):
                # Get pixel color from original image (all channels)
                pixel = sonar_image[range_idx, beam_idx, :]

                # Skip black pixels (outside actual data)
                if np.all(pixel == 0):
                    continue

                # Calculate distance in meters
                distance = range_idx * range_height_ratio

                # Convert polar (distance, angle) to Cartesian coordinates
                # Angle is measured from forward direction (up in image)
                px = int(sonar_x + distance * np.sin(beam_angle) * pixels_per_meter)
                py = int(sonar_y - distance * np.cos(beam_angle) * pixels_per_meter)
                cone_image[py, px, :] = pixel

        if not self.draw_distance_lines:
            return cone_image
        
        # Draw scale lines on both sides of the cone (red with 2-meter resolution)
        scale_color = (0, 0, 255)  # Red in BGR
        scale_line_length = 15  # Length of each scale tick
        scale_resolution = 2.0  # 2 meter resolution

        # Draw radial dotted lines connecting left and right edges at constant distances
        dot_spacing = 10  # Pixels between dots for dashed line
        dot_length = 5    # Length of each dot

        for dist in np.arange(0, self.sonar_range + 1, scale_resolution):
            # Generate points along the arc at this distance
            num_points = 100
            arc_points = []

            for i in range(num_points):
                # Interpolate angle from left to right
                angle = angle_start + (i / (num_points - 1)) * self.sonar_fov_rad

                # Convert polar to Cartesian
                px = int(sonar_x + dist * np.sin(angle) * pixels_per_meter)
                py = int(sonar_y - dist * np.cos(angle) * pixels_per_meter)

                # Only add points within bounds
                if 0 <= px < output_size and 0 <= py < output_size:
                    arc_points.append((px, py))

            # Draw dotted line connecting the arc points
            for j in range(len(arc_points) - 1):
                if (j // dot_spacing) % 2 == 0:  # Draw dot, skip space
                    pt1 = arc_points[j]
                    pt2 = arc_points[min(j + dot_length, len(arc_points) - 1)]
                    cv2.line(cone_image, pt1, pt2, scale_color, 1)

        # Left edge (negative angle limit)
        left_angle = angle_start
        for dist in np.arange(0, self.sonar_range + 1, scale_resolution):
            # Position on left edge
            px_center = int(sonar_x + dist * np.sin(left_angle) * pixels_per_meter)
            py_center = int(sonar_y - dist * np.cos(left_angle) * pixels_per_meter)

            # Perpendicular direction (inward from left edge)
            perp_angle = left_angle + np.pi / 2
            px_end = int(px_center + scale_line_length * np.cos(perp_angle))
            py_end = int(py_center + scale_line_length * np.sin(perp_angle))

            if 0 <= px_center < output_size and 0 <= py_center < output_size:
                # cv2.line(cone_image, (px_center, py_center), (px_end, py_end), scale_color, 2)
                cv2.circle(cone_image, (px_center, py_center), 1, scale_color, -1)
        # Right edge (positive angle limit)
        right_angle = angle_start + self.sonar_fov_rad
        for dist in np.arange(0, self.sonar_range + 1, scale_resolution):
            # Position on right edge
            px_center = int(sonar_x + dist * np.sin(right_angle) * pixels_per_meter)
            py_center = int(sonar_y - dist * np.cos(right_angle) * pixels_per_meter)

            # Perpendicular direction (inward from right edge)
            perp_angle = right_angle - np.pi / 2
            px_end = int(px_center + scale_line_length * np.cos(perp_angle))
            py_end = int(py_center + scale_line_length * np.sin(perp_angle))

            if 0 <= px_center < output_size and 0 <= py_center < output_size:
                # cv2.line(cone_image, (px_center, py_center), (px_end, py_end), scale_color, 2)
                cv2.circle(cone_image, (px_center, py_center), 1, scale_color, -1)
        # DEBUG: Visualize a random beam with yellow points
        random_beam_idx = int(width // 3)  # For consistent debugging, use the center beam
        beam_angle = angle_start + random_beam_idx * angle_increment
        rospy.logwarn(f"[DEBUG BEAM] Random beam index: {random_beam_idx}, angle: {beam_angle}")

        for range_idx in range(height):
            pixel = sonar_image[range_idx, random_beam_idx, :]
            if np.all(pixel == 0):
                continue

            distance = range_idx * range_height_ratio
            px = int(sonar_x + distance * np.sin(beam_angle) * pixels_per_meter)
            py = int(sonar_y - distance * np.cos(beam_angle) * pixels_per_meter)

            if 0 <= px < output_size and 0 <= py < output_size:
                cv2.circle(cone_image, (px, py), 1, (0, 255, 255), -1)  # Yellow points

        # DEBUG: Visualize specific ranges - small and large
        small_range_idx = int(height * 0.1)  # 10% from start (close to sensor)
        large_range_idx = int(height * 0.9)  # 90% from start (far from sensor)

        small_distance = small_range_idx * range_height_ratio
        large_distance = large_range_idx * range_height_ratio
        rospy.logwarn(f"[DEBUG RANGES] Small range idx: {small_range_idx} (dist: {small_distance:.2f}m), Large range idx: {large_range_idx} (dist: {large_distance:.2f}m)")

        # Mark all beams at small range with red circles
        for beam_idx in range(width):
            beam_angle = angle_start + beam_idx * angle_increment
            px = int(sonar_x + small_distance * np.sin(beam_angle) * pixels_per_meter)
            py = int(sonar_y - small_distance * np.cos(beam_angle) * pixels_per_meter)
            if 0 <= px < output_size and 0 <= py < output_size:
                cv2.circle(cone_image, (px, py), 1, (0, 0, 255), -1)  # Red circles

        # Mark all beams at large range with cyan circles
        for beam_idx in range(width):
            beam_angle = angle_start + beam_idx * angle_increment
            px = int(sonar_x + large_distance * np.sin(beam_angle) * pixels_per_meter)
            py = int(sonar_y - large_distance * np.cos(beam_angle) * pixels_per_meter)
            if 0 <= px < output_size and 0 <= py < output_size:
                cv2.circle(cone_image, (px, py), 1, (255, 255, 0), -1)  # Cyan circles

        return cone_image



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

           

            # Publish cone projection
            cone_projection = self.sonar_image_to_cone_projection(sonar_image)
            cone_msg = self.cv_bridge.cv2_to_imgmsg(cone_projection, encoding="bgr8")
            cone_msg.header.stamp = rospy.Time.now()
            cone_msg.header.frame_id = self.sonar_frame
            self.cone_projection_pub.publish(cone_msg)

            image_msg = self.cv_bridge.cv2_to_imgmsg(sonar_image, encoding="bgr8")
            image_msg.header.stamp = rospy.Time.now()
            image_msg.header.frame_id = self.sonar_frame
            self.image_pub.publish(image_msg)

            


if __name__ == "__main__":
    sonar = SonoptixEchoDriver()
    rospy.spin()
