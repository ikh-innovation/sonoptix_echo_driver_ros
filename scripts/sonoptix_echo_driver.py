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
        self.flip_output_x_sonar_image_cone = rospy.get_param(
            "~flip_output_x_sonar_image_cone", True
        )
        self.flip_output_y_sonar_image_cone = rospy.get_param(
            "~flip_output_y_sonar_image_cone", True
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
        self.mutex  = threading.Lock()
        self.mutex_sonar_capture = threading.Lock()
        self.cv_bridge = CvBridge()
        self.sonar_capture = None
        self.latest_sonar_image = None
        self.frame_reader_thread = None
        self.frame_reader_should_stop = False

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
        print("[DEBUG] debugging_image_callback: CALLED")
        sonar_image = self.cv_bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
        print("[DEBUG] debugging_image_callback: Image converted, shape:", sonar_image.shape)
        flip_opt = None
        if self.flip_input_x_sonar_image and self.flip_input_y_sonar_image:
            flip_opt = -1
        elif self.flip_input_y_sonar_image:
            flip_opt = 1
        elif self.flip_input_x_sonar_image:
            flip_opt = 0
        if flip_opt is not None:
            print("[DEBUG] debugging_image_callback: Flipping with option:", flip_opt)
            sonar_image = cv2.flip(sonar_image, flip_opt)


        print("[DEBUG] debugging_image_callback: Publishing to laserscan")
        self.publish_sonar_image_to_laserscan(sonar_image)
        print("[DEBUG] debugging_image_callback: Publishing to image")
        self.publish_sonar_image_to_image(sonar_image)
        print("[DEBUG] debugging_image_callback: DONE")

    def dyn_reconf_callback(self, config, _):
        print("[DEBUG] dyn_reconf_callback: CALLED")
        rospy.logwarn(config)
        print("[DEBUG] dyn_reconf_callback: Acquiring mutex")
        with self.mutex:
            print("[DEBUG] dyn_reconf_callback: Updating configuration")
            self.sonar_frame = config.sonar_frame
            self.image_data_factor = config.image_data_factor
            self.default_distance_value = config.default_distance_value
            self.publish_sonar_image = config.publish_sonar_image
            self.flip_input_x_sonar_image = config.flip_input_x_sonar_image
            self.flip_input_y_sonar_image = config.flip_input_y_sonar_image
            self.flip_output_x_sonar_image = config.flip_output_x_sonar_image
            self.flip_output_y_sonar_image = config.flip_output_y_sonar_image
            self.flip_output_x_sonar_image_cone = config.flip_output_x_sonar_image_cone
            self.flip_output_y_sonar_image_cone = config.flip_output_y_sonar_image_cone
            self.sonar_min_range = config.sonar_min_range
            self.mock_hardware = config.mock_hardware
            self.draw_distance_lines = config.draw_distance_lines
            self.minimum_obstacle_value = config.minimum_obstacle_value
            if config.sonar_range != self.sonar_range:
                print("[DEBUG] dyn_reconf_callback: Range changed, calling set_sonar_range")
                self.set_sonar_range(config.sonar_range)
            if config.enabled != self.sonar_enabled:
                print("[DEBUG] dyn_reconf_callback: Enabled state changed, calling set_sonar_state")
                self.set_sonar_state(config.enabled)
        print("[DEBUG] dyn_reconf_callback: DONE")
        return config

    
    
    def enable_sonar_callback(self, srv):
        print("[DEBUG] enable_sonar_callback: CALLED with data:", srv.data)
        print("[DEBUG] enable_sonar_callback: Acquiring mutex")
        with self.mutex:
            print("[DEBUG] enable_sonar_callback: Calling set_sonar_state")
            self.set_sonar_state(srv.data)
        print("[DEBUG] enable_sonar_callback: DONE")
        return True, ""

    def frame_reader_worker(self):
        """Continuously read frames from sonar capture and keep the latest one."""
        print("[DEBUG] frame_reader_worker: STARTED")
        while not self.frame_reader_should_stop:
            print("[DEBUG] frame_reader_worker: Loop iteration, enabled:", self.sonar_enabled)
            time.sleep(0.0001)
            if not self.sonar_enabled:
                print("[DEBUG] frame_reader_worker: Sonar not enabled, continuing")
                continue

            print("[DEBUG] frame_reader_worker: Attempting to acquire mutex_sonar_capture")
            with self.mutex_sonar_capture:
                print("[DEBUG] frame_reader_worker: Acquired mutex_sonar_capture")
                if not self.mock_hardware:
                    if self.sonar_capture is None or not self.sonar_capture.isOpened():
                        print("[DEBUG] frame_reader_worker: Setting up sonar capture")
                        self.setup_sonar_capture()
                        continue

                    print("[DEBUG] frame_reader_worker: Attempting to read frame")
                    success, frame = self.sonar_capture.read()
                    print("[DEBUG] frame_reader_worker: Read result - success:", success)
                    if not success:
                        self.sonar_capture.release()
                        self.sonar_capture = None
                        rospy.logerr("Frame reader thread: Error reading from sonar image stream")

                        continue

            # Store the latest frame (will overwrite older ones)
            print("[DEBUG] frame_reader_worker: Attempting to acquire mutex for storing frame")
            with self.mutex:
                print("[DEBUG] frame_reader_worker: Acquired mutex, storing frame")
                self.latest_sonar_image = frame
            print("[DEBUG] frame_reader_worker: Frame stored successfully")

    def start_frame_reader_thread(self):
        """Start the dedicated frame reader thread if not already running."""
        print("[DEBUG] start_frame_reader_thread: CALLED")
        if self.frame_reader_thread is None or not self.frame_reader_thread.is_alive():
            print("[DEBUG] start_frame_reader_thread: Creating new thread")
            if not self.mock_hardware:
                print("[DEBUG] start_frame_reader_thread: Setting up sonar capture")
                self.setup_sonar_capture()
            self.latest_sonar_image = None
            self.frame_reader_should_stop = False
            self.frame_reader_thread = threading.Thread(target=self.frame_reader_worker, daemon=True)
            self.frame_reader_thread.start()
            print("[DEBUG] start_frame_reader_thread: Thread started successfully")
            rospy.loginfo("Frame reader thread started")
        else:
            print("[DEBUG] start_frame_reader_thread: Thread already alive, skipping")

    def stop_frame_reader_thread(self):
        """Stop the dedicated frame reader thread."""
        print("[DEBUG] stop_frame_reader_thread: CALLED")
        if self.frame_reader_thread is not None and self.frame_reader_thread.is_alive():
            print("[DEBUG] stop_frame_reader_thread: Setting stop flag")
            self.frame_reader_should_stop = True
            print("[DEBUG] stop_frame_reader_thread: Waiting for thread to join (timeout=1.0)")
            self.frame_reader_thread.join(timeout=1.0)
            print("[DEBUG] stop_frame_reader_thread: Thread join completed")
            rospy.loginfo("Frame reader thread stopped")
        else:
            print("[DEBUG] stop_frame_reader_thread: Thread not alive or None")

    def sonar_read_laserscan_pub(self, _):
        print("[DEBUG] sonar_read_laserscan_pub: CALLED")
        sonar_image = None
        flip_opt = None
        print("[DEBUG] sonar_read_laserscan_pub: Attempting to acquire mutex")
        with self.mutex:
            print("[DEBUG] sonar_read_laserscan_pub: Acquired mutex")
            try:
                if not self.sonar_enabled:
                    print("[DEBUG] sonar_read_laserscan_pub: Sonar not enabled, returning")
                    return

                if self.mock_hardware:
                    print("[DEBUG] sonar_read_laserscan_pub: Using mock hardware")
                    height = 268
                    if self.sonar_range >= 30:
                        height = 1024
                    elif self.sonar_range >= 5:
                        height = 608
                    sonar_image = np.random.randint(
                        0, 256, size=(height, 256, 3), dtype=np.uint8
                    )
                else:
                    # Use the latest frame read by the dedicated thread
                    print("[DEBUG] sonar_read_laserscan_pub: Using real hardware, latest_image is:", self.latest_sonar_image is not None)
                    if self.latest_sonar_image is None:
                        print("[DEBUG] sonar_read_laserscan_pub: No latest image, returning")
                        return
                    sonar_image = self.latest_sonar_image


                if self.flip_input_x_sonar_image and self.flip_input_y_sonar_image:
                    flip_opt = -1
                elif self.flip_input_y_sonar_image:
                    flip_opt = 1
                elif self.flip_input_x_sonar_image:
                    flip_opt = 0
                print("[DEBUG] sonar_read_laserscan_pub: flip_opt:", flip_opt)

            except Exception as e:
                print("[DEBUG] sonar_read_laserscan_pub: Exception:", e)
                rospy.logerr(e)

        if flip_opt is not None:
            print("[DEBUG] sonar_read_laserscan_pub: Flipping image")
            sonar_image = cv2.flip(sonar_image, flip_opt)

        print("[DEBUG] sonar_read_laserscan_pub: Publishing to laserscan")
        self.publish_sonar_image_to_laserscan(sonar_image)
        print("[DEBUG] sonar_read_laserscan_pub: Publishing to image")
        self.publish_sonar_image_to_image(sonar_image)
        print("[DEBUG] sonar_read_laserscan_pub: DONE")

    def set_sonar_state(self, state):
        print("[DEBUG] set_sonar_state: CALLED with state:", state)
        self.sonar_enabled = state
        if self.sonar_enabled:
            print("[DEBUG] set_sonar_state: Enabling sonar, starting frame reader thread")
            self.start_frame_reader_thread()
        else:
            print("[DEBUG] set_sonar_state: Disabling sonar, stopping frame reader thread")
            self.stop_frame_reader_thread()

        if self.mock_hardware:
            print("[DEBUG] set_sonar_state: Mock hardware enabled, returning")
            return
        try:
            pass
            # requests.patch(
            #     self.api_url + "/transponder",
            #     json={
            #         "enable": self.sonar_enabled,
            #         "sonar_range": self.sonar_range,
            #     },
            # )
        except Exception as e:
            rospy.logerr("Failed while enabling sonar:" + str(e))

    def set_sonar_range(self, r):
        print("[DEBUG] set_sonar_range: CALLED with range:", r)
        previous_sonar_state = self.sonar_enabled
        self.sonar_range = r
        self.sonar_fov_rad = (
            120 * self.pi_deg_ratio
            if self.sonar_range <= 30
            else 90 * self.pi_deg_ratio
        )
        print("[DEBUG] set_sonar_range: sonar_fov_rad set to:", self.sonar_fov_rad)
        if self.mock_hardware:
            print("[DEBUG] set_sonar_range: Mock hardware, returning")
            return
        print("[DEBUG] set_sonar_range: Disabling sonar")
        self.set_sonar_state(False)
        if previous_sonar_state:
            print("[DEBUG] set_sonar_range: Waiting 1 second before re-enabling")
            time.sleep(1)
            print("[DEBUG] set_sonar_range: Re-enabling sonar")
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
        print("[DEBUG] setup_sonar_capture: CALLED")
        if self.mock_hardware:
            print("[DEBUG] setup_sonar_capture: Mock hardware, returning")
            return
        print("[DEBUG] setup_sonar_capture: Attempting to connect to rtsp://192.168.45.20:1945/")
        try:
            self.sonar_capture = cv2.VideoCapture("rtsp://192.168.45.20:1945/")
            print("[DEBUG] setup_sonar_capture: VideoCapture created successfully")
        except Exception as e:
            print("[DEBUG] setup_sonar_capture: Exception:", e)
            rospy.logerr("Failed to create sonar image capture:" + str(e))

    def sonar_image_to_ranges(self, sonar_image):
        print("[DEBUG] sonar_image_to_ranges: CALLED with shape:", sonar_image.shape)
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
        print("[DEBUG] sonar_image_to_ranges: DONE, ranges length:", len(ranges))
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
        print("[DEBUG] sonar_image_to_cone_projection: START")
        height, width, channels = sonar_image.shape
        rospy.logwarn(f"[CONE] Input image shape: {sonar_image.shape}")

        range_height_ratio = self.sonar_range / height

        # Calculate output image size based on actual sonar geometry
        print("[DEBUG] sonar_image_to_cone_projection: Calculating output size")
        if self.sonar_fov_rad > 0:
            cone_width_meters = 2.0 * self.sonar_range * np.sin(self.sonar_fov_rad / 2.0)
            pixels_per_meter = height / self.sonar_range
            output_width_pixels = int(cone_width_meters * pixels_per_meter)
            output_height_pixels = height
            output_size = max(output_width_pixels, output_height_pixels)
        else:
            output_size = height

        cone_image = np.zeros((output_size, output_size, channels), dtype=np.uint8)

        # Sonar position at center-bottom
        sonar_x = output_size / 2.0
        sonar_y = output_size - 1.0

        # Calculate angle increment
        print("[DEBUG] sonar_image_to_cone_projection: Calculating angles")
        if self.sonar_fov_rad > 0:
            angle_increment = self.sonar_fov_rad / (width - 1) if width > 1 else 0
            angle_start = -self.sonar_fov_rad / 2.0
        else:
            print("[DEBUG] sonar_image_to_cone_projection: FOV is 0, returning black image")
            return cone_image

        print(f"[DEBUG] sonar_image_to_cone_projection: Main loop - width:{width}, height:{height}")
        # Main beam projection - SIMPLIFIED (no debug visualization)
        for beam_idx in range(width):
            # Calculate angle for this beam
            beam_angle = angle_start + beam_idx * angle_increment

            for range_idx in range(height):
                pixel = sonar_image[range_idx, beam_idx, :]

                if np.all(pixel == 0):
                    continue

                distance = range_idx * range_height_ratio
                px = int(sonar_x + distance * np.sin(beam_angle) * pixels_per_meter)
                py = int(sonar_y - distance * np.cos(beam_angle) * pixels_per_meter)
                cone_image[py, px, :] = pixel

        if not self.draw_distance_lines:
            print("[DEBUG] sonar_image_to_cone_projection: DONE (no distance lines)")
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
        print("[DEBUG] publish_sonar_image_to_laserscan: CALLED")
        laserscan_msg = LaserScan()
        laserscan_msg.header.frame_id = self.sonar_frame
        laserscan_msg.header.stamp = rospy.Time.now()
        laserscan_msg.angle_min = -self.sonar_fov_rad / 2.0
        laserscan_msg.angle_max = self.sonar_fov_rad / 2.0
        laserscan_msg.angle_increment = self.sonar_fov_rad / 256
        laserscan_msg.scan_time = 1.0 / self.pub_hz
        laserscan_msg.range_min = self.sonar_min_range
        laserscan_msg.range_max = self.sonar_range
        print("[DEBUG] publish_sonar_image_to_laserscan: Converting image to ranges")
        laserscan_msg.ranges = self.sonar_image_to_ranges(sonar_image)
        print("[DEBUG] publish_sonar_image_to_laserscan: Publishing message")
        self.laserscan_pub.publish(laserscan_msg)
        print("[DEBUG] publish_sonar_image_to_laserscan: DONE")

    def publish_sonar_image_to_image(self, sonar_image):
        print("[DEBUG] publish_sonar_image_to_image: CALLED")
        if self.publish_sonar_image:
            print("[DEBUG] publish_sonar_image_to_image: Publishing enabled")
            if self.image_data_factor != 1.0:
                print("[DEBUG] publish_sonar_image_to_image: Applying image data factor")
                sonar_image = np.clip(
                    sonar_image.astype(np.float32) * self.image_data_factor, 0, 255
                ).astype(np.uint8)

            flip_opt_cone = None
            sonar_image_flipped = sonar_image
            if self.flip_output_x_sonar_image_cone and self.flip_output_y_sonar_image_cone:
                flip_opt_cone = -1
            elif self.flip_output_y_sonar_image_cone:
                flip_opt_cone = 1
            elif self.flip_output_x_sonar_image_cone:
                flip_opt_cone = 0
            if flip_opt_cone is not None:
                sonar_image_flipped = cv2.flip(sonar_image, flip_opt_cone)


            # Publish cone projection
            print("[DEBUG] publish_sonar_image_to_image: Creating cone projection")
            cone_projection = self.sonar_image_to_cone_projection(sonar_image_flipped)
            print("[DEBUG] publish_sonar_image_to_image: Converting cone to image message")
            cone_msg = self.cv_bridge.cv2_to_imgmsg(cone_projection, encoding="bgr8")
            cone_msg.header.stamp = rospy.Time.now()
            cone_msg.header.frame_id = self.sonar_frame
            print("[DEBUG] publish_sonar_image_to_image: Publishing cone projection")
            self.cone_projection_pub.publish(cone_msg)
            print("[DEBUG] publish_sonar_image_to_image: Cone projection published")


            flip_opt = None
            if self.flip_output_x_sonar_image and self.flip_output_y_sonar_image:
                flip_opt = -1
            elif self.flip_output_y_sonar_image:
                flip_opt = 1
            elif self.flip_output_x_sonar_image:
                flip_opt = 0
            if flip_opt is not None:
                sonar_image_flipped = cv2.flip(sonar_image, flip_opt)


            print("[DEBUG] publish_sonar_image_to_image: Converting sonar image to message")
            image_msg = self.cv_bridge.cv2_to_imgmsg(sonar_image_flipped, encoding="bgr8")
            image_msg.header.stamp = rospy.Time.now()
            image_msg.header.frame_id = self.sonar_frame
            print("[DEBUG] publish_sonar_image_to_image: Publishing sonar image")
            self.image_pub.publish(image_msg)
            print("[DEBUG] publish_sonar_image_to_image: DONE")



            


if __name__ == "__main__":
    sonar = SonoptixEchoDriver()
    rospy.spin()
