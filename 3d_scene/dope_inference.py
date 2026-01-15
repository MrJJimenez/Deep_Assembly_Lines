"""
DOPE (Deep Object Pose Estimation) inference module.

This module provides 6D pose estimation for objects using the DOPE framework.
It wraps the DOPE detector and provides a clean interface for detecting objects
and drawing their 3D bounding boxes.
"""

import os
import sys
import cv2
import yaml
import time
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

# Add DOPE framework to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "frameworks", "dope"))
from cuboid import Cuboid3d
from cuboid_pnp_solver import CuboidPNPSolver
from detector import ModelData, ObjectDetector
from utils import Draw


class DOPEDetector:
    """DOPE-based 6D pose estimator for object detection.
    
    This class wraps the DOPE framework to provide 6D pose estimation
    (position + orientation) for trained object classes.
    
    Attributes:
        class_name: Name of the object class being detected
        draw_color: RGB color tuple for drawing detections
        dimension: Object dimensions in cm (x, y, z)
    """
    
    def __init__(self, config_path, camera_info_path, weight_path, class_name="tool"):
        """Initialize the DOPE detector.
        
        Args:
            config_path: Path to DOPE config YAML file
            camera_info_path: Path to camera info YAML file
            weight_path: Path to trained weights (.pth file)
            class_name: Name of the object class to detect
        """
        self.class_name = class_name
        self.weight_path = weight_path
        
        # Load configurations
        with open(config_path) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        with open(camera_info_path) as f:
            self.camera_info = yaml.load(f, Loader=yaml.FullLoader)
        
        self.input_is_rectified = self.config["input_is_rectified"]
        self.downscale_height = self.config["downscale_height"]
        
        # Detection configuration
        self.config_detect = lambda: None
        self.config_detect.mask_edges = 1
        self.config_detect.mask_faces = 1
        self.config_detect.vertex = 1
        self.config_detect.threshold = 0.5
        self.config_detect.softmax = 1000
        self.config_detect.thresh_angle = self.config["thresh_angle"]
        self.config_detect.thresh_map = self.config["thresh_map"]
        self.config_detect.sigma = self.config["sigma"]
        self.config_detect.thresh_points = self.config["thresh_points"]
        
        # Load neural network model
        # Set parallel=True to handle DDP-trained weights with 'module.' prefix
        self.model = ModelData(
            name=class_name,
            net_path=weight_path,
            parallel=True
        )
        self.model.load_net_model()
        print(f"[DOPE] Model loaded for class: {class_name}")
        
        # Get draw color
        try:
            self.draw_color = tuple(self.config["draw_colors"][class_name])
        except:
            self.draw_color = (0, 255, 0)
        
        # Get object dimensions (in cm)
        self.dimension = tuple(self.config["dimensions"][class_name])
        self.class_id = self.config["class_ids"][class_name]
        
        # Create PNP solver
        self.pnp_solver = CuboidPNPSolver(
            class_name, 
            cuboid3d=Cuboid3d(self.config["dimensions"][class_name])
        )
        
        # Setup camera matrices
        self._setup_camera_matrices()
        
        print(f"[DOPE] Initialized - Object dimensions (cm): {self.dimension}")
    
    def _setup_camera_matrices(self):
        """Setup camera intrinsic matrices from camera info."""
        if self.input_is_rectified:
            P = np.matrix(
                self.camera_info["projection_matrix"]["data"], dtype="float64"
            ).copy()
            P.resize((3, 4))
            self.camera_matrix = P[:, :3]
            self.dist_coeffs = np.zeros((4, 1))
        else:
            self.camera_matrix = np.matrix(
                self.camera_info["camera_matrix"]["data"], dtype="float64"
            )
            self.camera_matrix.resize((3, 3))
            self.dist_coeffs = np.matrix(
                self.camera_info["distortion_coefficients"]["data"], dtype="float64"
            )
            self.dist_coeffs.resize((5, 1))
    
    def detect(self, frame):
        """Run 6D pose detection on a frame.
        
        Args:
            frame: BGR image (numpy array from OpenCV)
            
        Returns:
            dict with detection results containing:
                - detected: bool, True if object was detected
                - location: [x, y, z] in meters
                - quaternion: [x, y, z, w] orientation
                - projected_points: 2D bounding box corners
                - timestamp: detection timestamp
            Returns None if no detection
        """
        # Convert BGR to RGB
        frame_rgb = frame[..., ::-1].copy()
        
        # Get original dimensions
        height, width, _ = frame_rgb.shape
        
        # Calculate scaling factor
        scaling_factor = float(self.downscale_height) / height
        
        # Create a copy of camera matrix for this frame
        camera_matrix = self.camera_matrix.copy()
        
        if scaling_factor < 1.0:
            camera_matrix[:2] *= scaling_factor
            frame_rgb = cv2.resize(
                frame_rgb, 
                (int(scaling_factor * width), int(scaling_factor * height))
            )
        
        # Update PNP solver camera parameters
        self.pnp_solver.set_camera_intrinsic_matrix(camera_matrix)
        self.pnp_solver.set_dist_coeffs(self.dist_coeffs)
        
        # Run object detection
        results, _ = ObjectDetector.detect_object_in_image(
            self.model.net, 
            self.pnp_solver, 
            frame_rgb, 
            self.config_detect,
            grid_belief_debug=False
        )
        
        if not results:
            return None
        
        # Get the best detection (first result)
        for result in results:
            if result["location"] is not None:
                # Convert location from cm to meters
                location_m = [
                    result["location"][0] / 100.0,
                    result["location"][1] / 100.0,
                    result["location"][2] / 100.0
                ]
                
                return {
                    "detected": True,
                    "location": location_m,
                    "quaternion": list(result["quaternion"]),  # x, y, z, w
                    "projected_points": result["projected_points"],
                    "timestamp": time.time()
                }
        
        return None
    
    def draw_detection(self, frame, detection_result):
        """Draw 3D bounding box and coordinate axes on frame.
        
        Args:
            frame: BGR image to draw on
            detection_result: Detection result dict from detect()
            
        Returns:
            Frame with drawn annotations (BGR format)
        """
        if detection_result is None or not detection_result["detected"]:
            return frame
        
        # Convert BGR to RGB for PIL
        frame_rgb = frame[..., ::-1].copy()
        
        # Resize if needed
        height, width, _ = frame_rgb.shape
        scaling_factor = float(self.downscale_height) / height
        camera_matrix = self.camera_matrix.copy()
        
        if scaling_factor < 1.0:
            camera_matrix[:2] *= scaling_factor
            frame_rgb = cv2.resize(
                frame_rgb, 
                (int(scaling_factor * width), int(scaling_factor * height))
            )
        
        # Create PIL image for drawing
        im = Image.fromarray(frame_rgb)
        draw = Draw(im)
        
        # Draw the 3D bounding box
        projected_points = detection_result["projected_points"]
        if projected_points is not None and len(projected_points) > 0:
            # Check if any point is None (handle both list and numpy array cases)
            has_none = False
            try:
                for pt in projected_points:
                    if pt is None:
                        has_none = True
                        break
            except (TypeError, ValueError):
                has_none = False
            
            if not has_none:
                points2d = [tuple(pair) for pair in projected_points]
                draw.draw_cube(points2d, self.draw_color)
        
        # Draw coordinate axes at object centroid
        location = detection_result["location"]
        quaternion = detection_result["quaternion"]
        
        # Convert location back to cm for drawing
        location_cm = [loc * 100.0 for loc in location]
        self._draw_coordinate_system(draw, camera_matrix, self.dist_coeffs, 
                                      location_cm, quaternion, axis_length=10)
        
        # Convert back to BGR for OpenCV
        result_frame = np.array(im)[..., ::-1].copy()
        return result_frame
    
    def _draw_coordinate_system(self, draw, camera_matrix, dist_coeffs, location, quaternion, axis_length=10):
        """Draw 3D coordinate axes at object position.
        
        Args:
            draw: PIL Draw object
            camera_matrix: Camera intrinsic matrix
            dist_coeffs: Distortion coefficients
            location: Object location in cm [x, y, z]
            quaternion: Object orientation [x, y, z, w]
            axis_length: Length of coordinate axes in cm
        """
        # Convert quaternion to rotation matrix
        rot = Rotation.from_quat(quaternion)
        rotation_matrix = rot.as_matrix()
        
        # Define axis endpoints in object frame
        axes_3d = np.array([
            [axis_length, 0, 0],  # X axis (red)
            [0, axis_length, 0],  # Y axis (green)
            [0, 0, axis_length],  # Z axis (blue)
        ])
        
        # Transform axes to world frame
        axes_world = np.dot(rotation_matrix, axes_3d.T).T + location
        
        # Project centroid and axes endpoints to 2D
        centroid_3d = np.array(location).reshape(3, 1)
        axes_world_3d = axes_world.T
        
        centroid_2d, _ = cv2.projectPoints(centroid_3d.T, np.zeros(3), np.zeros(3), 
                                            camera_matrix, dist_coeffs)
        axes_2d, _ = cv2.projectPoints(axes_world_3d.T, np.zeros(3), np.zeros(3),
                                         camera_matrix, dist_coeffs)
        
        # Extract 2D coordinates
        centroid_2d = tuple(centroid_2d[0][0].astype(int))
        x_axis_2d = tuple(axes_2d[0][0].astype(int))
        y_axis_2d = tuple(axes_2d[1][0].astype(int))
        z_axis_2d = tuple(axes_2d[2][0].astype(int))
        
        # Draw axes
        draw.draw_line(centroid_2d, x_axis_2d, line_color=(255, 0, 0), line_width=3)
        draw.draw_line(centroid_2d, y_axis_2d, line_color=(0, 255, 0), line_width=3)
        draw.draw_line(centroid_2d, z_axis_2d, line_color=(0, 0, 255), line_width=3)


def load_dope_detector(weights_path, config_path, camera_info_path, class_name="tool"):
    """Load and initialize a DOPE detector.
    
    Args:
        weights_path: Path to trained weights (.pth file)
        config_path: Path to DOPE config YAML file
        camera_info_path: Path to camera info YAML file
        class_name: Name of the object class to detect
        
    Returns:
        DOPEDetector instance or None if loading fails
    """
    try:
        if not os.path.exists(weights_path):
            print(f"[DOPE] Weights not found: {weights_path}")
            return None
        if not os.path.exists(config_path):
            print(f"[DOPE] Config not found: {config_path}")
            return None
        if not os.path.exists(camera_info_path):
            print(f"[DOPE] Camera info not found: {camera_info_path}")
            return None
            
        print(f"[DOPE] Loading model: {weights_path}")
        detector = DOPEDetector(
            config_path=config_path,
            camera_info_path=camera_info_path,
            weight_path=weights_path,
            class_name=class_name
        )
        print(f"[DOPE] Model ready")
        return detector
        
    except Exception as e:
        print(f"[DOPE] Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_empty_pose():
    """Create an empty/default pose dictionary.
    
    Returns:
        dict with default pose values
    """
    return {
        "detected": False,
        "fresh": False,
        "location": [0, 0, 0],
        "quaternion": [0, 0, 0, 1],
        "projected_points": [],
        "timestamp": 0
    }
