# src/datasets/synthetic.py

import pyvips
import torch
from torch.utils.data import Dataset
import os
import cv2
import numpy as np
import kornia.geometry.transform as K
import kornia.augmentation as KA

from src.utils.dataset import read_scannet_gray # We can reuse this simple image reader

import cv2
import numpy as np
from shapely.geometry import Polygon
from math import tan, radians

def clean_polygon(poly):
    if poly is None:
        return None
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty:
        return None
    return poly

def get_overlap_ratio(poly1, poly2):
    P1, P2 = Polygon(poly1), Polygon(poly2)
    P1, P2 = clean_polygon(P1), clean_polygon(P2)

    if not P1.is_valid or not P2.is_valid:
        return 0.0, None

    if not P1.intersects(P2):
        return 0.0, None

    inter = P1.intersection(P2)
    if inter.is_empty:
        return 0.0, None

    return inter.area / min(P1.area, P2.area), inter


class SyntheticHomographyDataset(Dataset):
    def __init__(self, root_dir, list_path, img_resize=(640, 480), num_samples=100,
                 max_size_meters=150, min_size_meters=54, px_per_meter=15, **kwargs):
        """
        Dataset for training on synthetic homography warps.
        Args:
            root_dir (str): Path to the directory where images are stored.
            list_path (str): Path to a text file containing the names of images to use.
            img_resize (tuple): The size (W, H) to resize images to.
        """
        super().__init__()
        self.root_dir = root_dir
        self.img_resize = img_resize
        self.img_w, self.img_h = img_resize
        self.final_resolution = (self.img_h, self.img_w)
        self.num_samples = num_samples

        with open(list_path, 'r') as f:
            self.image_files = [os.path.join(self.root_dir, line.strip()) for line in f.readlines()]

        self.list_of_images = [
            pyvips.Image.new_from_file(el) for el in self.image_files]

        self.org_image_width = self.list_of_images[0].width
        self.org_image_height = self.list_of_images[0].height

        self.max_size = max_size_meters * px_per_meter
        self.min_size = min_size_meters * px_per_meter
        self.out_ratio = self.img_w / self.img_h
        self.px_per_meter = px_per_meter

        self.pitch_range = (-20, 20)  # degrees
        self.yaw_range = (-180, 180)    # degrees
        self.roll_range = (-5, 5)     # degrees
        self.z_range = (15., 200.0)      # meters
        self.x_std = 10.0  # meters
        self.y_std = 10.0  # meters
        self.z_std = 10.0  # meters
        self.pitch_std = 4.0  # degrees
        self.yaw_std = 180.0    # degrees
        self.roll_std = 1.0   # degrees
        self.overlap_threshold = 0.3

        self.debug = False

    def compute_planar_corners(
        self,
        camera_position: list,
        roll: float,
        pitch: float,
        yaw: float,
        fov_degrees: float,
        image_width: int,
        image_height: int
    ):
        """
        Computes the 3D world coordinates of the 4 corners of a camera's image
        plane projected onto a planar surface (Z=0).

        Args:
            height (float): The camera's height above the planar surface (Z=0).
            roll (float): Camera roll angle in degrees. Rotation about the X-axis.
            pitch (float): Camera pitch angle in degrees. Rotation about the Y-axis.
            yaw (float): Camera yaw angle in degrees. Rotation about the Z-axis.
            fov_degrees (float): The camera's horizontal field of view in degrees.
            image_width (int): The width of the camera image in pixels.
            image_height (int): The height of the camera image in pixels.

        Returns:
            dict: A dictionary containing the 3D coordinates of the four corners:
                'top_left', 'top_right', 'bottom_left', 'bottom_right'.
        """
        # Convert angles from degrees to radians
        roll_rad = radians(roll)
        pitch_rad = radians(pitch)
        yaw_rad = radians(yaw)
        fov_rad = radians(fov_degrees)

        C = np.asarray(camera_position, dtype=float).reshape(3)

        # 1. Compute the focal length from the horizontal FOV
        focal_length = (image_width / 2) / tan(fov_rad / 2)

        # 2. Camera Intrinsic Matrix (K)
        cx = image_width / 2
        cy = image_height / 2
        K = np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ])

        # 3. Extrinsic Parameters (Rotation and Translation)
        # Rotation matrices for each axis
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll_rad), -np.sin(roll_rad)],
            [0, np.sin(roll_rad), np.cos(roll_rad)]
        ])
        Ry = np.array([
            [np.cos(pitch_rad), 0, np.sin(pitch_rad)],
            [0, 1, 0],
            [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]
        ])
        Rz = np.array([
            [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
            [np.sin(yaw_rad), np.cos(yaw_rad), 0],
            [0, 0, 1]
        ])
        R_base = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, -1]])
        R = Rz @ Ry @ Rx @ R_base

        t = -R.dot(C).reshape(3,1)   # column vector

        H = K @ np.hstack((R[:, :2], t))
        
        # 5. Projecting the 4 corners of the image plane
        corners = {
            'top_left': [0, 0, 1],
            'top_right': [image_width, 0, 1],
            'bottom_left': [0, image_height, 1],
            'bottom_right': [image_width, image_height, 1]
        }

        world_corners = {}
        for corner_name, img_point in corners.items():
            # Get the inverse homography for projection from image to world
            H_inv = np.linalg.inv(H)
            
            # Multiply by the inverse homography
            world_point_homogeneous = H_inv @ np.array(img_point).reshape(3, 1)
            world_point_2d = world_point_homogeneous[:2] / world_point_homogeneous[2]

            # The z-coordinate should be 0, as we are on the planar surface.
            world_corners[corner_name] = world_point_2d.flatten()

        return world_corners

    def get_view(self, big_img, cam_pos, cam_angles, fov_deg=60, out_size=(400,400)):
        """
        Returns:
            view: warped view (out_size)
            footprint_on_bigimg: Nx2 quad in big_img pixel coordinates (float64)
        """
        out_w, out_h = out_size

        world_coords = self.compute_planar_corners(
            camera_position=cam_pos,
            roll=cam_angles[0],
            pitch=cam_angles[1],
            yaw=cam_angles[2],
            fov_degrees=fov_deg,
            image_width=out_w,
            image_height=out_h
        )
        view_corners = np.array([
            world_coords['top_left'],
            world_coords['top_right'],
            world_coords['bottom_right'],
            world_coords['bottom_left']
        ], dtype=np.float64) * self.px_per_meter
        # Get the source points (corners in the big image)
        src_pts = view_corners.astype(np.float32)
        
        # Define destination points (rectangle of size out_w x out_h)
        dst_pts = np.array([
            [0, 0],               # top-left
            [out_w-1, 0],         # top-right
            [out_w-1, out_h-1],   # bottom-right
            [0, out_h-1],          # bottom-left
        ], dtype=np.float32)
        # Calculate perspective transform matrix
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        corners = np.float32([
            [0, 0],
            [big_img.shape[1], 0],
            [big_img.shape[1], big_img.shape[0]],
            [0, big_img.shape[0]]
        ]).reshape(-1, 1, 2)

        x_min, y_min = src_pts.min(axis=0)

        x_max, y_max = src_pts.max(axis=0)

        # Check against original bounds
        out_of_bounds = (
            x_min < 0 or y_min < 0 or
            x_max > big_img.shape[1] or y_max > big_img.shape[0]
        )
        # Apply perspective transformation to get the view
        view = cv2.warpPerspective(big_img, M, (out_w, out_h), flags=cv2.INTER_LINEAR)
        #view = view[::-1]  # camera x axis is inverted in image space

        return view, view_corners, out_of_bounds

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        idx1 = np.random.randint(0, len(self.list_of_images))
        idx2 = np.random.randint(0, len(self.list_of_images))
        big_img1 = self.list_of_images[idx1].numpy()
        big_img2 = self.list_of_images[idx2].numpy()

        max_x = min(big_img1.shape[1], big_img2.shape[1]) / self.px_per_meter
        max_y = min(big_img1.shape[0], big_img2.shape[0]) / self.px_per_meter
        z_min, z_max = self.z_range

        pitch_min, pitch_max = self.pitch_range
        yaw_min, yaw_max = self.yaw_range
        roll_min, roll_max = self.roll_range

        overlap_ratio = 0.0
        out_of_bounds1 = True
        out_of_bounds2 = True
        while out_of_bounds1 or out_of_bounds2 or overlap_ratio < self.overlap_threshold:
            # Randomly sample camera positions and orientations
            cam1_pos = np.array([
                np.random.uniform(0., max_x),
                np.random.uniform(0., max_y),
                np.random.uniform(z_min, z_max)  # height in meters
            ], dtype=np.float32)
            cam1_angles = np.array([
                np.random.uniform(roll_min, roll_max),   # roll
                np.random.uniform(pitch_min, pitch_max), # pitch
                np.random.uniform(yaw_min, yaw_max)      # yaw
            ], dtype=np.float32)

            cam2_pos = cam1_pos + np.array([
                np.random.normal(0., self.x_std),
                np.random.normal(0., self.y_std),
                np.random.uniform(0., self.z_std)  # height in meters
            ], dtype=np.float32)
            cam2_angles = cam1_angles + np.array([
                np.random.normal(0., self.roll_std),   # roll
                np.random.normal(0., self.pitch_std), # pitch
                np.random.normal(0., self.yaw_std)      # yaw
            ], dtype=np.float32)
            cam2_pos = np.clip(cam2_pos, [0,0,z_min], [max_x, max_y, z_max])
            cam2_angles[2] = (cam2_angles[2] + 180) % 360 - 180  # wrap yaw to [-180,180]
            cam2_angles = np.clip(cam2_angles, [roll_min, pitch_min, yaw_min], [roll_max, pitch_max, yaw_max])

            # Generate two views
            view1, quad1, out_of_bounds1 = self.get_view(big_img1, cam1_pos, cam1_angles, out_size=self.img_resize)
            view2, quad2, out_of_bounds2 = self.get_view(big_img2, cam2_pos, cam2_angles, out_size=self.img_resize)

            # Compute overlap
            overlap_ratio, inter_poly = get_overlap_ratio(quad1, quad2)
        
        if self.debug:
            # Save the camera views
            cv2.imwrite("view1.jpg", view1[..., ::-1])
            cv2.imwrite("view2.jpg", view2[..., ::-1])

            # ------------------------------
            # Visualization of footprints
            # ------------------------------
            vis = big_img1.copy()

            # Draw quads
            cv2.polylines(vis, [quad1.astype(int)], True, (0,0,255), 3)  # red
            cv2.polylines(vis, [quad2.astype(int)], True, (0,255,0), 3)  # green
            
            # Collect all polygon points
            all_pts = np.vstack([quad1, quad2])
            if inter_poly and not inter_poly.is_empty:
                inter_pts = np.array(inter_poly.exterior.coords, dtype=np.float32)
                all_pts = np.vstack([all_pts, inter_pts])
                cv2.fillPoly(vis, [inter_pts.astype(np.int32)], (255, 0, 0))  # blue overlap

            # Compute bounding box of polygons
            x_min, y_min = np.min(all_pts, axis=0).astype(int)
            x_max, y_max = np.max(all_pts, axis=0).astype(int)

            # Add small padding
            pad = 20
            x_min = max(x_min - pad, 0)
            y_min = max(y_min - pad, 0)
            x_max = min(x_max + pad, vis.shape[1])
            y_max = min(y_max + pad, vis.shape[0])

            # Crop visualization
            vis_cropped = vis[y_min:y_max, x_min:x_max]

            # Save and show
            cv2.imwrite("footprints.jpg", vis_cropped[..., ::-1])

        
        # TODO get the homography between the two views
        homography = ...

        # The model's supervision expects certain keys. We'll provide dummy values for 3D-related data
        # to ensure compatibility with the data collator.
        dummy_T = torch.eye(4)
        dummy_K = torch.eye(3) # Intrinsics are not needed for homography supervision
        dummy_depth = torch.tensor([])
        scale = torch.tensor([1.0, 1.0]) # Assume no scaling beyond the initial resize

        data = {
            'image0': view1,  # (1, H, W)
            'image1': view2,  # (1, H, W)
            'homography': homography, # The ground truth homography
            'dataset_name': 'SyntheticHomography',
            'pair_names': ("x", "y"),
            'pair_id': idx,
            # --- Dummy values to satisfy the collate function ---
            'depth0': dummy_depth,
            'depth1': dummy_depth,
            'T_0to1': dummy_T,
            'T_1to0': torch.inverse(dummy_T),
            'K0': dummy_K,
            'K1': dummy_K,
            'scale0': scale,
            'scale1': scale,
        }
        return data