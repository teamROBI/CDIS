import numpy as np
from PIL import Image
import open3d as o3d
import imageio
from tqdm import tqdm
import math
import os
import json
import matplotlib.pyplot as plt

def get_number_of_images(poses_path):
    i = 0
    while os.path.isfile(os.path.join(poses_path, str(i) + '.txt')): i += 1 #20
    return i

class Camera:
    def __init__(self,
                 images,
                 intrinsic_path, 
                 intrinsic_resolution, 
                 poses_path, 
                 depths_path, 
                 extension_depth, 
                 depth_scale,
                 syn_depths_path,
                 extension_syn_depth,
                 depth_supplemented,
                 indices=None,
                 depths=None):
        self.images = images
        self.intrinsic = np.loadtxt(intrinsic_path)[:3, :3]
        self.intrinsic_original_resolution = intrinsic_resolution
        self.poses_path = poses_path
        self.depths_path = depths_path
        self.extension_depth = extension_depth
        self.depth_scale = depth_scale
        
        self.syn_depths_path = syn_depths_path
        self.extension_syn_depth = extension_syn_depth
        self.depth_supplemented = depth_supplemented
        
        if depths is not None:
            self.depths = depths
            print("[INFO] Depth images sampled!")
        elif indices is not None:
            self.depths = self.load_depths(indices)
            print("[INFO] Depth images loaded!")
        else:
            raise(NotImplementedError)
    
    def get_adapted_intrinsic(self, desired_resolution):
        '''Get adjusted camera intrinsics.'''
        if self.intrinsic_original_resolution == desired_resolution:
            return self.intrinsic
        
        resize_width = int(math.floor(desired_resolution[1] * float(
                        self.intrinsic_original_resolution[0]) / float(self.intrinsic_original_resolution[1])))
        
        adapted_intrinsic = self.intrinsic.copy()
        adapted_intrinsic[0, 0] *= float(resize_width) / float(self.intrinsic_original_resolution[0])
        adapted_intrinsic[1, 1] *= float(desired_resolution[1]) / float(self.intrinsic_original_resolution[1])
        adapted_intrinsic[0, 2] *= float(desired_resolution[0] - 1) / float(self.intrinsic_original_resolution[0] - 1)
        adapted_intrinsic[1, 2] *= float(desired_resolution[1] - 1) / float(self.intrinsic_original_resolution[1] - 1)
        return adapted_intrinsic
    
    def load_poses(self, indices):
        path = os.path.join(self.poses_path, str(0) + '.txt')
        shape = np.linalg.inv(np.loadtxt(path))[:3, :].shape
        poses = np.zeros((len(indices), shape[0], shape[1]))
        for i, idx in enumerate(indices):
            path = os.path.join(self.poses_path, str(idx) + '.txt')
            poses[i] = np.linalg.inv(np.loadtxt(path))[:3, :]
        return poses
    
    def load_depths(self, indices):
        depths = []
        for idx in tqdm(indices, desc="[INFO] Loading Depth images"):
            depths.append(self.load_depth(idx))
            
            # if idx in [0, 60, 1080]:
            #     # Visualize the depth map
            #     plt.figure(figsize=(6, 6))
            #     plt.imshow(depths[-1], cmap='viridis')  # You can change colormap as needed
            #     plt.colorbar(label="Depth Value")
            #     plt.title(f"Depth Map for Index {idx}")
            #     plt.axis("off")
            #     plt.show()
                
        return depths
      
    def load_depth(self, idx):
        depth_path = os.path.join(self.depths_path, str(idx) + self.extension_depth)
        sensor_depth = imageio.v2.imread(depth_path) / self.depth_scale
        
        if self.depth_supplemented:
            syn_depths_path = os.path.join(self.syn_depths_path, str(idx) + self.extension_syn_depth)
            if self.extension_syn_depth == '.png':
                syn_depth = imageio.v2.imread(syn_depths_path) / self.depth_scale
            else:
                syn_depth = np.load(syn_depths_path)["synthetic_depth"].astype(np.float32)
            
            supplemented_depth = np.where(sensor_depth == 0, syn_depth, sensor_depth)
            
            return supplemented_depth
        
        return sensor_depth


class Images:
    def __init__(self,
                 images_path,
                 extension,
                 indices,
                 images=None,
                 inst_matched_masks_2d=None,
                 load_rgb=True):
        self.images_path = images_path
        self.extension = extension
        self.indices = indices

        if images is not None:
            self.images = images
            print("[INFO] RGB images sampled!")
        elif load_rgb:
            self.images = self.load_images()
            print("[INFO] RGB images loaded!")
        else:
            # RGB only needed for debug visualization; skip the decode otherwise.
            self.images = None
        self.masks_2d = inst_matched_masks_2d

    def load_images(self):
        images = []
        for idx in tqdm(self.indices, desc="[INFO] Loading PIL RGB images"):
            img_path = os.path.join(self.images_path, str(idx) + self.extension)
            images.append(Image.open(img_path).convert("RGB"))
        return images
        
    def load_image(self, idx):
        img_path = os.path.join(self.images_path, str(idx) + self.extension)
        return Image.open(img_path).convert("RGB")
    
    def load_image_path(self, idx):
        return os.path.join(self.images_path, str(self.indices[idx]) + self.extension)
    
    def get_as_np_list(self):
        images = []
        for i in range(len(self.images)):
            images.append(np.asarray(self.images[i]))
        return images
    
class SuperPointMasks3D:
    def __init__(self, spp_path):
        self.masks = self.load_spp_masks(spp_path) # Shape: (num_points, num_superpoints)
        self.num_masks = self.masks.shape[1]
        
    def load_spp_masks(self, spp_path):
        # Load the JSON file
        with open(spp_path, 'r') as f:
            data = json.load(f)
        
        seg_indices = np.array(data["segIndices"])  # Shape: (num_points,)

        # Get unique superpoint IDs and create a mapping to contiguous indices
        unique_ids, inverse_indices = np.unique(seg_indices, return_inverse=True)
        # unique_ids: the actual superpoint IDs
        # inverse_indices: maps each original seg_id to a row index (0..N-1)

        num_points = len(seg_indices)
        num_superpoints = len(unique_ids)

        # Build one-hot mask: shape (num_points, num_superpoints)
        masks = np.zeros((num_points, num_superpoints), dtype=bool)
        masks[np.arange(num_points), inverse_indices] = True

        return masks
        
    def update_masks(self, new_masks):
        print(f"[INFO] New mask updated {self.num_masks} -> {new_masks.shape[1]}")
        self.masks = new_masks
        self.num_masks = new_masks.shape[1]
    
    
class PointCloud:
    def __init__(self, 
                 point_cloud_path,
                 superpoints_path=None,
                 visualize=False):
        self.pcd = o3d.io.read_point_cloud(point_cloud_path)
        self.points = np.asarray(self.pcd.points)
        self.colors = np.asarray(self.pcd.colors)
        self.num_points = self.points.shape[0]
        self.visualize = visualize
        
        if self.visualize:
            o3d.visualization.draw_geometries([self.pcd])
        
        if superpoints_path is not None:
            self.superpoints = self.load_super_points(superpoints_path)
            
    def get_homogeneous_coordinates(self):
        return np.append(self.points, np.ones((self.num_points,1)), axis = -1)
    
    def load_super_points(self, superpoints_path):
        with open(superpoints_path, 'r') as file:
            data = json.load(file)
            
        superpoints = np.array(data['segIndices'])
        print(f"[INFO] Superpoints loaded!")
        
        if self.visualize:
            self.visualize_3D_mask(superpoints)
        
        return superpoints
    
    def visualize_3D_mask(self, masks, title=None):
        """
        Visualizes the 3D point cloud with different colors based on the given masks.
        
        Parameters:
        - masks: A boolean numpy array of shape (num_points, num_masks), where each column represents a mask.
        """
        
        if masks.shape[0] != self.num_points:
            raise ValueError("Mask shape does not match the number of points in the point cloud.")
        if len(masks.shape) == 1:
            unique_ids = np.unique(masks)
            num_masks = len(unique_ids)
            mask_colors = np.random.rand(num_masks, 3)  # Random colors for each mask
            colored_points = np.zeros_like(self.colors)
            
            for idx, id in enumerate(unique_ids):
                colored_points[masks == id] = mask_colors[idx]  # Assign unique color to each mask
        else:
            # Assign colors based on masks
            num_masks = masks.shape[1]
            mask_colors = np.random.rand(num_masks, 3)  # Random colors for each mask
            colored_points = np.zeros_like(self.colors)
            
            for mask_idx in range(num_masks):
                mask = masks[:, mask_idx].astype(bool)
                colored_points[mask] = mask_colors[mask_idx]  # Assign unique color to each mask
            
        # Create a new point cloud object to visualize
        pcd_masked = o3d.geometry.PointCloud()
        pcd_masked.points = o3d.utility.Vector3dVector(self.points)
        pcd_masked.colors = o3d.utility.Vector3dVector(colored_points)
        
        # Visualize the point cloud
        if title is not None:
            o3d.visualization.draw_geometries([pcd_masked], window_name=title)
        else:
            o3d.visualization.draw_geometries([pcd_masked])
        
    