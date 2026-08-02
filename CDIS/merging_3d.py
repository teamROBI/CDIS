import numpy as np
import imageio
from tqdm import tqdm
import os
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from scipy.spatial import ConvexHull
import PIL

from utils.load import (
    Camera, SuperPointMasks3D, Images, PointCloud
)
from utils.util import (
    get_color_for_id
)
from CDIS.merge_ops import (
    create_mask_matrix, compute_iou_3d_matrix,
    find_continuous_ranges, frames_from_ranges,
    post_process_point_indices_dbscan, get_floor_indices_ransac,
    remove_floor_instances, compute_z_score,
)


class PointProjector:
    def __init__(self, camera: Camera, 
                 point_cloud: PointCloud, 
                 masks: SuperPointMasks3D,
                 vis_threshold,
                 indices,
                 debug_vis_mode,
                 debug_out_dir=None,
                 compute_mask_footprints=True):

        self.vis_threshold = vis_threshold
        self.indices = indices
        self.camera = camera
        self.point_cloud = point_cloud
        self.masks = masks
        self.debug_vis_mode = debug_vis_mode
        self.debug_out_dir=debug_out_dir
        self.parallel_processing = True
        self.color_map = {-1: [0, 0, 0]}
        if compute_mask_footprints:
            self.visible_points_in_view_in_mask, self.visible_points_view, self.projected_points, self.resolution = self.get_visible_points_in_view_in_mask()
        else:
            # Superpoint-projection merge only needs point visibility + 2D projections
            # (Eq. 5 is computed directly), not the per-mask HxW footprint array.
            self.visible_points_in_view_in_mask = None
            self.visible_points_view, self.projected_points, self.resolution = self.get_visible_points_view()
        # print(self.visible_points_in_view_in_mask.shape, self.visible_points_view.shape, self.projected_points.shape)
        # print(self.visible_points_in_view_in_mask[0][0])
        # print(self.visible_points_view[0][0])
        # print(self.projected_points[0][0])
        # assert False
    

    def get_visible_points_view(self):
        # -------------------------
        # Initialization
        # -------------------------
        vis_threshold = self.vis_threshold
        indices = self.indices
        poses = self.camera.load_poses(indices)
        X = self.point_cloud.get_homogeneous_coordinates()  # (n_points, 4)
        n_points = self.point_cloud.num_points
        depths_path = self.camera.depths_path        
        resolution = imageio.imread(os.path.join(depths_path, '0.png')).shape
        height = resolution[0]
        width = resolution[1]
        intrinsic = self.camera.get_adapted_intrinsic(resolution)
        
        # Preallocate arrays for results.
        projected_points = np.zeros((len(indices), n_points, 2), dtype=np.int32)
        visible_points_view = np.zeros((len(indices), n_points), dtype=bool)

        print(f"[INFO] Computing the visible points in each view...")
        
        # -------------------------
        # Helper: Process a Single View
        # -------------------------
        def process_view(i):
            idx = indices[i]
            # STEP 1: Project the 3D points to 2D
            projected_points_not_norm = (intrinsic @ poses[i] @ X.T).T
            mask = (projected_points_not_norm[:, 2] != 0)
            # Compute 2D coordinates by dividing by the third coordinate
            points2d = np.column_stack((
                projected_points_not_norm[mask, 0] / projected_points_not_norm[mask, 2],
                projected_points_not_norm[mask, 1] / projected_points_not_norm[mask, 2]
            ))
            proj_points_view = np.zeros((n_points, 2), dtype=np.int32)
            proj_points_view[mask] = points2d.astype(int)
            
            # STEP 2: Occlusion/Visibility check using sensor depth
            if hasattr(self.camera, "depths"):  # Check if 'depths' exists in self.camera
                sensor_depth = self.camera.depths[i]
            else:
                sensor_depth = self.camera.load_depth(idx)  # Load depth if not precomputed
                
            inside_mask = (
                (proj_points_view[:, 0] >= 0) &
                (proj_points_view[:, 1] >= 0) &
                (proj_points_view[:, 0] < width) &
                (proj_points_view[:, 1] < height)
            )
            point_depth = projected_points_not_norm[:, 2]
            if np.any(inside_mask):
                inside_indices = np.where(inside_mask)[0]
                xs = proj_points_view[inside_indices, 0]
                ys = proj_points_view[inside_indices, 1]
                sensor_depth_values = sensor_depth[ys, xs]
                point_depth_inside = point_depth[inside_indices]
                visibility_mask = (np.abs(sensor_depth_values - point_depth_inside) <= vis_threshold)
                inside_mask[inside_indices] = visibility_mask

            visible_points = inside_mask

            # Optional DEBUG visualization
            if self.debug_vis_mode == 1:
                self.visualize_visible_points_view(i, proj_points_view, visible_points, point_depth_inside[visibility_mask])
                
            return i, proj_points_view, visible_points

        # -------------------------
        # Process All Views: Batching frames if parallel_processing is True
        # -------------------------
        if getattr(self, 'parallel_processing', False):
            batch_size = 1  # Number of frames per batch (adjust as needed)
            num_views = len(indices)
            # Create a list of batches; each batch is a list of indices
            batches = [list(range(i, min(i + batch_size, num_views))) 
                    for i in range(0, num_views, batch_size)]
            
            start_time = time.time()
            with ThreadPoolExecutor() as executor:
                # Helper to process a batch of views
                def process_batch(batch):
                    batch_results = []
                    for i in batch:
                        batch_results.append(process_view(i))
                    return batch_results

                futures = [executor.submit(process_batch, batch) for batch in batches]
                completed = 0
                for future in as_completed(futures):
                    batch_results = future.result()  # A list of results for views in this batch
                    for (i, proj_points_view, visible_points) in batch_results:
                        projected_points[i] = proj_points_view
                        visible_points_view[i] = visible_points
                        completed += 1
                    elapsed = time.time() - start_time
                    print(f"\r=> [Processed views: {completed}/{num_views} - Elapsed time: {elapsed:.2f}s]", end='', flush=True)
            print()  # Move to next line after completion.
        else:
            # Serial processing if not using parallel_processing.
            start_time = time.time()
            for i, idx in enumerate(tqdm(indices)):
                _, proj_points_view, visible_points = process_view(i)
                projected_points[i] = proj_points_view
                visible_points_view[i] = visible_points
                elapsed = time.time() - start_time
                # print(f"\rProcessed views: {i+1}/{len(indices)} - Elapsed time: {elapsed:.2f}s", end='', flush=True)
            print()

        return visible_points_view, projected_points, resolution
    
    def visualize_visible_points_view(self, i, projected_points, visible_points_view, visible_point_depth_inside=None):
        if isinstance(self.camera.images.images[i], PIL.Image.Image):
            img = np.array(self.camera.images.images[i])  # Convert PIL image to NumPy array
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # Convert RGB (PIL) to BGR (OpenCV format) 
        else:
            img = cv2.cvtColor(self.camera.images.images[i], cv2.COLOR_RGB2BGR)
        
        # Extract visible points
        visible_points = projected_points[visible_points_view]
        
        if visible_point_depth_inside is not None:
                # Normalize depth values for colormap mapping
                depth_min, depth_max = np.min(visible_point_depth_inside), np.max(visible_point_depth_inside)
                if depth_max > depth_min:  # Prevent division by zero
                    normalized_depth = (visible_point_depth_inside - depth_min) / (depth_max - depth_min)
                else:
                    normalized_depth = np.zeros_like(visible_point_depth_inside)  # If all depths are the same, use a single color

                # Map depth values to colors using a colormap
                depth_colormap = (normalized_depth * 255).astype(np.uint8)  # Scale to 0-255
                depth_colormap = cv2.applyColorMap(depth_colormap[:, None], cv2.COLORMAP_JET)[:, 0]  # Apply colormap

                # Overlay visible points with colors corresponding to depth
                for pt, color in zip(visible_points, depth_colormap):
                    x, y = int(pt[0]), int(pt[1])
                    if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:  # Ensure points are within bounds
                        cv2.circle(img, (x, y), radius=2, color=tuple(int(c) for c in color), thickness=-1)  # Color-coded dot

        # Save the result
        output_filename = os.path.join(self.debug_out_dir, "debug_vis_1_visible_points_view")
        os.makedirs(output_filename, exist_ok=True)
        output_filename = os.path.join(output_filename, f"{self.indices[i]}.jpg")
        cv2.imwrite(output_filename, img)
    
    
    def get_visible_points_in_view_in_mask(self):  # for 238 frames 1 min -> 8.3 sec
        masks = self.masks
        num_view = len(self.indices)
        
        ### Get visible points in each image frames(view).
        if hasattr(self, "visible_points_view") and hasattr(self, "projected_points") and hasattr(self, "resolution"):
            visible_points_view, projected_points, resolution = self.visible_points_view, self.projected_points, self.resolution
            print(f"[INFO] Using pre-computed visible points in each view.")
        else:
            self.visible_points_view, self.projected_points, self.resolution = self.get_visible_points_view()  # e.g., (238, 237360) & (238, 237360, 2)
            visible_points_view, projected_points, resolution = self.visible_points_view, self.projected_points, self.resolution
        
        ### Get visible points in each view masks
        visible_points_in_view_in_mask = np.zeros((num_view, masks.num_masks, resolution[0], resolution[1]), dtype=bool)  # e.g., (238, 96, 480, 640)

        print(f"\r[INFO] Computing the visible points in each view in each of the {masks.num_masks} masks...")

        # Helper function to process a single view.
        def process_view(i):
            view_masks = np.zeros((masks.num_masks, resolution[0], resolution[1]), dtype=bool)
            for j in range(masks.num_masks):
                visible_masks_points = np.logical_and(masks.masks[:, j], visible_points_view[i])
                proj_points = projected_points[i][visible_masks_points]
                if proj_points.shape[0] != 0:
                    view_masks[j][proj_points[:, 1], proj_points[:, 0]] = True
                    if self.debug_vis_mode == 2:
                        self.visualize_visible_points_in_view_in_mask(i, j, view_masks[j])
            return i, view_masks

        # Helper function to process a batch of views.
        def process_batch(batch):
            batch_results = []
            for i in batch:
                batch_results.append(process_view(i))
            return batch_results

        # -------------------------
        # Parallel Processing with Batch
        # -------------------------
        if getattr(self, 'parallel_processing', False):
            batch_size = 1  # Number of frames per batch
            # Create batches: list of lists of view indices.
            batches = [list(range(i, min(i + batch_size, num_view)))
                    for i in range(0, num_view, batch_size)]
            start_time = time.time()
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(process_batch, batch) for batch in batches]
                completed = 0
                for future in as_completed(futures):
                    batch_results = future.result()  # List of (i, view_masks) for views in this batch.
                    for (i, view_masks) in batch_results:
                        visible_points_in_view_in_mask[i] = view_masks
                        completed += 1
                    elapsed = time.time() - start_time
                    print(f"\r=> [Processed views: {completed}/{num_view} - Elapsed time: {elapsed:.2f}s]", end='', flush=True)
            print()  # Move to next line after completion.
        else:
            # Serial processing
            start_time = time.time()
            for i in tqdm(range(num_view)):
                for j in range(masks.num_masks):
                    visible_masks_points = np.logical_and(masks.masks[:, j], visible_points_view[i])
                    proj_points = projected_points[i][visible_masks_points]
                    if proj_points.shape[0] != 0:
                        visible_points_in_view_in_mask[i][j][proj_points[:, 1], proj_points[:, 0]] = True
                        if self.debug_vis_mode == 2:
                            self.visualize_visible_points_in_view_in_mask(i, j, visible_points_in_view_in_mask[i][j])
                completed = i + 1
                elapsed = time.time() - start_time
                # print(f"\rProcessed views: {completed}/{num_view} - Elapsed time: {elapsed:.2f}s", end='', flush=True)
            print()

        self.visible_points_in_view_in_mask = visible_points_in_view_in_mask
        self.visible_points_view = visible_points_view
        self.projected_points = projected_points
        self.resolution = resolution
        return visible_points_in_view_in_mask, visible_points_view, projected_points, resolution
    
    
    def visualize_visible_points_in_view_in_mask(self, i, j, proj_points, dilation_size=3):        
        if isinstance(self.camera.images.images[i], PIL.Image.Image):
            img = np.array(self.camera.images.images[i])  # Convert PIL image to NumPy array
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # Convert RGB (PIL) to BGR (OpenCV format) 
        else:
            # img = cv2.cvtColor(self.camera.images.images[i], cv2.COLOR_RGB2BGR)
            # Create a color image from instance IDs
            mask = self.camera.images.masks_2d[i]
            h, w = mask.shape
            img = np.zeros((h, w, 3), dtype=np.uint8)
            for id_ in np.unique(mask):
                img[mask == id_] = get_color_for_id(id_, self.color_map)  # use instance's method
        
        # Convert mask to uint8 (0, 1, 2) for OpenCV processing
        mask_instance = (proj_points == 1).astype(np.uint8) * 255  # Red for instance
        mask_neighbor = (proj_points == 2).astype(np.uint8) * 255  # Blue for neighbor
        
        # Expand the masks using dilation to make points more visible
        kernel = np.ones((dilation_size, dilation_size), np.uint8)
        expanded_mask_instance = cv2.dilate(mask_instance, kernel, iterations=1)
        expanded_mask_neighbor = cv2.dilate(mask_neighbor, kernel, iterations=1)
        
        # Create an overlay: Red where instance is visible, Blue where neighbor is visible
        overlay = np.zeros_like(img, dtype=np.uint8)
        overlay[..., 2][expanded_mask_instance > 0] = 255  # Red channel
        overlay[..., 0][expanded_mask_neighbor > 0] = 255  # Blue channel
        
        # Blend overlay with the original image
        alpha = 0.5  # Transparency factor
        img = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)
        
        # Save the visualization
        if isinstance(j, tuple):
            output_dir = os.path.join(self.debug_out_dir, "debug_vis_2_visible_points_view_neighbor_masks")
        else:
            output_dir = os.path.join(self.debug_out_dir, "debug_vis_2_visible_points_view_mask")
        
        os.makedirs(output_dir, exist_ok=True)
        output_filename = os.path.join(output_dir, f"{i}_{j}.jpg")
        cv2.imwrite(output_filename, img)
        
        # Wait for key press; if Esc is pressed, exit visualization
        # key = cv2.waitKey(0)
        # if key == 27:  # Esc key code
        #     print("[INFO] Esc pressed. Exiting visualization.")
        #     cv2.destroyAllWindows()
        #     assert False
        
        # cv2.destroyAllWindows()
    
    def get_top_k_indices_per_mask_high_ram(self, k, visible_points_in_view_in_mask_neighbors=None, neighbor_pairs=None):
        if visible_points_in_view_in_mask_neighbors is not None:
            print(f"[INFO] Computing topk views for each neighbor masks.")
            # num_points_in_view_in_mask = visible_points_in_view_in_mask_neighbors.sum(axis=(2,3))
            
            # Separate the instance and neighbor visibility counts per view
            num_points_instance = (visible_points_in_view_in_mask_neighbors == 1).sum(axis=(2,3))  # Counts for ID 1
            num_points_neighbor = (visible_points_in_view_in_mask_neighbors == 2).sum(axis=(2,3))  # Counts for ID 2
            
            num_points_in_view_in_mask = np.minimum(num_points_instance, num_points_neighbor)
        else:
            print(f"[INFO] Computing topk views for each masks.")
            num_points_in_view_in_mask = self.visible_points_in_view_in_mask.sum(axis=(2,3))
            
        topk_indices_per_mask = np.argsort(-num_points_in_view_in_mask, axis=0)[:k,:].T
        
        if self.debug_vis_mode == 3:
            self.visualize_topk_visible_points_in_view_in_mask(topk_indices_per_mask, visible_points_in_view_in_mask_neighbors, neighbor_pairs)
        
        return topk_indices_per_mask
    
    def get_top_k_indices_per_mask(self, k, neighbor_pairs=None):
        if neighbor_pairs is not None:
            print(f"[INFO] Computing topk views for each neighbor pair...")
            # Get number of views and number of neighbor pairs.
            num_view = self.visible_points_in_view_in_mask.shape[0]
            num_pairs = len(neighbor_pairs)
            # Initialize an array to hold the visible point counts per view for each pair.
            # Shape: (num_view, num_pairs)
            num_points_in_view_for_pairs = np.zeros((num_view, num_pairs), dtype=int)
            
            # Helper function for threads.
            def compute_visible_counts(pair):
                mask_i, mask_j = pair
                visible_instance = self.visible_points_in_view_in_mask[:, mask_i].sum(axis=(1, 2))
                visible_neighbor = self.visible_points_in_view_in_mask[:, mask_j].sum(axis=(1, 2))
                # Use the minimum of the two counts as the effective visible count.
                return np.minimum(visible_instance, visible_neighbor)
            
            # Use a thread pool to compute visible counts in parallel.
            with ThreadPoolExecutor() as executor:
                # Submit tasks for each neighbor pair.
                futures = {executor.submit(compute_visible_counts, pair): idx for idx, pair in enumerate(neighbor_pairs)}
                completed = 0
                total = num_pairs
                start_time = time.time()
                for future in as_completed(futures):
                    idx = futures[future]
                    num_points_in_view_for_pairs[:, idx] = future.result()
                    completed += 1
                    elapsed = time.time() - start_time
                    print(f"\r=> [Processed pairs: {completed}/{total} - Elapsed time: {elapsed:.2f}s]", end='', flush=True)
            print()  # Move to next line after processing.
            
            # Compute top-k view indices for each neighbor pair.
            # np.argsort(-...) sorts in descending order.
            # The result has shape (num_pairs, k).
            topk_indices_per_mask = np.argsort(-num_points_in_view_for_pairs, axis=0)[:k, :].T
        else:
            print(f"[INFO] Computing topk views for each mask...")
            # Sum visible points over H and W for each view and each mask.
            num_points_in_view_in_mask = self.visible_points_in_view_in_mask.sum(axis=(2,3))
            # Compute top-k view indices for each mask.
            topk_indices_per_mask = np.argsort(-num_points_in_view_in_mask, axis=0)[:k, :].T

        if self.debug_vis_mode == 3:
            self.visualize_topk_visible_points_in_view_in_mask(topk_indices_per_mask, None, neighbor_pairs)

        return topk_indices_per_mask

    
    def visualize_topk_visible_points_in_view_in_mask(self, topk_indices_per_mask, visible_points_in_view_in_mask_neighbors=None, neighbor_pairs=None, draw_convex_hull=True):
        """
        Visualizes the top-k projected 3D instance masks onto 2D images.
        - Draws **red dots** for projected 3D points.
        - **Computes and draws convex hull** around them (green polygon).
        - Saves the images for debugging.

        Parameters:
            topk_indices_per_mask (list): List of top-k best views per mask.
            visible_points_in_view_in_mask_neighbors (optional): Alternative visibility mask.
            neighbor_pairs (optional): Neighbor information for filename saving.
        """
        for mask_idx, topk_view in enumerate(topk_indices_per_mask):
            for view in topk_view:
                if isinstance(self.camera.images.images[view], PIL.Image.Image):
                    img = np.array(self.camera.images.images[view])  # Convert PIL image to NumPy array
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # Convert RGB (PIL) to BGR (OpenCV format) 
                else:
                    img = cv2.cvtColor(self.camera.images.images[view], cv2.COLOR_RGB2BGR)
                
                # Select the correct visibility mask
                if visible_points_in_view_in_mask_neighbors is not None:
                    visibility_mask = visible_points_in_view_in_mask_neighbors[view][mask_idx]
                else:
                    visibility_mask = self.visible_points_in_view_in_mask[view][mask_idx]
                
                # Get coordinates of projected 3D points
                y_coords, x_coords = np.where(visibility_mask)
                projected_2D_points = np.column_stack((x_coords, y_coords))  # Format (x, y) for OpenCV

                # Overlay visible points as red dots
                for x, y in projected_2D_points:
                    if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:  # Ensure points are within bounds
                        cv2.circle(img, (x, y), radius=2, color=(0, 0, 255), thickness=-1)  # Red dot
                
                # Compute and draw convex hull if enough points exist
                if len(projected_2D_points) >= 3 and draw_convex_hull:
                    hull = ConvexHull(projected_2D_points)
                    hull_points = projected_2D_points[hull.vertices]  # Get hull boundary points
                    
                    # Draw convex hull as green polygon
                    cv2.polylines(img, [hull_points.astype(int)], isClosed=True, color=(0, 255, 0), thickness=2)
                
                # Prepare directory for saving
                base_dir = os.path.join(self.debug_out_dir, "topk_visible_points_in_view_in_neighbor_mask" if visible_points_in_view_in_mask_neighbors is not None else 
                                                                "topk_visible_points_in_view_in_mask")
                os.makedirs(base_dir, exist_ok=True)
                
                # Generate filename
                filename = f"{neighbor_pairs[mask_idx]}_{view}.jpg" if visible_points_in_view_in_mask_neighbors is not None else f"{mask_idx}_{view}.jpg"
                output_filename = os.path.join(base_dir, filename)
                
                # Save the visualization
                cv2.imwrite(output_filename, img)

                
                # Wait for key press; if Esc is pressed, exit visualization
                # key = cv2.waitKey(0)
                # if key == 27:  # Esc key code
                #     print("[INFO] Esc pressed. Exiting visualization.")
                #     cv2.destroyAllWindows()
                #     assert False
                
                # cv2.destroyAllWindows()
    
    def get_bbox(self, mask, view):
        if(self.visible_points_in_view_in_mask[view][mask].sum()!=0):
            true_values = np.where(self.visible_points_in_view_in_mask[view, mask])
            valid = True
            t, b, l, r = true_values[0].min(), true_values[0].max()+1, true_values[1].min(), true_values[1].max()+1 
        else:
            valid = False
            t, b, l, r = (0,0,0,0)
        return valid, (t, b, l, r)


def _instance_lookup(intensity_dict):
    """Map tracked global instance ids -> contiguous 0..K-1 (and the inverse)."""
    inst_ids = np.array(sorted(int(u) for u in intensity_dict.keys()))
    lut = np.full(int(inst_ids.max()) + 2, -1, dtype=np.int64) if len(inst_ids) else np.array([-1])
    for k, u in enumerate(inst_ids):
        lut[u] = k
    return inst_ids, lut


def merge_3d(cfg, scene_name, color_names, inst_match_list, intensity_dict):
    """Stage (b)+(c): 3D-Guided 2D Instance Merging and 3D Instance Consolidation.

    Implements the paper's superpoint-projection method (Sec. III-B / III-C):
      1. project every precomputed 3D superpoint into each tracked frame and
         associate it with the tracked 2D instance whose mask it most overlaps (Eq. 5);
      2. merge instances by 3D superpoint-set IoU (Eq. 6);
      3. absorb spatially-overlapping instances that are NOT co-observed in time,
         via 3D IoMin + temporal co-occurrence (Eq. 7-8);
      4. assign each superpoint the instance it is associated with most frequently
         across frames (voting), then expand to per-point labels.

    Returns a 1-D per-point instance-id array over the scene point cloud
    (length = #scene points, -1 = background/unassigned).
    """
    # ------------------------------------------------------------------
    # 1. Load scene geometry, superpoints, tracked masks, camera, projector
    # ------------------------------------------------------------------
    point_cloud_path = os.path.join(cfg.data.pcd_path, f"{scene_name}.ply")
    spp_path = cfg.data.spp_path.replace("scene_name", scene_name)
    process_path = os.path.join(cfg.data.scans_2d_path, scene_name)
    poses_path = os.path.join(process_path, cfg.data.camera.poses_path)
    intrinsic_path = os.path.join(process_path, cfg.data.camera.intrinsic_path)
    depths_path = os.path.join(process_path, cfg.data.depths.depths_path)
    syn_depths_path = os.path.join(cfg.data.depths.synthetic_depth_path, scene_name)
    images_path = os.path.join(process_path, cfg.data.images.images_path)

    spp_masks = SuperPointMasks3D(spp_path)                       # (N, num_sp) one-hot
    num_sp = spp_masks.num_masks
    sp_id = np.argmax(spp_masks.masks, axis=1).astype(np.int32)   # per-point superpoint id
    print(f"[INFO] Superpoints loaded: {num_sp}")

    total_images = len(color_names)
    indices = np.arange(0, total_images, step=cfg.merging_3d.image_iter)
    frames = inst_match_list[::cfg.merging_3d.image_iter]
    assert len(frames) == len(indices), f"frame/index mismatch {len(frames)} vs {len(indices)}"

    have_color = bool(frames) and "color" in frames[0]
    have_depth = bool(frames) and "depth" in frames[0]
    images = Images(
        images_path=images_path,
        extension=cfg.data.images.images_ext,
        indices=indices,
        images=[f["color"] for f in frames] if have_color else None,
        inst_matched_masks_2d=[f["mask_2d"] for f in frames],
        load_rgb=(cfg.debug.merging_3d_vis_mode != 0),  # RGB only used by debug viz
    )
    pointcloud = PointCloud(point_cloud_path, visualize=False)
    camera = Camera(
        images=images,
        intrinsic_path=intrinsic_path,
        intrinsic_resolution=cfg.data.camera.intrinsic_resolution,
        poses_path=poses_path,
        depths_path=depths_path,
        extension_depth=cfg.data.depths.depths_ext,
        depth_scale=cfg.data.depths.depth_scale,
        syn_depths_path=syn_depths_path,
        extension_syn_depth=cfg.data.depths.synthetic_depth_ext,
        depth_supplemented=cfg.data.depths.depth_supplemented,
        indices=indices,
        depths=[f["depth"] for f in frames] if have_depth else None,
    )
    projector = PointProjector(
        camera, pointcloud, spp_masks, cfg.merging_3d.vis_threshold, indices,
        cfg.debug.merging_3d_vis_mode, cfg.debug.debug_out_dir,
        compute_mask_footprints=False,
    )
    visible = projector.visible_points_view      # (V, N) bool
    proj = projector.projected_points            # (V, N, 2) int  (x, y)
    masks_2d = images.masks_2d                    # list of V (H, W) int arrays

    scene_coord = np.copy(np.asarray(pointcloud.points))
    scene_colors = np.copy(np.asarray(pointcloud.colors))
    num_points = scene_coord.shape[0]

    # ------------------------------------------------------------------
    # 2. Superpoint <-> instance association per frame (Eq. 5) + frame voting
    # ------------------------------------------------------------------
    inst_ids, lut = _instance_lookup(intensity_dict)
    K = len(inst_ids)
    if K == 0:
        print(f"[WARNING] {scene_name}: no tracked instances; empty prediction")
        return np.full(num_points, -1)
    max_id = int(inst_ids.max())

    assoc_overlap_th = float(getattr(cfg.merging_3d, "assoc_overlap_th", 0.0))
    vote_count = np.zeros((num_sp, K), dtype=np.int32)   # frames each superpoint -> instance k
    frame_track = {k: set() for k in range(K)}           # frames each instance appears (Eq. 8)
    H, W = masks_2d[0].shape
    tmp = np.zeros((num_sp, K), dtype=np.int32)

    for i in range(len(frames)):
        vis_i = visible[i]
        if not vis_i.any():
            continue
        pix = proj[i][vis_i]
        sp_i = sp_id[vis_i]
        xs = np.clip(pix[:, 0], 0, W - 1)
        ys = np.clip(pix[:, 1], 0, H - 1)
        lab = masks_2d[i][ys, xs]                         # tracked instance id at each projected point
        in_range = (lab >= 0) & (lab <= max_id)
        klab = np.full(lab.shape, -1, dtype=np.int64)
        klab[in_range] = lut[lab[in_range]]
        valid = klab >= 0
        if not valid.any():
            continue
        # Per-frame overlap counts -> per-superpoint argmax instance (Eq. 5).
        # Footprint |Pi_t(s)| = all visible projected points of the superpoint (incl. background).
        sp_footprint = np.bincount(sp_i, minlength=num_sp)
        # Scatter-count into (num_sp x K) via a flat bincount (much faster than np.add.at,
        # identical integer result).
        flat = sp_i[valid].astype(np.int64) * K + klab[valid]
        tmp = np.bincount(flat, minlength=num_sp * K).reshape(num_sp, K)
        active_sp = np.where(tmp.sum(axis=1) > 0)[0]
        a = tmp[active_sp].argmax(axis=1)
        # Gate weak associations: winning instance must cover >= assoc_overlap_th of the footprint.
        frac = tmp[active_sp, a] / np.maximum(sp_footprint[active_sp], 1)
        gate = frac >= assoc_overlap_th
        active_sp, a = active_sp[gate], a[gate]
        vote_count[active_sp, a] += 1
        for k in np.unique(a):
            frame_track[int(k)].add(i)

    # Min-association gate: drop votes from superpoints seen only fleetingly by an instance.
    min_assoc_frames = int(getattr(cfg.merging_3d, "min_assoc_frames", 1))
    if min_assoc_frames > 1:
        vote_count[vote_count < min_assoc_frames] = 0

    # ------------------------------------------------------------------
    # 3. 3D instance consolidation on superpoint sets (Eq. 6-8)
    # ------------------------------------------------------------------
    iou_3d_th = cfg.merging_3d.iou_3d_th
    iom_3d_th = cfg.merging_3d.iom_3d_th
    iom_frame_th = cfg.merging_3d.iom_frame_th
    detach_overlap = bool(getattr(cfg.merging_3d, "detach_overlap", False))  # B2 (Eq. 8)
    # #5: merge schedule. The paper merges Eq.6 hierarchically, pairwise across NEIGHBORING
    # frames; we merge globally greedily by descending IoU. "frame_prox" approximates the
    # paper's schedule (temporally-closest pairs first); "iou_asc" is a control probing
    # whether the schedule matters at all. "iou_desc" (default) is the current behaviour.
    merge_order = str(getattr(cfg.merging_3d, "merge_order", "iou_desc"))

    # Superpoint set per instance = superpoints associated to it in >= 1 frame (Eq. 5 union).
    instances_sp = [np.where(vote_count[:, k] > 0)[0] for k in range(K)]
    intensity_dict_k = {k: int(intensity_dict[int(inst_ids[k])]) for k in range(K)}
    intensity_3d_dict = {k: 1 for k in range(K)}
    frame_track_k = {k: np.array(sorted(frame_track[k])) if frame_track[k] else np.array([0]) for k in range(K)}

    # Drop instances with no associated superpoints, keeping vote_count columns in sync.
    keep_k = [k for k in range(K) if len(instances_sp[k]) > 0]
    vote_count = vote_count[:, keep_k]
    instances_sp = [instances_sp[k] for k in keep_k]
    intensity_dict_k = {new: intensity_dict_k[old] for new, old in enumerate(keep_k)}
    intensity_3d_dict = {new: 1 for new in range(len(keep_k))}
    frame_track_k = {new: frame_track_k[old] for new, old in enumerate(keep_k)}
    print(f"[INFO] Instances with associated superpoints: {len(instances_sp)} / {K}")

    if len(instances_sp) == 0:
        return np.full(num_points, -1)

    # Track how original (post-keep) instance columns fold together as we merge,
    # so the final per-superpoint voting reflects the merges.
    col_groups = [[k] for k in range(len(instances_sp))]

    def _merge_pass(denominator, threshold, use_cooccurrence):
        nonlocal instances_sp, intensity_dict_k, intensity_3d_dict, frame_track_k, col_groups
        merged = True
        while merged and len(instances_sp) > 1:
            mask_matrix = create_mask_matrix(instances_sp)
            score = compute_iou_3d_matrix(mask_matrix, denominator=denominator)
            pairs = np.argwhere(score > threshold)
            if len(pairs) == 0:
                break
            vals = score[pairs[:, 0], pairs[:, 1]]
            if merge_order == "iou_asc":
                order = np.argsort(vals)
            elif merge_order == "frame_prox":
                # Merge temporally-adjacent instances first (approximates the paper's
                # hierarchical neighboring-frame schedule); ties broken by higher IoU.
                med = np.array([np.median(frame_track_k[k]) if len(frame_track_k[k]) else 0.0
                                for k in range(len(instances_sp))])
                dist = np.abs(med[pairs[:, 0]] - med[pairs[:, 1]])
                order = np.lexsort((-vals, dist))
            else:
                order = np.argsort(-vals)
            pairs = pairs[order]
            changed = False
            while len(pairs) > 0:
                a, b = pairs[0]
                if intensity_dict_k[b] > intensity_dict_k[a]:
                    a, b = b, a
                if use_cooccurrence:
                    fa = frames_from_ranges(find_continuous_ranges(frame_track_k[a]))
                    fb = frames_from_ranges(find_continuous_ranges(frame_track_k[b]))
                    denom = min(len(fa), len(fb))
                    co = len(fa & fb) / denom if denom > 0 else 0.0
                    if co >= iom_frame_th:
                        # spatially overlapping but co-observed in time -> distinct objects.
                        # B2 (Eq. 8 detach): remove the shared superpoints from the LARGER
                        # instance, so the overlap (segmentation noise) is cleaned rather than
                        # left contaminating both. Final labels come from vote_count, so also
                        # zero those superpoints' votes in the larger instance's columns.
                        if detach_overlap:
                            inter = np.intersect1d(instances_sp[a], instances_sp[b])
                            if inter.size:
                                larger = a if len(instances_sp[a]) >= len(instances_sp[b]) else b
                                instances_sp[larger] = np.setdiff1d(instances_sp[larger], inter)
                                vote_count[np.ix_(inter, col_groups[larger])] = 0
                        pairs = pairs[1:]
                        continue
                # merge b into a
                instances_sp[a] = np.union1d(instances_sp[a], instances_sp[b])
                intensity_dict_k[a] += intensity_dict_k[b]
                intensity_3d_dict[a] += intensity_3d_dict[b]
                frame_track_k[a] = np.union1d(frame_track_k[a], frame_track_k[b])
                col_groups[a] = col_groups[a] + col_groups[b]
                instances_sp[b] = np.array([], dtype=int)
                pairs = pairs[(pairs[:, 0] != a) & (pairs[:, 1] != a) &
                              (pairs[:, 0] != b) & (pairs[:, 1] != b)]
                changed = True
            # compact emptied instances, keeping col_groups aligned
            valid = [i for i in range(len(instances_sp)) if len(instances_sp[i]) > 0]
            instances_sp = [instances_sp[i] for i in valid]
            intensity_dict_k = {n: intensity_dict_k[o] for n, o in enumerate(valid)}
            intensity_3d_dict = {n: intensity_3d_dict[o] for n, o in enumerate(valid)}
            frame_track_k = {n: frame_track_k[o] for n, o in enumerate(valid)}
            col_groups = [col_groups[i] for i in valid]
            merged = changed

    print("[PROCESSING] Merging instances by 3D superpoint IoU (Eq. 6)...")
    _merge_pass("union", iou_3d_th, use_cooccurrence=False)
    print(f"[INFO] After IoU merge: {len(instances_sp)} instances")
    print("[PROCESSING] Absorbing by IoMin + temporal co-occurrence (Eq. 7-8)...")
    _merge_pass("min", iom_3d_th, use_cooccurrence=True)
    print(f"[INFO] After IoMin absorb: {len(instances_sp)} instances")

    # ------------------------------------------------------------------
    # 4. Final voting: each superpoint -> its most-associated (merged) instance
    # ------------------------------------------------------------------
    G = len(instances_sp)
    grouped_votes = np.zeros((num_sp, G), dtype=np.int64)
    for g, cols in enumerate(col_groups):
        grouped_votes[:, g] = vote_count[:, cols].sum(axis=1)

    sp_final = np.full(num_sp, -1, dtype=np.int64)
    has_vote = grouped_votes.sum(axis=1) > 0
    sp_final[has_vote] = grouped_votes[has_vote].argmax(axis=1)

    # #3 vote purity: per merged instance, mean winning-vote fraction over its
    # superpoints (how "clean" the association was). Candidate confidence signal
    # (emitted only when emit_stats is set; never affects the prediction).
    emit_stats = bool(getattr(cfg.merging_3d, "emit_stats", False))
    purity_g = np.ones(G)
    if emit_stats:
        tot = grouped_votes.sum(axis=1)
        for g in range(G):
            sps = np.where(sp_final == g)[0]
            if len(sps):
                purity_g[g] = float(np.mean(grouped_votes[sps, g] / np.maximum(tot[sps], 1)))

    # Per-instance point-index sets over the scene point cloud.
    point_group = sp_final[sp_id]
    instances_points = [np.where(point_group == g)[0] for g in range(G)]
    intensity_g = {g: intensity_dict_k[g] for g in range(G)}

    # Floor removal: drop instances that mostly coincide with the RANSAC ground plane.
    if getattr(cfg.merging_3d, "floor_removal", False):
        floor_idx = get_floor_indices_ransac(pointcloud.pcd)
        instances_points, _ = remove_floor_instances(instances_points, floor_idx)

    # DBSCAN cleanup: keep only the largest connected cluster of each instance.
    # dbscan_split (out-of-paper): instead of discarding the non-largest clusters
    # (which lose their points to -1), keep EACH cluster as its own instance so a
    # 3D-merged blob of spatially separate objects is recovered as multiple instances.
    dbscan_split = bool(getattr(cfg.merging_3d, "dbscan_split", False))
    if dbscan_split:
        new_points, new_inten = [], {}
        for g in range(G):
            idx = instances_points[g]
            if len(idx) < 10:
                continue
            largest_local, others = post_process_point_indices_dbscan(scene_coord[idx])
            for cl in [largest_local] + list(others):
                new_points.append(idx[cl])
                new_inten[len(new_points) - 1] = intensity_g[g]
        instances_points, intensity_g, G = new_points, new_inten, len(new_points)
    elif getattr(cfg.merging_3d, "dbscan_cleanup", False):
        for g in range(G):
            idx = instances_points[g]
            if len(idx) < 10:
                instances_points[g] = np.array([], dtype=int)
                continue
            largest_local, _ = post_process_point_indices_dbscan(scene_coord[idx])
            instances_points[g] = idx[largest_local]

    # Assign per-point instance ids; higher-score instances win overlaps (intensity-weighted).
    valid_g = [g for g in range(G) if len(instances_points[g]) > 0]
    if not valid_g:
        print(f"[WARNING] {scene_name}: no instances survived cleanup")
        return np.full(num_points, -1)
    inst_list = [instances_points[g] for g in valid_g]
    inten_list = {n: intensity_g[valid_g[n]] for n in range(len(valid_g))}
    score = compute_z_score(inst_list, inten_list, weight=[1, 1])
    order = sorted(range(len(inst_list)), key=lambda x: score[x], reverse=True)

    scene_pred = np.full(num_points, -1, dtype=np.int64)
    for new_id, gi in enumerate(order):
        idx = inst_list[gi]
        free = scene_pred[idx] == -1
        scene_pred[idx[free]] = new_id

    # grow_unassigned (out-of-paper): flood unlabeled (-1) points to the nearest
    # labeled point within a radius, tightening instance coverage of GT boundaries
    # (raises per-instance IoU without adding instances).
    if getattr(cfg.merging_3d, "grow_unassigned", False):
        from scipy.spatial import cKDTree
        grow_radius = float(getattr(cfg.merging_3d, "grow_radius", 0.05))
        lab = scene_pred >= 0
        if lab.any() and (~lab).any():
            lab_ids = scene_pred[lab]
            tree = cKDTree(scene_coord[lab])
            dist, nn = tree.query(scene_coord[~lab], k=1, distance_upper_bound=grow_radius)
            ok = np.isfinite(dist)
            if ok.any():
                unlab_idx = np.where(~lab)[0]
                scene_pred[unlab_idx[ok]] = lab_ids[nn[ok]]

    # #3: dump per-final-instance candidate confidence signals (purity, size, frames)
    # for offline scoring experiments. new_id enumerates `order`, so index == label id.
    if emit_stats:
        K = len(order)
        conf_purity = np.array([purity_g[valid_g[order[nid]]] for nid in range(K)], dtype=np.float32)
        conf_size   = np.array([len(inst_list[order[nid]]) for nid in range(K)], dtype=np.float64)
        conf_frames = np.array([inten_list[order[nid]] for nid in range(K)], dtype=np.float64)
        stats_path = os.path.join(cfg.exp.save_path, f"{scene_name}_stats.npz")
        os.makedirs(cfg.exp.save_path, exist_ok=True)
        np.savez(stats_path, purity=conf_purity, size=conf_size, frames=conf_frames)

    n_inst = len(np.unique(scene_pred[scene_pred >= 0]))
    print(f"[INFO] Final 3D instances: {n_inst}")
    return scene_pred
