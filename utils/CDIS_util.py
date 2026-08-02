from __future__ import annotations
import os
import numpy as np
import open3d as o3d
import multiprocessing as mp
from PIL import Image
import imageio
from pathlib import Path
from typing import Tuple, Dict, Any, Iterable, List
from concurrent.futures import ThreadPoolExecutor

from utils.util import *
from utils.util_3d import *


### 3D instance matching
# Global variables to store large data
global_kd_tree = None
global_group_world = None

def init_globals(kd_tree, group_world):
    """
    Initialize global variables for worker processes.
    This function is called once per process, allowing workers to access global data without passing it.
    """
    global global_kd_tree, global_group_world
    global_kd_tree = kd_tree
    global_group_world = group_world

def process_voxel_batch(cfg):
    """
    Helper function to process a batch of voxels and find instance IDs in parallel.
    Accesses global kd_tree and group_world variables.
    :param voxel_batch: Batch of voxels to process
    :param start_index: Index in the original list of voxels
    :param voxel_size: Size of the voxel grid for matching instances to voxels
    :return: List of tuples where each contains (index, instance_ids)
    """
    voxel_batch, start_index, voxel_size = cfg
    result = []
    for i, voxel in enumerate(voxel_batch):
        indices = global_kd_tree.query_ball_point(voxel, voxel_size)
        instance_ids = np.unique(global_group_world[indices]) if indices else np.array([], dtype=global_group_world.dtype)
        result.append((start_index + i, instance_ids))
    return result

def assign_instances_to_voxels_by_index(scene_pcd, coords_world, group_world, voxel_size=0.05, batch_size=10):
    """
    Assign instance IDs to voxels in the scene_pcd based on proximity, using scene_pcd indices.
    :param scene_pcd: (M, 3) array representing the voxelized scene point cloud (GT coordinates)
    :param coords_world: (N, 3) array of 3D coordinates from the world
    :param group_world: (N,) array of instance IDs corresponding to coords_world
    :param voxel_size: Size of the voxel grid for matching instances to voxels
    :param batch_size: Number of voxels to process in a single batch
    :return: List where each index corresponds to the scene point and contains the instance IDs
    """
    kd_tree = KDTree(coords_world)
    
    # Initialize global variables in each process
    with mp.Pool(mp.cpu_count(), initializer=init_globals, initargs=(kd_tree, group_world)) as pool:        
        # Prepare batches of voxels to process
        num_voxels = len(scene_pcd)
        cfg = [(scene_pcd[i:i + batch_size], i, voxel_size) for i in range(0, num_voxels, batch_size)]
        
        # Run the process_voxel_batch function in parallel
        results = list(tqdm(pool.imap(process_voxel_batch, cfg), total=len(cfg)))

    # Initialize a list to store the instance IDs for each voxel
    voxel_to_instances = [None] * num_voxels
    
    # Populate the results in the correct order
    for batch in results:
        for i, instance_ids in batch:
            voxel_to_instances[i] = instance_ids

    return voxel_to_instances

# def assign_instances_to_voxels_by_index(scene_pcd, coords_world, group_world, voxel_size=0.05):
#     """
#     Assign instance IDs to voxels in the scene_pcd based on proximity, using scene_pcd indices.
#     :param scene_pcd: (M, 3) array representing the voxelized scene point cloud (GT coordinates)
#     :param coords_world: (N, 3) array of 3D coordinates from the world
#     :param group_world: (N,) array of instance IDs corresponding to coords_world
#     :param voxel_size: Size of the voxel grid for matching instances to voxels
#     :return: List where each index corresponds to the scene point and contains the instance IDs
#     """
#     # Create a KDTree for fast nearest neighbor search from coords_world
#     kd_tree = KDTree(coords_world)

#     # List to store instance IDs for each scene point (same size as scene_pcd)
#     voxel_to_instances = [None] * len(scene_pcd)

#     print("[INFO] Assigning instance IDs to each scene voxel...")
#     for i, voxel in enumerate(tqdm(scene_pcd)):
#         # Find the points within a voxel-sized radius of the voxel center
#         indices = kd_tree.query_ball_point(voxel, voxel_size)
        
#         # Get the unique instance IDs for points within this voxel
#         instance_ids = np.unique(group_world[indices])
        
#         # Assign the instance IDs to this voxel (use index `i`)
#         voxel_to_instances[i] = instance_ids
    
#     return voxel_to_instances

def match_instances_within_voxels_by_index(voxel_to_instances, coords_world, group_world):
    """
    Match instances within the same voxel based on the instance IDs assigned to each scene point index.
    :param voxel_to_instances: List where each index contains instance IDs for the corresponding voxel
    :param coords_world: (N, 3) array of 3D coordinates from the world
    :param group_world: (N,) array of instance IDs corresponding to coords_world
    :return: Dictionary containing the 3D matching results for each voxel
    """
    matched_instances = {}

    print("[INFO] Matching instance IDs in 3D space...")
    for voxel_idx, instance_ids in enumerate(tqdm(voxel_to_instances)):
        if len(instance_ids) < 2:
            continue  # Skip if there's only one or no instances in the voxel
        
        # Compare all instance pairs within this voxel
        for i, instance_id1 in enumerate(instance_ids):
            for instance_id2 in instance_ids[i + 1:]:
                # Get the points for each instance
                points1 = coords_world[group_world == instance_id1]
                points2 = coords_world[group_world == instance_id2]

                # Calculate IoU and centroid distance
                iou_3d = calculate_3d_iou(points1, points2)
                centroid1 = np.mean(points1, axis=0) if len(points1) > 0 else None
                centroid2 = np.mean(points2, axis=0) if len(points2) > 0 else None
                centroid_dist = calculate_centroid_distance(centroid1, centroid2)

                # Store the results
                matched_instances[(instance_id1, instance_id2)] = {
                    'iou_3d': iou_3d,
                    'centroid_distance': centroid_dist,
                    'voxel_idx': voxel_idx  # Store voxel index for reference
                }
    
    return matched_instances

def calculate_3d_iou(points1, points2, voxel_size=0.05):
    """
    Calculate 3D IoU between two sets of 3D points by voxelizing the point clouds.
    :param points1: (N, 3) array of 3D points for instance 1
    :param points2: (M, 3) array of 3D points for instance 2
    :param voxel_size: Voxel size to discretize the space for IoU calculation
    :return: IoU score
    """
    voxel_points1 = np.round(points1 / voxel_size).astype(np.int32)
    voxel_points2 = np.round(points2 / voxel_size).astype(np.int32)
    
    set1 = set(map(tuple, voxel_points1))
    set2 = set(map(tuple, voxel_points2))
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    if union == 0:
        return 0.0
    return intersection / union





def merge_3D_mask_id_per_frame(pcd_list, coords_path, group_path, do_filter=False):
    """Load or aggregate point cloud data from pcd_list and merge in a divide-and-conquer style."""    
    if os.path.exists(coords_path) and os.path.exists(group_path):
        print("[INFO] Use saved coords and group")
        coords_world = np.load(coords_path)
        group_world = np.load(group_path)
    else:
        if do_filter:
            # Parallelize the filtering step for each pcd_dict in pcd_list
            print("[INFO] Filtering point clouds in parallel...")
            
            with mp.Pool(mp.cpu_count()) as pool:
                # Directly assign the result to pcd_list since the order is preserved
                pcd_list = list(tqdm(pool.imap(filter_task, pcd_list), total=len(pcd_list)))

    # Perform divide-and-conquer merging of frames
    pcd_list, merge_info_list = divide_and_conquer_merge(pcd_list)
    
    # After merging all frames, save the aggregated result
    merged_pcd_dict = pcd_list[0]  # Final merged result
    coords_world = merged_pcd_dict["coord"]
    group_world = merged_pcd_dict["group"]
    
    np.save(coords_path, coords_world)
    np.save(group_path, group_world)

    return coords_world, group_world, merge_info_list

def divide_and_conquer_merge(pcd_list):
    """
    Perform divide-and-conquer merging of the point clouds in pcd_list.
    :param pcd_list: List of point cloud dictionaries (pcd_dict) to be merged.
    :return: A single merged point cloud dictionary.
    """
    merge_info_list = []  # List to store matching information
    
    while len(pcd_list) != 1:
        print(f"[INFO] Merging {len(pcd_list)} 3D frames...")
        # for idx, pcd_dict in enumerate(pcd_list):
        #     print(f"[DEBUG] {idx}: {np.unique(pcd_dict['group'])}")
        
        # If odd number of point clouds, keep the last unpaired frame separately
        last_unpaired = None
        if len(pcd_list) % 2 == 1:
            last_unpaired = pcd_list.pop()  # Remove the last element
        
        # Pair neighboring frames (0,1), (2,3), etc.
        pairs = [(pcd_list[i], pcd_list[i+1]) for i in range(0, len(pcd_list)-1, 2)]

        # Multiprocessing for merging pairs of point clouds and updating merge_info_list
        with mp.Pool(mp.cpu_count()) as pool:
            results = list(tqdm(pool.imap(merge_neighbor_frames, pairs), total=len(pairs)))

        # Unpack the merged pcd_list and matching information
        pcd_list, merge_info_batch = zip(*results)
        pcd_list = list(pcd_list)  # Convert tuple back to list
        merge_info_list.extend(merge_info_batch)  # Append batch matching information

        # Add back the last unpaired frame for the next round
        if last_unpaired is not None:
            pcd_list.append(last_unpaired)
    
    return pcd_list, merge_info_list

def merge_neighbor_frames(pairs, voxel_size=0.05, iou_threshold=0.5):
    """
    Merge two neighboring frames by matching 3D instances using IoU.
    :param pairs: Tuple containing two point cloud dictionaries (pcd_dict1, pcd_dict2).
    :param voxel_size: Voxel size to discretize space for IoU calculation.
    :param iou_threshold: Threshold for matching instances based on IoU.
    :return: Merged pcd_dict with combined coord and group.
    """
    pcd_dict1, pcd_dict2 = pairs

    coords1, group1 = pcd_dict1["coord"], pcd_dict1["group"]
    coords2, group2 = pcd_dict2["coord"], pcd_dict2["group"]

    merged_coords = []
    merged_groups = []

    unique_groups1 = np.unique(group1)
    unique_groups2 = np.unique(group2)
    
    merge_info = []  # Initialize list to store match information (group1 -> group2)

    # Skip matching if group ID already exists in both frames
    for gid1 in unique_groups1:
        if gid1 in unique_groups2:
            # If gid1 exists in both pcd_dict1 and pcd_dict2, concatenate the points
            points1 = coords1[group1 == gid1]
            points2 = coords2[group2 == gid1]

            # Simply concatenate matching points from both frames
            merged_coords.append(np.vstack([points1, points2]))
            merged_groups.append(np.full(len(points1) + len(points2), gid1))

            # Remove gid1 from unique_groups2 so it won't be checked again
            unique_groups2 = unique_groups2[unique_groups2 != gid1]
        else:
            # If gid1 doesn't exist in pcd_dict2, keep the points from pcd_dict1
            points1 = coords1[group1 == gid1]
            merged_coords.append(points1)
            merged_groups.append(np.full(len(points1), gid1))

    # Match remaining unmatched instances in pcd_dict2 with IoU-based criteria
    for gid2 in unique_groups2:
        points2 = coords2[group2 == gid2]
        best_iou = 0
        best_match = None

        # Check IoU against all remaining instances in pcd_dict1
        for gid1 in unique_groups1:
            if gid1 not in unique_groups2:  # Ensure we don't re-match
                points1 = coords1[group1 == gid1]
                iou = calculate_3d_iou(points1, points2, voxel_size=voxel_size)

                # Find the best IoU match
                if iou > best_iou and iou > iou_threshold:
                    best_iou = iou
                    best_match = gid1

        if best_match is not None:
            # If a match is found, merge points with gid1
            points1 = coords1[group1 == best_match]
            merged_coords.append(np.vstack([points1, points2]))
            merged_groups.append(np.full(len(points1) + len(points2), best_match))
            
            # Store matching information for intensity_dict update later
            merge_info.append((best_match, gid2))  # Matching gid1 with gid2
            if best_match == gid2:
                print(f"[DEBUG] PCD2: {gid2} matched PCD1: {best_match}, IoU={best_iou:.4f}")
        else:
            # If no match is found, add points2 as a new instance
            merged_coords.append(points2)
            merged_groups.append(np.full(len(points2), gid2))

    # Combine the merged coordinates and groups into the result
    merged_coords = np.vstack(merged_coords)
    merged_groups = np.hstack(merged_groups)

    return voxelize_global({"coord": merged_coords, "group": merged_groups}), merge_info


###################
def make_open3d_point_cloud(input_dict, voxelize, th):
    input_dict["group"] = remove_small_group(input_dict["group"], th)
    # input_dict = voxelize(input_dict)

    xyz = input_dict["coord"]
    if np.isnan(xyz).any():
        return None
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd


def compare_2_scene_clusters(pcd_list, index, voxel_size, voxelize):
    if len(index) == 1:
        return(pcd_list[index[0]])
    # print(index, flush=True)
    input_dict_0 = pcd_list[index[0]]
    input_dict_1 = pcd_list[index[1]]
    coord_0, group_0 = input_dict_0["coord"], input_dict_0["group"]
    coord_1, group_1 = input_dict_1["coord"], input_dict_1["group"]
    
    merged_coord, merged_group = [], []

    unique_groups_0 = np.unique(group_0)
    unique_groups_1 = np.unique(group_1)

    for i, cluster_0 in enumerate(unique_groups_0):
        coord_cluster_0 = coord_0[np.where(group_0 == cluster_0)[0]]
        for j, cluster_1 in enumerate(unique_groups_1):
            coord_cluster_1 = coord_1[np.where(group_1 == cluster_1)[0]]
            large_overlapping, overlapping_indices = calculate_cluster_overlap(coord_cluster_0, coord_cluster_1, voxel_size)
            # print(large_overlapping, len(overlapping_indices))

            if large_overlapping > 0.4:
                merged_coord.extend(coord_cluster_0)
                merged_coord.extend(coord_cluster_1)
                group_len = len(coord_cluster_0) + len(coord_cluster_1)
                merged_group.extend(np.full(group_len, cluster_0))
            elif len(overlapping_indices) > 0:
                if len(coord_cluster_0) < len(coord_cluster_1):
                    coord_cluster_0 = np.delete(coord_cluster_0, overlapping_indices, axis=0)
                    merged_coord.extend(coord_cluster_1)
                    merged_group.extend(np.full(len(coord_cluster_1), cluster_1))
                else:
                    coord_cluster_1 = np.delete(coord_cluster_1, overlapping_indices, axis=0)
                    merged_coord.extend(coord_cluster_1)
                    merged_group.extend(np.full(len(coord_cluster_1), cluster_1))
            else:
                merged_coord.extend(coord_cluster_1)
                merged_group.extend(np.full(len(coord_cluster_1), cluster_1))

        merged_coord.extend(coord_cluster_0)
        merged_group.extend(np.full(len(coord_cluster_0), cluster_0))
    
    pcd_dict = dict(coord=np.array(merged_coord), group=np.array(merged_group))
    pcd_dict = voxelize(pcd_dict)
    # dup = find_duplicate_coordinates(pcd_dict["coord"])
    # print(len(dup))

    return pcd_dict



################

# --- Small helpers ------------------------------------------------------------
def _norm_ext(ext: str) -> str:
    """Return extension with leading dot, e.g. 'png' -> '.png'."""
    return ext if ext.startswith('.') else f'.{ext}'

def _scale_to_meters(depth: np.ndarray, depth_scale: float) -> np.ndarray:
    """Scale depth to meters, preserving dtype where possible."""
    if depth_scale <= 0:
        raise ValueError("depth_scale must be > 0")
    # Use float32 for math; keeps memory down vs float64
    return (depth.astype(np.float32, copy=False) / float(depth_scale))

def _read_raw_depth(cfg, depth_path: str) -> np.ndarray:
    """Load raw depth based on dataset type and scale to meters."""
    if cfg.dataset_type in ["scannetv2", "real_world"]:
        # Stored as uint16 millimeters typically
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)  # keep 16-bit
        return _scale_to_meters(depth, cfg.data.depths.depth_scale)
    elif cfg.dataset_type == "replica":
        # Replica quirk kept as-is
        depth = imageio.v2.imread(depth_path)
        return depth.astype(np.float32) / 6533.5
    else:
        raise ValueError(f"Unknown dataset_type: {cfg.dataset_type}")

def _read_synthetic_depth(path_png_or_npz: str, depth_ext: str, depth_scale: float) -> np.ndarray:
    """Load precomputed synthetic depth and scale to meters if needed."""
    depth_ext = _norm_ext(depth_ext)
    if depth_ext == ".png":
        d = cv2.imread(path_png_or_npz, cv2.IMREAD_UNCHANGED)
        return _scale_to_meters(d, depth_scale)
    elif depth_ext == ".npz":
        d = np.load(path_png_or_npz)["synthetic_depth"]
        return d.astype(np.float32, copy=False)
    else:
        raise ValueError(f"Unsupported synthetic depth ext: {depth_ext}")

def _ensure_dir(p: str | Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)

def get_data_file_path(cfg, scene_name: str, file_name: str, iter: int = 0) -> Tuple[str, str, str]:
    """Build color/depth/pose paths for a given frame file name."""
    base = Path(cfg.data.scans_2d_path) / scene_name
    stem = str(int(file_name[:-4]) + iter)  # assumes file_name like '000123.png'
    color = base / cfg.data.images.images_path / f"{stem}{_norm_ext(cfg.data.images.images_ext)}"
    depth = base / cfg.data.depths.depths_path / f"{stem}{_norm_ext(cfg.data.depths.depths_ext)}"
    pose  = base / cfg.data.camera.poses_path / f"{stem}.txt"
    return str(color), str(depth), str(pose)

# --- Main readers -------------------------------------------------------------

def read_data_file(cfg, scene_name: str, color_name: str,
                   mask_generator, dense_scene_pcd, intrinsics) -> Dict[str, Any]:
    """
    Full reader: computes synthetic/supplemented depth and masks if missing.
    """
    frame: Dict[str, Any] = {"color_name": color_name}

    # Paths
    color_path, depth_path, pose_path = get_data_file_path(cfg, scene_name, color_name)
    frame["pose"] = np.loadtxt(pose_path)

    # Depth
    if cfg.depth_type == "raw_depth":
        depth_img = _read_raw_depth(cfg, depth_path)

    else:
        # synthetic (pc_depth) and supplemented (sup_depth) share base synthetic file
        syn_dir = Path(cfg.data.depths.synthetic_depth_path) / scene_name
        # keep the debug visualisation next to the synthetic depths instead of an absolute path
        syn_vis_dir = Path(cfg.data.depths.synthetic_depth_path).parent / "depth_from_pc_vis" / scene_name
        _ensure_dir(syn_dir); _ensure_dir(syn_vis_dir)
        syn_ext = _norm_ext(cfg.data.depths.synthetic_depth_ext)
        syn_stem = Path(color_name).stem
        syn_file = syn_dir / f"{syn_stem}{syn_ext}"

        if not syn_file.exists():
            # Render synthetic depth (meters, float32)
            depth_img = render_point_cloud_to_depth_image(
                np.asarray(dense_scene_pcd.points),
                frame["pose"],
                intrinsics,
                image_shape=tuple(cfg.data.img_size)  # (H, W)
            )
            # Save
            if syn_ext == ".png":
                cv2.imwrite(str(syn_file), (depth_img * cfg.data.depths.depth_scale).astype(np.uint16))
            else:
                np.savez_compressed(str(syn_file), synthetic_depth=depth_img.astype(np.float32))
            # Visualization
            vis = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            cv2.imwrite(str(syn_vis_dir / f"{syn_stem}.jpg"), vis)
        else:
            depth_img = _read_synthetic_depth(str(syn_file), syn_ext, cfg.data.depths.depth_scale)
            if cfg.debug.show_logs:
                print("[INFO] Use saved aggregated depth image extracted from the dense pcd")

        if cfg.depth_type == "sup_depth":
            raw_depth_img = _read_raw_depth(cfg, depth_path)
            depth_img = supplement_depth(raw_depth_img, depth_img)

    frame["depth"] = depth_img

    # RGB
    color_img = cv2.imread(color_path, cv2.IMREAD_COLOR)
    color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
    color_img = cv2.resize(color_img, (cfg.data.img_size[1], cfg.data.img_size[0]))
    frame["color"] = color_img

    # Masks
    mask_out_dir = Path(cfg.exp.mask2d_output) / scene_name
    _ensure_dir(mask_out_dir)
    group_path = mask_out_dir / f"{Path(color_name).stem}.png"

    if mask_generator is not None and not group_path.exists():
        if cfg.mask_model == "oneformer":
            group_ids, masks, masks_info = get_oneformer(color_img, mask_generator)
            # save_oneformer_result(...)

        elif cfg.mask_model == "cropformer":
            group_ids, masks = get_cropformer(
                color_img, mask_generator,
                score_threshold=cfg.vfm_config.cropformer_config["confidence-threshold"],
            )
            _ = save_cropformer_result(color_img, masks, str(mask_out_dir / f"{Path(color_name).stem}.jpg"))

        else:
            raise ValueError(f"Unsupported mask_model '{cfg.mask_model}'; only 'cropformer' is supported.")

        # Save mask as 16-bit label image
        Image.fromarray(num_to_natural(group_ids).astype(np.int16), mode="I;16").save(group_path)
    else:
        if cfg.debug.show_logs:
            print("[INFO] Use saved masks")
        group_ids = np.array(Image.open(group_path), dtype=np.int16)

    group_ids = num_to_natural(remove_group_ids_near_edge(group_ids))
    frame["mask_2d"] = group_ids

    return frame


def read_data_file_fast(cfg, scene_name: str, color_name: str) -> Dict[str, Any]:
    """
    Fast reader: assumes synthetic depth & masks already exist on disk.
    """
    frame: Dict[str, Any] = {"color_name": color_name}

    # Paths
    color_path, depth_path, pose_path = get_data_file_path(cfg, scene_name, color_name)
    frame["pose"] = np.loadtxt(pose_path)

    # Depth
    if cfg.depth_type == "raw_depth":
        depth_img = _read_raw_depth(cfg, depth_path)
    elif cfg.depth_type in ["pc_depth", "sup_depth"]:
        syn_dir = Path(cfg.data.depths.synthetic_depth_path) / scene_name
        syn_ext = _norm_ext(cfg.data.depths.synthetic_depth_ext)
        syn_file = syn_dir / f"{Path(color_name).stem}{syn_ext}"
        depth_img = _read_synthetic_depth(str(syn_file), syn_ext, cfg.data.depths.depth_scale)
        if cfg.depth_type == "sup_depth":
            raw_depth_img = _read_raw_depth(cfg, depth_path)
            depth_img = supplement_depth(raw_depth_img, depth_img)
    else:
        raise ValueError(f"Unknown depth_type: {cfg.depth_type}")

    frame["depth"] = depth_img

    # RGB is only consumed by optional debug visualization, never by tracking
    # (backproject/warp/IoU). Skip the decode entirely when no viz flag is set.
    need_color = (cfg.debug.visualize_warped_masks or cfg.debug.visualize_frame_track
                  or cfg.debug.visualize_instances_by_intensity)
    if need_color:
        color_img = cv2.imread(color_path, cv2.IMREAD_COLOR)
        color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        color_img = cv2.resize(color_img, (cfg.data.img_size[1], cfg.data.img_size[0]))
        frame["color"] = color_img
    else:
        frame["color"] = None

    # Mask (pre-saved). int32 keeps ids exact (they are far below 2**31) while halving
    # the per-frame mask bandwidth vs num_to_natural's default int64.
    mask_out_dir = Path(cfg.exp.mask2d_output) / scene_name
    group_path = mask_out_dir / f"{Path(color_name).stem}.png"
    group_ids = np.array(Image.open(group_path), dtype=np.int16)
    group_ids = num_to_natural(remove_group_ids_near_edge(group_ids)).astype(np.int32, copy=False)
    frame["mask_2d"] = group_ids

    return frame


def read_all_data_parallel_fast(cfg, scene_name: str, color_names: Iterable[str]) -> List[Dict[str, Any]]:
    """Parallel read of pre-saved data."""
    color_names = list(color_names)
    max_workers = min(16, max(1, (os.cpu_count() or 1)))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        results = list(tqdm(
            ex.map(lambda c: read_data_file_fast(cfg, scene_name, c), color_names),
            total=len(color_names),
            desc="Reading saved data in parallel"
        ))
    return results


def update_intensity(intensity, frame):
    ids = np.unique(frame["group"])   # derived from valid_mask; excludes background pixels
    ids = ids[ids != -1]
    for k in ids:
        intensity[k] = intensity.get(int(k), 0) + 1
    return intensity
