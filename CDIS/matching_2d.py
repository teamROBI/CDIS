import os
from os.path import join
import numpy as np
import cv2
from collections import deque, defaultdict
import pickle, time, copy, random
from tqdm import tqdm
from typing import Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.util import (
    visualize_3d_instances, visualize_warped_masks_over_time, visualize_inst_matching_over_time, color_reset, get_color_for_id
)
from utils.CDIS_util import (
    read_data_file, read_all_data_parallel_fast, update_intensity
)
from utils.util_3d import (
    backproject_to_3d, load_intrinsics, transform_points, project_to_2d
)

### 2D matching process
def match_2d(cfg, scene_name, color_names, mask_generator, dense_scene_pcd):
    """Main 2D matching loop (sequential or parallel) with optional visualization and saving."""
    # Intrinsics
    intrinsics = load_intrinsics(join(cfg.data.scans_2d_path, scene_name, 'intrinsics', 'intrinsic_depth.txt'))

    # Determine which frames to read (respect image_iter)
    if cfg.debug.developing:
        color_subset = color_names[:200:cfg.matching_2d.image_iter] # for faster loading for just testing
    else:
        color_subset = color_names[::cfg.matching_2d.image_iter]

    # Read frames
    if cfg.data.data_all_ready:
        frame_data = read_all_data_parallel_fast(cfg, scene_name, color_subset)
    else:
        frame_data = []
        for i in tqdm(range(0, len(color_names), cfg.matching_2d.image_iter), desc="Reading data sequentially"):
            frame_data.append(read_data_file(cfg, scene_name, color_names[i], mask_generator, dense_scene_pcd, intrinsics))

    # State
    inst_match_list = []
    intensity_dict = {}
    frame_queue = deque(maxlen=cfg.matching_2d.queue_size)
    max_past_id = 0

    # Viz color map (optional)
    if cfg.debug.visualize_warped_masks or cfg.debug.visualize_frame_track:
        color_map = {-1: [0, 0, 0]}

    # Timer stats
    full_process_start_time = time.time()
    frame_count = 0

    # Pre-init debug buffer to avoid NameError in parallel mode
    warped_masks_to_t = [] if cfg.debug.visualize_warped_masks else None

    # --- Main loop ---
    for i, t_frame in enumerate(frame_data):
        start_time = time.time()

        # Queue push
        frame_queue.append(t_frame)

        # Back-project current frame’s 2D instances to 3D
        p3d, group, valid = backproject_to_3d(
            frame_queue[-1]["mask_2d"],
            frame_queue[-1]["depth"],
            intrinsics,
            depth_scale=cfg.data.depths.depth_scale,
        )
        frame_queue[-1]["points_3d"] = p3d
        frame_queue[-1]["group"] = group
        frame_queue[-1]["valid_mask"] = valid

        # First frame: just update intensity and continue
        if i == 0:
            intensity_dict = update_intensity(intensity_dict, frame_queue[-1])
            if getattr(cfg.matching_2d, "fix_id_collision", False):
                # Frame 0 keeps its raw ids {0..M}; advance max_past_id past them so
                # ids minted in later frames cannot collide with frame-0's ids.
                m0 = frame_queue[-1]["mask_2d"]
                max_past_id = int(m0.max()) if (m0.size and m0.max() >= 0) else 0
            continue

        # Matching (parallel or sequential)
        if cfg.matching_2d.parallel_process:
            frame_queue[-1], max_past_id = match_mask2d_id_parallel_as_completed(
                frame_queue, intrinsics, cfg.matching_2d.iou_2d_th, max_past_id
            )
        else:
            if cfg.debug.visualize_warped_masks:
                warped_masks_to_t = []  # fresh list per frame for the viz function
                frame_queue[-1], max_past_id, warped_masks_to_t = match_mask2d_id(
                    cfg, frame_queue, intrinsics, max_past_id, warped_masks_debug=warped_masks_to_t
                )
            else:
                frame_queue[-1], max_past_id = match_mask2d_id(
                    cfg, frame_queue, intrinsics, max_past_id
                )

        # Output list once queue is filled. Store ONLY the tracked mask; merge_3d
        # re-reads color/depth/pose from disk, so keeping them here just wastes RAM
        # (long scenes have thousands of frames). Also free the exited frame's heavy
        # 3D buffers, which otherwise accumulate across the whole scene.
        if i >= cfg.matching_2d.queue_size - 1:
            f0 = frame_queue[0]
            inst_match_list.append({"mask_2d": f0["mask_2d"]})
            for k in ("color", "depth", "points_3d", "valid_mask", "group"):
                f0[k] = None

        # Update intensity
        intensity_dict = update_intensity(intensity_dict, frame_queue[-1])

        # Progress
        frame_count += 1
        elapsed = time.time() - start_time
        total_elapsed = time.time() - full_process_start_time
        avg_per_frame = total_elapsed / frame_count
        label = t_frame.get("color_name", color_subset[i] if i < len(color_subset) else f"frame_{i}")
        msg = (f"\r[PROCESSING] {label} | Time: {elapsed:.2f}s | "
               f"Avg/frame: {avg_per_frame:.2f}s | Total time: {total_elapsed:.2f}s")
        print(msg, end="", flush=True)

        # Visualization (optional)
        if cfg.debug.visualize_warped_masks or cfg.debug.visualize_frame_track:
            if cfg.debug.visualize_warped_masks and warped_masks_to_t is not None:
                img_warped = visualize_warped_masks_over_time(frame_queue[-1]["mask_2d"], warped_masks_to_t, color_map)
                cv2.imshow('Warped view t-1, t-2, ... to Frame t', img_warped)

            if cfg.debug.visualize_frame_track:
                img_match = visualize_inst_matching_over_time(inst_match_list, color_map)
                if img_match is not None:
                    cv2.imshow('Instance Matching t-1, t-2, ... to Frame t', img_match)

            key = cv2.waitKey(0)  # keep blocking if that’s your intended debug flow
            if key == 27:        # Esc
                assert False, "DEBUG"
            elif key == ord('r'):
                color_reset(color_map)

    print()  # newline after progress line

    # Flush remaining frames in queue (except the one already appended above)
    for j in range(1, len(frame_queue)):
        inst_match_list.append({"mask_2d": frame_queue[j]["mask_2d"]})

    # Post-filter by intensity
    inst_match_list, intensity_dict = remove_low_intensity_instances(cfg, inst_match_list, intensity_dict)

    # Optional viz by intensity
    if cfg.debug.visualize_instances_by_intensity:
        visualize_instances_by_intensity(inst_match_list, intensity_dict)

    # Save matching 2d result
    os.makedirs(cfg.exp.matching_2d_output, exist_ok=True)
    save_2d_matching_result(inst_match_list, intensity_dict, os.path.join(cfg.exp.matching_2d_output, scene_name))

    return inst_match_list, intensity_dict



### Sequential mask matching process
def match_mask2d_id(cfg, frame_queue, intrinsics, max_past_id: int, warped_masks_debug=None):
    """
    Match instance IDs in the latest frame against past frames using geometric warping + IoU.
    - Preserves existing IDs when IoU > threshold.
    - Mints new global IDs for any still-unmatched instances.
    - Updates frame_queue[-1]["mask_2d"] and ["group"] in-place.

    Returns:
        (updated_frame_dict, updated_max_past_id)  [and warped_masks_debug if provided]
    """
    # Current (target) frame
    tfd = frame_queue[-1]
    t_mask = tfd["mask_2d"]
    t_pose_inv = np.linalg.inv(tfd["pose"])

    # Collect current instance IDs (exclude -1)
    current_ids = np.unique(t_mask)
    current_ids = current_ids[current_ids != -1]

    # Mapping from original id -> final id (init as identity)
    id_map: Dict[int, int] = {int(i): int(i) for i in current_ids}

    # Track which IDs still need a match
    ids_to_find = list(map(int, current_ids))

    # A5 (one-to-one): a past track id may be claimed by at most one current
    # instance across the whole queue, so two distinct objects can't fuse into
    # the same track. `claimed` accumulates past ids already taken.
    one_to_one = getattr(cfg.matching_2d, "match_one_to_one", False)
    claimed = set()

    # Reusable warp buffer (signed int so -1 is valid)
    warped_mask_to_t = np.full_like(t_mask, -1, dtype=np.int32)

    # --- MATCHING / MAPPING PHASE: iterate past frames (newest → oldest) ---
    for past_frame in reversed(list(frame_queue)[:-1]):
        if not ids_to_find:
            break

        # Warp past frame labels into t-frame
        warp_past_frame_to_t_frame(past_frame, tfd, intrinsics, t_pose_inv, warped_mask_to_t,
                                   z_filter=getattr(cfg.matching_2d, "warp_z_filter", False),
                                   zbuffer=getattr(cfg.matching_2d, "warp_zbuffer", False))

        # Optional: keep a snapshot for debugging (avoid aliasing)
        if warped_masks_debug is not None:
            warped_masks_debug.append(warped_mask_to_t.copy())

        # Build a mask that keeps only the IDs we still care about
        relevant_mask = np.where(np.isin(t_mask, ids_to_find), t_mask, -1)

        # IoU only for those remaining IDs
        iou_scores: Dict[int, Tuple[Optional[int], float]] = calculate_instance_iou(
            relevant_mask, warped_mask_to_t,
            full_union=getattr(cfg.matching_2d, "iou_full_union", False))

        # Filter matched vs. still-unmatched
        th = cfg.matching_2d.iou_2d_th
        if one_to_one:
            # Gather all above-threshold candidates for this past frame, then let
            # each past id go to the current instance with the highest IoU; the
            # losers stay unmatched and may match an older frame (or mint a new id).
            cand = {}  # pid -> (oid, iou)
            for oid in ids_to_find:
                match = iou_scores.get(oid)
                if match is None:
                    continue
                pid, best_iou = match
                if best_iou > th and pid not in claimed:
                    if pid not in cand or best_iou > cand[pid][1]:
                        cand[pid] = (oid, best_iou)
            winner = {oid: pid for pid, (oid, _) in cand.items()}
            still_unmatched = []
            for oid in ids_to_find:
                if oid in winner:
                    id_map[oid] = winner[oid]
                    claimed.add(winner[oid])
                else:
                    still_unmatched.append(oid)
            ids_to_find = still_unmatched
        else:
            still_unmatched = []
            for oid in ids_to_find:
                match = iou_scores.get(oid)
                if match is None:
                    still_unmatched.append(oid)
                    continue
                pid, best_iou = match
                if best_iou > th:
                    id_map[oid] = pid
                else:
                    still_unmatched.append(oid)

            ids_to_find = still_unmatched

    # --- UPDATE PHASE: mint new IDs for any leftovers ---
    num_new = len(ids_to_find)
    if num_new:
        for offset, oid in enumerate(ids_to_find, start=1):
            id_map[oid] = max_past_id + offset
        max_past_id += num_new

    # Apply final mapping via a lookup table (identical to a per-id scan, but one pass).
    # id_map covers every non-background id in t_mask; index by id+1 so background (-1) -> 0.
    if id_map:
        lut = np.full(int(max(id_map)) + 2, -1, dtype=t_mask.dtype)
        for oid, fid in id_map.items():
            lut[oid + 1] = fid
        new_mask = lut[t_mask + 1]
    else:
        new_mask = np.full_like(t_mask, -1, dtype=t_mask.dtype)

    # Persist updates in the frame dict
    tfd["mask_2d"] = new_mask
    tfd["group"] = new_mask[tfd["valid_mask"]]

    if warped_masks_debug is None:
        return tfd, max_past_id
    else:
        return tfd, max_past_id, warped_masks_debug



def warp_past_frame_to_t_frame(frame_old, frame_new, K, t_pose_inv, warped_mask_to_t, z_filter=False, zbuffer=False):
    # Relative transform (old -> new)
    T = t_pose_inv @ frame_old["pose"]

    # Transform points
    P_new = transform_points(frame_old["points_3d"], T)  # (N,3)

    # Project (assumes your fast project_to_2d: returns (2, N))
    uv = project_to_2d(P_new, K)
    u, v = uv[0], uv[1]  # views, no copy

    # Round to nearest int32
    x = np.rint(u).astype(np.int32, copy=False)
    y = np.rint(v).astype(np.int32, copy=False)

    # Bounds check
    H, W = frame_new["depth"].shape[:2]
    inb = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    if z_filter:
        # Drop points behind the camera (z <= 0), which otherwise scatter wrong labels.
        inb &= (P_new[:, 2] > 0)

    # Scatter labels (reuse buffer)
    warped_mask_to_t.fill(-1)
    xv = x[inb]; yv = y[inb]
    labels = frame_old["group"][inb]
    if zbuffer:
        # A4 (z-buffer): several past points can fall on one current pixel; plain
        # assignment keeps whichever is written last (arbitrary point order), so a
        # far/occluded point can overwrite the nearer one. Splat far->near (stable)
        # so the nearest point wins the pixel — the standard occlusion resolution.
        zv = P_new[inb, 2]
        order = np.argsort(-zv, kind="stable")
        warped_mask_to_t[yv[order], xv[order]] = labels[order]
    else:
        warped_mask_to_t[yv, xv] = labels

    return warped_mask_to_t


def calculate_instance_iou(mask1: np.ndarray, mask2: np.ndarray, full_union: bool = False) -> Dict[int, Tuple[Optional[int], float]]:
    # Shared foreground
    valid = (mask1 != -1) & (mask2 != -1)

    m1 = mask1[valid]
    m2 = mask2[valid]

    # All ids in mask1 (for output keys)
    ids1_all = np.unique(mask1[mask1 != -1])
    out: Dict[int, Tuple[Optional[int], float]] = {int(i): (None, 0.0) for i in ids1_all}

    # Compact codes in the valid region
    ids1, inv1 = np.unique(m1, return_inverse=True)   # inv1 in [0..n1)
    ids2, inv2 = np.unique(m2, return_inverse=True)   # inv2 in [0..n2)
    n1 = ids1.shape[0]
    n2 = ids2.shape[0]

    if full_union:
        # Eq. 4 IoU: areas over the FULL masks (union spans non-co-visible pixels too).
        # Vectorized full-mask counts (bincount over id+1 so bg=-1 maps to index 0);
        # integer counts are identical to summing (mask==id) per id, but O(H*W) not O(n*H*W).
        cnt1 = np.bincount((mask1 + 1).ravel())
        cnt2 = np.bincount((mask2 + 1).ravel())
        a1 = cnt1[ids1 + 1].astype(np.float32, copy=False)
        a2 = cnt2[ids2 + 1].astype(np.float32, copy=False)
    else:
        # Current behaviour: areas restricted to the co-visible region.
        a1 = np.bincount(inv1, minlength=n1).astype(np.float32, copy=False)
        a2 = np.bincount(inv2, minlength=n2).astype(np.float32, copy=False)

    # Intersections for all (i,j) that occur
    pair = inv1.astype(np.int64, copy=False) * n2 + inv2  # 1D codes
    inter = np.bincount(pair)  # length <= n1*n2
    nz = inter.nonzero()[0]
    if nz.size == 0:
        return out

    # Map back to (i,j)
    i_idx = nz // n2
    j_idx = nz %  n2
    inter_v = inter[nz].astype(np.float32, copy=False)

    # IoU = inter / (a1[i] + a2[j] - inter)
    union = a1[i_idx] + a2[j_idx] - inter_v
    iou = inter_v / union

    # Best per i
    best_iou = np.zeros(n1, dtype=np.float32)
    best_j   = -np.ones(n1, dtype=np.int32)
    # One pass over the nonzeros
    for k in range(iou.size):
        i = i_idx[k]
        v = iou[k]
        if v > best_iou[i]:
            best_iou[i] = v
            best_j[i]   = j_idx[k]

    # Write results for ids that actually appeared in valid
    for i_c, j_c in zip(np.flatnonzero(best_j >= 0), best_j[best_j >= 0]):
        out[int(ids1[i_c])] = (int(ids2[j_c]), float(best_iou[i_c]))

    return out

### Parallel mask matching process
def warp_and_match_single_frame(past_frame, t_frame, intrinsics, t_pose_inv, current_ids):
    """
    1. Warps past frame to current frame
    2. Computes IoU with current frame instances
    3. Returns: dict[current_id] = (best_matching_past_id, best_iou)
    """
    # Each thread can create its own fresh mask
    warped_mask = np.full_like(t_frame["mask_2d"], -1)
    
    warp_past_frame_to_t_frame(past_frame, t_frame, intrinsics, t_pose_inv, warped_mask)
    relevant_mask = np.where(np.isin(t_frame["mask_2d"], current_ids), t_frame["mask_2d"], -1)
    iou_scores = calculate_instance_iou(relevant_mask, warped_mask)
    return iou_scores

def merge_iou_results_new_first(iou_dicts, threshold):
    """
    Merge multiple IoU match dicts into a single dict by keeping highest IoU per current_id
    """
    merged = {}
    for d in iou_dicts:
        for cid, (pid, iou) in d.items():
            if cid not in merged and iou > threshold:
                merged[cid] = (pid, iou)
    return merged


def match_mask2d_id_parallel(frame_queue, intrinsics, threshold, max_past_id):
    t_frame = frame_queue[-1]
    t_pose_inv = np.linalg.inv(t_frame["pose"])
    current_ids = np.unique(t_frame["group"][t_frame["group"] != -1])

    with ThreadPoolExecutor(max_workers=os.cpu_count() // 2) as executor:
        iou_results = list(executor.map(
            lambda args: warp_and_match_single_frame(*args),
            [
                (past_frame, t_frame, intrinsics, t_pose_inv, current_ids)
                for past_frame in reversed(list(frame_queue)[:-1])
            ]
        ))
    
    merged_matches = merge_iou_results_new_first(iou_results, threshold)

    # Assign IDs
    updated_group_ids = np.copy(t_frame["group"])
    updated_mask_2d = np.copy(t_frame["mask_2d"])
    new_id_count = 1

    for cid in current_ids:
        if cid in merged_matches:
            pid, best_iou = merged_matches[cid]
            new_group_id = pid
        else:
            new_group_id = max_past_id + new_id_count
            new_id_count += 1

        updated_group_ids[t_frame["group"] == cid] = new_group_id
        updated_mask_2d[t_frame["mask_2d"] == cid] = new_group_id

    t_frame["group"] = updated_group_ids
    t_frame["mask_2d"] = updated_mask_2d

    added = new_id_count - 1
    return t_frame, max_past_id + added


def merge_iou_best_score(current_best_matches, new_result, threshold):
    """
    Merges a new dictionary of IoU scores into the main dictionary,
    keeping only the best score found so far for each instance.
    """
    for cid, (pid, iou) in new_result.items():
        if iou > threshold:
            # If we haven't seen this ID, or if the new IoU is better, update it.
            if cid not in current_best_matches or iou > current_best_matches[cid][1]:
                current_best_matches[cid] = (pid, iou)
    # No return needed as the dictionary is modified in place

def match_mask2d_id_parallel_as_completed(frame_queue, intrinsics, threshold, max_past_id):
    t_frame = frame_queue[-1]
    t_pose_inv = np.linalg.inv(t_frame["pose"])
    current_ids = np.unique(t_frame["group"][t_frame["group"] != -1])
    
    merged_matches = {}
    
    with ThreadPoolExecutor(max_workers=max(16, (os.cpu_count() or 1) // 2)) as executor:
        # Step 1: Submit all jobs to the executor and get 'future' objects
        future_to_past_frame = {
            executor.submit(warp_and_match_single_frame, past_frame, t_frame, intrinsics, t_pose_inv, current_ids): past_frame
            for past_frame in reversed(list(frame_queue)[:-1])
        }

        # Step 2: Process results as they complete
        for future in as_completed(future_to_past_frame):
            try:
                # Get the result from the completed future
                iou_dict = future.result()
                # Immediately merge the result
                merge_iou_best_score(merged_matches, iou_dict, threshold)
            except Exception as exc:
                past_frame_info = future_to_past_frame[future]
                print(f'Frame processing generated an exception: {exc} for past frame {past_frame_info.get("color")}')

    # Assign IDs
    updated_group_ids = np.copy(t_frame["group"])
    updated_mask_2d = np.copy(t_frame["mask_2d"])
    new_id_count = 1

    for cid in current_ids:
        if cid in merged_matches:
            pid, best_iou = merged_matches[cid]
            new_group_id = pid
        else:
            new_group_id = max_past_id + new_id_count
            new_id_count += 1

        updated_group_ids[t_frame["group"] == cid] = new_group_id
        updated_mask_2d[t_frame["mask_2d"] == cid] = new_group_id

    t_frame["group"] = updated_group_ids
    t_frame["mask_2d"] = updated_mask_2d
    
    added = new_id_count - 1
    return t_frame, max_past_id + added

### Post process
def remove_low_intensity_instances(cfg, inst_match_list, intensity_dict):
    """
    Remove instance IDs with intensity < threshold by setting them to -1 in masks.
    Returns updated inst_match_list and intensity_dict.

    Args:
        inst_match_list (list): list of frames, each with 'mask_2d' and 'color'
        intensity_dict (dict): mapping of instance_id -> intensity value
        threshold (float or int): threshold for filtering
    """
    # Find IDs to remove
    remove_ids = {inst_id for inst_id, intensity in intensity_dict.items() if intensity < cfg.matching_2d.intensity_th}

    # Deep copy inst_match_list to avoid modifying original
    updated_inst_match_list = copy.deepcopy(inst_match_list)

    # Update masks: set removed IDs to -1
    for frame in updated_inst_match_list:
        mask = frame["mask_2d"]
        mask[np.isin(mask, list(remove_ids))] = -1

    # Update intensity_dict: remove those IDs
    updated_intensity_dict = {k: v for k, v in intensity_dict.items() if k not in remove_ids}
    
    if cfg.debug.visualize_removed_instances_by_instensity:
        visualize_removed_instances_by_instensity(inst_match_list, updated_inst_match_list, remove_ids)

    return updated_inst_match_list, updated_intensity_dict


### Process saving and visualization
def save_2d_matching_result(inst_match_list, intensity_dict, save_path):
    # Stack mask_2d into one array (frames, H, W)
    masks = np.stack([frame["mask_2d"].astype(np.int16) for frame in inst_match_list], axis=0)

    # Create directory
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save masks as compressed NumPy file
    mask_path = save_path+"_masks.npz"
    np.savez_compressed(mask_path, masks=masks)

    # Save intensity_dict separately
    dict_path = save_path+"_intensity.pkl"
    with open(dict_path, 'wb') as f:
        pickle.dump(intensity_dict, f)

    print(f"[INFO] Saved masks to {mask_path}")
    print(f"[INFO] Saved intensity_dict to {dict_path}")
    
def load_2d_matching_result(save_path):
    """
    Load masks and intensity_dict, reconstructing inst_match_list format.
    """
    mask_path = save_path+"_masks.npz"
    dict_path = save_path+"_intensity.pkl"

    # Load masks
    masks = np.load(mask_path)["masks"]

    # Load intensity_dict
    with open(dict_path, 'rb') as f:
        intensity_dict = pickle.load(f)

    # Reconstruct original format
    inst_match_list = [{"mask_2d": mask} for mask in masks]

    return inst_match_list, intensity_dict

def visualize_removed_instances_by_instensity(old_list, new_list, removed_ids):
    """
    Visualize removed instances by showing original vs updated masks side by side.
    Removed IDs are highlighted in red on the updated version.

    Args:
        old_list (list): original inst_match_list
        new_list (list): updated inst_match_list after removal
        removed_ids (set or list): IDs that were removed
        get_color_for_id (function): function to assign consistent colors
    """
    color_map = {-1: [0, 0, 0]}
    
    removed_ids = set(removed_ids)

    for frame_idx, (old_frame, new_frame) in enumerate(zip(old_list, new_list)):
        mask_old = old_frame["mask_2d"]
        mask_new = new_frame["mask_2d"]

        # ---- convert masks to color images ----
        h, w = mask_old.shape
        color_old = np.zeros((h, w, 3), dtype=np.uint8)
        color_new = np.zeros((h, w, 3), dtype=np.uint8)

        for id_ in np.unique(mask_old):
            if id_ == -1:
                continue
            color_old[mask_old == id_] = get_color_for_id(id_, color_map)

        for id_ in np.unique(mask_new):
            if id_ == -1:
                continue
            # If this instance was removed, highlight it in red
            if id_ in removed_ids:
                color_new[mask_new == id_] = [0, 0, 255]  # Red for removed
            else:
                color_new[mask_new == id_] = get_color_for_id(id_, color_map)

        # ---- concatenate for visualization ----
        combined = np.concatenate((color_old, color_new), axis=1)

        # ---- display ----
        window_name = f"Removed Instances with Intensity"
        cv2.imshow(window_name, combined)
        print(f"\r[DEBUGGING] Removed Instances Frame {frame_idx}", end="", flush=True)
        key = cv2.waitKey(0)
        
        if key == 27:  # ESC to break
            break
        
    cv2.destroyWindow(window_name)
    
def visualize_instances_by_intensity(inst_match_list, intensity_dict):
    color_map = {-1: [0, 0, 0]}
    
    # ---- group instance IDs by intensity ----
    value_to_ids = defaultdict(list)
    for inst_id, value in intensity_dict.items():
        value_to_ids[value].append(inst_id)

    # ---- sort by intensity (low to high) ----
    sorted_values = sorted(value_to_ids.items(), key=lambda x: x[0])
    print(sorted_values)

    for value, inst_ids in sorted_values:
        # if value <= 1:
        #     continue
        
        print(f"Processing intensity {value}, instance IDs {inst_ids}")

        # ---- loop through frames ----
        for frame_idx, frame in enumerate(inst_match_list):
            mask = frame["mask_2d"]
            color_img = cv2.cvtColor(frame["color"], cv2.COLOR_RGB2BGR)
            overlay = color_img.copy()
            drawn = False

            for inst_id in inst_ids:
                instance_mask = (mask == inst_id).astype(np.uint8)
                if instance_mask.sum() == 0:
                    continue

                # ---- color overlay ----
                color = get_color_for_id(inst_id, color_map)
                overlay[instance_mask > 0] = color

                # ---- centroid for label ----
                ys, xs = np.where(instance_mask > 0)
                if len(xs) > 0 and len(ys) > 0:
                    cx, cy = int(xs.mean()), int(ys.mean())
                    cv2.putText(overlay, f"{inst_id}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                drawn = True

            # ---- blend once ----
            if drawn:
                vis_img = cv2.addWeighted(overlay, 0.5, color_img, 0.5, 0)
                window_name = f"Value {value}"
                print(f"\r[DEBUGGING] Value {value} - Frame {frame_idx}", end="", flush=True)
                cv2.imshow(window_name, vis_img)
                key = cv2.waitKey(0)
                if key == 27:  # Esc to break early
                    return
                
    cv2.destroyWindow(window_name)