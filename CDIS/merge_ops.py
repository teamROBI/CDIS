"""3D instance merging operations for CDIS.

Helper functions used by the 3D-Guided 2D Instance Merging and 3D Instance
Consolidation stages (Sec. III-B / III-C of the paper). These operate on
per-instance point-index sets over the scene point cloud:

    instances_indices : list[np.ndarray]   # scene-point indices per instance
    intensity_dict    : dict[int, int]     # instance id -> #frames it was tracked
    intensity_3d_dict : dict[int, int]     # instance id -> #3D merges accumulated
    frame_track_dict  : dict[int, np.ndarray]  # instance id -> frame indices seen

The point-cloud aggregation helpers (``filter_pcd_by_intensity``,
``aggregate_pcd_list``) consume a ``pcd_list`` of per-frame voxelized clouds
``{"coord": (M,3) world xyz, "group": (M,) instance ids}``.
"""

import numpy as np
import open3d as o3d
from scipy.sparse import csr_matrix


# ---------------------------------------------------------------------------
# Point-cloud aggregation
# ---------------------------------------------------------------------------
def filter_pcd_by_intensity(pcd_list, intensity_dict, intensity_th=4):
    """Drop instances (and their points) whose tracked-frame count < intensity_th."""
    intensity_dict = {k: v for k, v in intensity_dict.items() if v >= intensity_th}
    filtered_pcd_list = []
    total_size = 0
    filtered_size = 0

    for pcd_dict in pcd_list:
        coords = pcd_dict['coord']
        group = pcd_dict['group']
        total_size += len(coords)

        mask = np.isin(group, list(intensity_dict.keys()))
        filtered_coords = coords[mask.flatten()]
        filtered_group = group[mask.flatten()]
        filtered_size += len(filtered_coords)

        filtered_pcd_list.append({'coord': filtered_coords, 'group': filtered_group})

    print(f"[INFO] Total size {total_size} -> Filtered to {filtered_size}")
    return filtered_pcd_list, intensity_dict


def aggregate_pcd_list(pcd_list):
    """Concatenate all per-frame clouds into one world cloud and record, for each
    instance id, the list of frame indices in which it appears."""
    coords_world = np.concatenate([p["coord"] for p in pcd_list], axis=0)
    group_world = np.concatenate([p["group"] for p in pcd_list], axis=0)

    frame_track_dict = {}
    for frame_index, pcd_dict in enumerate(pcd_list):
        for group_id in np.unique(pcd_dict["group"]):
            frame_track_dict.setdefault(group_id, []).append(frame_index)

    return coords_world, group_world, frame_track_dict


# ---------------------------------------------------------------------------
# 3D IoU / IoMin matrices
# ---------------------------------------------------------------------------
def _rows_to_csr(rows, n_cols):
    """Build a binary CSR membership matrix directly from a list of index arrays.

    Assumes each row's indices are unique (true for our instance sets, which come from
    np.where / np.union1d), so this is identical to lil scatter + tocsr but much faster.
    """
    lengths = np.fromiter((len(r) for r in rows), dtype=np.int64, count=len(rows))
    indptr = np.zeros(len(rows) + 1, dtype=np.int64)
    np.cumsum(lengths, out=indptr[1:])
    cols = (np.concatenate([np.asarray(r) for r in rows]).astype(np.int64)
            if len(rows) else np.zeros(0, dtype=np.int64))
    data = np.ones(cols.shape[0], dtype=int)
    return csr_matrix((data, cols, indptr), shape=(len(rows), n_cols))


def create_mask_matrix(instances_indices, second_indices=None):
    """Build a sparse (num_instances x num_points) binary membership matrix."""
    max_index = max(max(idx.max() for idx in instances_indices),
                    max(idx.max() for idx in second_indices) if second_indices else 0)

    mask_matrix = _rows_to_csr(instances_indices, max_index + 1)
    if second_indices is None:
        return mask_matrix

    return mask_matrix, _rows_to_csr(second_indices, max_index + 1)


def compute_iou_3d_matrix(mask_matrix, denominator="union"):
    """Pairwise 3D IoU (or intersection-over-min/max) between all instances."""
    intersection_array = (mask_matrix @ mask_matrix.transpose()).toarray()
    instance_sizes = mask_matrix.sum(axis=1).A1

    if denominator == "union":
        denominator_matrix = instance_sizes[:, None] + instance_sizes[None, :] - intersection_array
    elif denominator == "max":
        denominator_matrix = np.maximum(instance_sizes[:, None], instance_sizes[None, :])
    elif denominator == "min":
        denominator_matrix = np.minimum(instance_sizes[:, None], instance_sizes[None, :])
    else:
        raise ValueError("Invalid denominator type. Choose from 'union', 'max', or 'min'.")

    with np.errstate(divide='ignore', invalid='ignore'):
        iou_matrix = np.divide(
            intersection_array, denominator_matrix,
            out=np.zeros_like(intersection_array, dtype=float),
            where=denominator_matrix != 0,
        )

    np.fill_diagonal(iou_matrix, 0)
    return iou_matrix


# ---------------------------------------------------------------------------
# Instance bookkeeping
# ---------------------------------------------------------------------------
def clean_up_zero(instances_indices, intensity_dict, intensity_3d_dict, frame_track_dict):
    """Drop emptied instances and renumber all bookkeeping dicts to 0..K-1."""
    valid_gids = [gid for gid in range(len(instances_indices)) if len(instances_indices[gid])]
    new_gid_mapping = {old: new for new, old in enumerate(valid_gids)}

    instances_indices = [instances_indices[old] for old in valid_gids]
    intensity_dict = {new: intensity_dict[old] for old, new in new_gid_mapping.items()}
    intensity_3d_dict = {new: intensity_3d_dict[old] for old, new in new_gid_mapping.items()}
    frame_track_dict = {new: frame_track_dict[old] for old, new in new_gid_mapping.items()}

    return instances_indices, intensity_dict, intensity_3d_dict, frame_track_dict


def clean_up_zero_debug(instances_indices, intensity_dict, intensity_3d_dict, frame_track_dict, debug_id):
    valid_gids = [gid for gid in range(len(instances_indices)) if len(instances_indices[gid])]
    new_gid_mapping = {old: new for new, old in enumerate(valid_gids)}

    instances_indices = [instances_indices[old] for old in valid_gids]
    intensity_dict = {new: intensity_dict[old] for old, new in new_gid_mapping.items()}
    intensity_3d_dict = {new: intensity_3d_dict[old] for old, new in new_gid_mapping.items()}
    frame_track_dict = {new: frame_track_dict[old] for old, new in new_gid_mapping.items()}

    return instances_indices, intensity_dict, intensity_3d_dict, frame_track_dict, new_gid_mapping[debug_id]


# ---------------------------------------------------------------------------
# Merging (3D IoU) and absorbing (3D IoMin + temporal co-occurrence)
# ---------------------------------------------------------------------------
def merge_3d_instances(iou_matrix, instances_indices, intensity_dict, intensity_3d_dict, frame_track_dict, iou_3d_th):
    """Greedily merge instance pairs whose 3D IoU exceeds ``iou_3d_th`` (Eq. 6)."""
    iou_pairs = np.argwhere(iou_matrix > iou_3d_th)
    iou_values = iou_matrix[iou_pairs[:, 0], iou_pairs[:, 1]]
    sorted_pairs = iou_pairs[np.argsort(-iou_values)]

    while len(sorted_pairs) > 0:
        gid1, gid2 = sorted_pairs[0]
        if intensity_dict[gid2] > intensity_dict[gid1]:
            gid1, gid2 = gid2, gid1

        instances_indices[gid1] = np.union1d(instances_indices[gid1], instances_indices[gid2])
        intensity_dict[gid1] += intensity_dict[gid2]
        intensity_3d_dict[gid1] += intensity_3d_dict[gid2]
        frame_track_dict[gid1] = np.union1d(frame_track_dict[gid1], frame_track_dict[gid2])
        instances_indices[gid2] = []

        sorted_pairs = sorted_pairs[(sorted_pairs[:, 0] != gid1) & (sorted_pairs[:, 1] != gid1) &
                                    (sorted_pairs[:, 0] != gid2) & (sorted_pairs[:, 1] != gid2)]

    return clean_up_zero(instances_indices, intensity_dict, intensity_3d_dict, frame_track_dict)


def absorb_3d_instances(iom_matrix, instances_indices, intensity_dict, intensity_3d_dict, frame_track_dict, iom_3d_th, iom_frame_th, debug_id=None):
    """Absorb high-IoMin pairs that are NOT co-observed in time (Eq. 7-8): a
    spatially-overlapping pair seen at different frame times is the same object
    viewed across the sequence, so the smaller is absorbed into the larger."""
    iom_pairs = np.argwhere(iom_matrix > iom_3d_th)
    iom_values = iom_matrix[iom_pairs[:, 0], iom_pairs[:, 1]]
    sorted_pairs = iom_pairs[np.argsort(-iom_values)]

    while len(sorted_pairs) > 0:
        gid1, gid2 = sorted_pairs[0]
        if intensity_dict[gid2] > intensity_dict[gid1]:
            gid1, gid2 = gid2, gid1

        frames1 = frames_from_ranges(find_continuous_ranges(frame_track_dict[gid1]))
        frames2 = frames_from_ranges(find_continuous_ranges(frame_track_dict[gid2]))
        denominator = min(len(frames1), len(frames2))
        frame_iomin = len(frames1 & frames2) / denominator if denominator > 0 else 0

        if frame_iomin < iom_frame_th:
            instances_indices[gid1] = np.union1d(instances_indices[gid1], instances_indices[gid2])
            intensity_dict[gid1] += intensity_dict[gid2]
            intensity_3d_dict[gid1] += intensity_3d_dict[gid2]
            frame_track_dict[gid1] = np.union1d(frame_track_dict[gid1], frame_track_dict[gid2])
            instances_indices[gid2] = []

            sorted_pairs = sorted_pairs[(sorted_pairs[:, 0] != gid1) & (sorted_pairs[:, 1] != gid1) &
                                        (sorted_pairs[:, 0] != gid2) & (sorted_pairs[:, 1] != gid2)]
        else:
            sorted_pairs = sorted_pairs[1:]

    instances_indices, intensity_dict, intensity_3d_dict, frame_track_dict = clean_up_zero(
        instances_indices, intensity_dict, intensity_3d_dict, frame_track_dict)

    return instances_indices, intensity_dict, intensity_3d_dict, frame_track_dict, debug_id


def find_continuous_ranges(frame_indices, gap=5):
    """Group sorted frame indices into (start, end) runs, splitting on gaps > gap."""
    ranges = []
    start = frame_indices[0]
    for i in range(1, len(frame_indices)):
        if frame_indices[i] - frame_indices[i - 1] > gap:
            ranges.append((start, frame_indices[i - 1]))
            start = frame_indices[i]
    ranges.append((start, frame_indices[-1]))
    return ranges


def frames_from_ranges(range_list):
    """Expand a list of (start, end) ranges into a set of frame indices."""
    frames = set()
    for start, end in range_list:
        frames.update(range(start, end + 1))
    return frames


# ---------------------------------------------------------------------------
# DBSCAN cleanup / floor removal / scoring
# ---------------------------------------------------------------------------
def calculate_eps_based_on_flatness(points, min_eps=0.05, max_eps=0.06, flatness_min=0.025, flatness_max=0.1):
    """Pick a DBSCAN eps from the point set's flatness (eigenvalue ratio)."""
    points_centered = points - np.mean(points, axis=0)
    cov_matrix = np.cov(points_centered, rowvar=False)
    eigenvalues = np.sort(np.linalg.eigh(cov_matrix)[0])
    flatness_ratio = eigenvalues[0] / eigenvalues[2]
    normalized_ratio = np.clip((flatness_ratio - flatness_min) / (flatness_max - flatness_min), 0, 1)
    return min_eps + (max_eps - min_eps) * (1 - normalized_ratio)


def post_process_point_indices_dbscan(points, min_eps=0.05, max_eps=0.06, min_points=5, min_cluster_size=100):
    """Keep the largest DBSCAN cluster; return (largest_local_idx, [other_clusters])."""
    eps_value = calculate_eps_based_on_flatness(points, min_eps=min_eps, max_eps=max_eps)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    labels = np.array(pcd.cluster_dbscan(eps=eps_value, min_points=min_points, print_progress=False))

    unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
    if len(unique_labels) == 0:
        print(f"[WARNING] no cluster found. {len(points)}")
        return np.arange(len(points)), []

    order = np.argsort(-counts)
    sorted_labels, sorted_counts = unique_labels[order], counts[order]

    largest_cluster_indices = np.where(labels == sorted_labels[0])[0]
    other_clusters_indices = [
        np.where(labels == label)[0]
        for label, count in zip(sorted_labels[1:], sorted_counts[1:]) if count > min_cluster_size
    ]
    return largest_cluster_indices, other_clusters_indices


def get_floor_indices_ransac(scene_pcd, distance_threshold=0.06, ransac_n=3, num_iterations=3000, seed=0):
    """RANSAC plane fit -> indices of the dominant (floor) plane.

    RANSAC is randomized; seeding Open3D's global RNG makes floor removal (and thus the
    whole pipeline) deterministic and reproducible run-to-run.
    """
    o3d.utility.random.seed(seed)
    _, inliers = scene_pcd.segment_plane(distance_threshold=distance_threshold,
                                         ransac_n=ransac_n, num_iterations=num_iterations)
    return np.array(inliers)


def remove_floor_instances(instances_indices, floor_indices, threshold=0.5):
    """Delete instances that mostly overlap the floor; else subtract floor points."""
    floor_instances_indices = []
    for i, indices in enumerate(instances_indices):
        overlap = np.intersect1d(indices, floor_indices)
        overlap_ratio = len(overlap) / len(indices) if len(indices) > 0 else 0
        if overlap_ratio >= threshold:
            floor_instances_indices.append(indices)
            instances_indices[i] = []
        else:
            instances_indices[i] = np.setdiff1d(indices, floor_indices)
    return instances_indices, floor_instances_indices


def compute_z_score(instances_indices, intensity_dict, weight=[1, 2]):
    """Rank instances by a weighted z-score of (small size, high intensity)."""
    size_list = [len(instances_indices[i]) for i in range(len(instances_indices))]
    intensity_list = [intensity_dict[i] for i in range(len(instances_indices))]

    mean_size, std_size = np.mean(size_list), np.std(size_list)
    mean_intensity, std_intensity = np.mean(intensity_list), np.std(intensity_list)

    def z(value, mean, std):
        return (value - mean) / std if std > 0 else 0

    normalized_size = [-z(s, mean_size, std_size) for s in size_list]
    normalized_intensity = [z(v, mean_intensity, std_intensity) for v in intensity_list]

    w_size, w_intensity = weight[0], weight[1]
    return [(w_size * normalized_size[i] + w_intensity * normalized_intensity[i]) / (w_size + w_intensity)
            for i in range(len(size_list))]
