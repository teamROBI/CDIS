import os
from os.path import join
import glob
import torch
# import cProfile, pstats

from utils.util import (
    count_files, num_to_natural
)
from utils.util_3d import (
    build_scene_point_cloud
)
from CDIS.matching_2d import (
    match_2d, load_2d_matching_result
)
from CDIS.merging_3d import (
    merge_3d
)


def seg_inst(cfg, scene_name, mask_generator):
    ### Data preparation
    # Select images used for the segmentation process
    color_names = sorted(os.listdir(join(cfg.data.scans_2d_path, scene_name, 'color')), key=lambda a: int(os.path.basename(a).split('.')[0]))
    # color_names = color_names[::cfg.matching_2d.image_iter] #Iterate every 10 images
    print(f"[INFO] 2D merging stage, iterate {len(color_names)} images in every {cfg.matching_2d.image_iter} iter")

    # Build dense pcd from RGB-D data
    total_images_2D = len(color_names) / cfg.matching_2d.image_iter
    if cfg.depth_type == "pc_depth" or cfg.depth_type == "sup_depth":
        print(f"[INFO] Using {cfg.depth_type} image.")
        count_existing_depth_pc = count_files(os.path.join(cfg.data.depths.synthetic_depth_path, scene_name))
        if count_existing_depth_pc >= total_images_2D:
            print("[INFO] Use saved pc depth", count_existing_depth_pc, ">=", total_images_2D)
            dense_scene_pcd = None
        else:
            print("[INFO] Build new 3d scene for pc depth", count_existing_depth_pc, "<", total_images_2D)
            dense_scene_pcd = build_scene_point_cloud(cfg, color_names, scene_name, cfg.debug.visualize_dense_pcd)
    else:
        print("[INFO] Using raw depth image.")
        dense_scene_pcd = None
    
    ### 2D matching part
    matching_2d_cache_dir = os.path.join(cfg.exp.matching_2d_output, scene_name)
    if glob.glob(f"{matching_2d_cache_dir}*"):
        print("[INFO] Loaded 2D matching result")
        inst_match_list, intensity_dict = load_2d_matching_result(matching_2d_cache_dir)
    else:
        inst_match_list, intensity_dict = match_2d(cfg, scene_name, color_names, mask_generator, dense_scene_pcd)
    
    ### 3D merging part
    scene_inst_3d_pred = merge_3d(cfg, scene_name, color_names, inst_match_list, intensity_dict)
    
        
    os.makedirs(cfg.exp.save_path, exist_ok=True)
    torch.save(num_to_natural(scene_inst_3d_pred), join(cfg.exp.save_path, scene_name + ".pth"))
    

    
