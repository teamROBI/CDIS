import open3d as o3d
import numpy as np

# Load mesh
mesh = o3d.io.read_triangle_mesh("data/scannetv2/input/org_ply/org_val_ply/scene0011_00_vh_clean_2.ply")
mesh.compute_vertex_normals()

# Load intrinsics
K = np.loadtxt("data/scannetv2/input/scannetv2_images/val/scene0011_00/intrinsics/intrinsic_color.txt")
fx, fy = K[0, 0], K[1, 1]
cx, cy = K[0, 2], K[1, 2]
width, height = round(cx*2), round(cy*2)

# Load pose
pose = np.loadtxt("data/scannetv2/input/scannetv2_images/val/scene0011_00/pose/0.txt")
extrinsic = np.linalg.inv(pose)

# Set up intrinsic
intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

# Set up renderer
renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
material = o3d.visualization.rendering.MaterialRecord()
material.shader = "defaultLit"
renderer.scene.add_geometry("mesh", mesh, material)
renderer.scene.set_background([0, 0, 0, 1])  # black

# Set up camera
renderer.setup_camera(intrinsic, extrinsic)

# Render depth
depth = renderer.render_to_depth_image(z_in_view_space=True)
o3d.io.write_image("depth_000000.png", depth)

# Convert to NumPy for post-processing
depth_np = np.asarray(depth)

# Convert to uint16 in millimeters (ScanNet-style)
depth_mm = (depth_np * 1000).astype(np.uint16)

# Save using Open3D (or imageio/PIL)
depth_img = o3d.geometry.Image(depth_mm)
o3d.io.write_image("depth_000000_uint.png", depth_img)